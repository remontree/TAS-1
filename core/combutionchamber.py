# 필요한 라이브러리 임포트
import math
from dataclasses import dataclass
import CoolProp
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq
from rocketcea.cea_obj import CEA_Obj, add_new_fuel

R = 8.3143  # 보편 기체 상수

# 연소실 클래스

class CombutionChamber:
  def __init__(self, config):
    self.chamberT = config["T"]
    self.chamberP = config["P"]
    self.fuel = config["FUEL"]
    self.cstar_eta = config["CSTAR_ETA"]
    self.m = 0

    # ── C* 점화 상승 지연(startup transient) 상수 ──────────────────────
    # Cstar(t) = Cstar_steady * (1 - deficit * exp(-t / tau))
    # tau     : 시간 지연 상수 [s] (클수록 정상상태 도달이 느려짐)
    # deficit : t=0 시점의 초기 결손 비율 (0~1)
    self.cstar_tau     = config.get("CSTAR_TAU", 0.35)
    self.cstar_deficit = config.get("CSTAR_STARTUP_DEFICIT", 0.9)

    # ── 챔버 케이싱 반지름/길이 [m] — 실제 자유부피(V_free) 계산용 ──────
    # 없으면(구버전 config 등) free_volume()이 None을 반환하고
    # combustion_chamber_pressure()가 기존 누적질량 m_gas 방식으로 폴백한다.
    self.chamber_radius = config.get("CHAMBER_RADIUS")
    self.chamber_length = config.get("CHAMBER_LENGTH")

  def Cstar(self, k, R_univ, MW_mix, t):
    R_spec = R_univ / MW_mix  # J/(kg·K)
    term1 = (R_spec * self.chamberT / k) ** 0.5
    term2 = ((k + 1) / 2) ** ((k + 1) / (2 * (k - 1)))
    return term1 * term2 *  self.cstar_eta *  (1 - self.cstar_deficit*math.exp(-t/self.cstar_tau))

  def free_volume(self):
    """
    현재 시점 챔버 자유부피(가스가 실제로 차지하는 부피) [m^3].
    = 챔버 케이싱 내부 부피 - 남은 고체 연료 부피 (그레인이 타들어가며 증가).
    CHAMBER_RADIUS/CHAMBER_LENGTH가 없으면 None을 반환한다.
    """
    if self.chamber_radius is None or self.chamber_length is None:
        return None

    fuel = self.fuel
    if getattr(fuel, "grain_type", "PORT") in ("FIN", "FINOCYL"):
        port_area = math.pi * fuel.inner_radius ** 2
    else:
        port_area = math.pi * fuel.inner_radius ** 2 * max(getattr(fuel, "port_num", 1), 1)

    solid_vol   = max(math.pi * fuel.outer_radius ** 2 * fuel.length - port_area * fuel.length, 0.0)
    chamber_vol = math.pi * self.chamber_radius ** 2 * self.chamber_length
    return max(chamber_vol - solid_vol, 1e-9)

  def combustion_chamber_pressure(self, P_prev, m_gas, mox_dot, mf_dot, mex_dot,
                                rho_g, r_dot, A_b, dt,
                                Pa, At, c_star, gamma, inflow_fn=None):
    # 1) 초크 판정
    # Pb/Pc <= (2/(g+1))^(g/(g-1)) 이면 choked
    Pc = max(P_prev, 1.0)
    Pb = max(Pa, 1.0)
    crit = (2.0/(gamma+1.0)) ** (gamma/(gamma-1.0))
    choked = (Pb / Pc) <= crit

    #mex = max(0.0, Pc * At / max(c_star, 1e-9))
    mex = mex_dot

    # 3) 질량 source term
    mass_source = (max(mox_dot, 0.0) + max(mf_dot, 0.0) - mex)

    # 4) 하이브리드 최소 반영: dP/P = dm/m - dV/V
    #
    # [수정] 예전에는 m_gas로 "점화 시점부터 누적된 순유입질량"을 그대로 썼다.
    # m_gas(0)=0에서 출발해 dm/m 항의 분모가 "지금까지 쌓인 질량"이다 보니,
    # 실제 챔버 부피/유량과 무관하게 항상 수백 ms에 걸쳐 서서히 채워지는
    # 인위적인 지연이 생겼다 (m_gas가 유량에 비례해 커질 때까지 a=source/m_gas
    # 가 크게 유지되다가 서서히 줄어드는, 순전히 수식 구조상의 아티팩트).
    #
    # 대신 "고정된 실제 자유부피(V_free) × 현재 연소가스 밀도(rho_g)"를 그
    # 시점의 챔버 내 가스 질량으로 쓰면, 챔버가 실제로 채워지는 물리적 시간
    # 상수 τ ≈ (rho_g·V_free)/mass_source 로 수렴한다 — 자유부피가 작고
    # 유량이 크면(전형적인 하이브리드 모터) 수십~수백 ms 안에 거의 수직으로
    # 압력이 올라가는 실측 거동을 재현한다.
    V_free = self.free_volume()

    if V_free is not None and rho_g is not None and rho_g > 0.0:
        # V_free가 작을 때는 한 스텝(dt) 안에서도 압력이 몇 배씩 뛸 수 있다.
        # rho_g(밀도)와 mex(노즐 초크 유량 = At·P/c*, P에 비례)는 이번 스텝
        # 시작 시점(구 Pc 기준) 값 그대로라, 그 상태로 exp(a·dt)를 한 번에
        # 적분하면 "압력이 오를수록 밀도도 오르고 배출량도 늘어난다"는 되먹임이
        # 반영되지 않아 압력이 과도하게 치솟는다(N2O 임계압 초과 등).
        # Fuel 레벨셋의 CFL 서브스텝과 같은 방식으로, 서브스텝마다 밀도와
        # 초크 배출유량을 현재 P에 비례해 갱신하며 적분해 안정화한다.
        # (이 되먹임 덕에 평형 상태는 자연스럽게 Pc_eq = inflow·c*/At 로
        #  수렴한다 — 하이브리드 로켓 정상상태 챔버압 공식과 일치.)
        vdot   = max(0.0, A_b) * max(0.0, r_dot)
        inflow = max(mox_dot, 0.0) + max(mf_dot, 0.0)
        mex0   = max(mex_dot, 0.0)
        a0 = ((inflow - mex0) / max(rho_g * V_free, 1e-9)) - (vdot / V_free)

        if inflow_fn is not None:
            # [수정] 유입 유량을 압력의 함수 inflow_fn(P)로 받아 후진 오일러로
            # 암시적으로 푼다:  V·(ρ_new-ρ)/dt = inflow(P_new) - mex(P_new) - ρ_new·Vdot
            # (ρ, mex ∝ P). 인젝터 차압이 작아 유량이 압력에 매우 민감해도
            # 무조건 안정하고 Pc가 탱크압을 넘지 않으므로, 아래의 명시적 서브스텝과
            # 스텝 간 완화(blend)가 필요 없다.
            def F(P):
                r = P / Pc
                return (rho_g * r * (V_free + dt * vdot) - rho_g * V_free
                        - dt * (inflow_fn(P) - mex0 * r))
            lo, hi = 1.0, max(Pc, 1e5)
            while F(hi) < 0.0 and hi < 1e9:
                hi *= 2.0
            if F(lo) >= 0.0:
                return lo
            return brentq(F, lo, hi, xtol=1.0, rtol=1e-10)

        MAX_STEP_GROWTH = 0.3  # 서브스텝당 |a·dt_sub| 상한
        n_sub  = min(max(1, math.ceil(abs(a0) * dt / MAX_STEP_GROWTH)), 2000)
        dt_sub = dt / n_sub

        P = Pc
        for _ in range(n_sub):
            ratio   = (P / Pc) if Pc > 0 else 1.0
            rho_now = rho_g * ratio                 # 밀도 ≈ 압력에 비례
            mex_now = mex0 * ratio                   # 초크유량 At·P/c* ≈ 압력에 비례
            m_safe  = max(rho_now * V_free, 1e-9)
            a       = ((inflow - mex_now) / m_safe) - (vdot / V_free)
            x       = max(min(a * dt_sub, MAX_STEP_GROWTH), -MAX_STEP_GROWTH)
            P       = max(1.0, P * math.exp(x))
        P_target = P

        # ── 아우터 스텝 간 완화(relaxation) ──────────────────────────────
        # mox_dot(인젝터 유량)은 "직전 스텝 챔버압" 기준으로 한 스텝 지연되어
        # 계산된다. 챔버 충전 자체가 이제 매우 빨라졌기 때문에, 이번 스텝에서
        # (지연된) mox_dot 기준 평형까지 한번에 다 가버리면, 다음 스텝엔 그
        # 새 압력에서 다시 계산된 mox_dot(인젝터 압력차가 줄어 훨씬 작음) 기준
        # 평형으로 되튀어 스텝마다 압력이 크게 진동하는 불안정이 생긴다
        # (원래의 느린 누적질량 방식은 이 진동을 막아주는 관성 역할을 겸했었다).
        # 물리적으로 타당한 짧은 완화 시간(τ≈3·dt, 수십 ms)에 걸쳐 목표치로
        # 수렴시켜, 빠른 상승은 유지하면서 스텝 간 되튐은 감쇠시킨다.
        blend = 1.0 / 3.0
        P_new = Pc * (P_target / Pc) ** blend if Pc > 0 else P_target
    else:
        # CHAMBER_RADIUS/LENGTH가 없는 구버전 config용 폴백 (기존 누적질량 방식)
        m_safe = max(m_gas, 1e-9)
        vol_term = 0.0
        if rho_g is not None and rho_g > 0.0:
            V = max(m_safe / rho_g, 1e-9)
            Vdot = max(0.0, A_b) * max(0.0, r_dot)
            vol_term = Vdot / V
        a = (mass_source / m_safe) - vol_term
        x = max(min(a * dt, 2.0), -2.0)
        P_new = max(1.0, Pc * math.exp(x))

    return P_new


  def NHNE_rho2_h2(self,
    fluid: str,
    P1: float,     # 상류 압력 [Pa]
    Pv1: float,    # 상류 증기압 [Pa] (판별/확장용, 계산에는 직접 미사용)
    ryo1: float,   # 상류 밀도 [kg/m^3] (참고용)
    h1: float,     # 상류 엔탈피 [J/kg]
    P2: float,     # 하류 압력 [Pa]
    S1: float      # 상류 엔트로피 [J/kg/K]
):
    """
    반환:
      ryo2 : 하류 밀도 [kg/m^3]
      h2   : 하류 엔탈피 [J/kg]
    """

    # ─────────────────────────────
    # 1. P2에서 포화 물성 계산
    # ─────────────────────────────
    sf   = PropsSI("S", "P", P2, "Q", 0, fluid)
    sg   = PropsSI("S", "P", P2, "Q", 1, fluid)
    hf   = PropsSI("H", "P", P2, "Q", 0, fluid)
    hg   = PropsSI("H", "P", P2, "Q", 1, fluid)
    ryo_f = PropsSI("D", "P", P2, "Q", 0, fluid)
    ryo_g = PropsSI("D", "P", P2, "Q", 1, fluid)

    # ─────────────────────────────
    # 2. 등엔트로피 조건으로 건도 계산
    #    S2 = S1
    # ─────────────────────────────
    x2 = (S1 - sf) / (sg - sf)

    # ─────────────────────────────
    # 3. 상태 판별
    # ─────────────────────────────
    if 0.0 <= x2 <= 1.0:
        # ===== 2상 평형 (NHNE / HEM) =====
        h2 = hf + x2 * (hg - hf)
        ryo2 = 1.0 / (x2 / ryo_g + (1.0 - x2) / ryo_f)

    else:
        # ===== 단상 (플래싱 없음 또는 완전 기화) =====
        # 등엔트로피 상태를 직접 계산
        h2 = PropsSI("H", "P", P2, "S", S1, fluid)
        ryo2 = PropsSI("D", "P", P2, "S", S1, fluid)

    return ryo2, h2
