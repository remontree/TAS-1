# 필요한 라이브러리 임포트
import math
from dataclasses import dataclass
import CoolProp
from CoolProp.CoolProp import PropsSI
from rocketcea.cea_obj import CEA_Obj, add_new_fuel

R = 8.3143  # 보편 기체 상수

# 인젝터 클래스

class Injector:
  def __init__(self, config):
    self.A = config["A"] # 인젝터 면적 [m^2]
    self.Cd = config["CD"] # 유량제어계수
    self.N = config["N"]   # 오리피스 개수
    self.L = config["L"] # 오리피스 길이
    self.D = math.sqrt(self.A/3.14)*2



  def get_ryo_meu_Pv_NitrousOxide(self, P0, T, phase_hint="liquid"):
    """
    입력:
      - P0 : 압력 [Pa]
      - T  : 온도 [K]
      - phase_hint : "liquid" 또는 "gas" (기본 liquid)

    출력:
      - ryo : 밀도 [kg/m^3]
      - meu : 점도 [Pa*s]

    동작:
      1) rho는 CoolProp로 계산
      2) mu는 CoolProp로 시도
         - N2O 점도 모델 미지원이면 Chung류(저밀도 가스 점도) fallback 사용
      3) 필요 시 Psat(Pv)는 내부적으로 계산(현재 반환에는 미포함)
    """

    fluid = "NitrousOxide"

    # Psat (참고용: 필요하면 Cd 모델의 Pv로 사용 가능)
    Pv = None
    try:
        Pv = PropsSI("P", "T", T, "Q", 0, fluid)
    except Exception:
        Pv = None

    # 1) 밀도 rho
    if phase_hint == "liquid" and Pv is not None:
        # 탱크가 포화 근처 액상이라고 가정하면 포화액 밀도 사용
        ryo = PropsSI("D", "T", T, "Q", 0, fluid)
    elif phase_hint == "gas" and Pv is not None:
        ryo = PropsSI("D", "T", T, "Q", 1, fluid)
    else:
        # 일반적으로는 (P,T)
        ryo = PropsSI("D", "P", P0, "T", T, fluid)

    # 2) 점도 mu: CoolProp 우선
    try:
        meu = PropsSI("V", "P", P0, "T", T, fluid)  # Pa*s
        return ryo, meu  # (원하신 호출 형태 유지)

    except Exception:
        # 3) Fallback: Chung/Chapman-Enskog 기반 저밀도 가스 점도(temperature-dependent correlation)
        #    임계 몰부피 Vc는 PropsSI("Vcrit")가 아니라 rho_crit로부터 계산

        # 분자량 [kg/mol]
        M = PropsSI("M", fluid)
        M_gmol = M * 1000.0  # g/mol

        # 임계온도, 이심인자
        Tc = PropsSI("Tcrit", fluid)
        omega = PropsSI("acentric", fluid)

        # 임계밀도: 키가 빌드별로 달라서 2단계로 시도
        rho_c = None
        for key in ("rhocrit", "rhomass_critical"):
            try:
                rho_c = PropsSI(key, fluid)  # kg/m^3
                break
            except Exception:
                pass
        if rho_c is None:
            raise ValueError(
                "CoolProp에서 임계밀도(rhocrit)를 읽을 수 없습니다. "
                "Chung fallback에 필요합니다. (REFPROP 사용 또는 임계밀도 상수 입력 필요)"
            )

        # 임계 몰부피 [m^3/mol] = M / rho_c
        Vc = M / rho_c

        # dipole moment [Debye] (N2O ~ 0.166 D)
        mu_D = 0.166

        # sigma [Angstrom] 계산을 위해 Vc를 cm^3/mol로 변환
        Vc_cm3mol = Vc * 1e6
        sigma_A = 0.809 * (Vc_cm3mol ** (1.0 / 3.0))

        # eps/k [K]
        eps_over_k = Tc / (1.2593 + 0.8440 * omega - 0.0780 * (omega ** 2))

        # reduced temperature
        T_star = T / eps_over_k

        # collision integral (Neufeld type)
        Omega = (
            1.16145 * (T_star ** -0.14874)
            + 0.52487 * math.exp(-0.77320 * T_star)
            + 2.16178 * math.exp(-2.43787 * T_star)
        )

        # polarity correction
        mu_r = 131.3 * mu_D / math.sqrt(max(Vc_cm3mol * Tc, 1e-30))
        Fc = 1.0 - 0.2756 * omega + 0.059035 * (mu_r ** 4)

        # mu0 in microPoise
        mu_microP = 40.785 * Fc * math.sqrt(M_gmol * T) / (max(sigma_A, 1e-30) ** 2 * max(Omega, 1e-30))

        # microPoise -> Pa*s (1 microPoise = 1e-7 Pa*s)
        meu = mu_microP * 1e-7

        return ryo, meu

  # 레이놀즈 수 구하기
  def Re(self, ryo, V, meu):
    return ryo * V * self.D / meu

  # 인젝터 내 항력 계수 구하기
  def Cd_fluid(self, P0, Pv, Pc, Re_val):
      L = self.L
      D = self.D
      A0=self.A
      Ai=self.A
      if P0 <= Pc:
          return 0.0

      K = (P0 - Pv) / max(P0 - Pc, 1e-9)
      Kcav = -1000 / max(Re_val, 1e-9)
      Kflip = 1 + 1 / ((1 + L / D) * ((1 + 2000) / max(Re_val, 1e-9)))

      if K > Kcav and K >= Kflip:
          C_Du = 0.827 - 0.0085 * L / D
          C_D = (1 / C_Du + 20 * ((1 + 2.25 * L / D) / max(Re_val, 1e-9))) ** -1
      elif K <= Kcav and K >= Kflip:
          beta = Ai / A0
          C_C = 0.62 + 0.38 * beta ** 3
          C_D = C_C * math.sqrt(max(K, 0.0))
      else:
          C_D = 0.45

      return C_D

  # 비압축성 유동 모델
  def IncompressibleMassFlow(self, Cd, rho, P1, P2):
      dP = P1 - P2
      if dP <= 0:
          return 0.0
      return self.A * Cd * math.sqrt(2 * rho * dP)

  # 동차 유동 모델
  def HomogeneousMassFlow(self, Cd, rho, h1, h2):
      if h1 <= h2:
          return 0.0
      return self.A * Cd * rho * math.sqrt(2 * (h1 - h2))

  """"
  NHNE 모델
  파라메터:
    - Ac: 오리피스 면적
    - Cd: 항력 계수
    - P1: 오리피스 상류 압력
    - Pv1: 상류 유체 증기압
    - ryo1: 상류층 유체 밀도
    - h1: 오리피스 윗 부분 엔탈피
    - P2: 오리피스 하류 압력
    - ryo2: 하류층 유체 밀도
    - h2: 오리피스 아랫 부분 엔탈피
  """

  """def n_drain(self, Cd, P1, Pv1, ryo1, h1, P2, ryo2, h2):
      dP = P1 - P2
      den = Pv1 - P2
      if dP <= 0 or den <= 0:
          return 0.0

      ratio = max(dP / den, 0.0)
      k = math.sqrt(ratio)

      m_inc = self.IncompressibleMassFlow(Cd, ryo1, P1, P2)
      m_hem = self.HomogeneousMassFlow(Cd, ryo2, h1, h2)

      return -((m_inc / (1 + k)) + (m_hem * (1 - 1 / (1 + k))))*self.N"""

  # NHNE 모델로 변경 예정
  def n_drain(self, tank_pressure, chamber_pressure, MW, v_hat):
    # 챔버 압력이 탱크 압력보다 높거나 같으면 유량은 0
    if tank_pressure <= chamber_pressure:
      return 0.0
    return -self.Cd * self.A * self.N * math.sqrt((2*(tank_pressure-chamber_pressure))/(MW*v_hat))

  # ──────────────────────────────────────────────────────────────────────
  # 기체(증기) 블로우다운 유량 모델
  #
  # 액체 산화제가 모두 소진된 뒤, 탱크에 남은 N2O 증기가 오리피스를 통해
  # 빠져나가는 질유량. 액체 NHNE 모델(get_ryo_meu_Pv_NitrousOxide 등)과는
  # 완전히 별개의 경로 — 압축성(등엔트로피) 오리피스 유동의 초크/비초크
  # 표준식을 그대로 사용한다.
  # ──────────────────────────────────────────────────────────────────────
  def get_vapor_mdot(self, P_up, T_up, P_down, fluid="NitrousOxide"):
    """
    입력:
      - P_up   : 상류(탱크) 압력 [Pa]
      - T_up   : 상류(탱크) 온도 [K]
      - P_down : 하류(연소실) 압력 [Pa]
      - fluid  : CoolProp 유체명

    출력:
      - mdot : 오리피스를 통한 질유량 [kg/s] (전체 오리피스 N개 합산)
    """
    if P_up <= P_down or P_up <= 0 or T_up <= 0:
      return 0.0

    try:
      Cp = PropsSI("CPMOLAR", "T", T_up, "P", P_up, fluid)   # J/(mol·K)
      Cv = PropsSI("CVMOLAR", "T", T_up, "P", P_up, fluid)   # J/(mol·K)
      MW = PropsSI("M", fluid)                                # kg/mol
    except Exception:
      return 0.0

    if Cv <= 0 or MW <= 0:
      return 0.0

    gamma = Cp / Cv
    R_specific = R / MW  # J/(kg·K)

    crit_ratio = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    pr = P_down / P_up
    A_total = self.A * self.N

    if pr <= crit_ratio:
      # 초크(choked) 유동 — 하류 압력과 무관, 상류 조건에만 의존
      mdot = self.Cd * A_total * P_up * math.sqrt(
          gamma / (R_specific * T_up) *
          (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))
      )
    else:
      # 비초크(subsonic) 유동
      term = max(pr ** (2.0 / gamma) - pr ** ((gamma + 1.0) / gamma), 0.0)
      mdot = self.Cd * A_total * P_up * math.sqrt(
          (2.0 * gamma / (R_specific * T_up * (gamma - 1.0))) * term
      )

    return max(mdot, 0.0)
