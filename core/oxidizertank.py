import math
import CoolProp
from CoolProp.CoolProp import PropsSI

R = 8.3143  # 보편 기체 상수 [J/(mol·K)]


class OxidizerTank:
    def __init__(self, configs):

        self.radius = configs["RADIUS"]
        self.height = configs["HEIGHT"]

        self.bulkhead_A = self.radius ** 2 * math.pi
        self.V = self.bulkhead_A * self.height

        self.V = configs["V"]  # TODO: 나중에 삭제

        self.pressurant      = configs["PRESSURANT"]
        self.oxidizer_liquid = configs["OXIDIZER_LIQUID"]
        self.oxidizer_vapor  = configs["OXIDIZER_VAPOR"]
        self.material        = configs["MATERIAL"]

        self.tankT = configs["T"]
        self.tankP = configs["P"]
        self.m     = configs["M"]

    def tank_cp(self):
        """탱크 재질(알루미늄) 정압 비열 [J/(kg·K)]"""
        return (4.8 + 0.00322 * self.tankT) * 155.239

    def dPsatdT(self, dT=1e-5):
        """포화압력의 온도 미분 dPsat/dT [Pa/K] — 임계점 초과 시 0 반환"""
        T    = self.tankT
        name = self.oxidizer_liquid.name
        T_crit = PropsSI("Tcrit", name)
        P_crit = PropsSI("Pcrit", name)

        # 임계점 근방에서는 수치 미분 불안정 → 0으로 처리
        if T + dT >= T_crit or self.tankP >= P_crit:
            return 0.0

        P_plus  = PropsSI("P", "T", T + dT, "Q", 1, name)
        P_minus = PropsSI("P", "T", T - dT, "Q", 1, name)
        return (P_plus - P_minus) / (2 * dT)

    # ------------------------------------------------------------------
    #  탱크 상태 업데이트
    #
    #  [단위 정책] matter.py CoolProp 호출 기준:
    #    h     = PropsSI("H",...)        J/kg      → × MW 필요
    #    Cv    = PropsSI("CVMOLAR",...) J/(mol·K) → 변환 불필요
    #    v_hat = 1/PropsSI("Dmolar",...) m³/mol    → 변환 불필요
    # ------------------------------------------------------------------
    def tank_state_update(self, n_dot_drain, dt):

        name   = self.oxidizer_liquid.name
        MW     = self.oxidizer_liquid.MW    # [kg/mol]
        T      = self.tankT
        P_crit = PropsSI("Pcrit", name)
        T_crit = PropsSI("Tcrit", name)
        supercritical = (T >= T_crit or self.tankP >= P_crit)

        # ── h: 포화선(Q=0/Q=1)으로 직접 계산 ────────────────────────────
        #  props_with_tp()는 단상 H를 저장하므로 액/기 h가 같아짐
        #  → h_vap - h_liq = 0 → 잠열 항 소멸 오류
        #  초임계 시: T를 임계온도 직하로 클램핑 (잠열 → 0 수렴)
        T_query = T if not supercritical else (T_crit - 0.01)
        h_liq = PropsSI("H", "T", T_query, "Q", 0, name) * MW   # [J/mol]
        h_vap = PropsSI("H", "T", T_query, "Q", 1, name) * MW   # [J/mol]

        # ── Psat: 초임계이면 P_crit으로 대체 ─────────────────────────────
        Psat = self.oxidizer_liquid.Psat
        if Psat is None:
            Psat = P_crit

        # ── Cv, v_hat은 이미 per-mol ─────────────────────────────────────
        Cv_liq    = self.oxidizer_liquid.Cv     # J/(mol·K)
        Cv_vap    = self.oxidizer_vapor.Cv      # J/(mol·K)
        v_hat_liq = self.oxidizer_liquid.v_hat  # m³/mol
        Cv_press  = self.pressurant.Cv          # J/(mol·K)

        # ── f: 논문 공식은 음수(탱크 감소 방향) 기준. abs() 제거 ────────────
        # abs()를 씌우면 Z, W, dT_dt 부호 전부 반전 → 온도 상승 오류 발생
        f = n_dot_drain   # [mol/s], 음수 = 탱크에서 나가는 방향

        # ── dPsat/dT [Pa/K] ──────────────────────────────────────────────
        dPsat_dT = self.dPsatdT(dt)

        # ── 연립방정식 계수 ───────────────────────────────────────────────
        a_tank  = self.m * self.tank_cp()
        a_liq   = self.oxidizer_liquid.n * Cv_liq
        a_vap   = self.oxidizer_vapor.n  * Cv_vap
        a_press = self.pressurant.n      * Cv_press
        a = a_tank + a_liq + a_vap + a_press

        b       = self.tankP * v_hat_liq                              # [J/mol]
        e       = R * T - (h_vap - h_liq)                            # [J/mol]
        j       = -Psat * v_hat_liq                                   # [J/mol]
        V_gas   = max(self.V - self.oxidizer_liquid.n * v_hat_liq, 1e-9)
        k       = V_gas * dPsat_dT                                    # [J/K]
        m_coeff = R * T                                               # [J/mol]
        q       = R * self.oxidizer_vapor.n                           # [J/K]

        # ── [DEBUG] 계수 검증 ─────────────────────────────────────────────
        _errs = []
        if not (-30000 < e < 0):
            _errs.append(f"e={e:.2f} J/mol  (정상: -30000~0, 이상적: -14000~-16000)")
        if v_hat_liq <= 0:
            _errs.append(f"v_hat_liq={v_hat_liq:.4e} m³/mol  (정상: 양수)")
        if not (0 < V_gas < self.V * 1.01):
            _errs.append(f"V_gas={V_gas:.4e} m³  (정상: 0 ~ {self.V:.4e})")
        if a <= 0:
            _errs.append(f"a={a:.2f} J/K  (정상: 양수)")
        if _errs:
            raise ValueError(
                f"[TANK DEBUG] 계수 이상 @ T={T:.4f}K P={self.tankP:.2f}Pa\n"
                + "\n".join(f"  !! {err}" for err in _errs)
                + f"\n  h_liq={h_liq:.2f} h_vap={h_vap:.2f} J/mol"
                + f"\n  Cv_liq={Cv_liq:.4f} Cv_vap={Cv_vap:.4f} J/(mol·K)"
                + f"\n  v_hat_liq={v_hat_liq:.4e} m³/mol"
                + f"\n  n_liq={self.oxidizer_liquid.n:.4f} n_vap={self.oxidizer_vapor.n:.4f} mol"
                + f"\n  V_tank={self.V:.4e} V_gas={V_gas:.4e} m³"
                + f"\n  Psat={Psat:.2f} Pa  dPsat_dT={dPsat_dT:.2f} Pa/K"
                + f"\n  a={a:.2f} b={b:.4f} e={e:.2f} j={j:.4f} k={k:.4f}"
                + f"\n  m_coeff={m_coeff:.4f} q={q:.6f} f={f:.6f}"
                + f"\n  supercritical={supercritical}"
            )

        # ── 연립방정식 풀기 ───────────────────────────────────────────────
        denom_Z = a * (m_coeff + j) + (q - k) * (e - b)
        if abs(denom_Z) < 1e-20:
            raise ValueError(
                f"[TANK DEBUG] denom_Z≈0 @ T={T:.4f}K  denom_Z={denom_Z:.4e}"
            )

        d_oxidizer_vapor_n_dt = -f * (-j * a + (q - k) * b) / denom_Z
        Z = d_oxidizer_vapor_n_dt

        denom_W = -j * a + (q - k) * b
        if abs(denom_W) < 1e-20:
            raise ValueError(
                f"[TANK DEBUG] denom_W≈0 @ T={T:.4f}K  denom_W={denom_W:.4e}"
            )

        d_oxidizer_liquid_n_dt = -Z * (m_coeff * a + (q - k) * e) / denom_W
        W = d_oxidizer_liquid_n_dt

        dT_dt = (b * W + e * Z) / a

        # ── 매 스텝 핵심 로그 ─────────────────────────────────────────────
        print(
            f"[TANK] T={T:.4f}K | P={self.tankP/1e6:.4f}MPa | "
            f"e={e:.1f}J/mol | dT/dt={dT_dt:.5f}K/s | "
            f"Z={Z:.4e} W={W:.4e} f={f:.4e} mol/s"
        )

        # ── 몰수 업데이트 ─────────────────────────────────────────────────
        self.oxidizer_liquid.n = max(0.0, self.oxidizer_liquid.n + W * dt)
        self.oxidizer_vapor.n  = max(0.0, self.oxidizer_vapor.n  + Z * dt)

        self.oxidizer_liquid.update_props("N", self.oxidizer_liquid.n)
        self.oxidizer_vapor.update_props("N",  self.oxidizer_vapor.n)

        # ── 온도 업데이트 ─────────────────────────────────────────────────
        self.tankT += dT_dt * dt

        self.oxidizer_liquid.update_props("T", self.tankT)
        self.oxidizer_vapor.update_props("T",  self.tankT)

        # ── 압력 업데이트 ─────────────────────────────────────────────────
        V_gas_new = max(self.V - self.oxidizer_liquid.n * v_hat_liq, 1e-9)
        self.tankP = (self.pressurant.n + self.oxidizer_vapor.n) * R * self.tankT / V_gas_new

        #self.tankP = 5500000
        self.oxidizer_liquid.update_props("P", self.tankP)
        self.oxidizer_vapor.update_props("P",  self.tankP)

        self.oxidizer_liquid.props_with_tp(self.tankT, self.tankP)
        self.oxidizer_vapor.props_with_tp(self.tankT,  self.tankP)
        self.pressurant.props_with_tp(self.tankT,      self.tankP)

    # ------------------------------------------------------------------
    #  기체(증기) 블로우다운 갱신 — 액체 소진 이후 전용 경로
    #
    #  tank_state_update()(액상-기상 평형 ODE)는 액체가 남아있는 동안에만
    #  호출되며, 이 메서드와는 완전히 분리되어 있다 — 액체 계산 로직은
    #  건드리지 않는다.
    #
    #  탱크에 남은 N2O 증기가 등엔트로피적으로(단열·가역) 팽창하며 오리피스로
    #  빠져나간다고 가정한다: 현재 비엔트로피를 고정한 채 밀도 감소분만큼
    #  CoolProp의 (밀도, 엔트로피) 플래시로 새 온도·압력을 구한다.
    #  가압제는 소모되지 않고 고정 몰수로 남아 돌턴 분압으로 기여한다.
    # ------------------------------------------------------------------
    def vapor_blowdown_update(self, mdot_out, dt):
        name = self.oxidizer_vapor.name
        MW   = self.oxidizer_vapor.MW          # [kg/mol]
        n_v  = self.oxidizer_vapor.n           # [mol]
        V_gas = max(self.V, 1e-9)              # 액체가 없으므로 탱크 전체가 기체 공간

        if n_v <= 1e-9:
            # 증기까지 다 떨어짐 — 남은 압력은 가압제 분압뿐
            self.oxidizer_vapor.n  = 0.0
            self.oxidizer_liquid.n = 0.0
            self.tankP = self.pressurant.n * R * self.tankT / V_gas if self.pressurant.n > 0 else 0.0
            print(f"[VAPOR BLOWDOWN] 증기 소진 | P={self.tankP/1e6:.4f}MPa")
            return

        m_v  = n_v * MW                         # [kg]
        rho  = m_v / V_gas                      # [kg/m^3]
        T    = self.tankT

        dm    = min(max(mdot_out, 0.0) * dt, m_v)
        m_new = max(m_v - dm, 1e-9)
        rho_new = m_new / V_gas

        # [수정] 삼중점(N2O 182.33K) 아래에서는 CoolProp (D,S) 플래시가 예외 없이
        # 상태방정식을 외삽한 엉터리 값(분압 ~1 → ~15 bar)을 돌려줘 탱크 압력이
        # 튀고 유량·온도가 폭주했다. 삼중점 아래 결과는 버리고, 온도는 삼중점에
        # 고정(고체 석출 잠열이 온도를 붙잡음), N2O 분압은 그 포화압으로 제한한다.
        T_trip = PropsSI('Ttriple', name)
        try:
            s1 = PropsSI('S', 'T', T, 'D', rho, name)   # [J/(kg·K)] — 등엔트로피 기준
            T_new  = PropsSI('T', 'D', rho_new, 'S', s1, name)
            Pv_new = PropsSI('P', 'D', rho_new, 'S', s1, name)   # N2O 증기 분압
        except Exception:
            # 임계점 근방 등 (D,S) 플래시가 수렴하지 않는 경우:
            # 이상기체 등엔트로피 관계식(T·rho^(1-γ)=const)로 대체
            try:
                Cp = PropsSI('CPMASS', 'T', T, 'D', rho, name)
                Cv = PropsSI('CVMASS', 'T', T, 'D', rho, name)
                gamma_v = Cp / Cv if Cv > 0 else 1.3
            except Exception:
                gamma_v = 1.3
            # [수정] tankP에는 가압제 분압이 포함돼 있어, 그대로 쓰면 아래에서
            # P_press를 또 더할 때 가압제 분압이 이중으로 들어간다. N2O 분압만 사용.
            P_press_old = self.pressurant.n * R * T / V_gas if self.pressurant.n > 0 else 0.0
            T_new  = T * (rho_new / rho) ** (gamma_v - 1.0)
            Pv_new = max(self.tankP - P_press_old, 0.0) * (rho_new / rho) ** gamma_v

        if T_new < T_trip:
            T_new  = T_trip
            Pv_new = min(rho_new / MW * R * T_trip, PropsSI('P', 'T', T_trip, 'Q', 1, name))

        # [수정] 다 빠졌을 때 m_new는 하한 1e-9 kg인데, 몰수로는 2.3e-8 mol이라
        # 소진 판정(n <= 1e-9 mol)에 걸리지 않아 He 압력만으로 가짜 유량이 계속됐다.
        n_v_new = m_new / MW if dm < m_v else 0.0

        # 가압제(있다면) 같은 부피·온도에서 분압으로 기여 (돌턴의 법칙)
        P_press = self.pressurant.n * R * T_new / V_gas if self.pressurant.n > 0 else 0.0

        self.tankT = T_new
        self.tankP = Pv_new + P_press

        self.oxidizer_liquid.n = 0.0
        self.oxidizer_vapor.n  = n_v_new
        self.oxidizer_vapor.T   = T_new
        self.oxidizer_vapor.P   = self.tankP
        self.oxidizer_vapor.rho = rho_new
        if rho_new > 0:
            self.oxidizer_vapor.v_hat = MW / rho_new
        self.oxidizer_vapor.V = self.oxidizer_vapor.v_hat * n_v_new

        self.pressurant.props_with_tp(T_new, self.tankP)

        print(
            f"[VAPOR BLOWDOWN] T={T_new:.4f}K | P={self.tankP/1e6:.4f}MPa | "
            f"n_vapor={n_v_new:.4f}mol | mdot_out={mdot_out:.5f}kg/s"
        )
