import math
import CoolProp
from CoolProp.CoolProp import PropsSI

R = 8.3143  # 보편 기체 상수

class Matter:
    def __init__(self, Props):

        self.name  = Props["NAME"]
        fluids = CoolProp.__fluids__
        if self.name not in fluids:
            raise ValueError("물질 이름이 잘못되었습니다.")

        self.state = Props["STATE"]
        self.n     = Props["M"] / Props["MW"]
        self.MW    = Props["MW"]
        self.R     = Props["R"]

        self.T    = Props["T"]
        self.P    = Props["P"]
        self.Psat = Props["P_SAT"]
        self.h    = Props["H"]
        self.s    = Props["S"]
        self.Cp   = Props["CP"]
        self.Cv   = Props["CV"]
        self.mu   = Props["MU"]
        self.rho  = Props["RHO"]
        self.v_hat = Props["V_HAT"]
        self.V    = Props["V"]

        self.is_all_updated = False
        self.update_table = {
            "T": 0, "P": 0, "P_SAT": 0, "H": 0, "S": 0,
            "CP": 0, "CV": 0, "MU": 0, "RHO": 0, "V_HAT": 0, "V": 0
        }

        Tc = PropsSI("Tcrit", self.name)
        if self.T < Tc:
            self.P = PropsSI("P", "T", self.T, "Q", 0, self.name) * 1.001
        else:
            self.P = Props["P"]

        self.props_with_tp(self.T, self.P)

    def valid_update(self):
        for key in self.update_table.keys():
            if self.update_table[key] == 0:
                self.is_all_updated = False
                return
        self.is_all_updated = True

    def __str__(self):
        info  = f"물질 이름: {self.name}\n"
        info += f"물질 상: {self.state}\n"
        info += f"물질 몰 수: {self.n}\n"
        info += f"물질 몰당 부피: {self.v_hat}\n"
        info += f"물질 온도: {self.T}\n"
        info += f"물질 압력: {self.P}\n"
        info += f"물질 포화 증기압: {self.Psat}\n"
        info += f"물질 엔탈피: {self.h}\n"
        info += f"물질 엔트로피: {self.s}\n"
        info += f"물질 정압 비열: {self.Cp}\n"
        info += f"물질 정적 비열: {self.Cv}\n"
        info += f"물질 점성 계수: {self.mu}\n"
        info += f"물질 기체 상수: {self.R}\n"
        info += f"물질 밀도: {self.rho}\n"
        info += f"물질 분자량: {self.MW}\n"
        info += f"물질 질량: {self.n * self.MW}\n"
        info += f"물질 부피: {self.V}\n"
        return info

    def _is_supercritical(self, T, P):
        """임계점 초과 여부 확인."""
        P_crit = PropsSI("Pcrit", self.name)
        T_crit = PropsSI("Tcrit", self.name)
        return (T >= T_crit or P >= P_crit)

    def props_with_tp(self, T, P):
        # ── 기본 물성 (T, P 기반 단상) ──────────────────────────────────
        self.h  = PropsSI("H",       "T", T, "P", P, self.name)   # J/kg
        self.s  = PropsSI("S",       "T", T, "P", P, self.name)
        self.Cp = PropsSI("CPMOLAR", "T", T, "P", P, self.name)   # J/(mol·K)
        self.Cv = PropsSI("CVMOLAR", "T", T, "P", P, self.name)   # J/(mol·K)
        self.mu = 0

        supercritical = self._is_supercritical(T, P)

        # ── rho / v_hat: 임계점 초과 시 단상, 미만 시 포화선 기준 ────────
        try:
            if supercritical:
                # 초임계 단상: Q 지정 불가 → T, P 기반
                self.rho   = PropsSI("D",      "T", T, "P", P, self.name)
                self.v_hat = 1 / PropsSI("Dmolar", "T", T, "P", P, self.name)
            elif self.state == "Liquid":
                self.rho   = PropsSI("D",      "T", T, "Q", 0, self.name)
                self.v_hat = 1 / PropsSI("Dmolar", "T", T, "Q", 0, self.name)
            else:
                self.rho   = PropsSI("D",      "T", T, "Q", 1, self.name)
                self.v_hat = 1 / PropsSI("Dmolar", "T", T, "Q", 1, self.name)
        except Exception:
            # fallback: 단상 계산
            self.rho   = PropsSI("D",      "T", T, "P", P, self.name)
            self.v_hat = 1 / PropsSI("Dmolar", "T", T, "P", P, self.name)

        self.V = self.v_hat * self.n

        # ── 포화증기압 ────────────────────────────────────────────────────
        # 초임계 시 None 대신 P_crit 반환 — 다운스트림 None 크래시 방지
        try:
            if supercritical:
                self.Psat = PropsSI("Pcrit", self.name)
            else:
                self.Psat = PropsSI("P", "T", self.T, "Q", 0, self.name)
        except Exception:
            self.Psat = PropsSI("Pcrit", self.name)

        for key in self.update_table.keys():
            self.update_table[key] = 1
        self.is_all_updated = True

    def props_with_trho(self):
        self.P = PropsSI("P", "T", self.T, "D", self.rho, self.name)
        self.props_with_tp(self.T, self.P)

    def props_with_ph(self):
        self.T = PropsSI("T", "P", self.P, "H", self.h, self.name)
        self.props_with_tp(self.T, self.P)

    def props_with_ps(self):
        self.T = PropsSI("T", "P", self.P, "S", self.s, self.name)
        self.props_with_tp(self.T, self.P)

    def update_props(self, prop, value):
        if prop == "N":
            self.n = value
        elif prop == "T":
            self.T = value
        elif prop == "P":
            self.P = value
        elif prop == "P_SAT":
            self.Psat = value
        elif prop == "H":
            self.h = value
        elif prop == "S":
            self.s = value
        elif prop == "CP":
            self.Cp = value
        elif prop == "CV":
            self.Cv = value
        elif prop == "MU":
            self.mu = value
        elif prop == "RHO":
            self.rho = value
        elif prop == "V_HAT":
            self.v_hat = value
        elif prop == "V":
            self.V = value
        else:
            raise ValueError("잘못된 물성치에 대한 업데이트를 시도했습니다.")

        if self.is_all_updated:
            self.is_all_updated = False
            for key in self.update_table.keys():
                self.update_table[key] = 0

        self.update_table[prop] = 1
