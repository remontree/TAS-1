import functools
import math

from CoolProp.CoolProp import PropsSI

INJECTOR_MARGIN = 0.7
FUEL_MARGIN = 0.8
MAX_ORI = 0.1
# 연소 챔버 길이 ≥ (연료 길이 + 2 × 연료 외경 반지름) × CHAMBER_LENGTH_FACTOR
#   (2R = 프리 챔버 0.5R + 포스트 챔버 1.5R). 챔버 길이는 설계 변수가 아니라 항상 이 배수로 정해진다 (GA.derive).
CHAMBER_LENGTH_FACTOR = 1.3
BULKHEAD_NOZZLE_MARGIN = 1 / CHAMBER_LENGTH_FACTOR      # 예전 값 0.7 (= 1.43 배)

# 압력 용기(산화제 탱크, 연소 챔버) 두께 설계 안전 계수
SAFETY_FACTOR = 2.0

# 목표 평균 추력 / 임펄스 허용 오차 비율 (0.05 = ±5%)
TARGET_TOLERANCE = 0.05

# 산화제 탱크 초기 기체 공간(ullage) 최소 비율 = V_gas / V_tank
OXT_MIN_ULLAGE = 0.10

# 산화제 탱크 초기 압력 상한 = 산화제 임계압 × 이 비율 (N2O 7.245 MPa → 약 7.03 MPa)
# core 의 2상 탱크 모델은 탱크압이 임계압을 넘으면 계수 이상(ValueError)으로 멈춘다
# (학습 데이터 17,000개: 7.2 MPa 미만 탱크 오류 0%, 7.25 MPa 이상 74~95%)
OXT_MAX_P_RATIO = 0.97

# 다공 포트 사이 최소 웹 두께 [m]
MIN_PORT_WEB = 0.002

# 포트 반지름이 레벨셋 격자 몇 칸 이상이어야 하는지 (너무 작으면 연소면 인식 불가)
MIN_PORT_GRID_CELLS = 3

# 시뮬레이션이 simt 에 도달해서 끊기면(연소 미완료) 불합격 처리할지
REQUIRE_BURNOUT = True

# 인젝터 최소 차압 비율 = (탱크압 - 챔버압) / 챔버압 의 연소 중 최솟값 하한.
# 차압이 작으면 연소 불안정이 공급계로 전달되거나 역류(챔버 → 탱크) 위험이 있다.
MIN_INJECTOR_DP_RATIO = 0.20

# 대리모델 예측으로 검사할 때만 쓰는 안전 여유 (core 검증은 원래 기준)
#   dp_ratio         : 차압 하한을 이만큼 올림   (0.20 → 0.23)
#   target_tolerance : 목표 허용 오차를 이만큼 줄임 (±5% → ±4%)
SURROGATE_MARGIN = {"dp_ratio": 0.03, "target_tolerance": 0.01}


class Constrain:
    def __init__(self):
        pass

    # 제약 조건을 검증하는 함수.
    # 오버로딩 해서 사용함
    def validate(self):
        pass

    # 제약 위반 정도 (0 = 만족, 클수록 많이 위반). GA 가 불합격 후보끼리 비교할 때 사용.
    # 판정은 항상 validate() 기준이고, 위반 크기만 degree() 로 계산한다.
    def violation(self):
        if self.validate():
            return 0.0
        return max(self.degree(), 1e-6)

    # 상대 위반량. 오버로딩 해서 사용함
    def degree(self):
        return 1.0

    def name(self):
        return type(self).__name__


def _excess(value, limit):
    """value < limit 이어야 할 때의 상대 초과량"""
    return (value - limit) / max(abs(limit), 1e-12)


# 제약의 가정
# 산화제 탱크 및 연소 챔버는 상용 스토어에서 동일한 사이즈의 두 알루미늄 6061 파이프를 사서 가공한다.
# 초기 사용 파이프의 내경은 그대로 사용 하며, 외경은 cnc로 깍아서 두께를 조절한다.

# 산화제 탱크, 연소 챔버 두께 < 상용 스토어에서 구매한 초기 알루미늄 파이프 두께
class ChamberThickness(Constrain):
    def __init__(self, toxt, tcc, tal):
        self.toxt = toxt
        self.tcc = tcc
        self.tal = tal

    def validate(self):
        flag = self.toxt < self.tal and self.tcc < self.tal
        return flag

    def degree(self):
        return max(_excess(self.toxt, self.tal), _excess(self.tcc, self.tal))


# 산화제 탱크 반지름 = 연소 챔버 반지름 and 산화제 탱크 두께 = 연소 챔버 두께
class ChamberEquality(Constrain):
    def __init__(self, roxt, toxt, rcc, tcc):
        self.roxt = roxt
        self.toxt = toxt
        self.rcc = rcc
        self.tcc = tcc

    def validate(self):
        flag = self.roxt == self.rcc and self.toxt == self.tcc
        return flag

    def degree(self):
        return abs(self.roxt - self.rcc) / self.rcc + abs(self.toxt - self.tcc) / self.tcc



# 연소 챔버 두께 제한 사항
class Tcc(Constrain):
    def __init__(self, sigmaAL,Rcc, PccMEOP, FS, Tcc):
        self.sigmaAL = sigmaAL
        self.Rcc = Rcc
        self.PccMEOP = PccMEOP
        self.FS = FS
        self.Tcc = Tcc

    def validate(self):
        flag = self.Rcc*self.PccMEOP/(self.sigmaAL) * self.FS < self.Tcc
        return flag

    def degree(self):
        required = self.Rcc*self.PccMEOP/(self.sigmaAL) * self.FS
        return _excess(required, self.Tcc)



# 산화제 탱크 두께 제한 사항
class Toxt(Constrain):
    def __init__(self, sigmaAL, Roxt, PoxtMEOP, FS, Toxt):
        self.sigmaAL = sigmaAL
        self.Roxt = Roxt
        self.PoxtMEOP = PoxtMEOP
        self.FS = FS
        self.Toxt = Toxt


    def validate(self):
        flag = self.Roxt*self.PoxtMEOP/self.sigmaAL * self.FS < self.Toxt
        return flag

    def degree(self):
        required = self.Roxt*self.PoxtMEOP/self.sigmaAL * self.FS
        return _excess(required, self.Toxt)



# 인젝터 차압: 연소 중 (탱크압 - 챔버압) / 챔버압 >= MIN_INJECTOR_DP_RATIO
class InjectorDP(Constrain):
    def __init__(self, dp_ratio, limit=None):
        self.dp_ratio = dp_ratio
        self.limit = MIN_INJECTOR_DP_RATIO if limit is None else limit

    def validate(self):
        return self.dp_ratio is not None and self.dp_ratio >= self.limit

    def degree(self):
        if self.dp_ratio is None:
            return 1.0
        return (self.limit - self.dp_ratio) / self.limit


def injector_dp_ratio(sim):
    """
    시뮬레이션 결과 → 연소 중 최소 (Pt - Pc)/Pc.
    GA.simulate() 는 시계열에서 계산한 dp_ratio_min 을 준다. 예전 결과처럼 그 값이 없으면
    최대 탱크압/최대 챔버압으로 근사한다 (실측 대비 0.01~0.03 정도 크게 나온다).
    """
    if sim.get("dp_ratio_min") is not None:
        return sim["dp_ratio_min"]
    pt, pc = sim.get("max_tank_pressure"), sim.get("max_chamber_pressure")
    if pt is None or pc is None or pc <= 0:
        return None
    return (pt - pc) / pc


# 산화제 충진량의 액체 + 기체 부피 < 산화제 탱크 내부 부피
class OxtVolume(Constrain):
    def __init__(self, Mox, Msc, rhoox, rhosc, Roxt, Loxt):
        self.Mox = Mox
        self.Msc = Msc
        self.rhoox = rhoox
        self.rhosc = rhosc
        self.Roxt = Roxt
        self.Loxt = Loxt

    def validate(self):
        flag = self.Mox/self.rhoox + self.Msc/self.rhosc < math.pi * self.Roxt**2*self.Loxt
        return flag

    def degree(self):
        return _excess(self.Mox/self.rhoox + self.Msc/self.rhosc, math.pi * self.Roxt**2*self.Loxt)



# 인젝터 오리피스 면적 x 인젝터 오리피스 개수 < 연소 챔버 내원 넓이 x INJECTOR_MARGIN
class InjectorArea(Constrain):
    def __init__(self, Ro, No, Rcc):
        self.Ro = Ro
        self.No = No
        self.Rcc = Rcc

    def validate(self):
        flag = math.pi *  self.Ro**2 * self.No < INJECTOR_MARGIN*self.Rcc**2*math.pi
        return flag

    def degree(self):
        return _excess(math.pi * self.Ro**2 * self.No, INJECTOR_MARGIN*self.Rcc**2*math.pi)


# 산화제 탱크, 연소 챔버 길이 < 알루미늄 파이프 길이
class ChamberLength(Constrain):
    def __init__(self, Loxt, Lal, Lcc):
        self.Loxt = Loxt
        self.Lal = Lal
        self.Lcc = Lcc

    def validate(self):
        flag = self.Loxt < self.Lal and self.Lcc < self.Lal
        return flag

    def degree(self):
        return max(_excess(self.Loxt, self.Lal), _excess(self.Lcc, self.Lal))


# 연료 포트 넓이 x 포트 개수 < 연료 단면 넓이 x FUEL_MARGIN
class FuelArea(Constrain):
    def __init__(self, Rp, Np, Rf):
        self.Rp = Rp
        self.Np = Np
        self.Rf = Rf

    def validate(self):
        flag = self.Rp**2 * math.pi * self.Np < self.Rf**2 * math.pi * FUEL_MARGIN
        return flag

    def degree(self):
        return _excess(self.Rp**2 * math.pi * self.Np, self.Rf**2 * math.pi * FUEL_MARGIN)


# 포트 중앙 반지름 + 포트 반지름 < 연료 반지름
class FuelPort(Constrain):
    def __init__(self, Rcp, Rp, Rf):
        self.Rcp = Rcp
        self.Rp = Rp
        self.Rf = Rf

    def validate(self):
        flag = self.Rcp + self.Rp < self.Rf
        return flag

    def degree(self):
        return _excess(self.Rcp + self.Rp, self.Rf)


# 오리피스 최대 길이
class MaxOri(Constrain):
    def __init__(self, Lo):
        self.Lo = Lo

    def validate(self):
        flag = self.Lo <=MAX_ORI
        return flag

    def degree(self):
        return _excess(self.Lo, MAX_ORI)



# 연료 길이 + 포스트 챔버, 프리 챔버 ≤ 컴버션 챔버 길이 x BULKHEAD_NOZZLE_MARGIN
# (≤: 마지막에 챔버 길이를 하한에 딱 맞춰 줄인 형상도 합격)
class FuelLenght(Constrain):
    def __init__(self, Lf, Rf, Lcc):
        self.Lf = Lf
        self.Rf = Rf
        self.Lcc = Lcc

    def validate(self):
        flag = self.Lf+2*self.Rf <= self.Lcc * BULKHEAD_NOZZLE_MARGIN + 1e-12
        return flag

    def degree(self):
        return _excess(self.Lf+2*self.Rf, self.Lcc * BULKHEAD_NOZZLE_MARGIN)


# 연료 반지름 < 챔버 반지름
class FuelRadius(Constrain):
    def __init__(self, Rf, Rcc):
        self.Rf = Rf
        self.Rcc = Rcc

    def validate(self):
        flag = self.Rf < self.Rcc
        return flag

    def degree(self):
        return _excess(self.Rf, self.Rcc)


# 노즐 목 반지름 < 챔버 반지름
class NozzleThroatSize(Constrain):
    def __init__(self, Rnt, Rcc):
        self.Rnt = Rnt
        self.Rcc = Rcc
    def validate(self):
        flag = self.Rnt < self.Rcc
        return flag

    def degree(self):
        return _excess(self.Rnt, self.Rcc)


# 노즐 출구 반지름 > 노즐 목 반지름 (core 의 MeFromEps 는 eps < 1 이면 예외 발생)
class NozzleExpansion(Constrain):
    def __init__(self, Rne, Rnt):
        self.Rne = Rne
        self.Rnt = Rnt

    def validate(self):
        return self.Rne > self.Rnt

    def degree(self):
        return _excess(self.Rnt, self.Rne)


# 노즐 출구 반지름 <= 가공 후 기체 외경 반지름 (챔버 반지름 + 챔버 두께)
class NozzleExitSize(Constrain):
    def __init__(self, Rne, Rcc, Tcc):
        self.Rne = Rne
        self.Rcc = Rcc
        self.Tcc = Tcc

    def validate(self):
        return self.Rne <= self.Rcc + self.Tcc

    def degree(self):
        return _excess(self.Rne, self.Rcc + self.Tcc)


# 다공 포트일 때 이웃 포트 사이 웹 두께 >= MIN_PORT_WEB
# 포트 중심 간 거리 = 2 x 포트 중앙 반지름 x sin(pi / 포트 개수)
class PortWeb(Constrain):
    def __init__(self, Rcp, Rp, Np):
        self.Rcp = Rcp
        self.Rp = Rp
        self.Np = Np

    def web(self):
        if self.Np <= 1:
            return math.inf
        return 2*self.Rcp*math.sin(math.pi/self.Np) - 2*self.Rp

    def validate(self):
        return self.web() >= MIN_PORT_WEB

    def degree(self):
        return (MIN_PORT_WEB - self.web()) / MIN_PORT_WEB


# 포트 반지름 >= 레벨셋 격자 크기 x MIN_PORT_GRID_CELLS
# (core 의 연료 레벨셋 격자 = 2 x 연료 반지름 / (grid_size - 1))
class PortResolution(Constrain):
    def __init__(self, Rp, Rf, grid_size):
        self.Rp = Rp
        self.Rf = Rf
        self.grid_size = grid_size

    def min_radius(self):
        return MIN_PORT_GRID_CELLS * 2*self.Rf/(self.grid_size - 1)

    def validate(self):
        return self.Rp >= self.min_radius()

    def degree(self):
        return _excess(self.min_radius(), self.Rp)


# ──────────────────────────────────────────────────────────────────────────
# 산화제 탱크 초기 충진 상태
# core/rocket.py set_component() 의 N2O 액/기 분배 + 가압제 분압 계산을 그대로 옮긴 것.
# 이 계산이 성립하지 않으면 (과충진 / 전부 기체 / 임계 온도 초과) core 시뮬레이션이 깨진다.
# ──────────────────────────────────────────────────────────────────────────
R = 8.3143  # core/matter.py 와 동일한 보편 기체 상수

# 포화 물성은 (산화제, 온도) 에만 의존하므로 캐시 (PropsSI 가 이 함수 시간의 대부분)
_saturated = functools.lru_cache(maxsize=1024)(PropsSI)

def oxidizer_tank_state(oxidizer, oxidizer_MW, oxidizer_M, pressurant, pressurant_MW,
                        pressurant_M, T, V):
    Tcrit = _saturated("Tcrit", oxidizer)
    state = {"T_crit": Tcrit, "P_crit": _saturated("Pcrit", oxidizer), "valid_T": T < Tcrit}
    if not state["valid_T"]:
        return state

    # core/matter.py: 임계온도 미만이면 P = Psat(T) x 1.001, v_hat = 포화 액체 몰부피
    P = _saturated("P", "T", T, "Q", 0, oxidizer) * 1.001
    v_hat = 1 / _saturated("Dmolar", "T", T, "Q", 0, oxidizer)
    N = oxidizer_M / oxidizer_MW

    # core/rocket.py: N2O 액/기 분배
    N_L = (N*R*T - P*V)/(R*T - P*v_hat)
    N_G = N - N_L
    V_gas = V - N_L * v_hat

    # core/rocket.py: 가압제 분압, 총 탱크압
    n_pressurant = pressurant_M / pressurant_MW
    P_pressurant = n_pressurant * R * T / max(V_gas, 1e-9)

    state.update({
        "P_sat": P,
        "N": N,
        "N_liquid": N_L,
        "N_vapor": N_G,
        "V_liquid": N_L * v_hat,
        "V_gas": V_gas,
        "ullage": V_gas / V,
        "P_pressurant": P_pressurant,
        "P_tank": P + P_pressurant,
        "rho_liquid": _saturated("D", "T", T, "Q", 0, oxidizer),
        "rho_pressurant": math.inf,
    })

    # 가압제 밀도 (OxtVolume 용). 과충진으로 압력이 비정상이면 이상기체 근사
    if pressurant_M > 0:
        P_total = state["P_tank"]
        try:
            state["rho_pressurant"] = PropsSI("D", "T", T, "P", P_total, pressurant)
        except ValueError:
            state["rho_pressurant"] = P_total * pressurant_MW / (R * T)
    return state


# 산화제 탱크 초기 충진이 core 계산상 성립하는지
#   - 탱크 온도 < 산화제 임계 온도
#   - 액체 N2O 몰수 > 0 (전부 기체가 되지 않음)
#   - 기체 공간 비율 >= OXT_MIN_ULLAGE (과충진 아님)
class OxtFill(Constrain):
    def __init__(self, tank_state):
        self.s = tank_state

    def validate(self):
        s = self.s
        return s["valid_T"] and s["N_liquid"] > 0 and s["ullage"] >= OXT_MIN_ULLAGE

    def degree(self):
        s = self.s
        if not s["valid_T"]:
            return 10.0
        if s["N_liquid"] <= 0:
            return -s["N_liquid"] / s["N"] + 1e-3
        return (OXT_MIN_ULLAGE - s["ullage"]) / OXT_MIN_ULLAGE


# 산화제 탱크 초기 압력 (N2O 포화압 + 가압제 분압) <= 임계압 × OXT_MAX_P_RATIO
# 넘으면 core 탱크 모델이 성립하지 않음 (초임계)
class OxtPressure(Constrain):
    def __init__(self, tank_state):
        self.s = tank_state

    def limit(self):
        return self.s["P_crit"] * OXT_MAX_P_RATIO

    def validate(self):
        return self.s["P_tank"] <= self.limit()

    def degree(self):
        return self.s["P_tank"] / self.limit() - 1.0


# ──────────────────────────────────────────────────────────────────────────
# 시뮬레이션 결과로 확인하는 제약
# ──────────────────────────────────────────────────────────────────────────

# 평균 추력 제한 조건
# 목표한 평균 추력에 대해 5% 내외의 시뮬레이션 결과가 나왔는지 확인한다.
class ThrustAvg(Constrain):
    def __init__(self, simThrustAvg, targetThrustAvg, tolerance=None):
        self.simThrustAvg = simThrustAvg
        self.targetThrustAvg = targetThrustAvg
        self.tol = TARGET_TOLERANCE if tolerance is None else tolerance

    def validate(self):
        flag = self.targetThrustAvg - self.targetThrustAvg*self.tol <= self.simThrustAvg and self.simThrustAvg <= self.targetThrustAvg+self.targetThrustAvg*self.tol
        return flag

    def degree(self):
        return abs(self.simThrustAvg - self.targetThrustAvg) / self.targetThrustAvg - self.tol



# 임펄스 제한 조건
# 목표한 임펄스에 대해 5% 내외의 시뮬레이션 결과가 나왔는지 확인한다.
class Impulse(Constrain):
    def __init__(self, simImpulse, targetImpulse, tolerance=None):
            self.simImpulse = simImpulse
            self.targetImpulse = targetImpulse
            self.tol = TARGET_TOLERANCE if tolerance is None else tolerance

    def validate(self):
        flag = self.targetImpulse - self.targetImpulse*self.tol <= self.simImpulse and self.simImpulse <= self.targetImpulse+self.targetImpulse*self.tol
        return flag

    def degree(self):
        return abs(self.simImpulse - self.targetImpulse) / self.targetImpulse - self.tol


# 시뮬레이션이 예외 없이 끝났고, (REQUIRE_BURNOUT 이면) simt 에 걸려 잘리지 않았는지
class SimCompleted(Constrain):
    def __init__(self, error, termination_code):
        self.error = error
        self.termination_code = termination_code

    def validate(self):
        if self.error:
            return False
        return not (REQUIRE_BURNOUT and self.termination_code == "TIME_LIMIT_REACHED")

    def degree(self):
        return 10.0 if self.error else 1.0


# ──────────────────────────────────────────────────────────────────────────
# 제약 목록 생성
#   design : core Rocket config 와 같은 키를 가진 dict (fixedvalue + 설계 변수)
#   pipe   : managepipedb.load_pipes() 의 파이프 dict
# ──────────────────────────────────────────────────────────────────────────

def pre_sim_constraints(design, pipe):
    """
    시뮬레이션 전에 형상만으로 검사하는 제약 목록.
    반환: (constraints, tank_state)
    """
    d = design
    sigma = pipe["yield_strength"]
    tank_state = oxidizer_tank_state(
        d["oxidizer"], d["oxidizer_MW"], d["oxidizer_M"],
        d["pressurant"], d["pressurant_MW"], d["pressurant_M"],
        d["temperature"],
        math.pi * d["oxidizer_tank_radius"]**2 * d["oxidizer_tank_length"],
    )

    # 연소 챔버 두께는 산화제 탱크와 같은 파이프를 같은 두께로 깍는다고 가정
    tcc = d["oxidizer_tank_tickness"]

    constraints = [
        ChamberThickness(d["oxidizer_tank_tickness"], tcc, pipe["wall_thickness"]),
        ChamberEquality(d["oxidizer_tank_radius"], d["oxidizer_tank_tickness"],
                        d["combution_chamber_radius"], tcc),
        ChamberLength(d["oxidizer_tank_length"], pipe["length"], d["combution_chamber_length"]),
        OxtFill(tank_state),
        InjectorArea(d["injector_oriffice_radius"], d["injector_oriffice_n"],
                     d["combution_chamber_radius"]),
        MaxOri(d["injector_oriffice_length"]),
        FuelArea(d["inner_radius"], d["port_num"], d["outter_radius"]),
        FuelPort(d["port_center_radius"], d["inner_radius"], d["outter_radius"]),
        PortWeb(d["port_center_radius"], d["inner_radius"], d["port_num"]),
        PortResolution(d["inner_radius"], d["outter_radius"], d["grid_size"]),
        FuelLenght(d["fuel_length"], d["outter_radius"], d["combution_chamber_length"]),
        FuelRadius(d["outter_radius"], d["combution_chamber_radius"]),
        NozzleThroatSize(d["nozzle_throat_radius"], d["combution_chamber_radius"]),
        NozzleExpansion(d["nozzle_exit_radius"], d["nozzle_throat_radius"]),
        NozzleExitSize(d["nozzle_exit_radius"], d["combution_chamber_radius"], tcc),
    ]

    # 탱크 충진이 성립할 때만 압력/부피 기반 제약 추가 (초기 탱크압을 MEOP 추정치로 사용)
    if OxtFill(tank_state).validate():
        constraints += [
            OxtPressure(tank_state),
            Toxt(sigma, d["oxidizer_tank_radius"], tank_state["P_tank"], SAFETY_FACTOR,
                 d["oxidizer_tank_tickness"]),
            OxtVolume(d["oxidizer_M"], d["pressurant_M"], tank_state["rho_liquid"],
                      tank_state["rho_pressurant"], d["oxidizer_tank_radius"],
                      d["oxidizer_tank_length"]),
        ]
    return constraints, tank_state


def post_sim_constraints(design, pipe, sim, target_impulse, target_thrust_avg, tolerance=None, margin=None):
    """
    시뮬레이션 결과가 있어야 검사 가능한 제약 목록.
    sim: GA.simulate() 결과 dict
    tolerance: 목표 임펄스·평균 추력 허용 오차 비율 (None = TARGET_TOLERANCE). MDO 실행마다 바꿀 수 있다.
    margin: 대리모델 예측으로 검사할 때의 안전 여유 (SURROGATE_MARGIN). 최적화가 제약 경계에 붙은 설계를
            고르므로 예측 오차만큼 core 에서 떨어진다 → 예측에는 조금 더 엄격한 기준을 쓴다.
    """
    margin = margin or {}
    base_tol = TARGET_TOLERANCE if tolerance is None else tolerance
    tol = max(base_tol - margin.get("target_tolerance", 0.0), base_tol * 0.5)   # 허용 오차가 작아도 0 이 되지 않게
    dp_limit = MIN_INJECTOR_DP_RATIO + margin.get("dp_ratio", 0.0)
    d = design
    completed = SimCompleted(sim.get("error"), sim.get("termination_code"))
    if sim.get("error"):
        return [completed]

    sigma = pipe["yield_strength"]
    return [
        completed,
        Toxt(sigma, d["oxidizer_tank_radius"], sim["max_tank_pressure"], SAFETY_FACTOR,
             d["oxidizer_tank_tickness"]),
        Tcc(sigma, d["combution_chamber_radius"], sim["max_chamber_pressure"], SAFETY_FACTOR,
            d["oxidizer_tank_tickness"]),
        InjectorDP(injector_dp_ratio(sim), dp_limit),
        Impulse(sim["impulse"], target_impulse, tol),
        ThrustAvg(sim["thrust_avg"], target_thrust_avg, tol),
    ]


def check(constraints):
    """제약 목록 검사. 반환: (전부 만족 여부, 총 위반량, 실패한 제약 이름 목록)"""
    total = 0.0
    failed = []
    for c in constraints:
        v = c.violation()
        if v > 0:
            total += v
            failed.append(c.name())
    return not failed, total, failed


# ──────────────────────────────────────────────────────────────────────────
# 제약 여유 (견고성) — Isp 가 사실상 같은 형상들 중 경계에서 먼 것을 고를 때 사용 (GA.ROBUST_PICK)
# 항목별 여유를 0~1 로 정규화 (1 = 충분, 0 = 경계) 하고 가장 빠듯한 값을 점수로 쓴다
# ──────────────────────────────────────────────────────────────────────────
def robustness(design, pipe, sim, target_impulse, target_thrust_avg, tolerance=None):
    """반환: (점수, {항목: 여유}). 결과가 없으면 (None, {})"""
    if not sim or sim.get("error") or sim.get("impulse") is None:
        return None, {}
    d, sigma = design, pipe["yield_strength"]
    tol = TARGET_TOLERANCE if tolerance is None else tolerance
    m = {
        "impulse": 1 - abs(sim["impulse"] / target_impulse - 1) / tol,
        "thrust_avg": 1 - abs(sim["thrust_avg"] / target_thrust_avg - 1) / tol,
    }
    dp = injector_dp_ratio(sim)
    if dp is not None:
        m["dp_ratio"] = (dp - MIN_INJECTOR_DP_RATIO) / MIN_INJECTOR_DP_RATIO
    m["tank_wall"] = -Toxt(sigma, d["oxidizer_tank_radius"], sim["max_tank_pressure"], SAFETY_FACTOR,
                           d["oxidizer_tank_tickness"]).degree()
    m["chamber_wall"] = -Tcc(sigma, d["combution_chamber_radius"], sim["max_chamber_pressure"], SAFETY_FACTOR,
                             d["oxidizer_tank_tickness"]).degree()
    m = {k: max(min(v, 1.0), -1.0) for k, v in m.items()}
    return min(m.values()), m
