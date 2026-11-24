"""from .hybridengine import *
import pandas as pd


class Rocket:
    def __init__(self, config):

        # 산화제 종류
        self.oxidizer = config["oxidizer"]

        # 산화제 몰수 [mol]
        self.oxidizer_MW = config["oxidizer_MW"]

        # 산화제 질량 [kg]
        self.oxidizer_M = config["oxidizer_M"]

        # 가압제 종류
        self.pressurant = config["pressurant"]

        # 가압제 몰수 [mol]
        self.pressurant_MW = config["pressurant_MW"]

        # 가압제 질량 [kg]
        self.pressurant_M = config["pressurant_M"]

        # 연료 종류
        self.fuel = config["fuel"]

        # 연료 후퇴율 상수 a
        self.fuel_a = config["fuel_a"]

        # 연료 후퇴율 상수 n
        self.fuel_n = config["fuel_n"]

        # 연료 내부 반지름[m]
        self.inner_radius = config["inner_radius"]

        # 연료 외부 반짊[m]
        self.outter_radius = config["outter_radius"]

        # 대기압 [Pa]
        self.Pa = config["Pa"]

        # 온도 [k]
        self.temperature = config["temperature"]

        # cstar_eta
        self.cstar_eta = config["cstar_eta"]

        # 산화제 탱크 반지름 [m]
        self.oxidizer_tank_radius = config["oxidizer_tank_radius"]

        # 산화제 탱크 길이 [m]
        self.oxidizer_tank_length = config["oxidizer_tank_length"]

        # 산화제 탱크 재질 밀도
        self.oxidizer_tank_material_rho = config["oxidizer_tank_material_rho"]

        # 산화제 탱크 두께 [m]
        self.oxidizer_tank_tickness = config["oxidizer_tank_tickness"]

        # 인젝터 오리피스 반지름 [m]
        self.injector_oriffice_radius = config["injector_oriffice_radius"]

        # 인젝터 오리피스 길이 [m]
        self.injector_oriffice_length = config["injector_oriffice_length"]

        # 인젝터 오리피스 개수
        self.injector_oriffice_n = config["injector_oriffice_n"]

        # 인젝터 항력 계수
        self.injector_cd = config["injector_cd"]

        # 연소 챔버 목표 압력 [Pa]
        self.combution_chamber_pressure = config["combution_chamber_pressure"]

        # 연소 챔버 길이 [m]
        self.combution_chamber_length = config["combution_chamber_length"]

        # 연소 챔버 반지름 [m]
        self.combution_chamber_radius = config["combution_chamber_radius"]

        # 연료 길이 [m]
        self.fuel_length = config["fuel_length"]

        # 연료 밀도
        self.fuel_rho = config["fuel_rho"]

        # 연료 포트 개수
        self.port_num = config.get("port_num", 1)

        # 연료 중심으로 부터 포트의 거리
        self.port_center_radius = config.get("port_center_radius",
                                         config["outter_radius"] * 0.5)

        # 그리드 사이즈 (연료 burn back 시뮬레이션)
        self.grid_size = config.get("grid_size", 300)

        # 노즐 목 반지름 [m]
        self.nozzle_throat_radius = config["nozzle_throat_radius"]

        # 노즐 출구 반지름 [m]
        self.nozzle_exit_radius = config["nozzle_exit_radius"]

        # dt
        self.dt = config["dt"]

        # simt
        self.simt = config["simt"]

        # 추력 계수
        self.thrust_coeff = config["thrust_coeff"]

        # 산화제 탱크 부피 [m^3]
        self.oxidizer_tank_v =  self.oxidizer_tank_radius**2*math.pi*self.oxidizer_tank_length

        # 인젝터 오리피스 면적 [m^2]
        self.injector_oriffice_area = self.injector_oriffice_radius**2*math.pi

        # 노즐목 면적 [m^2]
        self.nozzle_throat_area = self.nozzle_throat_radius**2*math.pi

        # 노즐 출구 면적 [m^2]
        self.nozzle_exit_area = self.nozzle_exit_radius**2*math.pi

        # eps
        self.eps = self.nozzle_exit_area/self.nozzle_throat_area

        # 로켓 컴포넌트 설정
        self.set_component()


    def set_component(self):

        # 산화제 초기 설정
        N2O_prop = {
            "NAME" : self.oxidizer,
            "STATE" : "Liquid",
            "MW" : self.oxidizer_MW,
            "M" : self.oxidizer_M,
            "R" : R,
            "T" : self.temperature,
            "P" : None,
            "P_SAT" : None,
            "H" : None,
            "S" : None,
            "CP" : None,
            "CV" : None,
            "MU" : None,
            "RHO" : None,
            "V_HAT" : None,
            "V" : None
        }


        self.N2O = Matter(N2O_prop)

        N2O_N = self.N2O.n
        N2O_v_hat = self.N2O.v_hat
        N2O_P = self.N2O.P
        N2O_T = self.N2O.T
        N2O_MW = self.N2O.MW

        N20_N_L = (N2O_N*R*N2O_T - N2O_P*self.oxidizer_tank_v)/(R*N2O_T-N2O_P*N2O_v_hat)
        N2O_N_G = N2O_N - N20_N_L

        N2O_liquid_prop = {
            "NAME" : self.oxidizer,
            "STATE" : "Liquid",
            "MW" : self.oxidizer_MW,
            "M" : N20_N_L*N2O_MW,
            "R" : R,
            "T" : self.temperature,
            "P" : None,
            "P_SAT" : None,
            "H" : None,
            "S" : None,
            "CP" : None,
            "CV" : None,
            "MU" : None,
            "RHO" : None,
            "V_HAT" : None,
            "V" : None
        }

        N2O_vapor_prop = {
            "NAME" : self.oxidizer,
            "STATE" : "Vapor",
            "MW" : self.oxidizer_MW,
            "M" : N2O_N_G*N2O_MW,
            "R" : R,
            "T" : self.temperature,
            "P" : None,
            "P_SAT" : None,
            "H" : None,
            "S" : None,
            "CP" : None,
            "CV" : None,
            "MU" : None,
            "RHO" : None,
            "V_HAT" : None,
            "V" : None
        }

        Pressurant_prop = {
            "NAME" : self.pressurant,
            "STATE" : "Vapor",
            "MW" : self.pressurant_MW,
            "M" : self.pressurant_M,
            "R" : R,
            "T" : self.temperature,
            "P" : N2O_P,
            "P_SAT" : None,
            "H" : None,
            "S" : None,
            "CP" : None,
            "CV" : None,
            "MU" : None,
            "RHO" : None,
            "V_HAT" : None,
            "V" : None
        }

        self.N2O_liquid = Matter(N2O_liquid_prop)
        self.N2O_vapor = Matter(N2O_vapor_prop)
        self.Pressurant = Matter(Pressurant_prop)


        oxidizer_tank_config = {
            "RADIUS": self.oxidizer_tank_radius,
            "HEIGHT": self.oxidizer_tank_length,

            "V" : self.oxidizer_tank_v,

            "PRESSURANT" :  self.Pressurant,
            "OXIDIZER_LIQUID" : self.N2O_liquid,
            "OXIDIZER_VAPOR" : self.N2O_vapor,

            "MATERIAL" : "Al",

            "T" : N2O_T,
            "P" : N2O_P,
            "M" : self.oxidizer_tank_material_rho*math.pi*self.oxidizer_tank_length*(2*self.oxidizer_tank_radius*self.oxidizer_tank_tickness+self.oxidizer_tank_tickness**2)
        }

        self.oxidizer_tank = OxidizerTank(oxidizer_tank_config)

        injector_config = {
            "A" : self.injector_oriffice_area,
            "CD" : self.injector_cd,
            "N" : self.injector_oriffice_n,
            "L" : self.injector_oriffice_length
        }

        self.injector = Injector(injector_config)

        fuel_config = {
            "NAME"               : self.fuel,
            "LENGTH"             : self.fuel_length,
            "INNER_RADIUS"       : self.inner_radius,
            "OUTER_RADIUS"       : self.outter_radius,
            "RHO"                : self.fuel_rho,
            "A"                  : self.fuel_a,
            "N"                  : self.fuel_n,
            "PORT_NUM"           : self.port_num,
            "PORT_CENTER_RADIUS" : self.port_center_radius,
            "GRID_SIZE"          : self.grid_size,
        }

        self.fuel = Fuel(fuel_config)

        combution_chamber_config = {
            "T" : self.temperature,
            "P" : 101325,
            "FUEL" : self.fuel,
            "CSTAR_ETA" : self.cstar_eta
        }

        self.combution_chamber = CombutionChamber(combution_chamber_config)

        nozzle_config = {
            "EPS" : self.eps,
            "THROAT_RADIUS" : self.nozzle_throat_radius
        }

        self.nozzle = Nozzle(nozzle_config)

        self.engine = HybridEngine(self.oxidizer_tank, self.injector, self.combution_chamber, self.nozzle)


    def ignite(self):
        # 1. 시뮬레이션 수행 (핵심 물리 계산)
        self.engine.simulation(self.dt, self.simt, self.thrust_coeff)

        # 2. 데이터 수집
        t = self.engine.time_step[:-1]

        # 딕셔너리로 한꺼번에 묶기
        plot_vars = {
            "Time (s)": t, # 시간을 미리 포함
            "Tank Temperature (K)": self.engine.tank_temperature[:-1],
            "Tank Pressure (Pa)": self.engine.tank_pressure[:-1],
            "Chamber Temperature (K)": self.engine.chamber_temperature[:-1],
            "Chamber Pressure (Pa)": self.engine.chamber_pressure[:-1],
            "Thrust (N)": self.engine.thrust[:-1],
            "Exhaust Velocity Ve (m/s)": self.engine.ve[:-1],
            "Isp (s)": self.engine.Isp[:-1],
            "Fuel Regression r_dot (m/s)": self.engine.r_dot[:-1],
            "mox_dot (kg/s)": self.engine.mox_dot[:-1],
            "mf_dot (kg/s)": self.engine.mf_dot[:-1],
            "me_dot (kg/s)": self.engine.me_dot[:-1],
            "O/F Ratio": self.engine.of[:-1],
            "Characteristic Velocity c* (m/s)": self.engine.cstar[:-1],
            "Thrust Coeff": self.engine.cf[:-1],
        }


        df = pd.DataFrame(plot_vars)

        df.to_csv("../result.csv", index=False, encoding='utf-8-sig')

        return df
"""

from .hybridengine import *
import pandas as pd


class Rocket:
    def __init__(self, config):

        # 산화제 종류
        self.oxidizer = config["oxidizer"]

        # 산화제 몰수 [mol]
        self.oxidizer_MW = config["oxidizer_MW"]

        # 산화제 질량 [kg]
        self.oxidizer_M = config["oxidizer_M"]

        # 가압제 종류
        self.pressurant = config["pressurant"]

        # 가압제 몰수 [mol]
        self.pressurant_MW = config["pressurant_MW"]

        # 가압제 질량 [kg]
        self.pressurant_M = config["pressurant_M"]

        # 연료 종류
        self.fuel = config["fuel"]

        # 연료 후퇴율 상수 a
        self.fuel_a = config["fuel_a"]

        # 연료 후퇴율 상수 n
        self.fuel_n = config["fuel_n"]

        # 연료 내부 반지름[m]
        self.inner_radius = config["inner_radius"]

        # 연료 외부 반짊[m]
        self.outter_radius = config["outter_radius"]

        # 대기압 [Pa]
        self.Pa = config["Pa"]

        # 온도 [k]
        self.temperature = config["temperature"]

        # cstar_eta
        self.cstar_eta = config["cstar_eta"]

        # C* 점화 상승 지연 시간상수 [s] / 초기 결손 비율
        self.cstar_tau = config.get("cstar_tau", 0.35)
        self.cstar_startup_deficit = config.get("cstar_startup_deficit", 0.9)

        # 산화제 탱크 반지름 [m]
        self.oxidizer_tank_radius = config["oxidizer_tank_radius"]

        # 산화제 탱크 길이 [m]
        self.oxidizer_tank_length = config["oxidizer_tank_length"]

        # 산화제 탱크 재질 밀도
        self.oxidizer_tank_material_rho = config["oxidizer_tank_material_rho"]

        # 산화제 탱크 두께 [m]
        self.oxidizer_tank_tickness = config["oxidizer_tank_tickness"]

        # 인젝터 오리피스 반지름 [m]
        self.injector_oriffice_radius = config["injector_oriffice_radius"]

        # 인젝터 오리피스 길이 [m]
        self.injector_oriffice_length = config["injector_oriffice_length"]

        # 인젝터 오리피스 개수
        self.injector_oriffice_n = config["injector_oriffice_n"]

        # 인젝터 항력 계수
        self.injector_cd = config["injector_cd"]

        # 연소 챔버 목표 압력 [Pa]
        self.combution_chamber_pressure = config["combution_chamber_pressure"]

        # 연소 챔버 길이 [m]
        self.combution_chamber_length = config["combution_chamber_length"]

        # 연소 챔버 반지름 [m]
        self.combution_chamber_radius = config["combution_chamber_radius"]

        # 연료 길이 [m]
        self.fuel_length = config["fuel_length"]

        # 연료 밀도
        self.fuel_rho = config["fuel_rho"]

        # 연료 포트 개수
        self.port_num = config.get("port_num", 1)

        # 연료 중심으로 부터 포트의 거리
        self.port_center_radius = config.get("port_center_radius",
                                         config["outter_radius"] * 0.5)

        # 그리드 사이즈 (연료 burn back 시뮬레이션)
        self.grid_size = config.get("grid_size", 300)

        # 연료 그레인 형상: "PORT"(원형 다공 포트, 기본) | "FIN"(멀티핀/wagon-wheel)
        self.grain_type = config.get("grain_type", "PORT")

        # 멀티핀 그레인 핀 개수
        self.fin_num = config.get("fin_num", 6)

        # 멀티핀 그레인 핀의 반경 방향 길이 [m]
        self.fin_length = config.get(
            "fin_length", (config["outter_radius"] - config["inner_radius"]) * 0.6)

        # 멀티핀 그레인 핀 밑동(보어 쪽) 폭 [m]
        self.fin_width = config.get("fin_width", config["inner_radius"] * 0.6)

        # 멀티핀 그레인 핀 끝단(바깥쪽) 폭 [m] (기본값 = fin_width, 테이퍼 없음)
        self.fin_tip_width = config.get("fin_tip_width", self.fin_width)

        # 노즐 목 반지름 [m]
        self.nozzle_throat_radius = config["nozzle_throat_radius"]

        # 노즐 출구 반지름 [m]
        self.nozzle_exit_radius = config["nozzle_exit_radius"]

        # dt
        self.dt = config["dt"]

        # simt
        self.simt = config["simt"]

        # 추력 계수
        self.thrust_coeff = config["thrust_coeff"]

        # 산화제 탱크 부피 [m^3]
        self.oxidizer_tank_v =  self.oxidizer_tank_radius**2*math.pi*self.oxidizer_tank_length

        # 인젝터 오리피스 면적 [m^2]
        self.injector_oriffice_area = self.injector_oriffice_radius**2*math.pi

        # 노즐목 면적 [m^2]
        self.nozzle_throat_area = self.nozzle_throat_radius**2*math.pi

        # 노즐 출구 면적 [m^2]
        self.nozzle_exit_area = self.nozzle_exit_radius**2*math.pi

        # eps
        self.eps = self.nozzle_exit_area/self.nozzle_throat_area

        # 로켓 컴포넌트 설정
        self.set_component()


    def set_component(self):

        # 산화제 초기 설정
        N2O_prop = {
            "NAME" : self.oxidizer,
            "STATE" : "Liquid",
            "MW" : self.oxidizer_MW,
            "M" : self.oxidizer_M,
            "R" : R,
            "T" : self.temperature,
            "P" : None,
            "P_SAT" : None,
            "H" : None,
            "S" : None,
            "CP" : None,
            "CV" : None,
            "MU" : None,
            "RHO" : None,
            "V_HAT" : None,
            "V" : None
        }


        self.N2O = Matter(N2O_prop)

        N2O_N = self.N2O.n
        N2O_v_hat = self.N2O.v_hat
        N2O_P = self.N2O.P
        N2O_T = self.N2O.T
        N2O_MW = self.N2O.MW

        # ── N2O 액/기 분배 계산 ─────────────────────────────────────────
        # N2O_P = Psat(T): N2O 증기 분압은 가압제 유무와 무관하게 Psat으로 결정
        # (달톤 분압 법칙: 가압제는 N2O 상평형에 영향을 주지 않음)
        N20_N_L = (N2O_N*R*N2O_T - N2O_P*self.oxidizer_tank_v)/(R*N2O_T-N2O_P*N2O_v_hat)
        N2O_N_G = N2O_N - N20_N_L

        # ── 기체 공간 부피 및 가압제 분압 계산 ──────────────────────────
        # V_gas = 탱크 전체 - 액체 N2O 부피
        V_gas = self.oxidizer_tank_v - N20_N_L * N2O_v_hat

        # 가압제 몰수로부터 실제 분압 계산 (이상기체 법칙)
        # pressurant_M = 0 이면 P_pressurant = 0 → 기존 자가압 방식과 동일
        n_pressurant = self.pressurant_M / self.pressurant_MW
        P_pressurant = n_pressurant * R * N2O_T / max(V_gas, 1e-9)

        # 총 탱크압 = N2O 포화압 + 가압제 분압
        P_tank_total = N2O_P + P_pressurant

        N2O_liquid_prop = {
            "NAME" : self.oxidizer,
            "STATE" : "Liquid",
            "MW" : self.oxidizer_MW,
            "M" : N20_N_L*N2O_MW,
            "R" : R,
            "T" : self.temperature,
            "P" : None,
            "P_SAT" : None,
            "H" : None,
            "S" : None,
            "CP" : None,
            "CV" : None,
            "MU" : None,
            "RHO" : None,
            "V_HAT" : None,
            "V" : None
        }

        N2O_vapor_prop = {
            "NAME" : self.oxidizer,
            "STATE" : "Vapor",
            "MW" : self.oxidizer_MW,
            "M" : N2O_N_G*N2O_MW,
            "R" : R,
            "T" : self.temperature,
            "P" : None,
            "P_SAT" : None,
            "H" : None,
            "S" : None,
            "CP" : None,
            "CV" : None,
            "MU" : None,
            "RHO" : None,
            "V_HAT" : None,
            "V" : None
        }

        Pressurant_prop = {
            "NAME" : self.pressurant,
            "STATE" : "Vapor",
            "MW" : self.pressurant_MW,
            "M" : self.pressurant_M,
            "R" : R,
            "T" : self.temperature,
            # 가압제는 초임계 기체 → matter.py의 else 분기 실행
            # → self.P = Props["P"] 를 그대로 사용하므로 반드시 실제 분압을 넣어야 함
            # pressurant_M = 0 이면 P_pressurant = 0인데 CoolProp은 P=0 허용 안 함
            # → 최소값으로 대기압(101325) 사용 (물성 계산용, 탱크압에는 영향 없음)
            "P" : max(P_pressurant, 101325.0),
            "P_SAT" : None,
            "H" : None,
            "S" : None,
            "CP" : None,
            "CV" : None,
            "MU" : None,
            "RHO" : None,
            "V_HAT" : None,
            "V" : None
        }

        self.N2O_liquid = Matter(N2O_liquid_prop)
        self.N2O_vapor = Matter(N2O_vapor_prop)
        self.Pressurant = Matter(Pressurant_prop)


        oxidizer_tank_config = {
            "RADIUS": self.oxidizer_tank_radius,
            "HEIGHT": self.oxidizer_tank_length,

            "V" : self.oxidizer_tank_v,

            "PRESSURANT" :  self.Pressurant,
            "OXIDIZER_LIQUID" : self.N2O_liquid,
            "OXIDIZER_VAPOR" : self.N2O_vapor,

            "MATERIAL" : "Al",

            "T" : N2O_T,
            "P" : P_tank_total,   # N2O 포화압 + 가압제 분압
            "M" : self.oxidizer_tank_material_rho*math.pi*self.oxidizer_tank_length*(2*self.oxidizer_tank_radius*self.oxidizer_tank_tickness+self.oxidizer_tank_tickness**2)
        }

        self.oxidizer_tank = OxidizerTank(oxidizer_tank_config)

        injector_config = {
            "A" : self.injector_oriffice_area,
            "CD" : self.injector_cd,
            "N" : self.injector_oriffice_n,
            "L" : self.injector_oriffice_length
        }

        self.injector = Injector(injector_config)

        fuel_config = {
            "NAME"               : self.fuel,
            "LENGTH"             : self.fuel_length,
            "INNER_RADIUS"       : self.inner_radius,
            "OUTER_RADIUS"       : self.outter_radius,
            "RHO"                : self.fuel_rho,
            "A"                  : self.fuel_a,
            "N"                  : self.fuel_n,
            "PORT_NUM"           : self.port_num,
            "PORT_CENTER_RADIUS" : self.port_center_radius,
            "GRID_SIZE"          : self.grid_size,
            "GRAIN_TYPE"         : self.grain_type,
            "FIN_NUM"            : self.fin_num,
            "FIN_LENGTH"         : self.fin_length,
            "FIN_WIDTH"          : self.fin_width,
            "FIN_TIP_WIDTH"      : self.fin_tip_width,
        }

        self.fuel = Fuel(fuel_config)

        combution_chamber_config = {
            "T" : self.temperature,
            "P" : 101325,
            "FUEL" : self.fuel,
            "CSTAR_ETA" : self.cstar_eta,
            "CSTAR_TAU" : self.cstar_tau,
            "CSTAR_STARTUP_DEFICIT" : self.cstar_startup_deficit,
            "CHAMBER_RADIUS" : self.combution_chamber_radius,
            "CHAMBER_LENGTH" : self.combution_chamber_length,
        }

        self.combution_chamber = CombutionChamber(combution_chamber_config)

        nozzle_config = {
            "EPS" : self.eps,
            "THROAT_RADIUS" : self.nozzle_throat_radius
        }

        self.nozzle = Nozzle(nozzle_config)

        self.engine = HybridEngine(self.oxidizer_tank, self.injector, self.combution_chamber, self.nozzle)


    def ignite(self):
        # 1. 시뮬레이션 수행 (핵심 물리 계산)
        self.engine.simulation(self.dt, self.simt, self.thrust_coeff)

        # 시뮬레이션 종료 사유 (웹 UI 알람용)
        self.termination_code    = getattr(self.engine, "termination_code", None)
        self.termination_message = getattr(self.engine, "termination_message", None)

        # 2. 데이터 수집
        t = self.engine.time_step[:-1]

        # 딕셔너리로 한꺼번에 묶기
        plot_vars = {
            "Time (s)": t, # 시간을 미리 포함
            "Tank Temperature (K)": self.engine.tank_temperature[:-1],
            "Tank Pressure (Pa)": self.engine.tank_pressure[:-1],
            "Chamber Temperature (K)": self.engine.chamber_temperature[:-1],
            "Chamber Pressure (Pa)": self.engine.chamber_pressure[:-1],
            "Thrust (N)": self.engine.thrust[:-1],
            "Exhaust Velocity Ve (m/s)": self.engine.ve[:-1],
            "Isp (s)": self.engine.Isp[:-1],
            "Fuel Regression r_dot (m/s)": self.engine.r_dot[:-1],
            "mox_dot (kg/s)": self.engine.mox_dot[:-1],
            "mf_dot (kg/s)": self.engine.mf_dot[:-1],
            "me_dot (kg/s)": self.engine.me_dot[:-1],
            "O/F Ratio": self.engine.of[:-1],
            "Characteristic Velocity c* (m/s)": self.engine.cstar[:-1],
            "Thrust Coeff": self.engine.cf[:-1],
        }


        df = pd.DataFrame(plot_vars)

        df.to_csv("../result.csv", index=False, encoding='utf-8-sig')

        return df
