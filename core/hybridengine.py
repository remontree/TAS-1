# 필요한 라이브러리 임포트
import math
import json
import os
from dataclasses import dataclass
import CoolProp
from CoolProp.CoolProp import PropsSI
from rocketcea.cea_obj import CEA_Obj, add_new_fuel

from .combutionchamber import *
from .fuel import *
from .injector import *
from .matter import *
from .nozzle import *
from .oxidizertank import *

R = 8.3143  # 보편 기체 상수


# 하이브리드 엔진 클래스
class HybridEngine:
  def __init__(self, oxidizer_tank, injector, combution_chamber, nozzle):
    self.oxidizer_tank = oxidizer_tank
    self.injector = injector
    self.combution_chamber = combution_chamber
    self.nozzle = nozzle

    self.mox_dot_tmp = 0

  def simulation(self, dt, simt, thrust_coeff):
    t = 0

    # logger
    self.time_step = [0.0]
    #oxidizer_mass = []
    #fuel_mass = []
    self.tank_temperature = [self.oxidizer_tank.tankT]
    self.tank_pressure = [self.oxidizer_tank.tankP]
    self.chamber_temperature = [self.combution_chamber.chamberT]
    self.chamber_pressure = [self.combution_chamber.chamberP]
    self.thrust = [0.0]
    self.ve = [0.0]
    self.Isp = [0.0]
    self.of = [0.0]
    self.r_dot = [0.0]
    self.mox_dot = [0.0]
    self.mf_dot = [0.0]
    self.me_dot = [0.0]
    self.cstar = [0.0]
    self.cf = [0.0]


    self.impulse = 0

    # ── 종료 조건 추적 ────────────────────────────────────────────────
    # 시뮬레이션이 어떤 사유로 종료되었는지 웹 UI 알람에 노출하기 위한 상태
    self.termination_code    = None
    self.termination_message = None

    # 액체 소진 → 기체(증기) 블로우다운 전환 로그를 한 번만 찍기 위한 플래그
    self._vapor_phase_logged = False

    self.fuel = self.combution_chamber.fuel.name

    base_path = os.path.dirname(os.path.abspath(__file__))
    fuels_path = os.path.join(base_path, "../fuel_cea_card.json")

    with open(fuels_path, "r", encoding="utf-8") as f:
        fuels_data = json.load(f)

    # 2. 현재 설정된 연료(self.fuel)에 맞는 카드 문자열 가져오기
    # self.fuel이 "PARAFFIN"이면 PARAFFIN 카드를 가져옴
    current_fuel_card = fuels_data.get(self.fuel)



    if not current_fuel_card:
        raise ValueError(f"연료 '{self.fuel}'에 대한 CEA 카드가 데이터베이스에 없습니다.")

    # 3. [최소 수정] RocketCEA 등록 및 객체 생성
    # 연료 이름은 가독성을 위해 self.fuel을 그대로 사용하거나 카드 내 이름을 따름
    add_new_fuel(self.fuel, current_fuel_card)

    # 나사 CEA 객체 생성 (사용자 코드 반영)
    cea = CEA_Obj(oxName='N2O', fuelName=self.fuel)

    # TODO: 유효배기 속도, ISP, IMPULSE

    while t<=simt:
      t += dt
      #print(f"시간 {t}에서의 시뮬레이션 시작------------------------------------------------------------------------------------")

      """"
  NHNE 모델
  파라메터:
    - P1: 오리피스 상류 압력            self.oxidizer_tank.tankP
    - Pv1: 상류 유체 증기압             self.oxidizer_tank.oxidizer_liquid.Psat
    - ryo1: 상류층 유체 밀도           self.oxidizer_tank.oxidizer_liquid.rho
    - h1: 오리피스 윗 부분 엔탈피       self.oxidizer_tank.oxidizer_liquid.h
    - P2: 오리피스 하류 압력            self.combution_chamber.chamberP
    - ryo2: 하류층 유체 밀도
    - h2: 오리피스 아랫 부분 엔탈피
  """
      liquid_available = self.oxidizer_tank.oxidizer_liquid.n > 0

      if liquid_available:
        # 인젝터 내부로 흘러가는 유체의 레이놀즈 수 구하기
        ryo, meu = self.injector.get_ryo_meu_Pv_NitrousOxide(self.oxidizer_tank.tankP, self.oxidizer_tank.tankT, phase_hint="liquid")

        #ryo = PropsSI("D", "P", self.oxidizer_tank.tankP, "T", self.oxidizer_tank.tankT, self.oxidizer_tank.oxidizer_liquid.name)   # kg/m^3
        #meu = PropsSI("V", "P", self.oxidizer_tank.tankP, "T", self.oxidizer_tank.tankT, self.oxidizer_tank.oxidizer_liquid.name)

        #print(self.mox_dot_tmp)
        v = self.mox_dot_tmp / (ryo * self.injector.A)

        Re = self.injector.Re(v,ryo, meu)

        cd = self.injector.Cd_fluid(self.oxidizer_tank.tankP, self.oxidizer_tank.oxidizer_liquid.Psat, self.combution_chamber.chamberP, Re)

        h2, rho2 = self.combution_chamber.NHNE_rho2_h2(self.oxidizer_tank.oxidizer_liquid.name, self.oxidizer_tank.tankP, self.oxidizer_tank.oxidizer_liquid.Psat, self.oxidizer_tank.oxidizer_liquid.rho,
                                        self.oxidizer_tank.oxidizer_liquid.h, self.combution_chamber.chamberP, self.oxidizer_tank.oxidizer_liquid.s)

        #n_drain = self.injector.n_drain(cd, self.oxidizer_tank.tankP, self.oxidizer_tank.oxidizer_liquid.Psat, self.oxidizer_tank.oxidizer_liquid.rho, self.oxidizer_tank.oxidizer_liquid.h, self.combution_chamber.chamberP, rho2, h2) # 인젝터에서 챔버로 흘러나오는 산화제 몰유량


        #----------------------------------------------------------------------------------------
        fluid = self.oxidizer_tank.oxidizer_liquid.name  # "N2O"

        P1  = self.oxidizer_tank.tankP          # 탱크 압력 [Pa]
        P2  = self.combution_chamber.chamberP   # 연소실 압력 [Pa]
        T1  = self.oxidizer_tank.tankT          # 탱크 온도 [K]

        # ── 1. 포화 압력 (saturation pressure at T1) ──────────────────────────
        Pv = PropsSI('P', 'T', T1, 'Q', 0, fluid)   # [Pa]
        # 첫 번째 코드: p_v = 다항식(Tem_o) [MPa] 에 해당

        # ── 2. 탱크 상류 액체 상태량 (탱크 온도 T1 기준) ─────────────────────
        # [수정] 예전에는 P1 기준 포화 상태량을 썼는데, 가압제(He)로 P1 > Psat(T1)
        # 인 과냉 액체에서는 Tsat(P1)(≈290K)의 물성이 되어 실제 액체(≈255K)와
        # 크게 어긋났다. 인젝터로 들어가는 것은 바닥의 액체이므로 (P1, T1)
        # 압축액 상태를 쓰고, 자가가압(P1≈Psat)일 때는 T1 포화액으로 둔다.
        if P1 > Pv * (1.0 + 1e-6):
            rho_L      = PropsSI('D', 'P', P1, 'T', T1, fluid)   # 압축(과냉) 액체 밀도 [kg/m³]
            h_upstream = PropsSI('H', 'P', P1, 'T', T1, fluid)   # [J/kg]
            s_total    = PropsSI('S', 'P', P1, 'T', T1, fluid)   # [J/kg·K]
        else:
            rho_L      = PropsSI('D', 'T', T1, 'Q', 0, fluid)    # 포화 액체 밀도 [kg/m³]
            h_upstream = PropsSI('H', 'T', T1, 'Q', 0, fluid)
            s_total    = PropsSI('S', 'T', T1, 'Q', 0, fluid)

        # [수정] 인젝터 유량을 하류 압력 P2의 함수로 묶는다. 상류 상태량과 cd는
        # 이번 스텝 값으로 고정하고, 챔버압 갱신(combustion_chamber_pressure)이
        # 새 챔버압에서 이 함수를 다시 불러 유량과 압력을 함께(암시적으로) 푼다.
        # 예전처럼 직전 스텝 Pc로 구한 유량을 고정해 쓰면 인젝터 차압이 작을 때
        # (오리피스가 크면 ΔP가 Pc의 수 %) d(mox)/d(Pc)가 너무 가팔라 스텝마다
        # Pc가 탱크압을 넘나들며 진동했다.
        if cd <= 0.0:
          # 직전 Pc ≥ P1이면 Cd_fluid가 0을 돌려주므로 차압이 남는 쪽에서 다시 구한다
          cd = self.injector.Cd_fluid(P1, self.oxidizer_tank.oxidizer_liquid.Psat, 0.99 * P1, Re)
        A_inj, N_inj = self.injector.A, self.injector.N
        P2_MIN = 0.9e5   # N2O 삼중점 압력(0.878 bar) 아래에서는 포화 상태량이 없음

        def liquid_mox(P2):
          P2 = max(P2, P2_MIN)
          dP1 = P1 - P2   # 전체 압력 강하 [Pa]
          dP2 = Pv - P2   # 증기압 기준 압력 강하 [Pa]

          # [수정] Dyer NHNE: mdot = Cd·A·[ κ/(1+κ)·SPI + 1/(1+κ)·HEM ],
          #   κ = sqrt(dP1/dP2). 예전 식은 가중치가 뒤바뀌어(SPI에 1/(1+κ))
          #   Pc→Pv에서 SPI 기여가 sqrt(Pv-Pc)로 0이 되고, Pc>Pv면 mox=0으로
          #   끊겼다. 이 급경사/불연속 때문에 Pc가 Psat(T1) 바로 아래에 묶여
          #   스텝마다 진동했다. Pc ≥ Pv(플래싱 없음)는 κ→∞ 극한인 순수 SPI.
          if dP1 <= 0:
            return 0.0
          m_spi = math.sqrt(2.0 * rho_L * dP1)   # [kg/(m²·s)]
          if dP2 <= 0:
            return cd * A_inj * m_spi * N_inj

          # ── 4. 챔버 압력에서의 등엔트로피 팽창 하류 상태 (chi_star) ───────────
          s_L2   = PropsSI('S', 'P', P2, 'Q', 0, fluid)
          s_v2   = PropsSI('S', 'P', P2, 'Q', 1, fluid)
          h_L2   = PropsSI('H', 'P', P2, 'Q', 0, fluid)
          h_v2   = PropsSI('H', 'P', P2, 'Q', 1, fluid)
          rho_L2 = PropsSI('D', 'P', P2, 'Q', 0, fluid)
          rho_v2 = PropsSI('D', 'P', P2, 'Q', 1, fluid)

          chi_star = (s_total - s_L2) / (s_v2 - s_L2)   # 등엔트로피 팽창 후 건도
          chi_star = max(0.0, min(1.0, chi_star))

          h_out  = h_L2 * (1.0 - chi_star) + h_v2 * chi_star           # [J/kg]
          rho_2  = 1.0 / (chi_star / rho_v2 + (1.0 - chi_star) / rho_L2)   # [kg/m³]
          # 첫 번째 코드의 rho_l_2, rho_v_2, h_out 다항식에 해당

          h_diff = h_upstream - h_out   # [J/kg]

          # HEM 항 (균질 평형 모델 항): rho_2 * sqrt(2 * Δh)
          m_hem = rho_2 * math.sqrt(max(0.0, 2.0 * h_diff))

          # NHNE 가중치 κ/(1+κ) = sqrt(dP1) / (sqrt(dP1) + sqrt(dP2))
          w_spi = math.sqrt(dP1) / (math.sqrt(dP1) + math.sqrt(dP2))

          return cd * A_inj * (w_spi * m_spi + (1.0 - w_spi) * m_hem) * N_inj

        ox_mdot_fn = liquid_mox
        mox_dot = liquid_mox(P2)

        # mox_dot [kg/s] → n_drain [mol/s]
        MW = self.oxidizer_tank.oxidizer_liquid.MW  # [kg/mol]

        n_drain = -mox_dot / MW  # [mol/s]

      else:
        # ── 액체 소진 후: 남은 기체(증기) N2O만으로 이어지는 블로우다운 ──
        # (액체 NHNE 모델은 위 if 분기에서 그대로 유지, 손대지 않음)
        if not self._vapor_phase_logged:
          print("액체 산화제 전부 소진 - 잔여 기체(증기)로 블로우다운 계속")
          self._vapor_phase_logged = True

        if self.oxidizer_tank.oxidizer_vapor.n <= 1e-9:
          # 증기까지 완전히 소진 — 더 이상 흘려보낼 산화제가 없음
          ox_mdot_fn = lambda P2: 0.0
        else:
          P_up, T_up = self.oxidizer_tank.tankP, self.oxidizer_tank.tankT
          vap_name = self.oxidizer_tank.oxidizer_vapor.name
          ox_mdot_fn = lambda P2: self.injector.get_vapor_mdot(P_up, T_up, P2, vap_name)
        mox_dot = ox_mdot_fn(self.combution_chamber.chamberP)
        MW = self.oxidizer_tank.oxidizer_vapor.MW
        n_drain = -mox_dot / MW  # [mol/s] (부기용, vapor_blowdown_update는 mox_dot을 직접 사용)

      #----------------------------------------------------------------------------------------

      #n_drain = self.injector.n_drain(self.oxidizer_tank.tankP, self.combution_chamber.chamberP, self.oxidizer_tank.oxidizer_liquid.MW, self.oxidizer_tank.oxidizer_liquid.v_hat)

      #print("산화제 탱크 압력: ", self.oxidizer_tank.tankP)

      #print("산화제 탱크에서 인젝터를 거쳐 연소 챔버로 흘러오는 몰유량: ",n_drain)

      #mox_dot = -n_drain * self.oxidizer_tank.oxidizer_liquid.MW  # 인젝터에서 챔버로 흘러나오는 산화제 질유량

      self.mox_dot_tmp = mox_dot


      #print("산화제 탱크에서 인젝터를 거쳐 연소 챔버로 흘러오는 질유량: ",mox_dot, "kg/s")

      Aburn, mf_dot, r_dot = self.combution_chamber.fuel.burn(mox_dot, dt) # 연료 연소로 인한 연소 면적, 연료 질유량

      #print("연소 면적: ", Aburn, "연료 질유량: ", mf_dot, "후퇴율: ", r_dot)

      # 연료 전부 소진 시 시뮬레이션 종료
      if Aburn == 0:
        print("연료 전부 소진")
        self.termination_code    = "FUEL_DEPLETED"
        self.termination_message = "연료가 전부 소진되어 시뮬레이션이 종료되었습니다."
        self.time_step.append(t+dt)
        self.tank_temperature.append(self.tank_temperature[-1])
        self.tank_pressure.append(0)
        self.chamber_temperature.append(self.chamber_temperature[-1])
        self.chamber_pressure.append(0)
        self.thrust.append(0)
        self.ve.append(0.0)
        self.Isp.append(0.0)
        self.of.append(0.0)
        self.r_dot.append(0.0)
        self.mox_dot.append(0.0)
        self.mf_dot.append(0.0)
        self.me_dot.append(0.0)
        self.cstar.append(0.0)
        self.cf.append(0.0)
        break

      # 기체 블로우다운 말미에 산화제 유량이 사실상 0으로 수렴하면(연소가
      # 사실상 끝난 상태) OF_ratio가 0/0에 가까워져 CEA 호출이 불안정해지므로,
      # CEA/추력 계산으로 넘어가기 전에 여기서 추력 0N으로 마무리한다.
      if not liquid_available and mox_dot < 1e-6:
        print("기체 산화제까지 소진되어 추력 0N으로 종료")
        self.termination_code    = "OXIDIZER_FULLY_DEPLETED"
        self.termination_message = "액체와 기체 산화제가 모두 소진되어 추력이 0N에 도달, 시뮬레이션이 종료되었습니다."
        self.time_step.append(t+dt)
        self.tank_temperature.append(self.tank_temperature[-1])
        self.tank_pressure.append(self.tank_pressure[-1])
        self.chamber_temperature.append(self.chamber_temperature[-1])
        self.chamber_pressure.append(0)
        self.thrust.append(0)
        self.ve.append(0.0)
        self.Isp.append(0.0)
        self.of.append(0.0)
        self.r_dot.append(0.0)
        self.mox_dot.append(0.0)
        self.mf_dot.append(0.0)
        self.me_dot.append(0.0)
        self.cstar.append(0.0)
        self.cf.append(0.0)
        break

      OF_ratio = mox_dot/mf_dot if mf_dot > 1e-12 else 0.0 # O/F 비 구하기

      #print("OF 비: ", OF_ratio)

      #card_str = """
      #fuel HTPB(L)   C 10.0   H 15.0   wt%=100.0
      #h,cal=-32000.0    t(k)=298.15   rho.g/cc=0.92a
      #"""

      # RocketCEA에 HTPB 연료 등록
      #add_new_fuel('HTPB', card_str)

      # CEA 객체 생성
      #cea = CEA_Obj(oxName='N2O', fuelName='HTPB')

      Pc = self.combution_chamber.chamberP   # Chamber pressure [Pa]
      MR = OF_ratio    # Mixture ratio O/F

      rho_g = cea.get_Chamber_Density(Pc*0.000145037737, MR, self.nozzle.eps)

      factor = 0.4539237/0.028316846592

      rho_g *= factor

      #print("챔버 내부 압력: ", Pc, "챔버 내부 밀도: ", rho_g)

      mw, gamma = cea.get_Chamber_MolWt_gamma(Pc*0.000145037737, MR, self.nozzle.eps)

      mw /= 1000


      #print("챔버 내부 분자량: ", mw, "챔버 내부 비열비: ", gamma)

      # [수정] Cstar는 self.chamberT를 쓰는데, 예전엔 이 값이 직전 스텝 MR로
      # 계산된 온도라 이번 스텝 gamma/MW(현재 MR)와 짝이 맞지 않았다
      # (MR이 급변하면 c*가 807↔2078처럼 튀는 원인). 현재 Pc, MR로 먼저 갱신.
      self.combution_chamber.chamberT = cea.get_Tcomb(Pc*0.000145037737, MR)/1.8

      # Cstar 구하기
      cstar = self.combution_chamber.Cstar(gamma, R, mw, t)
      #cstar = cea.get_Cstar(Pc*0.000145037737,MR) 단위 변환 문제인지 값이 이상하게 나옴

      #print("특성 속도: ", cstar)

      # 노즐 출구로 나가는 질유량
      #me_dot = self.nozzle.m_dot(dt, self.combution_chamber.chamberP, cstar, OF_ratio)
      me_dot = self.nozzle.m_dot(self.combution_chamber.chamberP, cstar)

      """me_dot_ss = self.nozzle.m_dot(self.combution_chamber.chamberP, cstar)

      # (선택) 노즐 1차 지연
      tau_n = 0.003
      alpha = dt / (tau_n + dt)
      me_dot = (1 - alpha) * self.me_dot[-1] + alpha * me_dot_ss

      # 질량 업데이트 전 배출 질량 제한
      M_MIN = 1e-6
      dm_in = (mox_dot + mf_dot) * dt
      dm_out = me_dot * dt

      dm_out = min(dm_out, self.combution_chamber.m + dm_in - M_MIN)
      me_dot = dm_out / dt

      # 질량 업데이트
      self.combution_chamber.m = max(self.combution_chamber.m + dm_in - dm_out, M_MIN)

      # 압력 업데이트 (exp 폭주 제한은 함수 내부에 넣는 게 깔끔)
      self.combution_chamber.chamberP = self.combution_chamber.combustion_chamber_pressure(
          self.combution_chamber.chamberP,
          self.combution_chamber.m,
          mox_dot, mf_dot, me_dot,
          rho_g, r_dot, Aburn, dt
      )"""

      #print("노즐 출구로 나가는 질유량: ", me_dot)

      self.combution_chamber.m += mf_dot*dt + mox_dot*dt - me_dot*dt

      #print("챔버 내부 가스 질량: ", self.combution_chamber.m)

      # [수정] 챔버압을 인젝터 유량과 함께 암시적으로 푼다. 연료 유량은 이번
      # 스텝 그레인 형상 그대로 r ∝ G^n 이므로 mf ∝ mox^n 으로 스케일한다.
      fuel_n = self.combution_chamber.fuel.n
      mox_ref, mf_ref = mox_dot, mf_dot
      def mf_of(mx):
        if mox_ref <= 1e-12 or mf_ref <= 0.0:
          return 0.0
        return mf_ref * (mx / mox_ref) ** fuel_n
      def inflow_fn(P):
        mx = ox_mdot_fn(P)
        return mx + mf_of(mx)

      self.combution_chamber.chamberP = self.combution_chamber.combustion_chamber_pressure(self.combution_chamber.chamberP, self.combution_chamber.m, mox_dot, mf_dot, me_dot, rho_g, r_dot, Aburn, dt, 101325, self.nozzle.At, cstar, gamma, inflow_fn=inflow_fn)

      # 새 챔버압과 짝이 맞는 유량으로 기록·탱크 배출량을 맞춘다 (그레인 후퇴는
      # 위 burn()에서 스텝 시작 유량으로 이미 진행됨 — 정상 구간에선 차이 미미)
      mox_dot = ox_mdot_fn(self.combution_chamber.chamberP)
      mf_dot  = mf_of(mox_dot)
      OF_ratio = mox_dot/mf_dot if mf_dot > 1e-12 else 0.0
      self.mox_dot_tmp = mox_dot
      if liquid_available:
        n_drain = -mox_dot / self.oxidizer_tank.oxidizer_liquid.MW

      # [수정] 노즐 배출 유량도 새 챔버압으로 다시 구한다. 예전엔 갱신 전(직전 스텝)
      # Pc로 구한 me_dot을 그대로 기록해, 추력(새 Pc)/me_dot(옛 Pc) 비율로 Ve·Isp를
      # 계산했다. 점화 첫 스텝은 Pc가 1 atm → 3.5 MPa로 뛰어 Ve≈67,000 m/s,
      # Isp≈6,900 s 같은 가짜 스파이크가 나왔다.
      me_dot = self.nozzle.m_dot(self.combution_chamber.chamberP, cstar)

      #print("챔버 내부 압력 업데이트 후 값: ", self.combution_chamber.chamberP)

      # [수정] 압력 갱신 후 다시 맞춘 O/F로 연소 온도를 구한다 (예전엔 갱신 전 MR 사용)
      MR = OF_ratio
      self.combution_chamber.chamberT = cea.get_Tcomb(self.combution_chamber.chamberP*0.000145037737, MR)/1.8

      #print("챔버 내부 온도 업데이트 후 값: ", self.combution_chamber.chamberT)

      # 노즐 출구 마하수 구하기
      Me = self.nozzle.MeFromEps(gamma)

      #print("노즐 출구 마하수: ", Me)

      # 노즐 출구 압력 구하기

      Pe = self.combution_chamber.chamberP/self.nozzle.PressureRatio(gamma, Me)


      # 추력 계수 구하기
      Cf = self.nozzle.Cf(gamma, self.combution_chamber.chamberP, Pe)

      # 추력 구하기

      Thrust = thrust_coeff*Cf*self.combution_chamber.chamberP*self.nozzle.At

      #print(thrust_coeff,Cf,self.combution_chamber.chamberP,self.nozzle.At)

      #print("추력: ", Thrust)

      # 상태 로깅
      self.time_step.append(t)
      self.tank_temperature.append(self.oxidizer_tank.tankT)
      self.tank_pressure.append(self.oxidizer_tank.tankP)
      self.chamber_temperature.append(self.combution_chamber.chamberT)
      self.chamber_pressure.append(self.combution_chamber.chamberP)
      self.thrust.append(Thrust)
      self.impulse += Thrust*dt
      self.ve.append(Thrust/(me_dot))
      self.Isp.append(Thrust/(me_dot*9.8))
      self.of.append(OF_ratio)
      self.r_dot.append(r_dot)
      self.mox_dot.append(mox_dot)
      self.mf_dot.append(mf_dot)
      self.me_dot.append(me_dot)
      self.cstar.append(cstar)
      self.cf.append(Cf)



      if liquid_available:
        self.oxidizer_tank.tank_state_update(n_drain, dt)
      else:
        self.oxidizer_tank.vapor_blowdown_update(mox_dot, dt)

      # 외부 압력이 더 커지면 시뮬레이션 종료
      if Pe >= self.combution_chamber.chamberP:
        print("압력이 외부보다 작아서 종료 됨")
        self.termination_code    = "CHAMBER_PRESSURE_BELOW_AMBIENT"
        self.termination_message = "챔버 압력이 외부 대기압보다 낮아져 시뮬레이션이 종료되었습니다."
        self.time_step.append(t+dt)
        self.tank_temperature.append(self.tank_temperature[-1])
        self.tank_pressure.append(0)
        self.chamber_temperature.append(self.chamber_temperature[-1])
        self.chamber_pressure.append(0)
        self.thrust.append(0)
        self.ve.append(0.0)
        self.Isp.append(0.0)
        self.of.append(0.0)
        self.r_dot.append(0.0)
        self.mox_dot.append(0.0)
        self.mf_dot.append(0.0)
        self.me_dot.append(0.0)
        self.cstar.append(0.0)
        self.cf.append(0.0)

        break

      # (기존 "액체 산화제 전부 소진 시 시뮬레이션 종료" 지점이었던 곳.
      #  이제 액체 소진은 종료가 아니라 기체 블로우다운으로의 전환일 뿐이며,
      #  다음 반복에서 liquid_available=False로 자동 전환된다. 실제 최종
      #  종료는 위쪽의 "기체 산화제까지 소진되어 추력 0N" 분기에서 처리한다.)

    else:
      # break 없이 while 조건이 거짓이 되어 루프가 끝난 경우 (설정 시간 도달, 정상 종료)
      self.termination_code    = "TIME_LIMIT_REACHED"
      self.termination_message = f"설정된 시뮬레이션 시간({simt}s)에 도달하여 정상 종료되었습니다."
