# 필요한 라이브러리 임포트
import math
from dataclasses import dataclass
import CoolProp
from CoolProp.CoolProp import PropsSI
from rocketcea.cea_obj import CEA_Obj, add_new_fuel

R = 8.3143  # 보편 기체 상수

# 노즐 클래스

class Nozzle:
  def __init__(self, config):
    self.eps = config["EPS"]     # 노즐 확장비 A/At
    self.throat_radius = config["THROAT_RADIUS"]       # 노즐 목 반지름 [m]
    self.At = self.throat_radius**2*math.pi            # 노즐 목 면적 [m^2]

  # P0/P
  def PressureRatio(self, k, Me):
    return (1+(k-1)/2*Me**2)**(k/(k-1))

  # 출구 질유량
  def m_dot(self, Pc, Cstar):
    return self.At*Pc/Cstar

  # 출구 마하수로 부터 노즐 확장비 구하는 함수
  def EpsFromMe(self, k, Me):
      term = (2/(k+1)) * (1 + (k-1)/2 * Me**2)
      return (1/Me) * term**((k+1)/(2*(k-1)))

  # 노즐 확장비로 부터 출구 마하수 구하는 함수
  def MeFromEps(self, k, M_guess=3.0):
    if self.eps < 1:
      raise ValueError("eps must be greater than 1")

    M = M_guess
    for i in range(200):
      f = self.EpsFromMe(k, M) - self.eps
      h = 1e-6
      df = (self.EpsFromMe(k, M + h) - self.EpsFromMe(k, M - h)) / (2 * h)
      M_new = M - f / df
      if M_new <= 1.0:
        M_new = 1.0 + 1e-6
      if abs(M_new - M) < 1e-10:
        M = M_new
        break
      M = M_new
    return M

  # 추력 계수
  def Cf(self, k, Pc, Pe, Pa = 101325):
    return k*(2/(k+1))**((k+1)/(2*(k-1)))*math.sqrt((2/(k-1))*(1-(Pe/Pc)**((k-1)/k)))+self.eps*(Pe-Pa)/Pc

"""
# 필요한 라이브러리 임포트
import math
from dataclasses import dataclass
import CoolProp
from CoolProp.CoolProp import PropsSI
from rocketcea.cea_obj import CEA_Obj, add_new_fuel

R = 8.3143  # 보편 기체 상수

# 노즐 클래스
class Nozzle:
  def __init__(self, config):
    self.eps = config["EPS"]                           # 초기 노즐 확장비 A/At
    self.throat_radius_0 = config["THROAT_RADIUS"]     # 초기 노즐 목 반지름 [m] (기준값)

    self.throat_radius = self.throat_radius_0          # 현재 노즐 목 반지름 [m]
    self.At = self.throat_radius**2 * math.pi          # 초기 노즐 목 면적 [m^2]

    # 출구 면적은 침식과 무관하게 물리적으로 고정되어 있으므로 미리 계산해 둡니다.
    self.Ae = self.eps * self.At

  # P0/P
  def PressureRatio(self, k, Me):
    return (1+(k-1)/2*Me**2)**(k/(k-1))

  # 출구 질유량 (삭마율 및 At, eps 업데이트 하드 코딩)
  def m_dot(self, t, Pc, Cstar):
    # 1. 삭마율 하드 코딩 (예: 초당 0.1 mm = 0.0001 m/s)
    # 원하시는 수치로 직접 수정해서 사용하시면 됩니다.
    erosion_rate = 0.00008

    # 2. 시간 t[s]에 따른 목 반지름 및 면적 업데이트
    self.throat_radius = self.throat_radius_0 + (erosion_rate * t)
    self.At = self.throat_radius**2 * math.pi

    # 3. 출구 면적(Ae)은 고정되어 있으므로, 커진 목 면적(At)을 반영해 확장비(eps) 갱신
    self.eps = self.Ae / self.At

    # 4. 업데이트된 At를 바탕으로 질유량 계산
    return self.At * Pc / Cstar

  # 출구 마하수로 부터 노즐 확장비 구하는 함수
  def EpsFromMe(self, k, Me):
      term = (2/(k+1)) * (1 + (k-1)/2 * Me**2)
      return (1/Me) * term**((k+1)/(2*(k-1)))

  # 노즐 확장비로 부터 출구 마하수 구하는 함수
  def MeFromEps(self, k, M_guess=3.0):
    if self.eps < 1:
      raise ValueError("eps must be greater than 1")

    M = M_guess
    for i in range(200):
      f = self.EpsFromMe(k, M) - self.eps
      h = 1e-6
      df = (self.EpsFromMe(k, M + h) - self.EpsFromMe(k, M - h)) / (2 * h)
      M_new = M - f / df
      if M_new <= 1.0:
        M_new = 1.0 + 1e-6
      if abs(M_new - M) < 1e-10:
        M = M_new
        break
      M = M_new
    return M

  # 추력 계수
  def Cf(self, k, Pc, Pe, Pa = 101325):
    # 여기서 사용되는 self.eps는 m_dot 실행 시 자동으로 업데이트된 최신 값입니다.
    return k*(2/(k+1))**((k+1)/(2*(k-1)))*math.sqrt((2/(k-1))*(1-(Pe/Pc)**((k-1)/k)))+self.eps*(Pe-Pa)/Pc"""

"""  # 필요한 라이브러리 임포트
import math
from dataclasses import dataclass
import CoolProp
from CoolProp.CoolProp import PropsSI
from rocketcea.cea_obj import CEA_Obj, add_new_fuel

R = 8.3143  # 보편 기체 상수

# 노즐 클래스
class Nozzle:
  def __init__(self, config):
    self.eps = config["EPS"]                           # 초기 노즐 확장비 A/At
    self.throat_radius_0 = config["THROAT_RADIUS"]     # 초기 노즐 목 반지름 [m] (기준값)

    # 설정값에 흑연 밀도가 없을 경우 논문의 기본값인 1850.0 kg/m^3 사용
    self.rho_s = config.get("RHO_S", 1850.0)

    self.throat_radius = self.throat_radius_0          # 현재 노즐 목 반지름 [m]
    self.At = self.throat_radius**2 * math.pi          # 초기 노즐 목 면적 [m^2]

    # 출구 면적은 침식과 무관하게 물리적으로 고정되어 있으므로 미리 계산해 둡니다.
    self.Ae = self.eps * self.At

  # P0/P
  def PressureRatio(self, k, Me):
    return (1+(k-1)/2*Me**2)**(k/(k-1))

  # 출구 질유량 (논문 기반 삭마율 반영)
  def m_dot(self, dt, Pc, Cstar, of_ratio=5.0):

    # 1. 논문(Eq. 15) 기반 삭마율 계산 (N2O-HDPE 기준)
    # 연소실 압력(Pc)은 Pa 단위로 가정하여 bar로 변환
    pc_bar = Pc / 100000.0

    # 삭마 질량 플럭스 (kg/(m^2 * s))
    term1 = of_ratio ** -1.466
    term2 = math.exp((-19.82 / of_ratio) + 1.568)
    power_pc = 0.0003627 * (of_ratio ** 2) - 0.01624 * of_ratio + 0.8168
    term3 = pc_bar ** power_pc
    mass_flux = term1 * term2 * term3

    # 선형 침식률 (m/s) = 삭마 질량 플럭스 / 흑연 밀도(kg/m^3)
    erosion_rate = mass_flux / self.rho_s

    # 2. 시간 t[s]에 따른 목 반지름 및 면적 업데이트
    # (참고: 만약 t가 누적 시간이라면 현재 코드가 맞으나, 매 타임스텝마다 O/F와 Pc가 바뀌는
    # 정밀한 과도 해석에서는 dt 단위로 삭마율을 누적하는 방식을 권장합니다.)
    self.throat_radius += erosion_rate * dt
    self.At = self.throat_radius**2 * math.pi

    # 3. 출구 면적(Ae)은 고정되어 있으므로, 커진 목 면적(At)을 반영해 확장비(eps) 갱신
    self.eps = self.Ae / self.At

    # 4. 업데이트된 At를 바탕으로 질유량 계산
    return self.At * Pc / Cstar

  # 출구 마하수로 부터 노즐 확장비 구하는 함수
  def EpsFromMe(self, k, Me):
      term = (2/(k+1)) * (1 + (k-1)/2 * Me**2)
      return (1/Me) * term**((k+1)/(2*(k-1)))

  # 노즐 확장비로 부터 출구 마하수 구하는 함수
  def MeFromEps(self, k, M_guess=3.0):
    if self.eps < 1:
      raise ValueError("eps must be greater than 1")

    M = M_guess
    for i in range(200):
      f = self.EpsFromMe(k, M) - self.eps
      h = 1e-6
      df = (self.EpsFromMe(k, M + h) - self.EpsFromMe(k, M - h)) / (2 * h)
      M_new = M - f / df
      if M_new <= 1.0:
        M_new = 1.0 + 1e-6
      if abs(M_new - M) < 1e-10:
        M = M_new
        break
      M = M_new
    return M

  # 추력 계수
  def Cf(self, k, Pc, Pe, Pa = 101325):
    # 여기서 사용되는 self.eps는 m_dot 실행 시 자동으로 업데이트된 최신 값입니다.
    return k*(2/(k+1))**((k+1)/(2*(k-1)))*math.sqrt((2/(k-1))*(1-(Pe/Pc)**((k-1)/k)))+self.eps*(Pe-Pa)/Pc"""
