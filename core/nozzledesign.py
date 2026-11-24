import math
from dataclasses import dataclass

def A_ratio_from_M(M, gamma):
    term = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * M**2)
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return (1.0 / M) * term**exponent

def M_from_A_ratio(target_A_ratio, gamma, M_guess=3.0):
    if target_A_ratio < 1.0:
        raise ValueError("A/A* must be >= 1")
    M = M_guess
    for i in range(200):
        f = A_ratio_from_M(M, gamma) - target_A_ratio
        h = 1e-6
        df = (A_ratio_from_M(M + h, gamma) - A_ratio_from_M(M - h, gamma)) / (2*h)
        if abs(df) < 1e-12:
            break
        M_new = M - f/df
        if M_new <= 1.0:
            M_new = 1.0 + 1e-6
        if abs(M_new - M) < 1e-9:
            M = M_new
            break
        M = M_new
    return M

def pressure_ratio_from_M(M, gamma):
    return (1.0 + 0.5*(gamma-1.0)*M**2)**(-gamma/(gamma-1.0))

def temperature_ratio_from_M(M, gamma):
    return (1.0 + 0.5*(gamma-1.0)*M**2)**-1.0

def exit_velocity(M, gamma, R, T0):
    Te = T0 * temperature_ratio_from_M(M, gamma)
    a_e = math.sqrt(gamma * R * Te)
    return M * a_e

def choked_mass_flow_coeff(gamma):
    return math.sqrt(gamma) * (2.0/(gamma+1.0))**((gamma+1.0)/(2.0*(gamma-1.0)))

@dataclass
class DesignParams:
    F_req: float
    A_t: float
    epsilon0: float
    pa: float
    gamma: float
    R: float
    Tc: float
    max_iters: int = 50
    tol: float = 1e-5

def iterative_pc_and_epsilon(params: DesignParams, verbose=True):
    eps = params.epsilon0
    pc = None
    history = []
    for iteration in range(params.max_iters):
        M_e = M_from_A_ratio(eps, params.gamma, M_guess=3.0)
        p_e_over_pc = pressure_ratio_from_M(M_e, params.gamma)
        choked_coeff = choked_mass_flow_coeff(params.gamma)
        mdot_over_pcAt = math.sqrt(params.gamma/(params.R*params.Tc)) * choked_coeff
        Ve = exit_velocity(M_e, params.gamma, params.R, params.Tc)

        term1 = mdot_over_pcAt * Ve

        numerator = params.F_req + params.pa * eps * params.A_t
        denominator = params.A_t * (term1 + eps * p_e_over_pc)
        pc_new = numerator / denominator

        mdot = params.A_t * pc_new * math.sqrt(params.gamma/(params.R*params.Tc)) * choked_coeff

        target_pe_over_pc = params.pa / pc_new

        if not (0.0 < target_pe_over_pc < 1.0):
            ideal_eps = eps
        else:
            M_low = 1.000001
            M_high = 200.0
            M_mid = None
            for _ in range(200):
                M_mid = 0.5*(M_low + M_high)
                val = pressure_ratio_from_M(M_mid, params.gamma) - target_pe_over_pc
                if abs(val) < 1e-12:
                    break
                if val > 0:
                    M_low = M_mid
                else:
                    M_high = M_mid
            M_target = M_mid
            ideal_eps = A_ratio_from_M(M_target, params.gamma)

        history.append({
            "iter": iteration+1,
            "eps": eps,
            "M_e": M_e,
            "p_e_over_pc": p_e_over_pc,
            "pc_Pa": pc_new,
            "mdot_kg_s": mdot,
            "ideal_eps_for_pe_eq_pa": ideal_eps
        })

        #if verbose:
        #    print(f"반복 {iteration+1:02d} | eps={eps:.4f} | Me={M_e:.4f} | p_e/p_c={p_e_over_pc:.5f} | p_c={pc_new/1e5:.4f} bar | mdot={mdot:.3f} kg/s | ideal_eps(pe=pa)={ideal_eps:.4f}")

        eps_next = 0.5*eps + 0.5*ideal_eps
        if abs(eps_next - eps) < params.tol:
            eps = eps_next
            pc = pc_new
            break
        eps = eps_next
        pc = pc_new

    return history, pc, eps

params = DesignParams(
    F_req = 1500.0,
    A_t   = 0.00015,
    epsilon0 = 6.0,
    pa = 101325.0,
    gamma = 1.15,
    R = 300.0,
    Tc = 3048.0,
    max_iters=60,
    tol=1e-6
)

history, pc_final, eps_final = iterative_pc_and_epsilon(params, verbose=True)

print("\n최종 요약:")
print(f"  수렴한 팽창비 epsilon = {eps_final:.4f}")
print(f"  노즐 목 반지름 = {math.sqrt(params.A_t/math.pi)}")
print(f"  노즐 출구 반지름 = {math.sqrt(params.A_t*eps_final/math.pi)}")
print(f"  수렴한 챔버압 p_c = {pc_final/1e5:.4f} bar ({pc_final:.1f} Pa)")
print("  최근 반복 결과:")
for row in history[-6:]:
    print(f"   반복 {row['iter']:2d}: eps={row['eps']:.4f}, Me={row['M_e']:.4f}, p_c={row['pc_Pa']/1e5:.4f} bar, mdot={row['mdot_kg_s']:.3f} kg/s, ideal_eps={row['ideal_eps_for_pe_eq_pa']:.4f}")

first = history[0]
print("\n첫 반복(초기) 상세값:")
print(f"  초기 eps = {first['eps']:.4f}")
print(f"  초기 출력 마하 Me = {first['M_e']:.4f}")
print(f"  초기 p_e/p_c = {first['p_e_over_pc']:.6f}")
print(f"  계산된 초기 챔버압 p_c = {first['pc_Pa']:.3f} Pa")
print(f"  계산된 초기 질량유량 mdot = {first['mdot_kg_s']:.3f} kg/s")
