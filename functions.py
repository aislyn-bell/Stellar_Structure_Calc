from constants import *
from properties import *
from new_interpolator import *
from scipy.integrate import solve_ivp
import scipy

def P_from_kappa(rho,T,g):
    return (2 / 3) * g / interp(log_R, log_T, solar_table, rho, T)

def normal_P(rho, T):
    return (1/3)*a*T**4 + rho*Na*k*T/mu

def P_diff(rho,T,g):
    Pdiff = 1 - (P_from_kappa(rho, T, g) / normal_P(rho, T))
    return Pdiff

def density(P, T):
    rho = mu/(Na * k * T) * (P - 1/3 * a * (T**4))
    return rho

def f11(rho, T):
    T7 = T / 1e7
    if T7 < 1:
        psi = 1
    elif T7 > 3.5:
        psi = 1.4
    else:
        psi = .15 * T7 + .85
    return np.exp(5.92e-3 * (rho / T7 ** 3) ** (1 / 2)) * psi

def nuclear_en_gen(T, rho):
    T9 = T / 1e9
    g11 = 1 + (3.82 * T9) + (1.51 * T9 ** 2) + (.144 * T9 ** 3) - (.0114 * T9 ** 4)
    epp = 2.57e4 * f11(rho, T) * g11 * rho * (X ** 2) * (T9 ** (-2 / 3)) * np.exp(-3.381 / T9 ** (1 / 3))
    g141 = 1 - (2.00 * T9) + (3.41 * T9 ** 2) - (2.43 * T9 ** 3)
    ecno = 8.24e25 * g141 * XCNO * X * rho * T9 ** (-2 / 3) * np.exp(-15.231 * T9 ** (-1 / 3) - (T9 / .8) ** 2)
    return epp + ecno


def load1(Pc, Tc):
    '''core values for solve_IVP'''
    if not np.isfinite(Pc) or Pc <= 0:
        print(f"Invalid Pc = {Pc}, using default")
        Pc = 1e17
    
    if not np.isfinite(Tc) or Tc <= 0:
        print(f"Invalid Tc = {Tc}, using default")
        Tc = 1.5e7

    rhoc = density(Pc, Tc)
    if not np.isfinite(rhoc) or rhoc <= 0:
        print(f"density() returned invalid rhoc = {rhoc}")
        rhoc = 100.0  # Reasonable central density for solar-type star

    energy_center = nuclear_en_gen(Tc, rhoc)
    if not np.isfinite(energy_center):
        print(f"nuclear_en_gen returned nan")
        energy_center = 0.0
    kappa_c = interp(log_R, log_T, solar_table, rhoc, Tc) ### this is breaking everything 
    if not np.isfinite(kappa_c) or kappa_c <= 0:
        print(f"Invalid kappa_c, using default")
        kappa_c = 1.0

    Lr = energy_center * Mc
    Pr = Pc - 3*G/(8*np.pi)*((4*np.pi*rhoc / 3))**(4/3) * Mc **(2/3)
    Rr= (3/(4*np.pi*rhoc))**(1/3) * Mc **(1/3)
    if not np.isfinite(Lr):
        Lr = 0.0
    if not np.isfinite(Pr) or Pr <= 0:
        Pr = 0.9 * Pc
    if not np.isfinite(Rr) or Rr <= 0:
        Rr = 1e9

    ### Is this portion of the star radiative or convective? ###
    nabla_rad = (3 / (16 * np.pi * a * c)) * (Pc * kappa_c / (Tc ** 4)) * (Lr / (G * Mc))
    nabla_ad = .4

    Tr = 0
    if nabla_ad >= nabla_rad:
        Tr = (Tc **4 - 1/(2*a*c) * (3/(4*np.pi))**(2/3) * kappa_c * energy_center * rhoc**(4/3) * Mc**(2/3))**(1/4)
    elif nabla_rad > nabla_ad:
        logTr = np.log(Tc) - (np.pi/6)**(1/3) * G * nabla_ad * rhoc**(4/3) * Mc**(2/3) / Pc
        Tr = np.exp(logTr)
    else:
        print("Radiative transport is broken :/")

    return [Lr, Pr, Rr, Tr]


def load2(Ltot, Rtot):
    '''surface values for solve_IVP'''
    g = G * M_star / (Rtot ** 2)
    Ttot = (Ltot/(4*np.pi*(Rtot**2)*sb))**(1/4)
    rhotot = scipy.optimize.fsolve(P_diff, x0=[1e-8], args=(Ttot, g))[0]
    kappas = interp(log_R, log_T, solar_table, rhotot, Ttot)
    Ptot = 2*g / (3*kappas)
    if Ltot <= 0 or Rtot <= 0 or not np.isfinite(Ltot) or not np.isfinite(Rtot):
        print(f'Invalid inputs in load 2: Ltot={Ltot}, Rtot={Rtot}')
        return (1e33, 1e10, 5000, Rtot if Rtot > 0 else 1e10, 1e-10)

    return [Ltot, Ptot, Rtot, Ttot]


def derivs(m, y):
    rho = density(y[1], y[3])
    kappa = interp(log_R, log_T, solar_table, rho, y[3])
    del_rad = (3 / (16 * np.pi * a * c)) * (y[1] * kappa / (y[3] ** 4)) * (y[0] / (G*m))
    del_ad = .4
    nabla = np.minimum(del_rad, del_ad)

    dldm = nuclear_en_gen(y[3], rho)
    dPdm = -(G*m)/(4*np.pi*y[2]**4)
    drdm = 1/(4*np.pi*y[2]**2*rho)
    dTdm = -(G*m*y[3])/(4*np.pi*(y[2]**4)*y[1]) * nabla

    return [dldm, dPdm, drdm, dTdm]

def radgrad(P, T, L, kappa, M):
    return (3 / (16 * np.pi * a * c)) * (P * kappa / (T ** 4)) * (L / (G * M))