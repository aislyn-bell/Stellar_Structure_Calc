from functions import *
from constants import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def integrate(params):
    Lr, Pr, Rr, Tr = params
    inner_conditions = load1(Pr, Tr)
    outer_conditions = load2(Lr, Rr)

    mass_range = np.linspace(Mc, M_star, num=1000) #mass from center to surface of star
    mass_out = mass_range[mass_range < M_fitting] ### the masses for the outward integration
    mass_in = np.flipud(mass_range[mass_range >= M_fitting]) ### the masses for the inward integration

    tspan_out = [mass_out[0], mass_out[-1]]
    tspan_in = [mass_in[0], mass_in[-1]]

    ### actually run the integrations ###
    print('Starting integration')
    int_out_soln = solve_ivp(derivs, t_span = tspan_out, y0 = inner_conditions, t_eval = mass_out, method='Radau',
                             rtol=1e-6, atol=1e-10)
    print('Outward integration worked.')
    int_in_soln = solve_ivp(derivs, t_span = tspan_in, y0 = outer_conditions, t_eval = mass_in, method='Radau',
                            rtol=1e-6, atol=1e-10)
    print('Inward integration worked.')

    return mass_out, mass_in, int_out_soln, int_in_soln

def shootf(params):
    _,_, int_in_soln, int_out_soln = integrate(params) #parameters from integrating inwards and outwards
    
    Lin_array = int_in_soln.y[0]
    Lout_array = int_out_soln.y[0]
    Pin_array = int_in_soln.y[1]
    Pout_array = int_out_soln.y[1]
    Tin_array = int_in_soln.y[2]
    Tout_array = int_out_soln.y[2]
    Rin_array = int_in_soln.y[3]
    Rout_array = int_out_soln.y[3]
    Min_array = np.flipud(int_in_soln.t)
    Mout_array = int_out_soln.t

    np.save('Lin_array.npy', Lin_array)
    np.save('Lout_array.npy', Lout_array)
    np.save('Pin_array.npy', Pin_array)
    np.save('Pout_array.npy', Pout_array)
    np.save('Tin_array.npy', Tin_array)
    np.save('Tout_array.npy', Tout_array)
    np.save('Rin_array.npy', Rin_array)
    np.save('Rout_array.npy', Rout_array)
    np.save('Min_array.npy', Min_array)
    np.save('Mout_array.npy', Mout_array)

    L_resid = (int_in_soln.y[0, -1] - int_out_soln.y[0, -1])
    P_resid = (int_in_soln.y[1, -1] - int_out_soln.y[1, -1])
    R_resid = (int_in_soln.y[2, -1] - int_out_soln.y[2, -1])
    T_resid = (int_in_soln.y[3, -1] - int_out_soln.y[3, -1])

    return np.array([L_resid, P_resid, R_resid, T_resid]), Lin_array, Lout_array, Pin_array, Pout_array, Tin_array, Tout_array, Rin_array, Rout_array, Min_array, Mout_array

