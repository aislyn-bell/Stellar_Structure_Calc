from constants import *
import numpy as np

M_factor = 1.5

M_sun = 1.989e+33
R_sun = 6.957e+10
L_sun = 3.839e+33
g_sun = G * M_sun / (R_sun ** 2)
rho_sun = M_sun / ((4 / 3) * np.pi * (R_sun ** 3))

M_star = M_sun * M_factor
L_star = L_sun * M_factor ** 3.9
R_star = R_sun * M_factor ** 0.2
g_star = G * M_star / (R_star ** 2)

Mc = 1e-8 * M_star
M_fitting = M_star * 0.2

X = 0.70
Y = 0.28
Z = 0.02
XCNO = (2/3) * Z
mu = 4 / (3 + 5 * X)

Pc_guess = 1e17
Tc_guess = 1e7
# Pc_guess = (3 / (8 * np.pi)) * G * (M_star ** 2) / (R_star ** 4)
# Tc_guess = (1 / 2) * (mu / (Na * k)) * (G * M_star / R_star)

Lc_guess = L_star
Rc_guess = R_star

init_guess = np.array([Lc_guess, Pc_guess, Rc_guess, Tc_guess])