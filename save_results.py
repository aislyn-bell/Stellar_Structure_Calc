from functions import *
from new_interpolator import *
from fsolve import *
import numpy as np
from pandas.api.types import is_numeric_dtype
import pandas as pd

def concatenate_arrays():
    Lin_array = np.load('Lin_array.npy')
    Lout_array = np.load('Lout_array.npy')
    Pin_array = np.load('Pin_array.npy')
    Pout_array = np.load('Pout_array.npy')
    Tin_array = np.load('Tin_array.npy')
    Tout_array = np.load('Tout_array.npy')
    Rin_array = np.load('Rin_array.npy')
    Rout_array = np.load('Rout_array.npy')
    Min_array = np.load('Min_array.npy')
    Mout_array = np.load("Mout_array.npy")
    
    # Concatenate them
    L_full = np.concatenate([Lout_array, Lin_array])
    P_full = np.concatenate([Pout_array, Pin_array])
    T_full = np.concatenate([Tout_array, Tin_array])
    R_full = np.concatenate([Rout_array, Rin_array])
    M_full = np.concatenate([Mout_array, Min_array])
    
    return L_full, P_full, T_full, R_full, M_full


## now, use those arrays to make a table ###
def saveresults(L_full, P_full, T_full, R_full, M_full, df_return=True):
  
    #Only save a subset of the lines
    r_want = np.linspace(0, 1, 10) * np.max(R_full)
    ids = []
    for i in r_want:
        ids.append(np.argmin(abs(R_full - i)))
    
    L_total = L_full[ids]
    P_total = P_full[ids]
    r_total = R_full[ids]
    T_total = T_full[ids]
    M_total = M_full[ids]
    
 
    rho_total = density(P_total, T_total)
    e_total = np.array([nuclear_en_gen(rho, T) for rho, T in zip(rho_total, T_total)])
    ad_total = np.full(len(M_total), 0.4)
    kappas_total = np.array([interp(log_R, log_T, solar_table, rho, T) 
                         for rho, T in zip(rho_total, T_total)])
    rads = radgrad(P_total, T_total, L_total, kappas_total, M_total)

    gradients = []
    radvcon = []
    for j, k in zip(ad_total, rads):
        gradients.append(np.min([j, k]))
        if np.min([j, k]) == 0.4:
            radvcon.append('Convective')
        else:
            radvcon.append('Radiative')
    
    gradients = np.array(gradients)
    radvcon = np.array(radvcon)
    
    data = {
        'Mass': M_total / M_star,
        'Luminosity': L_total / np.max(L_total),
        'log Pressure': np.log10(P_total),
        'Radius': r_total / np.max(r_total),
        'log Temperature': np.log10(T_total),
        'log Density': np.log10(rho_total),
        'Energy Generation': e_total,
        'Adiabatic': ad_total,
        'Radiative': rads,
        'Gradients': gradients,
        'Transport': radvcon
    }

    df = pd.DataFrame(data)

    # Round to 3 sigfigs for readability
    def round_to_significant(col, digits=3):
        if not np.issubdtype(col.dtype, np.number):
            return col
        arr = col.astype(float)
        nonzero = arr != 0
        if np.any(nonzero):
            orders = np.floor(np.log10(np.abs(arr[nonzero])))
            factors = 10 ** (digits - 1 - orders)
            arr[nonzero] = np.round(arr[nonzero] * factors) / factors
        return arr
    
    for cols in df.columns:
        df[cols] = round_to_significant(df[cols].values, digits=3)
              
    df.to_csv("subset10_results_table.csv", index=False)
    
    if df_return:
        return df
    
### actually call the functions ###
if __name__ == "__main__":
    L_full, P_full, T_full, R_full, M_full = concatenate_arrays()
    df = saveresults(L_full, P_full, T_full, R_full, M_full)

