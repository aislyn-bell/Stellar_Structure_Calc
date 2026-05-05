import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
import pandas as pd

'''
reads the OPALtable_solar.txt, which is just the OPAL table for a star w/solar composition
'''
def opaldf(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    log_R = []
    log_T = []
    arr = []
    
    # Find the header with log R values
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('logT'):
            # Get log R values from header
            parts = line.strip().split()
            for val in parts[1:]:
                log_R.append(float(val))
            data_start = i + 1
            break
    
    # Read in data
    for i in range(data_start, len(lines)):
        line = lines[i].strip() #removes white space
        if not line:
            continue #skips blank lines
        list = line.split() # turn into a list
        if len(list) > 1:
            try:
                log_T.append(float(list[0]))
                row_kappa = [] #empty list to fill with opacities
                for val in list[1:]:
                    row_kappa.append(float(val))
                arr.append(np.array(row_kappa)) #make it an array
            except ValueError:
                pass  # Skip unparseable lines
    
    # Create DataFrame
    table = pd.DataFrame(arr)
    
    return table, log_R, log_T


def interp(R_arr, T_arr, table, rho_i, T_i):
    """
    Interpolates the opacity table at a given density and temperature
    """
    pairs = [] #TR pairs
    ops = [] #opacities
    for i in range(len(R_arr)):
        for j in range(len(T_arr)):
            opacity_val = table.iloc[j, i]
            # Skip 9.999 values
            if opacity_val < 9.0:
                pairs.append([R_arr[i], T_arr[j]])
                ops.append(opacity_val)
    
    log_R_i = np.log10(rho_i / (T_i / 1e6) ** 3)  # find value of R given rho_i and T_i

    kappa = 10 ** griddata(pairs, ops, (log_R_i, np.log10(T_i)), method='linear')  # non-log kappa value
    
    # Handle nans
    if np.isnan(kappa):
        kappa = 1.0  # default opacity
    
    return kappa


# Load the opacity table
solar_table, log_R, log_T = opaldf('OPALtable_solar.txt')
