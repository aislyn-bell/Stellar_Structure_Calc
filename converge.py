import numpy as np
from scipy.optimize import least_squares

'''
Use the Newton-Raphson method to shoot for f
'''
resid_scales = np.array([1e33, 1e16, 1e10, 1e7])
param_scales = np.array([1e34, 1e17, 1e11, 1e7])
lower_boundary = np.array([1e20, 1e10, 1e8, 1e3])

def residuals_only(f, y):
    """Return just the [L, P, R, T] shooting residuals."""
    residuals = f(np.array(y, dtype=float))
    if isinstance(residuals, tuple):
        residuals = residuals[0]
    return np.asarray(residuals, dtype=float)

def stepforward(f,y):
    '''step forward in Newton-Raphson method'''
    y = np.array(y, dtype=float).copy()
    fy_0 = residuals_only(f, y) #calcs diff from shoot for f at the convergence point
    y[1] += 2e15 * (fy_0[1] / resid_scales[1]) #update pressure guess
    y[3] += 2e6 * (fy_0[3] / resid_scales[3]) #update temperature guess
    return np.maximum(y, lower_boundary)

def tryagian(f,y, constant = 0.2, threshold = 0.1):
    '''step forward in Newton-Raphson method with overshooting. 
    Convergence occurs when the residuals are less than the threshold. y is the initial guess'''
    step_values = resid_scales  #step size to change each parameter by, based on expected values of L, P, R, T
    num = len(y) #number of parameters
    J = np.zeros((num, num)) #initialize Jacobian matrix for the number of parameters we have
    fy = f(y) #calculate the residuals at the current guess
    max_resid = np.max(np.abs(fy / step_values)) #calculate the absolute value of the  maximum fractional residual
    if max_resid < threshold:
        print("Converged!")
        return y
    else:
        print("residual is", np.abs(fy / step_values,",continuing"))
        for i in range(num):
            copy=y.copy() #copy the current guess
            copy[i] += step_values[i] #change one parameter at a time by it's step size
            J[:, i] = (f(copy) - fy) / step_values[i] #calculate the Jacobian 
        #newton_step = np.linalg.solve(J, -fy)  # Solve J· delta y  = -fy
        #y_new = y + overshoot * newton_step
            y_new = y + (np.dot(np.linalg.inv(J), -fy) * constant)
        return y_new
        
     
    
def find_convergence(f,y, max_iterations=50, tolerance = 0.1):
    '''use the newton-raphson method to find a converged solution'''
    for i in range(max_iterations):
        y_new = tryagian(f,y)

        ### Check for convergence using a tolerance value)
        if np.allclose(y, y_new, rtol=tolerance, atol=tolerance):
            print(f'Converged in {i + 1} iterations')
            return y_new
        
        y = y_new

    print(f'Did not converge in {max_iterations} iterations. Final guess is {y}')
    return y
    
def newton_raphson(f, y):
    """
    Iterates to find a converged solution.
    """
    i = 0
    while i < 50:
        y_new = tryagian(f, y)
        if np.array(y).tolist() == np.array(y_new).tolist():
            return y
        else:
            y = y_new
    i += 1

    print('Convergence is taking too long.')
