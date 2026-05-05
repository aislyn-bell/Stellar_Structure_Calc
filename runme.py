from fsolve import *
from converge import *
#from save_results import *

'''
This is the file that actually runs the stellar structure calculation!
'''
# guess = stepforward(shootf, init_guess)
# final_solution = find_convergence(shootf, init_guess)
# print("I finished running. Your final solution is", final_solution)

try:
    guess = stepforward(shootf, init_guess)
    final_solution = find_convergence(shootf, init_guess)
    mass_out, mass_in, int_in_soln, int_out_soln = integrate(final_solution)

    print("I finished running. Your final solution is", final_solution)

except Exception as e:
    print(f"Fatal error occured. Trying last solution")
    
    if 'guess' in locals():
        mass_out, mass_in, int_in_soln, int_out_soln = integrate(guess)
    else:
        print("Failed immediatley")


print("Attempt ran")

