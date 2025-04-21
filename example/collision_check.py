import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize

def collision_checker(gap_array, junction_time):
    return np.any((gap_array < junction_time/2) & (gap_array > -junction_time/2))

def gapsolver(X, gap_array, junction_time):
    gap_trial_array = gap_array + X
    result = collision_checker(gap_trial_array, junction_time)
    print(f"X: {X}, Gap array: {gap_trial_array}, Result: {result}")
    return result

def objective_function(X, gap_array, junction_time):
    return -int(gapsolver(X, gap_array, junction_time) ^ True)

def find_positive_x(gap_array, junction_time, tolerance=1e-5):
    lower_bound, upper_bound = 0.0, 1.0
    while gapsolver(upper_bound, gap_array, junction_time):
        lower_bound, upper_bound = upper_bound, upper_bound * 2.0

    while upper_bound - lower_bound > tolerance:
        mid_point = (upper_bound + lower_bound) / 2.0
        if gapsolver(mid_point, gap_array, junction_time):
            lower_bound = mid_point
        else:
            upper_bound = mid_point

    return upper_bound

# Example values
junction_time = 10
gap_array = np.array([1, 2, -3, 4, -5])

# Find the smallest positive value of X that makes gapsolver return False
X_solution = find_positive_x(gap_array, junction_time)

constraints = [{'type': 'ineq', 'fun': lambda X: X}]

# Initial guess
initial_guess = 0.0

# Minimize the negative of gapsolver
result = minimize(objective_function, initial_guess, args=(gap_array, junction_time), constraints=constraints)

X_solution = result.x[0]

print("The smallest positive value of X:", X_solution)
