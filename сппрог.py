import numpy as np
from scipy.optimize import minimize_scalar, minimize
import sys
import numdifftools as nd
import matplotlib.pyplot as plt

# Target function
def f(x):
    x1, x2 = x
    return (x1 - 4)**2 + (x2 - 1)**2 + x1 * x2

# Gradient of target function
def gradient(x):
    x1, x2 = x
    df_dx1 = 2*(x1 - 4) + x2
    df_dx2 = 2*(x2 - 1) + x1
    return np.array([df_dx1, df_dx2])

# Hessian of target function
def hessian_f(x):
    x1, x2 = x
    return np.array([[2, 1], [1, 2]])

# Function for Sylvester's minimization
def minimize_sylvester(x0, f, grad_f, hessian_f):
    res = minimize(f, x0, method='Newton-CG', jac=grad_f, hess=hessian_f)
    if res.success:
        return res.x
    else:
        return None

# Initial data
tolerance = 1e-6
starting_point = np.array([-10, -10])

# Output data
relaxation_sequence = [starting_point]
function_values = [f(starting_point)]
calculated_values = 0
iterations = 0

# Calculating the gradient at the starting point
gradient_vec = gradient(starting_point)
calculated_values += 2

# Vector p defining the direction of descent
p = -gradient_vec

# Input parameters
a = 4
b = 1
c = 1
x0 = np.array([-10.0, -10.0])
tol = 1e-6

print(f"Входные параметры: a = {a}, b = {b}, c = {c}, x0 = {x0}, tol = {tol}")

# Stop criteria
while iterations < 2:
    current_point = relaxation_sequence[-1]

    # One-dimensional minimization along the direction p
    def psi(alpha):
        new_point = current_point + alpha * p
        return f(new_point)

    # Golden section search for alpha
    result = minimize_scalar(psi, bounds=(0, 1), method='bounded')
    alpha = result.x

    # Update new point and gradient
    new_point = current_point + alpha * p
    gradient_vec = gradient(new_point)
    calculated_values += 2

    # Update p using the Polak-Ribiere formula
    beta = np.dot(gradient_vec, gradient_vec - gradient(current_point)) / np.dot(gradient(current_point), gradient(current_point))
    p = -gradient_vec + beta * p

    # Output the current iteration
    if iterations == 0:
        print(f"Первая итерация методом сопряженных градиентов: {new_point}")
    elif iterations == 1:
        print(f"Вторая итерация методом сопряженных градиентов: {new_point}")

    # Update relaxation sequence and function values
    relaxation_sequence.append(new_point)
    function_values.append(f(new_point))
    iterations += 1

# Sylvester's minimization check
sylvester_optimal_point = minimize_sylvester(starting_point, f, gradient, hessian_f)
if sylvester_optimal_point is not None:
    print(f"Проверка методом Сильвестра найдена точка оптимума при (x1, x2) = {sylvester_optimal_point}")
else:
    print("Проверить методом Сильвестра не удалось найти точку оптимума.")

# Plotting
x1 = np.linspace(-15, 15, 400)
x2 = np.linspace(-15, 15, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = (X1 - 4)**2 + (X2 - 1)**2 + X1 * X2

plt.figure(figsize=(8, 6))
plt.contour(X1, X2, Z, levels=[10, 20, 30, 70, 100], colors='gray', linestyles='dashed')

# Removing annotation for the last iteration
plt.scatter(*zip(*relaxation_sequence), c='red', label='Iteration points')
for i in range(1, len(relaxation_sequence)):
    plt.plot([relaxation_sequence[i-1][0], relaxation_sequence[i][0]], [relaxation_sequence[i-1][1], relaxation_sequence[i][1]], 'k')

# Setting ticks for x-axis and y-axis
plt.xticks(np.arange(-15, 16, 5))
plt.yticks(np.arange(-15, 16, 5))

# Setting axes to cross at (0, 0)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('')
plt.grid(True)
plt.show()
