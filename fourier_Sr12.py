import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
from math import factorial, exp, sin, cos, pi
import matplotlib.pyplot as plt
import warnings
import time

start_time = time.time()
warnings.filterwarnings("ignore", category=UserWarning)

def eta_val(n):
    return np.pi * n / 6

def alpha_val(eta):
    return (2 * eta + 1)**2 / (eta - 1)**4

def beta_val(eta):
    return -3 * eta * (2 + eta)**2 / (2 * (eta - 1)**4)

def gamma_val(eta):
    return eta * (2 * eta + 1)**2 / (2 * (eta - 1)**4)

def psi_1(eta):
    return -(alpha_val(eta) + beta_val(eta) + gamma_val(eta))

def psi_2(eta):
    return -(alpha_val(eta) + 2 * beta_val(eta) + 4 * gamma_val(eta))

def psi_3(eta):
    return -(2 * beta_val(eta) + 12 * gamma_val(eta))

def psi_4(eta):
    return -24 * gamma_val(eta)

def psi_5(eta):
    return -24 * gamma_val(eta)

def psi_3_0(eta):
    return -2 * beta_val(eta)

def alpha_new(f0, f1, eta):
    return (1/6) * (f0 * psi_2(eta) - f1 * psi_1(eta)) - (1/2) * f0 * psi_1(eta)

def beta_new(f0, f1, eta):
    return f0 * psi_1(eta) - (1/2) * (f0 * psi_2(eta) - f1 * psi_1(eta))

def gamma_new(f0, f1, eta):
    return (psi_2(eta) - psi_1(eta)) * f0 - psi_1(eta) * f1

def delta_new(f0, f1, eta):
    return f1 * psi_1(eta) - f0 * psi_2(eta)

# системы уравнений
def solve_k(vars, eta):
    k1, k2 = vars  
    integrand_k1 = lambda x: -x * (alpha_val(eta) + beta_val(eta) * x + gamma_val(eta) * x**3) * np.sinh(k1 * x) * np.cos(k2 * x)
    integrand_k2 = lambda x: -x * (alpha_val(eta) + beta_val(eta) * x + gamma_val(eta) * x**3) * np.cosh(k1 * x) * np.sin(k2 * x) 
    integral_k1 = 24 * eta * quad(integrand_k1, 0, 1, limit=100)[0]
    integral_k2 = 24 * eta * quad(integrand_k2, 0, 1, limit=100)[0]
    return [integral_k1 - k1, integral_k2 - k2]

def system_f(vars, eta):
    f0, f1, f2, f3, f4, f5, f6 = vars
    equations = [
        f0 - 12 * eta * sum([(-1)**m * [f0, f1, f2, f3, f4, f5, f6][m] * (
            psi_1(eta) / factorial(m + 2) -
            psi_2(eta) / factorial(m + 3) +
            psi_3(eta) / factorial(m + 4) -
            psi_4(eta) / factorial(m + 5) +
            psi_5(eta) / factorial(m + 6)
        ) for m in range(7)]) + 12 * eta * alpha_new(f0, f1, eta),
        f1 + 12 * eta * sum([(-1)**m * [f0, f1, f2, f3, f4, f5, f6][m] * (
            psi_1(eta) / factorial(m + 1) -
            psi_2(eta) / factorial(m + 2) +
            psi_3(eta) / factorial(m + 3) -
            psi_4(eta) / factorial(m + 4) +
            psi_5(eta) / factorial(m + 5)
        ) for m in range(7)]) - 12 * eta * beta_new(f0, f1, eta),
        f2 + 12 * eta * (psi_1(eta) + sum([(-1)**m * [f0, f1, f2, f3, f4, f5, f6][m] * (
            psi_2(eta) / factorial(m + 1) -
            psi_3(eta) / factorial(m + 2) +
            psi_4(eta) / factorial(m + 3) -
            psi_5(eta) / factorial(m + 4)
        ) for m in range(7)])) + 12 * eta * gamma_new(f0, f1, eta),
        f3 + 12 * eta * (psi_2(eta) + sum([(-1)**m * [f0, f1, f2, f3, f4, f5, f6][m] * (
            psi_3(eta) / factorial(m + 1) -
            psi_4(eta) / factorial(m + 2) +
            psi_5(eta) / factorial(m + 3)
        ) for m in range(7)])) - 12 * eta * delta_new(f0, f1, eta),
        f4 + 12 * eta * (2 * f0 * psi_3_0(eta) + psi_3(eta) + sum([(-1)**m * [f0, f1, f2, f3, f4, f5, f6][m] * (
            psi_4(eta) / factorial(m + 1) -
            psi_5(eta) / factorial(m + 2)
        ) for m in range(7)])),
        f5 + 12 * eta * (2 * f1 * psi_3_0(eta) + psi_4(eta) + sum([(-1)**m * [f0, f1, f2, f3, f4, f5, f6][m] * (
            psi_5(eta) / factorial(m + 1)
        ) for m in range(7)])),
        f6 + 12 * eta * (2 * f2 * psi_3_0(eta) + 2 * f0 * psi_5(eta) + psi_5(eta))
    ]
    return equations

# вычисление параметров для каждого n
n_values = np.arange(0.05, 0.85, 0.1)
data = []

for n in n_values:
    eta = eta_val(n)
    
    k1, k2 = fsolve(solve_k, [4.07, 4.76], args=(eta,))

    f_solution = fsolve(system_f, [1.0]*7, args=(eta,))
    f0, f1, f2, f3, f4, f5, f6 = f_solution

    data.append({
        'n_value': float(n),
        'n': float(eta),
        'f0': float(f0),
        'f1': float(f1),
        'f2': float(f2),
        'f3': float(f3),
        'f4': float(f4),
        'f5': float(f5),
        'f6': float(f6),
        'k1': float(k1),
        'k2': float(k2)
    })

# S(r12)
def f(x, params):
    if x <= 1:
        result = 0
        derivatives = [params['f0'], params['f1'], params['f2'], 
                     params['f3'], params['f4'], params['f5'], params['f6']]
        for m in range(7):
            result += ((-1) ** m) * (derivatives[m] / factorial(m)) * ((1-x) ** m)
        return result
    else:
        return exp(-params['k1'] * (x-1)) * ((params['f1'] + params['k1'] * params['f0']) * 
               (sin(params['k2'] * (x-1)) / params['k2']) + params['f0'] * cos(params['k2'] * (x-1)))

def f_prime(x, params):
    if x <= 1:
        result = 0
        derivatives = [params['f1'], params['f2'], params['f3'], 
                     params['f4'], params['f5'], params['f6'], 0]
        for m in range(5):
            result += ((-1) ** m) * (derivatives[m + 1] / factorial(m)) * ((1 - x) ** m)
        return result
    else:
        return exp(-params['k1'] * (x - 1)) * ((-(params['f1'] + params['k1'] * params['f0']) * 
               (sin(params['k2'] * (x - 1))) / params['k2']) + params['f1'] * cos(params['k2'] * (x - 1)))

def f_double_prime(x, params):
    if x <= 1:
        result = 0
        derivatives = [params['f1'], params['f2'], params['f3'], 
                     params['f4'], params['f5'], params['f6'], 0]
        for m in range(4):
            result += ((-1) ** m) * (derivatives[m + 2] / factorial(m)) * ((1 - x) ** m)
        return result
    else:
        return exp(-params['k1'] * (x - 1)) * (((params['f1'] + params['k1'] * params['f0']) * 
               (sin(params['k2'] * (x - 1))) / params['k2']) + params['f2'] * cos(params['k2'] * (x - 1)))

def F(z, params):
    numerator = f(z, params) * f_prime(z, params)
    denominator = 2 * np.pi * params['n'] * (1 + f(z, params))
    if abs(denominator) < 1e-10:
        return 0.0
    return numerator / denominator

def P(z, params):
    numerator = f_prime(z, params)**2 + f(z, params) * f_double_prime(z, params) * (1+ f(z, params))
    denominator = 2 * np.pi * params['n'] * (1 + f(z, params))**2
    if abs(denominator) < 1e-10:
        return 0.0
    return numerator / denominator

k_max = 300
r12_min, r12_max = 0.1, 10.0
num_points = 500

plt.figure(figsize=(12, 8))

with open('S_r12_results_python.txt', 'w') as f_out:
    f_out.write("n r12 S(r12)\n")
    
    for params in data:
        k_grid = np.linspace(0.01, k_max, 500)
        Fk_cache = {}
        fk_cache = {}

        def compute_Fk(k):
            if k not in Fk_cache:
                integrand = lambda z: P(z, params) * np.sin(k * z)
                Fk_cache[k] = 2 * quad(integrand, 1e-6, k_max, limit=5000, epsabs=1e-10, epsrel=1e-10)[0]
            return Fk_cache[k]

        def compute_fk(k):
            if k not in fk_cache:
                integrand = lambda z: f_prime(z, params) * np.cos(k * z)
                fk_cache[k] = 2 * quad(integrand, 1e-6, k_max, limit=5000, epsabs=1e-10, epsrel=1e-10)[0]
            return fk_cache[k]

        S_k_grid = np.zeros_like(k_grid)
        for i, k in enumerate(k_grid):
            try:
                Fk = compute_Fk(k)
                fk = compute_fk(k)
                f0 = f(0, params)
                S_k_grid[i] = Fk / (1 + f0 + fk)
            except:
                S_k_grid[i] = 0.0

        S_k_interp = CubicSpline(k_grid, S_k_grid, bc_type='natural')

        def S(r12):
            if r12 == 0:
                return 0 
            integrand = lambda k: S_k_interp(k) * np.sin(k * r12) / r12
            try:
                integral = quad(integrand, 0.01, k_max, limit=5000, epsabs=1e-10, epsrel=1e-10)[0]
                return integral / np.pi
            except:
                return 0.0

        r12_values = np.linspace(r12_min, r12_max, num_points)
        S_values = [S(r) for r in r12_values]
        
        n_val = params["n_value"]
        for r, s in zip(r12_values, S_values):
            f_out.write(f"{n_val:.2f} {r:.4f} {s:.8f}\n")
        
        plt.plot(r12_values, S_values, linewidth=2, label=f'n = {n_val:.1f}')

plt.xlabel('r12', fontsize=12)
plt.ylabel('S(r12)', fontsize=12)
plt.title('S(r12) для разных значений n', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

end_time = time.time()
execution_time = end_time - start_time
print(f"Общее время выполнения: {execution_time:.2f} секунд")

plt.savefig("S_r12_calculated_Python.png", dpi=300, bbox_inches='tight')
plt.show()