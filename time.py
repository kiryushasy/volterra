import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from math import factorial, exp, sin, cos, pi
import matplotlib.pyplot as plt
import warnings
import time

start_time = time.time()

# данные
data = [
    {
        'n_value': 0.1,
        'n': 0.05235987755982988,
        'f0': -0.010896299227692,
        'f1': 0.03533871802446059,
        'f2': 0.7375798016904298,
        'f3': 0.4803943311511456,
        'f4': -0.17149042380406146,
        'f5': 0.6040419153779827,
        'f6': -0.17603764656165466,
        'k1': 4.07155209680121,
        'k2': 4.76077138142061
    },
    {
        'n_value': 0.2,
        'n': 0.10471975511965977,
        'f0': -0.040955430040710734,
        'f1': 0.14046818273894418,
        'f2': 1.5705072406894023,
        'f3': 0.290760706585611,
        'f4': -0.29726760769015437,
        'f5': 3.2993377898684724,
        'f6': -5.249382884498001,
        'k1': 3.093382794870304,
        'k2': 5.115610312103863
    },
    {
        'n_value': 0.3,
        'n': 0.15707963267948966,
        'f0': -0.07937662775464152,
        'f1': 0.309341378413055,
        'f2': 2.20864472248648,
        'f3': -1.1853851064431131,
        'f4': 1.2037907201856726,
        'f5': 9.223225337000306,
        'f6': -25.939365178223138,
        'k1': 2.486506582624432,
        'k2': 5.398200639846574
    },
    {
        'n_value': 0.4,
        'n': 0.20943951023931953,
        'f0': -0.10922581443753625,
        'f1': 0.5135761556026422,
        'f2': 2.175732478808618,
        'f3': -4.465794619477479,
        'f4': 7.908170728448949,
        'f5': 19.122638943377872,
        'f6': -60.433398136695736,
        'k1': 2.0338135055198534,
        'k2': 5.6506553082306885
    },
    {
        'n_value': 0.5,
        'n': 0.2617993877991494,
        'f0': -0.10939238541917934,
        'f1': 0.6820299487786214,
        'f2': 0.7826461830054611,
        'f3': -9.50051023478937,
        'f4': 25.963924602703283,
        'f5': 37.594136439099046,
        'f6': -6.262541451779483,
        'k1': 1.6673892924723686,
        'k2': 5.888677094718416
    },
    {
        'n_value': 0.6,
        'n': 0.3141592653589793,
        'f0': -0.05724077661682139,
        'f1': 0.6767397591554315,
        'f2': -2.72443698126277,
        'f3': -14.554544245304914,
        'f4': 63.4380969951293,
        'f5': 99.83816262213533,
        'f6': 619.4154447014668,
        'k1': 1.3575848883484734,
        'k2': 6.120359242206799
    },
    {
        'n_value': 0.7,
        'n': 0.3665191429188092,
        'f0': 0.05624866068181238,
        'f1': 0.30848844513431417,
        'f2': -8.266654410734068,
        'f3': -14.357080355861644,
        'f4': 128.07626765123518,
        'f5': 368.5622356348904,
        'f6': 3181.982947178201,
        'k1': 1.0894096285297696,
        'k2': 6.3506224973383585
    },
    {
        'n_value': 0.8,
        'n': 0.41887902047863906,
        'f0': 0.18519599692546337,
        'f1': -0.43488961582191904,
        'f2': -13.204179656897638,
        'f3': -1.4729083827571146,
        'f4': 237.28168978059932,
        'f5': 1291.1753409262808,
        'f6': 9583.619392264429,
        'k1': 0.8549302040496924,
        'k2': 6.582679041325202
    },
    {
        'n_value': 0.9,
        'n': 0.47123889803846897,
        'f0': 0.24602516674204053,
        'f1': -1.155160916468779,
        'f2': -14.273806747444455,
        'f3': 25.71895221593576,
        'f4': 471.4503752608885,
        'f5': 3550.364370015452,
        'f6': 20132.826900627897,
        'k1': 0.650217090786169,
        'k2': 6.818520944399062
    }
]

# Параметры расчета
k_max = 30
r12_min, r12_max = 0.1, 10.0
num_points = 500

plt.figure(figsize=(12, 8))

for params in data:
    def f(x):
        if x <= 1:
            result = 0
            derivatives = [params['f0'], params['f1'], params['f2'], 
                         params['f3'], params['f4'], params['f5'], params['f6']]
            for m in range(7):
                result += ((-1) ** m) * (derivatives[m] / factorial(m)) * (x ** m)
            return result
        else:
            return exp(-params['k1'] * x) * ((params['f1'] + params['k1'] * params['f0']) * 
                   (sin(params['k2'] * x) / params['k2']) + params['f0'] * cos(params['k2'] * x))

    def f_prime(z):
        if z <= 1:
            result = 0
            derivatives = [params['f1'], params['f2'], params['f3'], 
                         params['f4'], params['f5'], params['f6'], 0]
            for m in range(6):
                result += ((-1) ** m) * (derivatives[m + 1] / factorial(m)) * ((1 - z) ** m)
            return result
        else:
            return exp(-params['k1'] * (z - 1)) * (((params['f1'] + params['k1'] * params['f0']) * 
                   (sin(params['k2'] * (z - 1))) / params['k2']) + params['f1'] * cos(params['k2'] * (z - 1)))

    def F(z):
        denominator = 2 * pi * params['n'] * (1 + f(z))
        if abs(denominator) < 1e-10:
            return 0.0
        return (f(z) * f_prime(z)) / denominator

    f0 = f(0)
    k_grid = np.linspace(0.01, k_max, 500)
    
    def F_k(k):
        integrand = lambda z: F(z) * np.sin(k * z)
        return 2 * quad(integrand, 1e-6, k_max, limit=2000, epsabs=1e-8, epsrel=1e-8)[0]

    def f_k(k):
        integrand = lambda z: f(z) * np.cos(k * z)
        return 2 * quad(integrand, 1e-6, k_max, limit=2000, epsabs=1e-8, epsrel=1e-8)[0]

    S_k_grid = np.zeros_like(k_grid)
    for i, k in enumerate(k_grid):
        try:
            Fk = F_k(k)
            fk = f_k(k)
            S_k_grid[i] = Fk / (1 + f0 + fk)
        except:
            S_k_grid[i] = 0.0

    S_k_interp = interp1d(k_grid, S_k_grid, kind='cubic', fill_value="extrapolate")

    def S(r12):
        if r12 == 0:
            return 0 
        integrand = lambda k: S_k_interp(k) * np.sin(k * r12) / r12
        try:
            integral = quad(integrand, 0.01, k_max, limit=2000, epsabs=1e-8, epsrel=1e-8)[0]
            return -integral / np.pi
        except:
            return 0.0

    r12_values = np.linspace(r12_min, r12_max, num_points)
    S_values = [S(r) for r in r12_values]

    plt.plot(r12_values, S_values, linewidth=2, 
             label=f'n = {params["n_value"]:.1f}')

plt.xlabel('r12', fontsize=12)
plt.ylabel('S(r12)', fontsize=12)
plt.title('S(r12) для разных значений n', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

end_time = time.time()
execution_time = end_time - start_time
print(f"Общее время выполнения: {execution_time:.2f} секунд")

plt.show()