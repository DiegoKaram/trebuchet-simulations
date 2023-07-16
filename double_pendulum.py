from scipy.integrate import RK45, solve_ivp, simps
import numpy as np
import matplotlib.pyplot as plt

PARAM_G = 9.81
PARAM_L = 1.0
CTE = PARAM_G / PARAM_L


def f(t, y):
    """
    ODE for an harmonic oscilator:
    d(x)/dt = dx/dt
    d(dx/dt)/dt = g/l * sin(x)
    """
    return np.array([y[1], - CTE * np.sin(y[0])])


def E(state):
    """Return the Lagrangian for a simple pendulum with length = 1, mass = 1 and angle measured from y=-1 upwards"""
    return 0.5 * PARAM_L * PARAM_L * state[1] * state[1] + PARAM_G * PARAM_L * (1. - np.cos(state[0]))


t0 = 0.0
t1 = 6*np.pi  # 0.0
n_outputs = int((t1 - t0) * 10000) + 1
y0 = np.array([np.pi*0.5, 0.])

int_methods = ['RK23', 'RK45', 'DOP853', 'LSODA']

results_dict = {}
for int_method in int_methods:
    print(f"integration methos: {int_method}")
    results = solve_ivp(fun=f,
                        t_span=(t0, t1),
                        y0=y0,
                        method=int_method,
                        t_eval=np.arange(t0, t1, (t1-t0)/n_outputs),
                        vectorized=True
                        )
    results_dict[int_method] = results


print(f"times shape: {results_dict['RK45']['t'].shape}")
print(f"coords shape: {results_dict['RK45']['y'].shape}")


# def sol(t, y_init):
#     y_init
#     # return np.pi * 0.5 * np.array([np.cos(t), -np.sin(t)])
#     return

# plt.figure()
# plt.plot(results['t'], results['y'][0], color='red')
# plt.plot(results['t'], results['y'][1], color='blue')
#
# plt.scatter(results['t'], [sol(t)[0] for t in results['t']], color='red')
# plt.scatter(results['t'], [sol(t)[1] for t in results['t']], color='blue')
#
# # plt.legend()
# plt.show()


dt = (t1-t0)/n_outputs

plt.figure()
plt.title("Normalized energy values for different integration methods")

exact_E = E(y0)
for int_method in int_methods:
    result = results_dict[int_method]
    energies = np.array([E(state) - exact_E for state in zip(result['y'][0], result['y'][1])])
    plt.plot(result['t'], energies / exact_E, label=int_method)

    # plt.plot(result['t'], result['y'][0], label='position')
    # plt.plot(result['t'], result['y'][1], label='vel')

plt.legend()
plt.show()
