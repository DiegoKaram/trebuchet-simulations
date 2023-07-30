from scipy.integrate import RK45, solve_ivp, simps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
sns.set()

int_methods = ['RK23', 'RK45']#, 'DOP853', 'LSODA']
movie = True
total_frames = 1000

PARAM_G = 9.81
L1 = 1.0
L2 = 1.0
M1 = 1.0
M2 = 1.0
#CTE = PARAM_G / PARAM_L


def alpha1(t1, t2):
    return L2 / L1 * (M2/(M1+M2)) * np.cos(t1 - t2)

def alpha2(t1, t2):
    return L1 / L2 * np.cos(t1 - t2)

def f1(t1, t2, w2):
    a = - L2/L1 * (M2/(M1+M2)) * w2**2 * np.sin(t1 - t2)
    b = PARAM_G / L1 * np.sin(t1)
    return a - b

def f2(t1, t2, w1):
    a = - L1/L2 * w1**2 * np.sin(t1 - t2)
    b = PARAM_G / L2 * np.sin(t2)
    return a - b

def g1(y):
    t1 = y[0]
    t2 = y[1]
    w1 = y[2]
    w2 = y[3]
    return (f1(t1, t2, w2) - alpha1(t1, t2) * f2(t1, t2,w1)) / (1 - alpha1(t1, t2) * alpha2(t1, t2))

def g2(y):
    t1 = y[0]
    t2 = y[1]
    w1 = y[2]
    w2 = y[3]
    return (-alpha2(t1, t2) * f1(t1, t2, w2) + f2(t1, t2,w1)) / (1 - alpha1(t1, t2) * alpha2(t1, t2))


def derivs_system(t, y):
    """
    ODE for a double pendulum:
    dx1/dt = w1
    dx2/dt = w2
    dw1/dt = g1
    dw2/dt = g2
    """
    return np.array([y[2], y[3], g1(y), g2(y)])

def E(state):
    """Return the energy for a double pendulum"""
    t1 = state[0]
    t2 = state[1]
    w1 = state[2]
    w2 = state[3]
    T = 0.5*M1*(L1*w1)**2 + 0.5*M2 * ((L1*w1)**2 * (L2*w2)**2 + 2*L1*L2*w1*w2*np.cos(t1-t2))
    V = -(M1+M2)*PARAM_G*L1*np.cos(t1) - M2*PARAM_G*L2*np.cos(t2)
    return T + V


t0 = 0.0
t1 = 3*np.pi  # 0.0
n_outputs = int((t1 - t0) * 100000) + 1
y0 = np.array([np.pi*0.25, np.pi*0.75, 0., 0.])
dt_eval = (t1-t0)/n_outputs

results_dict = {}
for int_method in int_methods:
    print(f"integration methos: {int_method}")
    results = solve_ivp(fun=derivs_system,
                        t_span=(t0, t1),
                        y0=y0,
                        method=int_method,
                        t_eval=np.arange(t0, t1, (t1-t0)/n_outputs),
                        vectorized=True
                        )
    results_dict[int_method] = results


print(f"times shape: {results_dict['RK45']['t'].shape}")
print(f"coords shape: {results_dict['RK45']['y'].shape}")


dt = (t1-t0)/n_outputs


##################
# Energies plots
##################
'''
plt.figure()
plt.title("Normalized energy values for different integration methods")

exact_E = E(y0)
for int_method in int_methods:
    result = results_dict[int_method]
    energies = np.array([E(state) - exact_E for state in zip(result['y'][0], result['y'][1], result['y'][2], result['y'][3])])
    plt.plot(result['t'], energies / exact_E, label=int_method)

    # plt.plot(result['t'], result['y'][0], label='position')
    # plt.plot(result['t'], result['y'][1], label='vel')

plt.legend()
plt.show()
'''

##################
# Coordinates plots
##################
angles = pd.DataFrame({'t1': results_dict['RK45']['y'][0],
                       't2': results_dict['RK45']['y'][1]})

vels = pd.DataFrame({'w1': results_dict['RK45']['y'][2],
                       'w2': results_dict['RK45']['y'][3]})

fig, axes = plt.subplots(3, 1, figsize=(10,5))
# angles
axes[0].set_title("Angles")
sns.lineplot(ax=axes[0], x=angles.index, y=angles['t1'].values, label='theta 1')
sns.lineplot(ax=axes[0], x=angles.index, y=angles['t2'].values, label='theta 2')
# vels
axes[1].set_title("Velocities")
sns.lineplot(ax=axes[1], x=vels.index, y=vels['w1'].values, label='w1')
sns.lineplot(ax=axes[1], x=vels.index, y=vels['w2'].values, label='w2')
# energies
axes[2].set_title("Energy")
exact_E = E(y0)
result = results_dict['RK45']
energies = np.array([E(state) - exact_E for state in zip(result['y'][0], result['y'][1], result['y'][2], result['y'][3])])
sns.lineplot(ax=axes[2], x=list(range(len(energies))), y=energies)
plt.show()

if movie:
    def get_coords(t1, t2):
        """Return the (x1, y1, x2, y2) coordinates of the bob at angle th."""
        return L1*np.sin(t1), -L1*np.cos(t1), L1*np.sin(t1) + L2*np.sin(t2), -L1*np.cos(t1) - L2*np.cos(t2)

        # Initialize the animation plot. Make the aspect ratio equal so it looks right.
    fig = plt.figure()
    ax = fig.add_subplot(aspect='equal')
    # The pendulum rod, in its initial position.
    plot_x10, plot_y10, plot_x20, plot_y20 = get_coords(y0[0], y0[1])
    print(f"px={plot_x10}")
    print(f"py={plot_y10}")
    print(f"px={plot_x20}")
    print(f"py={plot_y20}")
    line1, = ax.plot([0, plot_x10], [0, plot_y10], lw=3, c='k')
    line2, = ax.plot([plot_x10, plot_x20], [plot_y10, plot_y20], lw=3, c='k')
    # The pendulum bob: set zorder so that it is drawn over the pendulum rod.
    bob_radius = 0.08

    theta1 = results_dict['RK45']['y'][0,:]
    theta2 = results_dict['RK45']['y'][1,:]
    #theta10 = y0[0]
    #theta20 = y0[1]
    circle1 = ax.add_patch(plt.Circle((plot_x10, plot_y10), bob_radius,
                          fc='r', zorder=3))
    circle2 = ax.add_patch(plt.Circle((plot_x20, plot_y20), bob_radius,
                          fc='r', zorder=3))
    # Set the plot limits so that the pendulum has room to swing!
    ax.set_xlim(-(L1+L2)*1.2, (L1+L2)*1.2)
    ax.set_ylim(-(L1+L2)*1.2, (L1+L2)*1.2)

    def animate(i):
        """Update the animation at frame i."""
        x1, y1, x2, y2 = get_coords(theta1[i], theta2[i])
        line1.set_data([0, x1], [0, y1])
        line2.set_data([x1, x2], [y1, y2])
        circle1.set_center((x1, y1))
        circle2.set_center((x2, y2))

    nframes = len(theta1)
    interval = dt_eval * 10000
    print(interval)
    print(nframes)
    #plot_positions = []
    reduction_factor = int(nframes / total_frames)
    frames = [idx for idx in range(nframes) if idx % reduction_factor == 0]
    print(len(frames), frames)
    ani = animation.FuncAnimation(fig, animate, frames=frames, repeat=True,
                              interval=interval)
    plt.show()
