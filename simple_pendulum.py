from scipy.integrate import RK45, solve_ivp, simps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PARAM_G = 9.81
PARAM_L = 1.0
CTE = PARAM_G / PARAM_L

plot = False
movie = not plot
total_frames = 1000

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
t1 = 3*np.pi  # 0.0
n_outputs = int((t1 - t0) * 1000) + 1
y0 = np.array([np.pi*0.5, 0.])
dt_eval = (t1-t0)/n_outputs


int_methods = ['RK23', 'RK45', 'DOP853', 'LSODA']

results_dict = {}
for int_method in int_methods:
    print(f"integration methos: {int_method}")
    results = solve_ivp(fun=f,
                        t_span=(t0, t1),
                        y0=y0,
                        method=int_method,
                        t_eval=np.arange(t0, t1, dt_eval),
                        vectorized=True
                        )
    results_dict[int_method] = results


print(f"times shape: {results_dict['RK45']['t'].shape}")
print(f"coords shape: {results_dict['RK45']['y'].shape}")
#print(results_dict['RK45']['y'][0, :5])
#print(results_dict['RK45']['y'][1,:5])



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


#dt = (t1-t0)/n_outputs


if plot:
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
elif movie:
    def get_coords(th):
        """Return the (x, y) coordinates of the bob at angle th."""
        return PARAM_L * np.sin(th), -PARAM_L * np.cos(th)

        # Initialize the animation plot. Make the aspect ratio equal so it looks right.
    fig = plt.figure()
    ax = fig.add_subplot(aspect='equal')
    # The pendulum rod, in its initial position.
    plot_x0, plot_y0 = get_coords(y0[0])
    print(f"px={plot_x0}")
    print(f"py={plot_y0}")
    line, = ax.plot([0, plot_x0], [0, plot_y0], lw=3, c='k')
    # The pendulum bob: set zorder so that it is drawn over the pendulum rod.
    bob_radius = 0.08

    theta = results_dict['RK45']['y'][0,:]
    theta0 = y0[0]
    circle = ax.add_patch(plt.Circle(get_coords(theta0), bob_radius,
                          fc='r', zorder=3))
    # Set the plot limits so that the pendulum has room to swing!
    ax.set_xlim(-PARAM_L*1.2, PARAM_L*1.2)
    ax.set_ylim(-PARAM_L*1.2, PARAM_L*1.2)

    def animate(i):
        """Update the animation at frame i."""
        x, y = get_coords(theta[i])
        line.set_data([0, x], [0, y])
        circle.set_center((x, y))

    nframes = len(theta)
    interval = dt_eval * 1000
    print(interval)
    print(nframes)
    #plot_positions = []
    reduction_factor = int(nframes / total_frames)
    frames = [idx for idx in range(nframes) if idx % reduction_factor == 0]
    print(len(frames), frames)
    ani = animation.FuncAnimation(fig, animate, frames=frames, repeat=True,
                              interval=interval)
    plt.show()
