import numpy as np
import matplotlib.pyplot as plt
from models.container import container, Lcontainer
from models.utils.displayVehicleData import display_vehicle_data
"""
 This script simulates the dynamics of a container ship under feedback 
 control. The script concurrently simulates the ship using both a linear 
 model, defined in 'Lcontainer.py', and a nonlinear model, defined in 
 'container.py'. The outcomes of both simulations are then plotted side by 
 side for comparative analysis.

 Dependencies:
   container.py     - Nonlinear container ship model
   Lcontainer.py    - Linearized container ship model
"""


def rk4(func, h, x, *args):
    """Runge-Kutta 4th order method"""
    x = np.asarray(x)
    k1 = func(x, *args)
    k1 = np.asarray(k1)
    k2 = func(x + h / 2 * k1, *args)
    k2 = np.asarray(k2)
    k3 = func(x + h / 2 * k2, *args)
    k3 = np.asarray(k3)
    k4 = func(x + h * k3, *args)
    k4 = np.asarray(k4)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def ssa(angle, unit='deg'):
    """Normalize angle to the range [-180, 180] degrees or [-π, π] radians."""
    if unit == 'deg':
        return (angle + 180) % 360 - 180
    else:
        return (angle + np.pi) % (2 * np.pi) - np.pi


def display_control_method():

    print(
        "--------------------------------------------------------------------")
    print("MSS toolbox: High-Speed Container Ship")
    print("Linearized and nonlinear models")
    print("PD heading autopilot")
    print(
        "--------------------------------------------------------------------")
    print("Simulating...")


if __name__ == "__main__":
    """USER INPUTS"""
    T_final = 600  # Final simulation time [s]
    h = 0.1  # Sample time (sec)

    Kp = 1  # Controller P gain
    Td = 10  # Controller derivative time

    # Initial states
    x1 = np.array([7, 0, 0, 0, 0, 0, 0, 0, 0,
                   70]).T  # x1 = [u v r x y psi p phi delta n]'
    x2 = np.array([7, 0, 0, 0, 0, 0, 0, 0,
                   0]).T  # x2 = [u v r x y psi p phi delta]'

    # Time vector initialization
    t = np.arange(0, T_final + h, h)  # Time vector from 0 to T_final
    n_time_steps = len(t)  # Number of time steps

    # Display simulation options
    display_control_method()
    """MAIN LOOP"""
    simdata1 = np.zeros((n_time_steps, len(x1)))  # Preallocate table
    simdata2 = np.zeros((n_time_steps, len(x2)))  # Preallocate table

    for i in range(n_time_steps):

        r = x1[2]
        psi = x1[5]

        # Control system (constant thrust + PD heading controller)
        psi_ref = np.deg2rad(5)  # Desired heading
        delta_c = -Kp * (ssa(psi - psi_ref) + Td * r)  # PD controller
        n_c = 70

        # Store data for presentation
        simdata1[i] = x1.T
        simdata2[i] = x2.T

        # RK4 method (k+1)
        x1 = rk4(container, h, x1, np.array([delta_c, n_c]))
        x2 = rk4(Lcontainer, h, x2, np.array([delta_c]))

        # Euler's integration method (k+1)
        # xdot1 = container(x1,[delta_c n_c]);
        # xdot2 = Lcontainer(x2,delta_c)
        # x1 = euler2(xdot1, x1, h)
        # x2 = euler2(xdot2, x2, h)
        
    """PLOTS"""
    u1 = simdata1[:, 0]
    v1 = simdata1[:, 1]
    r1 = simdata1[:, 2]
    x1 = simdata1[:, 3]
    y1 = simdata1[:, 4]
    psi1 = np.rad2deg(simdata1[:, 5])
    p1 = np.rad2deg(simdata1[:, 6])
    phi1 = np.rad2deg(simdata1[:, 7])
    delta1 = np.rad2deg(simdata1[:, 8])
    n1 = simdata1[:, 9]

    u2 = simdata2[:, 0]
    v2 = simdata2[:, 1]
    r2 = simdata2[:, 2]
    x2 = simdata2[:, 3]
    y2 = simdata2[:, 4]
    psi2 = np.rad2deg(simdata2[:, 5])
    p2 = np.rad2deg(simdata2[:, 6])
    phi2 = np.rad2deg(simdata2[:, 7])
    delta2 = np.rad2deg(simdata2[:, 8])

    # North-East positions
    plt.figure(1, figsize=(10, 10))
    plt.plot(y1, x1, 'r', label='Nonlinear model')
    plt.plot(y2, x2, 'b', label='Linear model')
    plt.grid()
    plt.axis('Equal')
    plt.xlabel('East')
    plt.ylabel('North')
    plt.title('Ship position (m)')
    plt.legend()

    plt.figure(2, figsize=(16, 8))
    plt.subplot(221)
    plt.plot(t, r1, 'r', label='Nonlinear model')
    plt.plot(t, r2, 'b', label='Linear model')
    plt.xlabel('Time (s)')
    plt.ylabel('r (deg/s)')
    plt.grid()

    plt.subplot(222)
    plt.plot(t, phi1, 'r', label='Nonlinear model')
    plt.plot(t, phi2, 'b', label='Linear model')
    plt.xlabel('Time (s)')
    plt.ylabel('$\phi$ (deg/s)')
    plt.grid()

    plt.subplot(223)
    plt.plot(t, psi1, 'r', label='Nonlinear model')
    plt.plot(t, psi2, 'b', label='Linear model')
    plt.xlabel('Time (s)')
    plt.ylabel('$\psi$ (deg)')
    plt.grid()

    plt.subplot(224)
    plt.plot(t, delta1, 'r', label='Nonlinear model')
    plt.plot(t, delta2, 'b', label='Linear model')
    plt.xlabel('Time (s)')
    plt.ylabel('$\delta$ (deg)')
    plt.grid()

    vesselData = {('Length', '175 m'), ('Beam', '25.4 m'), ('Draft', '8.5 m'),
                  ('Mass', '21,750 tonnes'),
                  ('Volume displacement', '21,222 m3'),
                  ('Service speed', '7.0 m/s'), ('Max rudder angle', '10 deg'),
                  ('Max propeller speed', '160 RPM')}
    vesselName = "High-Speed Container Ship"
    imageFile = "container.jpg"
    figNO = 3
    display_vehicle_data(vesselName, vesselData, imageFile, figNO)
