import numpy as np
from numpy import unwrap
import matplotlib.pyplot as plt
from models.otter import otter
from GNC.gnc import ALOS_psi, ILOS_psi, add_intermediate_waypoints, crosstrack_hermite_LOS, hermite_spline, ref_model, LOS_observer
from GNC.utils.controlMethod import control_method
from models.utils.displayVehicleData import display_vehicle_data

"""
This script simulates the Otter Uncrewed Surface Vehicle (USV) under 
various control strategies to handle path following in the presence of 
ocean currents. This script allows the user to select from several control
methods and simulatesthe USV's performance using a cubic Hermite spline
or straight-line paths.

The simulation covers:
1. PID heading autopilot without path following.
2. Adaptive Line-of-Sight (ALOS) control for path following using
    straight lines and waypoint switching.
3. Integral Line-of-Sight (ILOS) control for path following using
    straight lines and waypoint switching.
4. ALOS control for path following using Hermite spline interpolation.

Dependencies:
otter                 - Dynamics of the Otter USV
ref_model              - Reference model for autopilot systems
ALOS_psi               - ALOS guidance algorithm for path following
ILOS_psi               - ILOS guidance algorithm for path following
hermite_spline         - Cubic Hermite spline computations
crosstrackHermiteLOS  - Cross-track error and LOS guidance law for
                            cubic Hermite splines
LOS_observer           - Observer for LOS guidance 
controlMethods        - Menu for choosing control law. 
"""

def ssa(angle, unit=None):
    if unit == 'deg':
        angle = (angle + 180) % 360 - 180
    else:
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle

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


def plot_straight_lines_and_circles(waypoints, R_switch):

    # Plot straight lines and circles for straight-line path following
    for idx in range(len(waypoints) - 1):
        x_coords = np.array([waypoints[idx, 0], waypoints[idx + 1, 0]])
        y_coords = np.array([waypoints[idx, 1], waypoints[idx + 1, 1]])

        if idx == 0:
            plt.plot(y_coords, x_coords, 'r--', label='Straight-line path')
        else:
            plt.plot(y_coords, x_coords, 'r--')

    theta = np.linspace(0, 2 * np.pi, 100)

    for idx in range(len(waypoints) - 1):
        x_circle = R_switch * np.cos(theta) + waypoints[idx, 0]
        y_circle = R_switch * np.sin(theta) + waypoints[idx, 1]

        if idx == 0:
            plt.plot(y_circle, x_circle, 'k', label='Circle of Acceptance')
        else:
            plt.plot(y_circle, x_circle, 'k')

def plot_hermite_splines(y_path, x_path, wpt):

    plt.plot(y_path, x_path, 'r', label='Hermite spline')
    plt.plot(wpt['pos']['y'], wpt['pos']['x'], 'ko', markerfacecolor='g', markersize=10, label='Waypoints')


def display_control_method(ControlFlag, R_switch, Delta_h):

    print(
        "--------------------------------------------------------------------")
    print("MSS toolbox: Otter USV")

    if ControlFlag == 1:
        print("PID course autopilot with reference feedforward")
    elif ControlFlag == 2:
        print("LOS path-following control using straight lines and waypoint switching")
        print(f"Circle of acceptance: R = {R_switch} m")
        print(f"Look-ahead distance: Delta_h = {Delta_h} m")
    elif ControlFlag == 3:
        print("ILOS path-following control using straight lines and waypoint switching")
        print(f"Circle of acceptance: R = {R_switch} m")
        print(f"Look-ahead distance: Delta_h = {Delta_h} m")
    else:
        print("ALOS path-following control using Hermite splines")
        print(f"Look-ahead distance: Delta_h = {Delta_h} m")   

    print(
        "--------------------------------------------------------------------")
    print("Simulating...")

    
if __name__ == "__main__":
    ## USER INPUTS
    h  = 0.05                 # Sampling time [s]
    T_final = 1000	                 # Final simulation time [s]

    # Load condition
    mp = 25                         # Payload mass (kg), maximum value 45 kg
    rp = np.array([0.05, 0, -0.35])            # Location of payload (m)

    # Ocean current
    V_c = 0.3                       # Ocean current speed (m/s)
    beta_c = np.deg2rad(30)            # Ocean current direction (rad)

    # Waypoints
    wpt = {
        'pos': {
            'x': np.array([0, 0, 150, 150, -100, -100, 200]),
            'y': np.array([0, 200, 200, -50, -50, 250, 250])
        }
    }

    waypoints = np.array([[0, 0], [0, 200], [150, 200], [150, 200],
                          [-100, 250], [200, 250]])

    # Add intermediate waypoints along the line segments between for better resolution
    wpt = add_intermediate_waypoints(wpt, 2)

    # ALOS and ILOS parameters
    Delta_h = 10                    # Look-ahead distance
    gamma_h = 0.001                 # ALOS adaptive gain
    kappa = 0.001                   # ILOS integral gain

    # Additional parameter for straight-line path following
    R_switch = 5                    # Radius of switching circle
    K_f = 0.3                       # LOS observer gain

    # Initial heading, vehicle points towards next waypoint
    psi0 = np.arctan2(wpt['pos']['y'][1] - wpt['pos']['y'][0],
                      wpt['pos']['x'][1] - wpt['pos']['x'][0])
    
    print(f"psi0 = {np.rad2deg(psi0):.2f}")

    # Additional parameters for Hermite spline path following
    Umax = 2                        # Maximum speed for Hermite spline LOS
    idx_start = 0                   # Initial index for Hermite spline
    w_path, x_path, y_path, dx, dy, pi_h, pp_x, pp_y, N_horizon = hermite_spline(wpt, Umax, h) # Compute Hermite spline for path following

    # Otter USV input matrix
    M = np.array([
        [85.2815,  0,      0,      0,      -11,       0],
        [0,      162.5,    0,     11,        0,      11],
        [0,        0,    135,      0,      -11,       0],
        [0,       11,      0,     15.0775,   0,       2.5523],
        [-11,      0,    -11,      0,       31.5184,  0],
        [0,       11,      0,      2.5523,   0,      41.4451]
    ])
    B_prop = np.array([
        [0.0111,     0.0111],
        [0.0044,  -0.0044]
    ])

    Binv = np.array([
        [45.1264,   114.2439],
        [45.1264,  -114.2439]
    ])


    # PID heading autopilot parameters (Nomoto model: M(6,6) = T/K)
    T = 1                           # Nomoto time constant
    K = T / M[5,5]                 # Nomoto gain constant

    wn = 1.5                        # Closed-loop natural frequency (rad/s)
    zeta = 1.0                      # Closed-loop relative damping factor (-)

    Kp = M[5,5] * wn**2                     # Proportional gain
    Kd = M[5,5] * (2 * zeta * wn - 1/T)    # Derivative gain
    Td = Kd / Kp                           # Derivative time constant
    Ti = 10.0 / wn                           # Integral time constant

    # Reference model parameters
    wn_d = 1.0                      # Natural frequency (rad/s)
    zeta_d = 1.0                    # Relative damping factor (-)
    r_max = np.deg2rad(10.0)           # Maximum turning rate (rad/s)

    # Propeller dynamics
    T_n = 0.1                       # Propeller time constant (s)
    n = np.array([0.0, 0.0])                      # Initial propeller speed, [n_left n_right]'

    # Initial states
    x = np.zeros(12)                 # x = [u v w p q r xn yn zn phi theta psi]'
    x[11] = psi0                    # Heading angle
    z_psi = 0.0                       # Integral state for heading control
    psi_d = psi0                    # Desired heading angle
    r_d = 0.0                         # Desired rate of turn
    a_d = 0.0                         # Desired acceleration

    # Time vector initialization
    t = np.arange(0.0, T_final + h, h)  # Time vector from 0 to T_final
    nTimeSteps = len(t)         # Number of time steps

    # Choose control method and display simulation options
    methods = ["PID heading autopilot, no path following",
            "ALOS path-following control using straight lines and waypoint switching",
            "ILOS path-following control using straight lines and waypoint switching",
            "ALOS path-following control using Hermite splines"]
    ControlFlag = control_method(methods)
    display_control_method(ControlFlag, R_switch, Delta_h)

    ## MAIN LOOP
    simdata = np.zeros((nTimeSteps, 14))    # Preallocate table for simulation data

    for i in range(nTimeSteps):

        # Measurements with noise
        # r = x[5] 
        # xn = x[6] 
        # yn = x[7]
        # psi = x[11] 

        r = x[5] + 0.001 * np.random.randn()       # Yaw rate 
        xn = x[6] + 0.01 * np.random.randn()       # North position
        yn = x[7] + 0.01 * np.random.randn()       # East position
        psi = x[11] + 0.001 * np.random.randn()    # Yaw angle

        # Guidance and control system
        if ControlFlag == 1:  # PID heading autopilot with reference model
            # Reference model, step input adjustments
            if t[i] > 500:
                psi_ref = np.deg2rad(-90)
            elif t[i] > 100:
                psi_ref = np.deg2rad(0)
            else:
                psi_ref = psi0

            # Reference model propagation
            psi_d, r_d, a_d = ref_model(psi_d, r_d, a_d, psi_ref, r_max, zeta_d, wn_d, h, 1)

        elif ControlFlag == 2:  # ALOS heading autopilot straight-line path following
            psi_ref, _ = ALOS_psi(xn, yn, Delta_h, gamma_h, h, R_switch, wpt)
            psi_d, r_d = LOS_observer(psi_d, r_d, psi_ref, h, K_f)

        elif ControlFlag == 3:  # ILOS heading autopilot straight-line path following
            psi_ref, _ = ILOS_psi(xn, yn, Delta_h, kappa, h, R_switch, wpt)
            psi_d, r_d = LOS_observer(psi_d, r_d, psi_ref, h, K_f)

        else:  # ALOS heading autopilot, cubic Hermite spline interpolation
            psi_ref, idx_start = crosstrack_hermite_LOS(w_path, x_path, y_path, dx, dy, pi_h, xn, yn, h, Delta_h, pp_x, pp_y, idx_start, N_horizon, gamma_h)
            psi_d, r_d = LOS_observer(psi_d, r_d, psi_ref, h, K_f)
        

        # PID heading (yaw moment) autopilot and forward thrust
        tau_X = 100                              # Constant forward thrust
        tau_N = (T/K) * a_d + (1/K) * r_d - Kp * (ssa(psi - psi_d) + Td * (r - r_d) + (1/Ti) * z_psi) # Derivative and integral terms

        # Control allocation
        u = Binv @ np.array([tau_X, tau_N])      # Compute control inputs for propellers
        n_c = np.sign(u) * np.sqrt(np.abs(u))  # Convert to required propeller speeds

        # Debug log every 5 steps
        if i % 50 == 0:
            print(f"[t={t[i]:.1f}s] tau_N = {tau_N:.2f}, psi = {np.rad2deg(psi):.2f}, psi_d = {np.rad2deg(psi_d):.2f}")
            print(f"         u = {u}, n_c = {n_c}, n = {n}")
            print(f"         ref_model: psi_d = {np.rad2deg(psi_d):.2f}, r_d = {np.rad2deg(r_d):.2f}, a_d = {np.rad2deg(a_d):.2f}")

        # Store simulation data
        simdata[i, :] = np.concatenate((x.T, [r_d, psi_d]))

        # RK4 method x(k+1)
        x = rk4(lambda x_, *args: otter(x_, *args)[0], h, x, n, mp, rp, V_c, beta_c)

        # Euler's method
        n = n + h/T_n * (n_c - n)              # Update propeller speeds
        z_psi = z_psi + h * ssa(psi - psi_d)   # Update integral state


    ## PLOTS

    # Simulation data structure
    nu   = simdata[:,0:6] 
    eta  = simdata[:,6:12] 
    r_d = simdata[:,12]    
    psi_d = simdata[:,13]  

    # positions
    plt.figure(1, figsize=(10, 10))
    plt.plot(eta[:,1],eta[:,0],'b', label='Vehicle position')  # vehicle position

    # Control method specific plots
    if ControlFlag in [2, 3]:  # Heading autopilot
        plot_straight_lines_and_circles(wpt, R_switch)
    elif ControlFlag == 4:  # Hermite splines
        plot_hermite_splines(y_path, x_path, wpt)


    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('North-East Positions (m)')

    # plt.plot velocities
    plt.figure(2, figsize=(4, 10))
    plt.subplot(611)
    plt.plot(t,nu[:,0])
    plt.xlabel('Time (s)')
    plt.ylabel('Surge velocity (m/s)')
    plt.subplot(612)
    plt.plot(t,nu[:,1])
    plt.xlabel('Time (s)')
    plt.ylabel('Sway velocity (m/s)')
    plt.subplot(613)
    plt.plot(t,nu[:,2])
    plt.xlabel('Time (s)')
    plt.ylabel('Heave velocity (m/s)')
    plt.subplot(614)
    plt.plot(t,np.rad2deg(nu[:,3]))
    plt.xlabel('Time (s)')
    plt.ylabel('Roll rate (deg/s)')
    plt.subplot(615)
    plt.plot(t,np.rad2deg(nu[:,4]))
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch rate (deg/s)')
    plt.subplot(616)
    plt.plot(t,np.rad2deg(nu[:,5]),label='r')
    plt.plot(t,np.rad2deg(r_d),label='r_d')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw rate (deg/s)')
    plt.legend()

    # plt.plot speed, heave position and Euler angles
    plt.figure(3, figsize=(4, 10))
    plt.subplot(511)
    plt.plot(t, np.sqrt(nu[:,0]**2 + nu[:,1]**2))
    plt.ylabel('Speed (m/s)')
    plt.subplot(512)
    plt.plot(t,eta[:,2])
    plt.ylabel('Heave position (m)')
    plt.subplot(513)
    plt.plot(t,np.rad2deg(eta[:,3]))
    plt.ylabel('Roll angle (deg)')
    plt.subplot(514)
    plt.plot(t,np.rad2deg(eta[:,4]))
    plt.ylabel('Pitch angle (deg)')
    plt.subplot(515)
    plt.plot(t,np.rad2deg(unwrap(eta[:,5])),label='$\psi$')
    plt.plot(t,np.rad2deg(unwrap(psi_d)),label='$\psi_d$')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw angle (deg)')
    plt.legend()

    plt.show()

    # Display the vehicle data and an image of the vehicle
    vehicleName = "Maritime Robotics Otter USV"
    imageFile = "otter.jpg"
    figNo = 4
    vehicleData = [('len', '2.0 m'), ('Beam', '1.08 m'), ('Draft (no payload)', '13.4 cm'), ('Draft (25 kg payload)', '19.5 cm'), ('Mass (no payload)', '55.0 kg'), ('Max speed', '3.0 m/s'), ('Max pos. propeller speed', '993 RPM'), ('Max neg. propeller speed', '-972 RPM')]

    # display_vehicle_data(vehicleName, vehicleData, imageFile, 4)


