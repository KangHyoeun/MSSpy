import numpy as np
import matplotlib.pyplot as plt
from .models.mariner import mariner
from ..GNC.gnc import EKF_5states, ref_model, LOS_chi, LOS_observer, sat
from ..GNC.utils.controlMethod import control_method
from .models.utils.displayVehicleData import display_vehicle_data

"""
 This script simulates the dynamic behavior of a Mariner-Class Cargo Vessel, 
 length 160.93 m, under PID heading control, and waypoint path-following 
 control using a course autopilot (Fossen 2022). The Speed Over Ground 
 (SOG) and Course over Ground (COG) are estimated during path following 
 using the 5-state Extended Kalman Filter (EKF) by Fossen and Fossen (2021). 

 Dependencies:
   mariner.py           - Vessel dynamics.  
   EKF_5states.py       - EKF for estimation of SOG and COG.
   ref_model.py          - Reference model for autopilot systems.
   LOSchi.py         - LOS guidance algorithm for path following.
   LOS_observer.py       - Observer for LOS guidance. 
   controlMethods.py    - Menu for choosing control law.

 References: 
   T. I. Fossen (2022). Line-of-sight Path-following Control utilizing an 
      Extended Kalman Filter for Estimation of Speed and Course over Ground 
      from GNSS Positions. Journal of Marine Science and Technology 27, 
      pp. 806–813.
   T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion 
       Control, 2nd edition, John Wiley & Sons. Ltd., Chichester, UK.
   S. Fossen and T. I. Fossen (2021). Five-state Extended Kalman 
      Filter for Estimation of Speed Over Ground (SOG), Course Over Ground 
      (COG) and Course Rate of Unmanned Surface Vehicles (USVs): 
      Experimental Results. Sensors 21(23). 
"""
def ssa(angle, unit=None):
    """Normalize angle to the range [-180, 180] degrees or [-π, π] radians."""
    if unit == 'deg':
        return (angle + 180) % 360 - 180 
    else:
        return (angle + np.pi) % (2 * np.pi) - np.pi
    

def rk4(func, h, x, *args):
    """Runge-Kutta 4th order method"""
    x = np.asarray(x)
    k1 = func(x, *args)
    k1 = np.asarray(k1)
    k2 = func(x + h/2 * k1, *args)
    k2 = np.asarray(k2)
    k3 = func(x + h/2 * k2, *args)
    k3 = np.asarray(k3)
    k4 = func(x + h * k3, *args)
    k4 = np.asarray(k4)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)

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


def display_control_method(ControlFlag, R_switch, Delta_h):

    print("--------------------------------------------------------------------")
    print("MSS toolbox: Mariner-Class Cargo Vessel")
    print("Five-state EKF for estimation of SOG and COG")
    
    if ControlFlag == 1:
        print("PID course autopilot with reference feedforward")
    elif ControlFlag == 2:
        print("LOS path-following using straight lines and waypoint switching")
        print(f"Circle of acceptance: R = {R_switch} m")
        print(f"Look-ahead distance: Delta_h = {Delta_h} m")
    
    print("--------------------------------------------------------------------")
    print("Simulating...")

if __name__ == "__main__":

    T_final = 3000  # Final simulation time (s)
    h = 0.05        # Sampling time (s)
    Z = 2           # GNSS measurement frequency (2 times slower)

    # waypoints
    wpt = {
        'pos': {
            'x': np.array([0, 2000, 5000, 3000, 6000, 10000]),
            'y': np.array([0, 0, 5000, 8000, 12000, 12000])
        }
    }

    waypoints = np.array([
        [0, 0],
        [2000, 0],
        [5000, 5000],
        [3000, 8000],
        [6000, 12000],
        [10000, 12000]
    ])

    # LOS parameters
    Delta_h = 500   # Look-ahead distance
    R_switch = 400  # Radius of switching circle
    K_f = 0.2       # LOS observer gain

    # Initial heading, vehicle points towards next waypoint
    psi0 = np.arctan2(wpt['pos']['y'][1] - wpt['pos']['y'][0], wpt['pos']['x'][1] - wpt['pos']['x'][0])

    # PID pole placement algorithm
    wn = 0.05                             # Closed-loop natural frequency
    T = 107.3                             # Nomoto time constant
    K = 0.185                             # Nomoto gain constant
    Kp = (T / K) * wn**2                  # Proportional gain
    Td = T / (K * Kp) * (2 * wn - 1 / T)  # Derivative time constant
    Ti = 10 / wn                          # Integral time constant

    # Reference model specifying the heading autopilot closed-loop dynamics
    wn_d = 0.1               # Natural frequency (rad/s)
    zeta_d = 1.0             # Relative damping factor (-)
    r_max = np.deg2rad(1.0)  # Maximum turning rate (rad/s)

    # Initialization for mariner vessel states
    x = np.array([0, 0, 0, 0, 0, psi0, 0])  # x = [u v r x y psi delta]'

    U0 = 7.7175        # Nominal speed
    e_int = 0.0        # Autopilot integral state
    delta_c = 0.0      # Initial rudder angle command
    psi_d = x[5]       # Initial desired heading angle
    chi_d = x[5]       # Initial desired course angle
    omega_chi_d = 0.0  # Initial desired course rate
    r_d = 0.0          # Initial desired rate of turn
    a_d = 0.0          # Initial desired acceleration

    # Initial EKF states x_hat = [x, y, U, chi, omega_chi]
    x_prd_init = np.array([x[3], x[4], U0, x[5], 0])  
    x_hat = x_prd_init  # Corrector equals initial state vector

    # Time vector initialization
    t = np.arange(0.0, T_final + h, h)  # Time vector from 0 to T_final
    nTimeSteps = len(t)                 # Number of time steps

    # Choose control method and display simulation options
    methods = [
        'PID heading autopilot, no path-following',
        'LOS path-following',
    ]
    ControlFlag = control_method(methods)
    display_control_method(ControlFlag, R_switch, Delta_h)

    """MAIN LOOP"""
    simdata = np.zeros((nTimeSteps, 15))  # Preallocate table

    for i in range(nTimeSteps):

        # Measurements with measurement noise
        r    = x[2] + 0.0001 * np.random.randn()
        xpos = x[3] + 0.01 * np.random.randn()
        ypos = x[4] + 0.01 * np.random.randn()
        psi  = x[5] + 0.0001 * np.random.randn()

        # EKF estimates used for path-following control
        U_hat = x_hat[2]
        chi_hat = x_hat[3]
        omega_chi_hat = x_hat[4]

        # Guidance and control system
        if ControlFlag == 1:
            # PID heading autopilot
            psi_ref = psi0  # Reference model, step input adjustments
            if t[i] > 100: psi_ref = np.deg2rad(30)
            if t[i] > 1000: psi_ref = np.deg2rad(-30)

            # PID heading autopilot
            e = ssa(psi - psi_d)
            delta_PID = (T/K) * a_d + (1/K) * r_d - Kp * (e + Td * (r - r_d) + (1/Ti) * e_int)  # Feedforward, PID

            # Reference model propagation
            psi_d, r_d, a_d = ref_model(psi_d, r_d, a_d, psi_ref, r_max, zeta_d, wn_d, h, 1)

        elif ControlFlag == 2:
            # LOS course autopilot for straight-line path following
            chi_ref, y_e = LOS_chi(xpos, ypos, Delta_h, R_switch, wpt)

            # LOS observer for estimation of chi_d and omega_chi_d
            chi_d, omega_chi_d = LOS_observer(chi_d, omega_chi_d, chi_ref, h, K_f)

            omega_chi_d = sat(omega_chi_d, np.deg2rad(1))  # Max value

            # PID course autopilot
            e = ssa(chi_hat - chi_d)
            delta_PID = (1/K) * omega_chi_d - Kp * (e + Td * (omega_chi_hat - omega_chi_d) + (1/Ti) * e_int)

        delta_c = sat(delta_c, np.deg2rad(40))  # Maximum rudder angle

        # Store data for presentation
        simdata[i] = np.array(x.tolist() + [psi_d, r_d, chi_d, omega_chi_d, delta_c, U_hat, chi_hat, omega_chi_hat])

        # RK4 method x(k+1)
        x = rk4(mariner, h, x, delta_c)  # RK4 method x(k+1)

        # Euler's method
        e_int += h * e
        delta_c += h * (delta_PID - delta_c) / 1.0

        # Propagation of the EKF states
        if i == 0:
            x_hat = EKF_5states(xpos, ypos, h, Z, 'NED', 100*np.diag([0.1,0.1]), 1000*np.diag([1,1]), 0.00001, 0.00001, x_prd_init)
        else:
            x_hat = EKF_5states(xpos, ypos, h, Z, 'NED', 100*np.diag([0.1,0.1]), 1000*np.diag([1,1]), 0.00001, 0.00001, None)

    """PLOTS"""

    # simdata[i] = [*x[:7], U, psi_d, r_d, chi_d, omega_chi_d, delta_c]
    u     = simdata[:,0]
    v     = simdata[:,1]
    r     = np.rad2deg(simdata[:,2])
    x     = simdata[:,3]
    y     = simdata[:,4]
    psi   = np.rad2deg(simdata[:,5])
    delta = -np.rad2deg(simdata[:,6])  # delta = -delta_r (physical_angle)
    psi_d = np.rad2deg(simdata[:,7])
    r_d   = np.rad2deg(simdata[:,8])
    chi_d = np.rad2deg(simdata[:,9])
    omega_chi_d = np.rad2deg(simdata[:,10])
    delta_c = np.rad2deg(simdata[:,11])
    U_hat = simdata[:,12]
    chi_hat = np.rad2deg(simdata[:,13])
    omega_chi_hat = np.rad2deg(simdata[:,14])

    U = np.sqrt((U0 + u)**2 + v**2)  # SOG
    beta_c = np.rad2deg(np.arctan2(v, U0 + u))  # Crab angle
    chi = ssa(psi + beta_c, 'deg')   # COG

    # Plot and animation of the North-East positions
    plt.figure(1, figsize=(10,10))
    plt.plot(y, x, 'b', label='Vessel Position')
    if ControlFlag == 2:  # Path-following controller, straight line and circles
        plot_straight_lines_and_circles(waypoints, R_switch)

    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('North-East Positions (m)')
    plt.axis('equal')
    plt.grid()
    plt.legend()

    plt.figure(2, figsize=(16,8))

    plt.subplot(221)
    if ControlFlag == 1:
        # Heading autopilot
        plt.plot(t, r_d, label='Desired')
        plt.plot(t, r, label='True')
        plt.ylabel('Yaw rate (deg/s)')
    else:
        # Course autopilot
        plt.plot(t, omega_chi_hat, label='Estimated')
        plt.plot(t, omega_chi_d, label='Desired')
        plt.ylabel('Course rate (deg/s)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid()

    plt.subplot(222)
    plt.plot(t, U_hat, label='Estimated')
    plt.plot(t, U, label='True')
    plt.ylabel('Speed (m/s)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(223)
    if ControlFlag == 1:
        # Heading autopilot
        plt.plot(t, psi_d, label='Desired')
        plt.plot(t, psi, label='True')
        plt.ylabel('Yaw angle (deg)')
    else:
        # Course autopilot
        plt.plot(t, chi_hat, label='Estimated')
        plt.plot(t, chi, label='True')
        plt.plot(t, chi_d, 'k', label='Desired')
        plt.ylabel('Course angle (deg)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(224)
    plt.plot(t, delta, label='Actual')
    plt.plot(t, delta_c, label='Commanded')
    plt.ylabel('Rudder angle (deg)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    for ax in plt.gcf().get_axes():
        ax.tick_params(labelsize=14)


    vehicleName = "Mariner-Class Cargo Vessel"
    vehicleData = [('Length', '160.93m'), ('Mass', '17,045 tonnes'), ('Max speed', '7.71m/s'), ('Max rudder angle', '40 deg')]
    imageFile = "mariner.jpg"
    figNo = 3
    display_vehicle_data(vehicleName, vehicleData, imageFile, figNo)

    plt.show()
