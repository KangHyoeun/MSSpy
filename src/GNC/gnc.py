import numpy as np

def ssa(angle, unit=None):
    """Normalize angle to the range [-180, 180] degrees or [-π, π] radians."""
    if unit == 'deg':
        return (angle + 180) % 360 - 180 
    else:
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
def sat(x, x_max):
    """
    Saturates the input x at the specified maximum absolute value x_max.
    """
    if np.all(np.array(x_max) <= 0):
        raise ValueError("x_max must be a vector of positive elements.")

    return np.minimum(np.maximum(x, -np.array(x_max)), np.array(x_max))

def ref_model(x_d, v_d, a_d, x_ref, v_max, zeta_d, w_d, h, eulerAngle):
    """
    Position, velocity, and acceleration reference model.
    """
    if eulerAngle == 1:
        e_x = ssa(x_d - x_ref)  # Smallest signed angle
    else:
        e_x = x_d - x_ref

    a_d_dot = -w_d**3 * e_x - (2 * zeta_d + 1) * w_d**2 * v_d - (2 * zeta_d + 1) * w_d * a_d

    x_d += h * v_d
    v_d += h * a_d
    a_d += h * a_d_dot

    if abs(v_d) > v_max:
        v_d = np.sign(v_d) * v_max

    return x_d, v_d, a_d

def LOS_observer(LOSangle, LOSrate, LOScommand, h, K_f):
    T_f = 1 / (K_f + 2 * np.sqrt(K_f) + 1)
    xi = LOSangle - LOSrate

    LOSangle += h * (LOSrate + K_f * ssa(LOScommand - LOSangle))
    
    PHI = np.exp(-h / T_f)
    xi = (PHI * xi)  + (1 - PHI) * LOSangle
    
    LOSrate = LOSangle - xi

    return LOSangle, LOSrate
def LOS_chi(x, y, Delta_h, R_switch, wpt):
    """
    Compute the desired course angle (chi_ref) and cross-track error (y_e).
    """
    if not hasattr(LOS_chi, 'persistent'):
        LOS_chi.persistent = {}

    persistent = LOS_chi.persistent

    if 'k' not in persistent:
        dist_between_wpts = np.sqrt(np.diff(wpt['pos']['x'])**2 + np.diff(wpt['pos']['y'])**2)
        if R_switch > min(dist_between_wpts):
            raise ValueError("The distances between the waypoints must be larger than R_switch")

        if R_switch < 0:
            raise ValueError("R_switch must be larger than zero")

        if Delta_h < 0:
            raise ValueError("Delta_h must be larger than zero")

        persistent['k'] = 0
        persistent['xk'] = wpt['pos']['x'][0]
        persistent['yk'] = wpt['pos']['y'][0]

    k = persistent['k']
    xk = persistent['xk']
    yk = persistent['yk']

    n = len(wpt['pos']['x'])

    if k < n - 1:
        xk_next = wpt['pos']['x'][k + 1]
        yk_next = wpt['pos']['y'][k + 1]
    else:
        bearing = np.arctan2(wpt['pos']['y'][-1] - wpt['pos']['y'][-2], wpt['pos']['x'][-1] - wpt['pos']['x'][-2])
        R = 1e10
        xk_next = wpt['pos']['x'][-1] + R * np.cos(bearing)
        yk_next = wpt['pos']['y'][-1] + R * np.sin(bearing)

    pi_h = np.arctan2(yk_next - yk, xk_next - xk)

    x_e = (x - xk) * np.cos(pi_h) + (y - yk) * np.sin(pi_h)
    y_e = -(x - xk) * np.sin(pi_h) + (y - yk) * np.cos(pi_h)

    d = np.sqrt((xk_next - xk)**2 + (yk_next - yk)**2)

    if d - x_e < R_switch and k < n - 1:
        persistent['k'] += 1
        persistent['xk'] = xk_next
        persistent['yk'] = yk_next

    chi_ref = pi_h - np.arctan(y_e / Delta_h)

    return chi_ref, y_e

def EKF_5states(position1, position2, h, Z, frame, Qd, Rd, alpha_1=0.01, alpha_2=0.1, x_prd_init=None):
    if not hasattr(EKF_5states, 'persistent'):
        EKF_5states.persistent = {}

    persistent = EKF_5states.persistent

    # WGS-84 데이터
    a = 6378137  # 반장축 (적도 반지름)
    f = 1 / 298.257223563  # 평평도
    e = np.sqrt(2 * f - f**2)  # 지구 이심률
    
    # 초기 상태 설정
    I5 = np.eye(5)

    if 'x_prd' not in persistent:
        if x_prd_init is None:
            print(f"Using default initial EKF states: x_prd = [{position1}, {position2}, 0, 0, 0]")
            persistent['x_prd'] = np.array([position1, position2, 0, 0, 0])
        else:
            print(f"Using user specified initial EKF states: x_prd = {x_prd_init}")
            persistent['x_prd'] = np.array(x_prd_init)
        persistent['P_prd'] = I5
        persistent['count'] = 1

    x_prd = persistent['x_prd']
    P_prd = persistent['P_prd']
    count = persistent['count']

    Cd = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    Ed = h * np.array([[0, 0], [0, 0], [1, 0], [0, 0], [0, 1]])

    if count == 1:
        y = np.array([position1, position2])
        K = P_prd @ Cd.T @ np.linalg.inv(Cd @ P_prd @ Cd.T + Rd)
        IKC = I5 - K @ Cd
        P_hat = IKC @ P_prd @ IKC.T + K @ Rd @ K.T
        eps = y - Cd @ x_prd

        if frame == 'LL':
            eps = eps % (2 * np.pi) - np.pi  # Smallest signed angle  
        
        x_hat = x_prd + K @ eps
        count = Z
    else:
        x_hat = x_prd
        P_hat = P_prd
        count -= 1

    if frame == 'NED':
        f = np.array([
            x_hat[2] * np.cos(x_hat[3]),
            x_hat[2] * np.sin(x_hat[3]),
            -alpha_1 * x_hat[2],
            x_hat[4],
            -alpha_2 * x_hat[4]
        ])

        Ad = I5 + h * np.array([
            [0, 0, np.cos(x_hat[3]), -x_hat[2] * np.sin(x_hat[3]), 0],
            [0, 0, np.sin(x_hat[3]), x_hat[2] * np.cos(x_hat[3]), 0],
            [0, 0, -alpha_1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, -alpha_2]
        ])

    elif frame == 'LL':
        Rn = a / np.sqrt(1 - e**2 * np.sin(x_hat[0])**2)
        Rm = Rn * ((1 - e**2) / (1 - e**2 * np.sin(x_hat[0])**2))
        
        f = np.array([
            (1 / Rm) * x_hat[2] * np.cos(x_hat[3]),
            (1 / (Rn * np.cos(x_hat[0]))) * x_hat[2] * np.sin(x_hat[3]),
            -alpha_1 * x_hat[2],
            x_hat[4],
            -alpha_2 * x_hat[4]
        ])
        
        Ad = I5 + h * np.array([
            [0, 0, (1 / Rm) * np.cos(x_hat[3]), -(1 / Rm) * x_hat[2] * np.sin(x_hat[3]), 0],
            [np.tan(x_hat[0]) / (Rn * np.cos(x_hat[0])) * x_hat[2] * np.sin(x_hat[3]), 0, (1 / (Rn * np.cos(x_hat[0]))) * np.sin(x_hat[3]), (1 / (Rn * np.cos(x_hat[0]))) * x_hat[2] * np.cos(x_hat[3]), 0],
            [0, 0, -alpha_1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, -alpha_2]
        ])

    x_prd = x_hat + h * f
    P_prd = Ad @ P_hat @ Ad.T + Ed @ Qd @ Ed.T

    persistent['x_prd'] = x_prd
    persistent['P_prd'] = P_prd
    persistent['count'] = count

    return x_hat
