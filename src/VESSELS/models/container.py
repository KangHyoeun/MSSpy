"""
 [xdot,U] = container(x,ui) returns the speed U in m/s (optionally) and the 
 time derivative of the state vector: x = [ u v r x y psi p phi delta n ]'
 for a container ship L = 175 m, where

 u     = surge velocity          (m/s)
 v     = sway velocity           (m/s)
 r     = yaw velocity            (rad/s)
 x     = position in x-direction (m)
 y     = position in y-direction (m)
 psi   = yaw angle               (rad)
 p     = roll velocity           (rad/s)
 phi   = roll angle              (rad)
 delta = actual rudder angle     (rad)
 n     = actual shaft velocity   (rpm)

 The input vector is:

 ui      = [ delta_c n_c ]'  where

 delta_c = commanded rudder angle (rad)
 n_c     = commanded shaft speed (rpm)  

 Reference:  Son og Nomoto (1982). On the Coupled Motion of Steering and 
             Rolling of a High Speed Container Ship, Naval Architect of 
             Ocean Engineering 20:73-83. From J.S.N.A., Japan, Vol. 150, 1981.
"""
import numpy as np


def container(x, ui):
    if len(x) != 10:
        raise ValueError("x-vector must have dimension 10!")
    if len(ui) != 2:
        raise ValueError("u-vector must have dimension 2!")

    print(x)

    # Normalization variables
    L = 175  # Length of ship (m)
    U = np.sqrt(x[0]**2 + x[1]**2)  # Service speed (m/s)

    if np.all(U <= 0):
        raise ValueError("The ship must have speed greater than zero")
    if np.all(x[9]) <= 0:
        raise ValueError("The propeller RPM must be greater than zero")

    delta_max = 10  # Max rudder angle (deg)
    Ddelta_max = 5  # Max rudder rate (deg/s)
    n_max = 160  # Max shaft velocity (RPM)

    # Nondimensional states and inputs
    delta_c = ui[0]
    n_c = ui[1] / 60 * L / U

    u = x[0] / U
    v = x[1] / U
    p = x[6] * L / U
    r = x[2] * L / U
    phi = x[7]
    psi = x[5]
    delta = x[8]
    n = x[9] / 60 * L / U

    # Parameters, hydrodynamic derivatives and main dimensions
    m, mx, my = 0.00792, 0.000238, 0.007049
    Ix, alphay, lx = 0.0000176, 0.05, 0.0313
    ly, Iz = 0.0313, 0.000456
    Jx, Jz = 0.0000034, 0.000419
    g = 9.81
    nabla = 21222
    AR = 33.0376
    Delta, D, GM = 1.8219, 6.533, 0.3 / L
    rho, t = 1025, 0.175

    W = rho * g * nabla / (rho * L**2 * U**2 / 2)

    Xuu, Xvr, Xrr = -0.0004226, -0.00311, 0.00020
    Xphiphi, Xvv = -0.00020, -0.00386

    Kv, Kr, Kp = 0.0003026, -0.000063, -0.0000075
    Kphi, Kvvv, Krrr = -0.000021, 0.002843, -0.0000462
    Kvvr, Kvrr, Kvvphi = -0.000588, 0.0010565, -0.0012012
    Kvphiphi, Krrphi, Krphiphi = -0.0000793, -0.000243, 0.00003569

    Yv, Yr, Yp = -0.0116, 0.00242, 0
    Yphi, Yvvv, Yrrr = -0.000063, -0.109, 0.00177
    Yvvr, Yvrr, Yvvphi = 0.0214, -0.0405, 0.04605
    Yvphiphi, Yrrphi, Yrphiphi = 0.00304, 0.009325, -0.001368

    Nv = -0.0038545
    Nr = -0.00222
    Np = 0.000213
    Nphi = -0.0001424
    Nvvv = 0.001492
    Nrrr = -0.00229
    Nvvr = -0.0424
    Nvrr = 0.00156
    Nvvphi = -0.019058
    Nvphiphi = -0.0053766
    Nrrphi = -0.0038592
    Nrphiphi = 0.0024195

    kk = 0.631
    epsilon = 0.921
    xR = -0.5
    wp = 0.184
    tau = 1.09
    xp = -0.526
    cpv = 0.0
    cpr = 0.0
    ga = 0.088
    cRr = -0.156
    cRrrr = -0.275
    cRrrv = 1.96
    cRX = 0.71
    aH = 0.237
    zR = 0.033
    xH = -0.48

    # Masses and moments of inertia
    m11 = (m + mx)
    m22 = (m + my)
    m32 = -my * ly
    m42 = my * alphay
    m33 = (Ix + Jx)
    m44 = (Iz + Jz)

    # Rudder saturation and dynamics
    if abs(delta_c) >= np.deg2rad(delta_max):
        delta_c = np.sign(delta_c) * np.deg2rad(delta_max)

    delta_dot = delta_c - delta
    if abs(delta_dot) >= np.deg2rad(Ddelta_max):
        delta_dot = np.sign(delta_dot) * np.deg2rad(Ddelta_max)

    # Shaft velocity saturation and dynamics
    n_c *= U / L
    n *= U / L

    if abs(n_c) >= n_max / 60:
        n_c = np.sign(n_c) * n_max / 60

    if n > 0.3:
        Tm = 5.65 / n
    else:
        Tm = 18.83

    n_dot = (1 / Tm) * (n_c - n) * 60

    # Calculation of state derivatives
    vR = ga * v + cRr * r + cRrrr * r**3 + cRrrv * r**2 * v
    uP = u * ((1 - wp) + tau * ((v + xp * r)**2 + cpv * v + cpr * r))
    J = uP * U / (n * D)
    KT = 0.527 - 0.455 * J
    uR = uP * epsilon * np.sqrt(1 + 8 * kk * KT / (np.pi * J**2))
    alphaR = delta + np.arctan(vR / uR)
    FN = -((6.13 * Delta) /
           (Delta + 2.25)) * (AR / L**2) * (uR**2 + vR**2) * np.sin(alphaR)
    T = 2 * rho * D**4 / (U**2 * L**2 * rho) * KT * n * abs(n)

    # Forces and moments
    X = Xuu * u**2 + (
        1 - t
    ) * T + Xvr * v * r + Xvv * v**2 + Xrr * r**2 + Xphiphi * phi**2 + cRX * FN * np.sin(
        delta) + (m + my) * v * r

    Y = Yv * v + Yr * r + Yp * p + Yphi * phi + Yvvv * v**3 + Yrrr * r**3 + Yvvr * v**2 * r + Yvrr * v * r**2 + Yvvphi * v**2 * phi + Yvphiphi * v * phi**2 + Yrrphi * r**2 * phi + Yrphiphi * r * phi**2 + (
        1 + aH) * FN * np.cos(delta) - (m + mx) * u * r

    K = Kv * v + Kr * r + Kp * p + Kphi * phi + Kvvv * v**3 + Krrr * r**3 + Kvvr * v**2 * r + Kvrr * v * r**2 + Kvvphi * v**2 * phi + Kvphiphi * v * phi**2 + Krrphi * r**2 * phi + Krphiphi * r * phi**2 - (
        1 + aH) * zR * FN * np.cos(delta) + mx * lx * u * r - W * GM * phi

    N = Nv * v + Nr * r + Np * p + Nphi * phi + Nvvv * v**3 + Nrrr * r**3 + Nvvr * v**2 * r + Nvrr * v * r**2 + Nvvphi * v**2 * phi + Nvphiphi * v * phi**2 + Nrrphi * r**2 * phi + Nrphiphi * r * phi**2 + (
        xR + aH * xH) * FN * np.cos(delta)

    # Dimensional state derivatives xdot = [ u v r x y psi p phi delta n ]'

    detM = m22 * m33 * m44 - m32**2 * m44 - m42**2 * m33

    xdot = np.array([
        X * (U**2 / L) / m11,
        -((-m33 * m44 * Y + m32 * m44 * K + m42 * m33 * N) / detM) *
        (U**2 / L),
        ((-m42 * m33 * Y + m32 * m42 * K + N * m22 * m33 - N * m32**2) / detM)
        * (U**2 / L**2), (np.cos(psi) * u - np.sin(psi) * np.cos(phi) * v) * U,
        (np.sin(psi) * u + np.cos(psi) * np.cos(phi) * v) * U,
        np.cos(phi) * r * (U / L),
        ((-m32 * m44 * Y + K * m22 * m44 - K * m42**2 + m32 * m42 * N) / detM)
        * (U**2 / L**2), p * (U / L), delta_dot, n_dot
    ]).T

    return xdot


def Lcontainer(x, ui, U0=7.0):

    # Check input dimensions
    if len(x) != 9:
        raise ValueError("x-vector must have dimension 9!")
    if U0 <= 0:
        raise ValueError("The ship must have speed greater than zero")

    # Normalization variables
    L = 175  # Length of ship (m)
    U = np.sqrt(U0**2 + x[1]**2)  # Ship speed (m/s)

    # Rudder limitations
    delta_max = 10  # Max rudder angle (deg)
    Ddelta_max = 5  # Max rudder rate (deg/s)

    # States and inputs
    delta_c = ui[0]
    v, y = x[1], x[4]
    r, psi = x[2], x[5]
    p, phi = x[6], x[7]
    nu = np.array([v, r, p]).T
    eta = np.array([y, psi, phi]).T
    delta = x[8]

    # Linear model matrices
    T = np.diag([1, 1 / L, 1 / L])
    Tinv = np.diag([1, L, L])

    M = np.array([[0.01497, 0.0003525, -0.0002205], [0.0003525, 0.000875, 0],
                  [-0.0002205, 0, 0.0000210]])

    N = np.array([[0.012035, 0.00522, 0], [0.0038436, 0.00243, -0.000213],
                  [-0.000314, 0.0000692, 0.0000075]])

    G = np.array([[0, 0, 0.0000704], [0, 0, 0.0001468], [0, 0, 0.0004966]])

    b = np.array([-0.002578, 0.00126, 0.0000855]).T

    # Rudder saturation and dynamics
    if abs(delta_c) >= np.deg2rad(delta_max):
        delta_c = np.sign(delta_c) * np.deg2rad(delta_max)

    delta_dot = delta_c - delta
    if abs(delta_dot) >= np.deg2rad(Ddelta_max):
        delta_dot = np.sign(delta_dot) * np.deg2rad(Ddelta_max)

    nudot = np.linalg.inv(T @ M @ Tinv) @ ((U**2 / L) * T @ b * delta -
                                           (U / L) * T @ N @ Tinv @ nu -
                                           (U / L)**2 * T @ G @ Tinv @ eta)

    # Dimensional state derivatives
    xdot = np.array([
        0, nudot[0], nudot[1],
        np.cos(psi) * U - np.sin(psi) * np.cos(phi) * v,
        np.sin(psi) * U + np.cos(psi) * np.cos(phi) * v,
        np.cos(phi) * r, nudot[2], p, delta_dot
    ]).T

    return xdot
