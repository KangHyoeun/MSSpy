import numpy as np
from math import sin, cos, atan, sqrt, radians, degrees
"""
 [xdot,U] = tanker(x,ui) returns the speed U in m/s (optionally) and the 
 time derivative of the state vector: x = [ u v r x y psi delta n ]'  for
 a the Esso 190,000-dwt tanker L = 304.8 m (Berlekom and Goddhard 1972, 
 Appendix A) where

   u:     surge velocity, must be positive (m/s) - design speed u = 8.23 m/s
   v:     sway velocity (m/s)
   r:     yaw velocity (rad/s)
   x:     position in x-direction (m)
   y:     position in y-direction (m)
   psi:   yaw angle (rad)
   delta: actual rudder angle (rad)
   n:     actual shaft velocity (rpm)  - nominal propeller speed is 80 rpm
 
 The input vector is

   ui = [ delta_c  n_c h ]'  where

   delta_c: commanded rudder angle (rad)
   n_c:     commanded shaft velocity (rpm)
   h:       water depth, must be larger than draft (m) - draft is 18.46 m

 Reference: 
   W. B. Van Berlekom and T. A. and Goddard (1972). Maneuvering of Large
     Tankers, Transaction of SNAME, 80:264-298
"""
def tanker(x, ui):
    if len(x) != 8:
        raise ValueError("x-vector must have dimension 8!")
    if len(ui) != 3:
        raise ValueError("u-vector must have dimension 3!")
    
    # Normalization variables
    L = 304.8  # Length of ship (m)
    g = 9.8    # Acceleration of gravity (m/s^2)
    
    # Dimensional states and inputs
    u = x[0]
    v = x[1]
    r = x[2]
    psi = x[5]
    delta = x[6]
    n = x[7] / 60  # rps
    
    U = sqrt(x[0]**2 + x[1]**2)
    
    delta_c = ui[0]
    n_c = ui[1] / 60  # rps
    h = ui[2]
    
    # Parameters, hydrodynamic derivatives and main dimensions
    delta_max = radians(10)  # Max rudder angle (rad)
    Ddelta_max = radians(2.33)  # Max rudder derivative (rad/s)
    n_max = 80 / 60  # Max shaft speed (rps)
    
    t = 0.22
    Tm = 50
    T = 18.46
    
    cun = 0.605
    cnn = 38.2
    
    Tuu = -0.00695
    Tun = -0.00063
    Tnn = 0.0000354
    
    m11 = 1.050
    m22 = 2.020
    m33 = 0.1232
    
    d11 = 2.020
    d22 = -0.752
    d33 = -0.231
    
    Xuu = -0.0377
    Yvv = -2.400
    Nvr = -0.300
    
    if h < T:
        raise ValueError("The depth must be larger than the draft (18.5 m)")
    
    z = T / (h - T)
    
    if z >= 0.8:
        Yuvz = -0.85 * (1 - 0.8 / z)
    
    # Rudder saturation and dynamics
    if abs(delta_c) >= delta_max:
        delta_c = np.sign(delta_c) * delta_max
    
    delta_dot = delta_c - delta
    
    if abs(delta_dot) >= Ddelta_max:
        delta_dot = np.sign(delta_dot) * Ddelta_max
    
    # Shaft saturation and dynamics
    if abs(n_c) >= n_max:
        n_c = np.sign(n_c) * n_max
    
    n_dot = (1 / Tm) * (n_c - n) * 60
    
    if u <= 0:
        raise ValueError("u must be larger than zero")
    
    beta = atan(v / u)
    
    gT = (1 / L * Tuu * u**2 + Tun * u * n + L * Tnn * abs(n) * n)
    
    c = sqrt(cun * u * n + cnn * n**2)
    
    gX = (1 / L * (Xuu * u**2 + L * d11 * v * r + Xuu * v**2 + L * gT * (1 - t)))
    
    gY = (1 / L * (Yvv * abs(v) * v + L * d22 * u * r))
    
    gLN = Nvr * abs(v) * r + L * d33 * u * r
    
    m11 -= Xuu * z
    m22 -= Yvv * z
    m33 -= Nvr * z
    
    # Dimensional state derivative
    xdot = np.array([
        gX / m11,
        gY / m22,
        gLN / (L**2 * m33),
        cos(psi) * u - sin(psi) * v,
        sin(psi) * u + cos(psi) * v,
        r,
        delta_dot,
        n_dot
    ])
    
    return xdot, U
