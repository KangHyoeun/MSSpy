import numpy as np


def mariner(x, ui, U0=7.7175):
    # 입력값 검증
    if len(x) != 7:
        raise ValueError("x 벡터는 길이가 7이어야 합니다!")
    if not isinstance(ui, (int, float)):
        raise ValueError("ui는 스칼라 값이어야 합니다!")

    # 정규화 변수
    L = 160.93
    U = np.sqrt((U0 + x[0])**2 + x[1]**2)

    # 무차원 상태 및 입력
    delta_c = -ui  # delta_c = -ui, 양의 delta_c가 양의 r로 연결됨
    u = x[0] / U
    v = x[1] / U
    r = x[2] * L / U
    psi = x[5]
    delta = x[6]

    # 매개변수, 유체역학적 미분 계수 및 주요 치수
    delta_max = 40  # 최대 러더 각도 (deg)
    Ddelta_max = 5  # 최대 러더 변화율 (deg/s)
    m = 798e-5
    Iz = 39.2e-5
    xG = -0.023

    Xudot, Yvdot, Nvdot = -42e-5, -748e-5, 4.646e-5
    Xu, Yrdot, Nrdot = -184e-5, -9.354e-5, -43.8e-5
    Xuu, Yv, Nv = -110e-5, -1160e-5, -264e-5
    Xuuu, Yr, Nr = -215e-5, -499e-5, -166e-5
    Xvv, Yvvv, Nvvv = -899e-5, -8078e-5, 1636e-5
    Xrr, Yvvr, Nvvr = 18e-5, 15356e-5, -5483e-5
    Xdd, Yvu, Nvu = -95e-5, -1160e-5, -264e-5
    Xudd, Yru, Nru = -190e-5, -499e-5, -166e-5
    Xrv, Yd, Nd = 798e-5, 278e-5, -139e-5
    Xvd, Yddd, Nddd = 93e-5, -90e-5, 45e-5
    Xuvd, Yud, Nud = 93e-5, 556e-5, -278e-5
    Yuud, Nuud = 278e-5, -139e-5
    Yvdd, Nvdd = -4e-5, 13e-5
    Yvvd, Nvvd = 1190e-5, -489e-5
    Y0, N0 = -4e-5, 3e-5
    Y0u, N0u = -8e-5, 6e-5
    Y0uu, N0uu = -4e-5, 3e-5

    # 질량 및 관성 모멘트 계산
    m11 = m - Xudot
    m22 = m - Yvdot
    m23 = m * xG - Yrdot
    m32 = m * xG - Nvdot
    m33 = Iz - Nrdot

    # 러더 포화 및 동역학 처리
    if abs(delta_c) >= np.deg2rad(delta_max):
        delta_c = np.sign(delta_c) * np.deg2rad(delta_max)

    delta_dot = delta_c - delta

    if abs(delta_dot) >= np.deg2rad(Ddelta_max):
        delta_dot = np.sign(delta_dot) * np.deg2rad(Ddelta_max)

    # 힘 및 모멘트 계산
    X = Xu * u + Xuu * u**2 + Xuuu * u**3 + Xvv * v**2 + Xrr * r**2 + Xrv * r * v + Xdd * delta**2 + Xudd * u * delta**2 + Xvd * v * delta + Xuvd * u * v * delta

    Y = Yv * v + Yr * r + Yvvv * v**3 + Yvvr * v**2 * r + Yvu * v * u + Yru * r * u + Yd * delta + Yddd * delta**3 + Yud * u * delta + Yuud * u**2 * delta + Yvdd * v * delta**2 + Yvvd * v**2 * delta + (
        Y0 + Y0u * u + Y0uu * u**2)

    N = Nv * v + Nr * r + Nvvv * v**3 + Nvvr * v**2 * r + Nvu * v * u + Nru * r * u + Nd * delta + Nddd * delta**3 + Nud * u * delta + Nuud * u**2 * delta + Nvdd * v * delta**2 + Nvvd * v**2 * delta + (
        N0 + N0u * u + N0uu * u**2)

    # 상태 미분 계산 (차원 복원)
    detM22 = m22 * m33 - m23 * m32

    xdot = np.array([
        X * (U**2 / L) / m11, -(-m33 * Y + m23 * N) * (U**2 / L) / detM22,
        (-m32 * Y + m22 * N) * (U**2 / L**2) / detM22,
        (np.cos(psi) * (U0 / U + u) - np.sin(psi) * v) * U,
        (np.sin(psi) * (U0 / U + u) + np.cos(psi) * v) * U, r * (U / L),
        delta_dot
    ]).T

    return xdot


# 사용 예시:
# x_initial_state는 초기 상태 벡터입니다.
# ui는 명령된 러더 각도입니다.
# U0는 기본 속도입니다.
