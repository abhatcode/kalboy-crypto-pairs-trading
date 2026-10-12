import numpy as np

from kalboy.kalman import KalmanState


def matrix_kalman(y, x, beta0, alpha0, q, r):
    """Textbook matrix form of the filter, as a reference."""
    theta = np.array([beta0, alpha0], dtype=float)
    P = np.diag([(0.5 * abs(beta0)) ** 2, (0.5 * abs(alpha0)) ** 2])
    Q = np.eye(2) * q
    out = []
    for yt, xt in zip(y, x):
        P = P + Q
        H = np.array([xt, 1.0])
        v = yt - H @ theta
        F = H @ P @ H + r
        K = P @ H / F
        theta = theta + K * v
        P = (np.eye(2) - np.outer(K, H)) @ P
        out.append((theta[0], theta[1], v, F))
    return np.array(out)


def test_scalar_filter_matches_matrix_form():
    rng = np.random.default_rng(0)
    x = 1 + np.cumsum(rng.normal(0, 0.01, 2000))
    y = 0.7 * x + 0.3 + rng.normal(0, 0.01, 2000)
    ref = matrix_kalman(y, x, 0.6, 0.2, 1e-5, 1e-4)
    kf = KalmanState(0.6, 0.2, 1e-5, 1e-4)
    got = []
    for yt, xt in zip(y, x):
        v, F = kf.update(yt, xt)
        got.append((kf.beta, kf.alpha, v, F))
    np.testing.assert_allclose(np.array(got), ref, rtol=1e-9, atol=1e-12)


def test_filter_recovers_true_hedge_ratio():
    rng = np.random.default_rng(1)
    x = 1 + np.cumsum(rng.normal(0, 0.01, 5000))
    y = 0.7 * x + 0.3 + rng.normal(0, 0.002, 5000)
    kf = KalmanState(1.0, 0.0, 1e-7, 4e-6)
    for yt, xt in zip(y, x):
        kf.update(yt, xt)
    assert abs(kf.beta - 0.7) < 0.02


def matrix_kalman_general(y, x, beta0, alpha0, q_beta, q_alpha, r0, lam=0.0, floor_frac=0.1):
    theta = np.array([beta0, alpha0], dtype=float)
    P = np.diag([(0.5 * abs(beta0)) ** 2, (0.5 * abs(alpha0)) ** 2])
    Q = np.diag([q_beta, q_alpha])
    r = r0
    out = []
    for yt, xt in zip(y, x):
        P = P + Q
        H = np.array([xt, 1.0])
        v = yt - H @ theta
        hph = H @ P @ H
        F = hph + r
        K = P @ H / F
        theta = theta + K * v
        P = (np.eye(2) - np.outer(K, H)) @ P
        if lam:
            r = max(floor_frac * r0, (1 - lam) * r + lam * (v * v - hph))
        out.append((theta[0], theta[1], v, F))
    return np.array(out)


def test_separate_q_and_adaptive_r_match_matrix_form():
    rng = np.random.default_rng(3)
    x = 1 + np.cumsum(rng.normal(0, 0.01, 3000))
    noise = rng.normal(0, 0.01, 3000) * np.where(np.arange(3000) % 500 < 50, 5.0, 1.0)
    y = 0.7 * x + 0.3 + noise
    ref = matrix_kalman_general(y, x, 0.6, 0.2, 1e-6, 1e-4, 1e-4, lam=0.02)
    kf = KalmanState(0.6, 0.2, 1e-6, 1e-4, process_noise_alpha=1e-4, r_forgetting=0.02)
    got = []
    for yt, xt in zip(y, x):
        v, F = kf.update(yt, xt)
        got.append((kf.beta, kf.alpha, v, F))
    np.testing.assert_allclose(np.array(got), ref, rtol=1e-9, atol=1e-12)


def test_defaults_reproduce_the_original_filter():
    rng = np.random.default_rng(4)
    x = 1 + np.cumsum(rng.normal(0, 0.01, 1000))
    y = 0.7 * x + 0.3 + rng.normal(0, 0.01, 1000)
    a, b = KalmanState(0.6, 0.2, 1e-5, 1e-4), KalmanState(0.6, 0.2, 1e-5, 1e-4, process_noise_alpha=1e-5)
    for yt, xt in zip(y, x):
        assert a.update(yt, xt) == b.update(yt, xt)


def test_adaptive_r_rises_in_a_volatile_burst_and_falls_back():
    rng = np.random.default_rng(5)
    x = np.ones(1500)
    noise = rng.normal(0, 0.01, 1500)
    noise[500:600] *= 10
    kf = KalmanState(1.0, 0.0, 1e-8, 1e-4, r_forgetting=0.05)
    rs = []
    for t in range(1500):
        kf.update(1.0 + noise[t], x[t])
        rs.append(kf.r)
    assert rs[599] > 20 * rs[499]
    assert rs[1499] < 2 * rs[499]


def test_mle_chooses_static_r_on_homoskedastic_noise():
    from kalboy.calibration import fit_noise
    rng = np.random.default_rng(6)
    x = 1 + np.cumsum(rng.normal(0, 0.01, 2000))
    y = 0.7 * x + 0.3 + rng.normal(0, 0.01, 2000)
    assert fit_noise(y, x, 0.7, 0.3, adaptive_r=True)['r_forgetting'] == 0.0
    noisy = y + rng.normal(0, 0.01, 2000) * np.where(np.arange(2000) % 400 < 80, 6.0, 0.0)
    assert fit_noise(noisy, x, 0.7, 0.3, adaptive_r=True)['r_forgetting'] > 0.0
