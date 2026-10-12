"""Two-state Kalman filter for a time-varying hedge ratio and intercept."""
import math


class KalmanState:
    def __init__(self, initial_beta, initial_alpha, process_noise=1e-4, measurement_noise=1e-3,
                 process_noise_alpha=None, r_forgetting=0.0, r_floor_frac=0.1):
        self.beta = initial_beta
        self.alpha = initial_alpha
        # prior covariance: 50% of each starting value
        self.P00 = (0.5 * abs(initial_beta)) ** 2 if initial_beta != 0 else 1.0
        self.P01 = 0.0
        self.P10 = 0.0
        self.P11 = (0.5 * abs(initial_alpha)) ** 2 if initial_alpha != 0 else 1.0
        self.q = process_noise
        self.q_alpha = process_noise if process_noise_alpha is None else process_noise_alpha
        self.r = measurement_noise
        self.r_forgetting = r_forgetting
        self.r_floor = r_floor_frac * measurement_noise

    def update(self, y, x):
        p00 = self.P00 + self.q
        p01 = self.P01
        p10 = self.P10
        p11 = self.P11 + self.q_alpha

        v = y - (x * self.beta + self.alpha)
        ph0 = p00 * x + p01
        ph1 = p10 * x + p11
        hph = x * ph0 + ph1
        F = hph + self.r
        k0 = ph0 / F
        k1 = ph1 / F

        self.beta += k0 * v
        self.alpha += k1 * v
        self.P00 = (1.0 - k0 * x) * p00 - k0 * p10
        self.P01 = (1.0 - k0 * x) * p01 - k0 * p11
        self.P10 = -k1 * x * p00 + (1.0 - k1) * p10
        self.P11 = -k1 * x * p01 + (1.0 - k1) * p11

        if self.r_forgetting:
            lam = self.r_forgetting
            self.r = max(self.r_floor, (1.0 - lam) * self.r + lam * (v * v - hph))
        return v, F


def log_likelihood(y, x, beta0, alpha0, q, r, skip=20, q_alpha=None, r_forgetting=0.0):
    """Gaussian log-likelihood of the one-step-ahead prediction errors."""
    kf = KalmanState(beta0, alpha0, process_noise=q, measurement_noise=r,
                     process_noise_alpha=q_alpha, r_forgetting=r_forgetting)
    ll = 0.0
    for i in range(len(y)):
        v, F = kf.update(y[i], x[i])
        if i >= skip:
            if F <= 0:
                return -math.inf
            ll -= 0.5 * (math.log(2 * math.pi * F) + v * v / F)
    return ll
