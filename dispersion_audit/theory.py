"""Linear water-wave theory used as the reference ("truth") of the audit."""
import numpy as np

G = 9.81


def wavenumber(omega, h, g=G):
    """Exact linear wavenumber k from omega^2 = g k tanh(k h) (Newton iteration)."""
    omega = np.asarray(omega, float)
    k = np.maximum(omega ** 2 / g, omega / np.sqrt(g * h))       # deep / shallow start
    for _ in range(50):
        th = np.tanh(k * h)
        f = g * k * th - omega ** 2
        df = g * th + g * k * h * (1 - th ** 2)
        step = f / df
        k = k - step
        if np.all(np.abs(step) < 1e-14 * k):
            break
    return k


def omega(k, h, g=G):
    return np.sqrt(g * k * np.tanh(k * h))


def lambda_phys(h):
    """Quartic coefficient of the long-wave expansion: omega^2 = g h k^2 (1 + Lambda k^2 + ...)."""
    return -h ** 2 / 3


def y_of(omega_, k, h, g=G):
    """y = omega^2 / (g h k^2) - 1, the normalized dispersion deviation."""
    return omega_ ** 2 / (g * h * k ** 2) - 1
