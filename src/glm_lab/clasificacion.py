import numpy as np


def momios(p):
    """Momios (odds): cuántas veces es más probable que pase a que no pase."""
    p = np.asarray(p, dtype=float)
    return p / (1 - p)


def logit(p):
    """Logaritmo de los momios: lleva una probabilidad de (0, 1) a toda la recta."""
    return np.log(momios(p))


def inv_logit(z):
    """Curva logística: lleva cualquier número a una probabilidad entre 0 y 1."""
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def umbral_por_costos(costo_fp, costo_fn):
    """Umbral que minimiza el costo esperado si las probabilidades están calibradas (Elkan, 2001)."""
    return costo_fp / (costo_fp + costo_fn)


def costo_total(y, p, umbral, costo_fp, costo_fn):
    """Costo de decidir 'sí' cuando p ≥ umbral."""
    y, decide = np.asarray(y), np.asarray(p) >= umbral
    return costo_fp * np.sum(decide & (y == 0)) + costo_fn * np.sum(~decide & (y == 1))
