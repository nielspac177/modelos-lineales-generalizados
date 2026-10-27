import numpy as np
from scipy import stats


def dispersion_pearson(y, mu, n_parametros):
    """Suma de residuos de Pearson al cuadrado entre grados de libertad. Cerca de 1 si Poisson es razonable."""
    y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
    return np.sum((y - mu) ** 2 / mu) / (len(y) - n_parametros)


def prob_cero_poisson(mu):
    return np.exp(-np.asarray(mu, dtype=float))


def _nb(mu, alpha):
    """Binomial negativa con media mu y varianza mu + alpha·mu² (la de statsmodels) en la forma de scipy."""
    r = 1 / alpha
    return stats.nbinom(r, r / (r + np.asarray(mu, dtype=float)))


def prob_cero_nb(mu, alpha):
    return _nb(mu, alpha).pmf(0)


def cuantil_nb(mu, alpha, q):
    """Menor conteo k con P(Y ≤ k) ≥ q: un máximo de inventario con nivel de servicio q."""
    return _nb(mu, alpha).ppf(q)
