"""Pruebas con valores calculados a mano (las cuentas están en cada comentario)."""
import numpy as np
import pandas as pd
import pytest

from glm_lab import (momios, logit, inv_logit, umbral_por_costos, costo_total, brier, tabla_calibracion,
                     dispersion_pearson, prob_cero_poisson, prob_cero_nb, cuantil_nb, cargar_tequila,
                     separar_por_fecha)


# ---------- momios y logit
def test_momios_de_una_probabilidad():
    assert momios(0.2) == pytest.approx(0.25)            # 0.2 / 0.8


def test_logit_e_inversa():
    assert logit(0.5) == pytest.approx(0.0)
    assert inv_logit(np.log(3)) == pytest.approx(0.75)   # 3 / (1 + 3)
    assert inv_logit(logit(0.137)) == pytest.approx(0.137)


# ---------- decisiones
def test_umbral_por_costos_cuando_un_fn_cuesta_cinco_fp():
    assert umbral_por_costos(costo_fp=1, costo_fn=5) == pytest.approx(1 / 6)   # 1 / (1 + 5)


def test_costo_total_cuenta_falsos_positivos_y_negativos():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.4, 0.1, 0.05])
    # con umbral 0.3 se predice [1, 1, 0, 0]: un FP (fila 2) y un FN (fila 3) -> 1·1 + 1·5
    assert costo_total(y, p, umbral=0.3, costo_fp=1, costo_fn=5) == pytest.approx(6)


# ---------- probabilidades bien calibradas
def test_brier():
    assert brier([1, 0], [0.8, 0.4]) == pytest.approx(0.10)   # (0.2² + 0.4²) / 2


def test_tabla_calibracion_con_dos_grupos():
    t = tabla_calibracion([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9], n_grupos=2)
    assert list(t["n"]) == [2, 2]
    assert list(t["p_media"].round(3)) == [0.15, 0.8]
    assert list(t["frecuencia"]) == [0.0, 1.0]


# ---------- conteos
def test_dispersion_pearson():
    # ((0-2)²/2 + 0 + (4-2)²/2) / (3 - 1) = 4 / 2
    assert dispersion_pearson([0, 2, 4], [2, 2, 2], n_parametros=1) == pytest.approx(2.0)


def test_prob_cero_poisson():
    assert prob_cero_poisson(2.0) == pytest.approx(np.exp(-2))


def test_prob_cero_binomial_negativa():
    # (1 / (1 + α·μ))^(1/α) = (1/2)² con μ = 2, α = 0.5
    assert prob_cero_nb(2.0, 0.5) == pytest.approx(0.25)


def test_cuantil_binomial_negativa():
    # r = 2, p = 0.5: P(0) = 0.25, P(1) = 0.25, P(2) = 0.1875 -> acumuladas 0.25, 0.50, 0.6875
    assert cuantil_nb(2.0, 0.5, 0.5) == 1
    assert cuantil_nb(2.0, 0.5, 0.6) == 2


# ---------- datos
def test_cargar_tequila_y_separar_por_fecha(tmp_path):
    f = tmp_path / "t.csv"
    pd.DataFrame({"semana": ["2015-08-24", "2015-08-31", "2015-09-07"], "tienda": [1, 1, 1],
                  "pidio": [0, 1, 1], "botellas": [0, 12, 3], "mes": [8, 8, 9]}).to_csv(f, index=False)
    d = cargar_tequila(f)
    assert d["semana"].dtype.kind == "M"
    assert d["mes"].dtype.name == "category"
    entrena, prueba = separar_por_fecha(d, "2015-09-01")
    assert len(entrena) == 2 and len(prueba) == 1
    assert entrena["semana"].max() < prueba["semana"].min()
