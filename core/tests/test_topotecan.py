import pytest
import jax.numpy as jnp
from identifiability.core.models.pkpd.topotecan import TopotecanModel


base_params = {
    "ko_m": 0.0289, "kc_m": 1.06e-4, "ki": 3.09e-4, "ke": 1.0140,
    "ko_c": 0.026553, "kc_c": 0.18637, "kb": 8.5341e-4, "kd": 4.4489,
    "BT": 28.9, "v1": 100.0, "v2": 0.1
}

def test_params_nonnegativity():
    bad = dict(base_params)
    bad["ki"] = -0.1
    with pytest.raises(ValueError):
        TopotecanModel(bad, initial_dose=1.0)

def test_f_and_g_shapes():
    model = TopotecanModel(base_params, initial_dose=1.0)
    x = model.initial_state
    t0 = 0.0
    f_out = model.f(x, None, t0)
    assert f_out.shape == (model.n,)
    g_out = model.g(x, None, t0)
    assert g_out.shape == (model.p,)

def test_mass_conservation():
    model = TopotecanModel(base_params, initial_dose=10.0)
    t = jnp.linspace(0, 60, 61)
    traj = model.simulate(model.initial_state, t, None)
    total = traj.sum(axis=1)
    assert jnp.allclose(total, total[0], atol=1e-6)

def test_no_cell_influx():
    p = {**base_params, "ki": 0.0}
    model = TopotecanModel(p, initial_dose=5.0)
    t = jnp.linspace(0, 60, 61)
    traj = model.simulate(model.initial_state, t, None)
    # Lc index=2, Ln index=4
    assert jnp.allclose(traj[:, 2], 0.0)
    assert jnp.allclose(traj[:, 4], 0.0)
