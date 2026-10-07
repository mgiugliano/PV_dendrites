import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuron import h  # noqa: E402

from pvdend import Config, calcium, get_cell, run_sweep  # noqa: E402
from pvdend.cell import PVCell  # noqa: E402
from pvdend.morphology import side_branches  # noqa: E402

BIOPHYS_HOC = ROOT / "original_model" / "biophys_HL23PV.hoc"


def hoc_blocks():
    """Parse {forsec <list>: {name: value}} from the original biophys file."""
    text = BIOPHYS_HOC.read_text()
    blocks = {}
    for lst, body in re.findall(r"forsec \$o1\.(\w+) \{(.*?)\n\t\}", text, re.S):
        blocks[lst] = {k: float(v) for k, v in re.findall(r"(\w+) = ([-\d.e]+)", body)}
    return blocks


@pytest.fixture(scope="module")
def short():
    return PVCell("short")


def test_biophysics_match_original_hoc(short):
    blocks = hoc_blocks()
    assert set(blocks) == {"all", "somatic", "axonal"}
    for lst_name, params in blocks.items():
        for sec in getattr(short, lst_name):
            for name, value in params.items():
                if name.startswith("vshift_"):
                    assert getattr(h, f"{name}") == value
                else:
                    assert getattr(sec, name) == pytest.approx(value, rel=1e-12), (sec, name)


def test_no_active_channels_in_dendrites(short):
    for sec in short.basal:
        for mech in ("NaTg", "Ca_LVA", "CaDynamics", "K_T"):
            assert not h.ismembrane(mech, sec=sec)


def test_target_path_lengths():
    s, l = PVCell("short"), PVCell("long")
    assert s.tip_distance == pytest.approx(100.808, abs=1e-3)
    assert l.tip_distance == pytest.approx(400.0, abs=1e-3)
    assert [x.name().split(".")[-1] for x in s.target_path] == ["dend[27]", "dend[29]", "dend[30]"]
    # the long path starts with the same three sections, then the grown part
    assert [x.name().split(".")[-1] for x in l.target_path[:3]] == ["dend[27]", "dend[29]", "dend[30]"]
    # everything off the target path is identical
    assert len(list(l.basal)) == len(list(s.basal)) + len(l.target_path) - 3


def test_target_path_is_unbranched_after_pruning():
    c = PVCell("long", prune_side_branches=True)
    assert side_branches(c, c.target_path) == []
    assert c.tip_distance == pytest.approx(400.0, abs=1e-3)
    assert len(c.pruned) > 0


@pytest.mark.parametrize("profile", ["uniform", "hotspot", "increasing", "decreasing"])
def test_ca_only_on_long_target_path(profile):
    cfg = Config(ca_profile=profile)
    for morph in ("short", "long"):
        cell = get_cell(morph, cfg)
        g = calcium.configure(cell, cfg)
        on_path = set(cell.target_path)
        for sec in cell.basal:
            has = h.ismembrane("Ca_LVA", sec=sec)
            assert has == (morph == "long" and sec in on_path), (morph, sec)
            assert h.ismembrane("CaDynamics", sec=sec) == has
        assert (g > 0).any() == (morph == "long")


def test_profiles_shape_and_total_normalisation():
    cfg = Config()
    cell = get_cell("long", cfg)
    d = np.array([x for _, x in cell.path_segments()])
    a = np.array([seg.area() for seg, _ in cell.path_segments()])
    totals = []
    for prof in ("uniform", "hotspot", "increasing", "decreasing"):
        g = calcium.path_gbar(cfg.replace(ca_profile=prof, ca_norm="total"), d, a)
        totals.append(np.sum(g * a))
    assert np.allclose(totals, cfg.g_total_equiv * a.sum(), rtol=1e-12)

    g = calcium.path_gbar(cfg.replace(ca_profile="increasing"), d, a)
    assert g[0] < g[-1] and g.max() == pytest.approx(cfg.g_peak, rel=5e-3)  # last segment centre ~1 um before the tip
    g = calcium.path_gbar(cfg.replace(ca_profile="hotspot"), d, a)
    inside = np.abs(d - cfg.hotspot_center_um) <= cfg.hotspot_width_um / 2
    assert np.all(g[inside] == cfg.g_peak) and np.all(g[~inside] == 0)


def test_switches_restore_model_values():
    cfg = Config()
    cell = get_cell("short", cfg)
    calcium.configure(cell, cfg.replace(ttx=True, somatic_ca_lva=False))
    assert cell.soma[0](0.5).NaTg.gbar == 0 and cell.soma[0](0.5).Ca_LVA.gbar == 0
    assert cell.axon[0](0.5).Nap.gbar == 0
    calcium.configure(cell, cfg)
    assert cell.soma[0](0.5).NaTg.gbar == pytest.approx(0.49958525078702043)
    assert cell.soma[0](0.5).Ca_LVA.gbar == pytest.approx(0.09250008555398015)


def test_sweep_runs_and_is_at_rest():
    cfg = Config(sites_um=[50, 100], n_events=3, freq_hz=50)
    res = run_sweep(cfg)
    assert len(res.summary) == 4
    for tr in res.traces.values():
        pre = tr["v_soma"][tr["t"] < cfg.onset_ms]
        assert np.ptp(pre) < 1e-6
    # identical passive dendrites near the soma: short and long respond similarly
    s = res.summary.set_index(["morphology", "site_um"])["peak_soma_mV"]
    assert s[("short", 50)] > s[("long", 50)] > 0
