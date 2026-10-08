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


@pytest.mark.parametrize("profile", ["uniform", "increasing"])
def test_ca_only_on_long_target_path(profile):
    cfg = Config(ca_profile=profile)
    for morph in ("short", "long"):
        cell = get_cell(morph, cfg)
        g = calcium.configure(cell, cfg)
        on_path = set(cell.target_path)
        for sec in cell.basal:
            has = h.ismembrane("Ca_LVA_dend", sec=sec)
            assert has == (morph == "long" and sec in on_path), (morph, sec)
            assert h.ismembrane("CaDynamics", sec=sec) == has
        assert (g > 0).any() == (morph == "long")


def test_profiles_shape():
    cfg = Config(g_ca_mS_cm2=2.5)
    cell = get_cell("long", cfg)
    d = np.array([x for _, x in cell.path_segments()])
    a = np.array([seg.area() for seg, _ in cell.path_segments()])
    gu = calcium.path_gbar(cfg.replace(ca_profile="uniform"), d)
    gi = calcium.path_gbar(cfg.replace(ca_profile="increasing"), d)
    assert np.allclose(gu, 2.5e-3)
    mid = int(np.argmin(np.abs(d - cfg.gradient_span_um / 2)))
    assert gi[mid] == pytest.approx(2.5e-3, rel=0.01) and gi[0] < 1e-4
    assert np.sum(gi * a) == pytest.approx(np.sum(gu * a), rel=0.03)  # about the same total conductance


def test_wider_mouth_only_changes_the_proximal_diameter():
    base, wide = PVCell("long"), PVCell("long", mouth_scale=1.5)
    for sb, sw in zip(base.target_path, wide.target_path):
        for i in range(sb.n3d()):
            d = base.distance(sb(0)) + sb.arc3d(i)
            expected = sb.diam3d(i) * (1 + 0.5 * max(0.0, 1 - d / 13.0))
            assert sw.diam3d(i) == pytest.approx(expected, rel=1e-4)
    assert wide.target_path[0].diam3d(0) == pytest.approx(1.5 * base.target_path[0].diam3d(0), rel=1e-4)


def test_shifted_activation_lowers_the_event_threshold_and_zero_shift_is_identical():
    from pvdend.protocols import run_site
    out = {}
    for shift in (0.0, -15.0):
        cfg = Config(ca_profile="increasing", g_ca_mS_cm2=1.0, ca_act_shift_mV=shift, morphologies=["long"])
        cell = get_cell("long", cfg)
        calcium.configure(cell, cfg)
        out[shift] = run_site(cell, cfg, 300.0)["v_syn"].max()
    assert out[-15.0] > out[0.0] + 5


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


@pytest.mark.parametrize("shift", [0.0, -15.0])
def test_ca_lva_rates_copy_matches_mod(shift):
    """mechanism.ca_lva_rates must reproduce the steady states of mod/Ca_LVA_dend.mod."""
    from pvdend.mechanism import ca_lva_rates
    sec = h.Section(name=f"calva_probe_{abs(shift):g}")
    sec.insert("Ca_LVA_dend")
    sec(0.5).Ca_LVA_dend.vshift_act = shift
    h.celsius = 34.0
    for v in (-90.0, -60.0, -40.0, -20.0):
        h.finitialize(v)
        m_inf, _, h_inf, _ = ca_lva_rates(v, 34.0, shift)
        assert sec(0.5).m_Ca_LVA_dend == pytest.approx(float(m_inf), rel=1e-9)
        assert sec(0.5).h_Ca_LVA_dend == pytest.approx(float(h_inf), rel=1e-9)


def test_dendritic_copy_with_zero_shift_matches_original_channel():
    sec = h.Section(name="calva_pair")
    sec.insert("Ca_LVA")
    sec.insert("Ca_LVA_dend")
    h.celsius = 34.0
    for v in (-80.0, -50.0, -30.0):
        h.finitialize(v)
        assert sec(0.5).m_Ca_LVA_dend == pytest.approx(sec(0.5).m_Ca_LVA, rel=1e-12)
        assert sec(0.5).h_Ca_LVA_dend == pytest.approx(sec(0.5).h_Ca_LVA, rel=1e-12)
