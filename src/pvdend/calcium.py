"""Dendritic Ca_LVA distributions, somatic Ca_LVA switch, and TTX.

Dendritic Ca_LVA (with CaDynamics) is placed only on the target path, and
only in the 'long' morphology; the 'short' morphology keeps the original,
Ca-free dendrite. Densities are functions of the absolute path distance d
from the soma centre, with g = cfg.g_ca_mS_cm2 and L = cfg.gradient_span_um:

    uniform     g(d) = g
    increasing  g(d) = 2 g min(d / L, 1)     (g at the midpoint L/2, 2 g at L)

Over a dendrite of near-constant diameter, both profiles carry about the same
total conductance. The dendritic channel (mod/Ca_LVA_dend.mod) is the model's
Ca_LVA with an optional shift of its activation gate (cfg.ca_act_shift_mV).
"""
from __future__ import annotations

import numpy as np
from neuron import h

from .cell import AXONAL, SOMATIC
from .config import Config

TTX_MECHS = ("NaTg", "Nap")
DEND_CA = "Ca_LVA_dend"


def shape(cfg: Config, d: np.ndarray) -> np.ndarray:
    """Profile relative to g (1 = g)."""
    d = np.asarray(d, dtype=float)
    p = cfg.ca_profile
    if p == "none":
        return np.zeros_like(d)
    if p == "uniform":
        return np.ones_like(d)
    if p == "increasing":
        return 2 * np.clip(d / cfg.gradient_span_um, 0, 1)
    raise ValueError(p)


def path_gbar(cfg: Config, distances, areas=None) -> np.ndarray:
    """Ca_LVA gbar (S/cm2) for path segments at `distances`."""
    return shape(cfg, distances) * cfg.g_ca_mS_cm2 * 1e-3


def apply_dendritic_ca(cell, cfg: Config) -> np.ndarray:
    """Set Ca_LVA + CaDynamics on the target path. Returns gbar per path segment.

    Cells are cached and reused across conditions, so the dendrite is first returned to its original
    Ca-free state; channels are then inserted only for the long cell and a profile other than 'none'.
    """
    segs = cell.path_segments()  # (segment, distance from the soma) from the soma to the tip
    for sec in cell.target_path:  # start from the original, Ca-free dendrite
        for mech in (DEND_CA, "CaDynamics"):
            if h.ismembrane(mech, sec=sec):
                sec.uninsert(mech)
    if cell.label != "long" or cfg.ca_profile == "none":
        return np.zeros(len(segs))

    g = path_gbar(cfg, [d for _, d in segs])  # density of each segment from its distance (uniform / increasing)
    for sec in cell.target_path:
        sec.insert(DEND_CA)
        sec.insert("CaDynamics")
    for (seg, _), gi in zip(segs, g):  # per segment: density, activation shift, Ca2+ buffering and removal
        seg.Ca_LVA_dend.gbar = gi
        seg.Ca_LVA_dend.vshift_act = cfg.ca_act_shift_mV
        seg.CaDynamics.gamma = cfg.cadyn_gamma
        seg.CaDynamics.decay = cfg.cadyn_decay_ms
    return g


def apply_somatic_and_ttx(cell, cfg: Config):
    """Somatic Ca_LVA on/off and TTX; always restores the model values first.

    TTX is mimicked by setting the Na+ conductances (NaTg, Nap) to zero in the soma and the axon (the only
    places that have them); otherwise their original densities are restored.
    """
    for lst, params in ((cell.somatic, SOMATIC), (cell.axonal, AXONAL)):
        for sec in lst:
            for mech in TTX_MECHS:
                setattr(sec, f"gbar_{mech}", 0.0 if cfg.ttx else params[f"gbar_{mech}"])
    for sec in cell.somatic:
        sec.gbar_Ca_LVA = SOMATIC["gbar_Ca_LVA"] if cfg.somatic_ca_lva else 0.0


def configure(cell, cfg: Config) -> np.ndarray:
    """Apply all channel switches of `cfg` to `cell`: somatic Ca_LVA and TTX first, then the dendritic
    Ca_LVA profile. Returns the dendritic Ca_LVA density (S/cm²) of every target-path segment."""
    apply_somatic_and_ttx(cell, cfg)
    return apply_dendritic_ca(cell, cfg)
