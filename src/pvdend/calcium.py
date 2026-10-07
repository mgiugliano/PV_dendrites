"""Dendritic Ca_LVA distributions, somatic Ca_LVA switch, and TTX.

Dendritic Ca_LVA (with CaDynamics) is placed only on the target path, and
only in the 'long' morphology; the 'short' morphology keeps the original,
Ca-free dendrite. Densities are functions of the absolute path distance d
from the soma centre:

    uniform     g(d) = g
    hotspot     g(d) = g  for |d - center| <= width/2, else 0
    increasing  g(d) = g * min(d / span, 1)
    decreasing  g(d) = g * max(1 - d / span, 0)

With ca_norm = 'peak', g is cfg.g_uniform (uniform) or cfg.g_peak (others).
With ca_norm = 'total', g is scaled so that every profile carries the same
total conductance on the path, sum(g(d) * area), as a uniform density of
cfg.g_total_equiv.
"""
from __future__ import annotations

import numpy as np
from neuron import h

from .cell import AXONAL, SOMATIC
from .config import Config

TTX_MECHS = ("NaTg", "Nap")


def shape(cfg: Config, d: np.ndarray) -> np.ndarray:
    """Unscaled profile (peak 1) at distances d."""
    d = np.asarray(d, dtype=float)
    p = cfg.ca_profile
    if p == "none":
        return np.zeros_like(d)
    if p == "uniform":
        return np.ones_like(d)
    if p == "hotspot":
        return (np.abs(d - cfg.hotspot_center_um) <= cfg.hotspot_width_um / 2).astype(float)
    if p == "increasing":
        return np.clip(d / cfg.gradient_span_um, 0, 1)
    if p == "decreasing":
        return np.clip(1 - d / cfg.gradient_span_um, 0, 1)
    raise ValueError(p)


def path_gbar(cfg: Config, distances, areas) -> np.ndarray:
    """Ca_LVA gbar (S/cm2) for path segments at `distances` with membrane `areas`."""
    s = shape(cfg, distances)
    if cfg.ca_profile == "none":
        return s
    if cfg.ca_norm == "peak":
        return s * (cfg.g_uniform if cfg.ca_profile == "uniform" else cfg.g_peak)
    areas = np.asarray(areas, dtype=float)
    if np.sum(s * areas) == 0:
        raise ValueError("Ca profile is empty on the target path (check hotspot position)")
    return s * cfg.g_total_equiv * np.sum(areas) / np.sum(s * areas)


def apply_dendritic_ca(cell, cfg: Config) -> np.ndarray:
    """Set Ca_LVA + CaDynamics on the target path. Returns gbar per path segment."""
    segs = cell.path_segments()
    for sec in cell.target_path:  # start from the original, Ca-free dendrite
        for mech in ("Ca_LVA", "CaDynamics"):
            if h.ismembrane(mech, sec=sec):
                sec.uninsert(mech)
    if cell.label != "long" or cfg.ca_profile == "none":
        return np.zeros(len(segs))

    g = path_gbar(cfg, [d for _, d in segs], [seg.area() for seg, _ in segs])
    for sec in cell.target_path:
        sec.insert("Ca_LVA")
        sec.insert("CaDynamics")
    for (seg, _), gi in zip(segs, g):
        seg.Ca_LVA.gbar = gi
        seg.CaDynamics.gamma = cfg.cadyn_gamma
        seg.CaDynamics.decay = cfg.cadyn_decay_ms
    return g


def apply_somatic_and_ttx(cell, cfg: Config):
    """Somatic Ca_LVA on/off and TTX; always restores the model values first."""
    for lst, params in ((cell.somatic, SOMATIC), (cell.axonal, AXONAL)):
        for sec in lst:
            for mech in TTX_MECHS:
                setattr(sec, f"gbar_{mech}", 0.0 if cfg.ttx else params[f"gbar_{mech}"])
    for sec in cell.somatic:
        sec.gbar_Ca_LVA = SOMATIC["gbar_Ca_LVA"] if cfg.somatic_ca_lva else 0.0


def configure(cell, cfg: Config) -> np.ndarray:
    apply_somatic_and_ttx(cell, cfg)
    return apply_dendritic_ca(cell, cfg)
