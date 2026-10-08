"""ipywidgets front end for the notebook (works in Jupyter, VS Code and Colab).

The widgets only build a `Config`; every button calls the same functions as the
command-line scripts (run_sweep, plotting.*), so a figure explored here can be
reproduced exactly with `python scripts/run.py <saved config>`.
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict

import ipywidgets as W
import matplotlib.pyplot as plt
from IPython.display import display

from . import plotting
from ._paths import CONFIG_DIR, FIGURES_DIR
from .config import CA_PROFILES, Config
from .protocols import get_cell, run_sweep

IN_COLAB = "google.colab" in sys.modules
_WIDE = W.Layout(width="330px")
_STYLE = {"description_width": "150px"}


def _download(path):
    """On Colab, offer the file at `path` as a browser download (no-op elsewhere)."""
    if IN_COLAB:
        from google.colab import files
        files.download(str(path))


class Explorer:
    """Widget panel of the explorer notebook: tabs of parameters, action buttons, progress and output."""
    def __init__(self, cfg: Config | None = None):
        """Create one widget per Config field (initialised from `cfg`), the buttons and the layout."""
        cfg = cfg or Config(name="explorer", site_step_um=20)
        self.fig = None
        self.fig_stem = None
        self.last_config = None

        def fl(value, desc, lo, hi, step, fmt=".3g"):
            """A float slider with a fixed layout."""
            return W.FloatSlider(value=value, min=lo, max=hi, step=step, description=desc,
                                 readout_format=fmt, continuous_update=False,
                                 layout=_WIDE, style=_STYLE)

        def num(value, desc, step):
            """A bounded float text box with a fixed layout."""
            return W.BoundedFloatText(value=value, min=0, max=1, step=step, description=desc,
                                      layout=_WIDE, style=_STYLE)

        w = self.w = {}
        # Ca channels
        w["ca_profile"] = W.Dropdown(options=CA_PROFILES, value=cfg.ca_profile,
                                     description="Dendritic Ca_LVA", layout=_WIDE, style=_STYLE)
        w["g_ca_mS_cm2"] = fl(cfg.g_ca_mS_cm2, "Density g (mS/cm²)", 0.1, 5.0, 0.1, ".1f")
        w["ca_act_shift_mV"] = fl(cfg.ca_act_shift_mV, "Activation shift (mV)", -25, 0, 1, ".0f")
        w["gradient_span_um"] = fl(cfg.gradient_span_um, "Gradient span (µm)", 50, 400, 10, ".0f")
        w["mouth_scale"] = fl(cfg.mouth_scale, "Mouth diameter (×)", 1.0, 2.0, 0.1, ".1f")
        # switches
        w["somatic_ca_lva"] = W.Checkbox(value=cfg.somatic_ca_lva, description="Somatic Ca_LVA on")
        w["ttx"] = W.Checkbox(value=cfg.ttx, description="TTX (NaTg, Nap = 0)")
        w["soma_bias_frac"] = fl(cfg.soma_bias_frac, "Somatic bias (× rheobase)", 0.0, 0.98, 0.01, ".2f")
        w["prune_side_branches"] = W.Checkbox(value=cfg.prune_side_branches,
                                              description="Prune side branches")
        w["short"] = W.Checkbox(value="short" in cfg.morphologies, description="Short (100 µm)")
        w["long"] = W.Checkbox(value="long" in cfg.morphologies, description="Long (400 µm)")
        # synapse
        w["syn_weight_nS"] = fl(cfg.syn_weight_uS * 1e3, "Weight (nS)", 0.5, 50, 0.5, ".1f")
        w["syn_tau1_ms"] = fl(cfg.syn_tau1_ms, "τ rise (ms)", 0.05, 5, 0.05, ".2f")
        w["syn_tau2_ms"] = fl(cfg.syn_tau2_ms, "τ decay (ms)", 0.5, 50, 0.5, ".1f")
        w["n_events"] = W.BoundedIntText(value=cfg.n_events, min=1, max=100, description="Events",
                                         layout=_WIDE, style=_STYLE)
        w["freq_hz"] = fl(cfg.freq_hz, "Frequency (Hz)", 1, 200, 1, ".0f")
        # sites
        w["site_step_um"] = W.Dropdown(options=[5.0, 10.0, 20.0, 25.0, 50.0], value=cfg.site_step_um,
                                       description="Site spacing (µm)", layout=_WIDE, style=_STYLE)
        w["site_start_um"] = fl(cfg.site_start_um, "First site (µm)", 0, 100, 5, ".0f")
        w["site_um"] = fl(200, "Single site (µm)", 0, 400, 5, ".0f")
        w["name"] = W.Text(value=cfg.name, description="Run name", layout=_WIDE, style=_STYLE)

        w["ca_profile"].observe(lambda _: self._refresh(), "value")

        b = self.b = {
            "sweep": W.Button(description="Run sweep", button_style="primary", icon="play"),
            "compare": W.Button(description="Compare all profiles", button_style="info", icon="bars"),
            "site": W.Button(description="Run single site", icon="dot-circle-o"),
            "export": W.Button(description="Export figure", icon="download"),
            "save": W.Button(description="Save config", icon="save"),
        }
        b["sweep"].on_click(lambda _: self._guard(self.run_sweep))
        b["compare"].on_click(lambda _: self._guard(self.run_comparison))
        b["site"].on_click(lambda _: self._guard(self.run_site))
        b["export"].on_click(lambda _: self._guard(self.export))
        b["save"].on_click(lambda _: self._guard(self.save_config))

        self.progress = W.IntProgress(min=0, max=1, layout=W.Layout(width="330px"))
        self.status = W.HTML()
        self.out = W.Output()

        tabs = W.Tab(children=[
            W.VBox([w["ca_profile"], w["g_ca_mS_cm2"], w["ca_act_shift_mV"], w["gradient_span_um"]]),
            W.VBox([w["somatic_ca_lva"], w["ttx"], w["soma_bias_frac"], W.HTML("<b>Morphologies</b>"), w["short"],
                    w["long"], w["mouth_scale"], w["prune_side_branches"]]),
            W.VBox([w["syn_weight_nS"], w["syn_tau1_ms"], w["syn_tau2_ms"], w["n_events"], w["freq_hz"]]),
            W.VBox([w["site_start_um"], w["site_step_um"], w["site_um"], w["name"]]),
        ])
        for i, t in enumerate(("Ca channels", "Switches", "Synapse", "Sites / run")):
            tabs.set_title(i, t)
        buttons = W.HBox([b["sweep"], b["compare"], b["site"], b["export"], b["save"]],
                         layout=W.Layout(flex_flow="row wrap"))
        self.ui = W.VBox([tabs, buttons, W.HBox([self.progress, self.status]), self.out])
        self._refresh()

    # --- widgets -> Config -----------------------------------------------------------
    def _refresh(self):
        """Enable only the widgets that matter for the selected Ca_LVA profile."""
        w, prof = self.w, self.w["ca_profile"].value
        for k in ("g_ca_mS_cm2", "ca_act_shift_mV"):
            w[k].disabled = prof == "none"
        w["gradient_span_um"].disabled = prof != "increasing"

    def config(self, **overrides) -> Config:
        """Read the widgets into a Config (any keyword overrides a field)."""
        w = self.w
        fields = {k: w[k].value for k in asdict(Config()) if k in w}
        fields["syn_weight_uS"] = w["syn_weight_nS"].value / 1e3
        fields["morphologies"] = [m for m in ("short", "long") if w[m].value]
        if not fields["morphologies"]:
            raise ValueError("Select at least one morphology.")
        return Config(**{**fields, **overrides})

    # --- actions -----------------------------------------------------------------------
    def _guard(self, fn):
        """Run a button action with all buttons disabled, and show any error in the status line."""
        for btn in self.b.values():
            btn.disabled = True
        try:
            fn()
        except Exception as exc:  # show errors in the panel instead of losing them
            self.status.value = f"<span style='color:#c0392b'>Error: {exc}</span>"
            raise
        finally:
            for btn in self.b.values():
                btn.disabled = False

    def _progress(self, i, n, text):
        """Update the progress bar and the status text during a sweep."""
        self.progress.max, self.progress.value = n, i
        self.status.value = f"&nbsp;{i}/{n} &nbsp; {text}"

    def _show(self, fig, stem, cfg):
        """Display a figure in the output area and remember it for 'Export figure'."""
        self.fig, self.fig_stem, self.last_config = fig, stem, cfg
        with self.out:
            self.out.clear_output(wait=True)
            display(fig)
        plt.close(fig)

    def run_sweep(self):
        """Button 'Run sweep': all sites of the current Config, then the overview figure."""
        cfg = self.config()
        t0 = time.time()
        res = run_sweep(cfg, progress=self._progress)
        self.result = res
        cells = {m: get_cell(m, cfg) for m in cfg.morphologies}
        self._show(plotting.overview_figure(res, cells), f"overview_{cfg.name}", cfg)
        self.status.value = f"&nbsp;done in {time.time() - t0:.0f} s ({len(res.summary)} simulations)"

    def run_comparison(self):
        """Button 'Compare all profiles': one sweep per Ca_LVA profile, then the comparison figure."""
        base = self.config()
        results = {}
        n = len(CA_PROFILES)
        for k, prof in enumerate(CA_PROFILES):
            cfg = base.replace(ca_profile=prof, name=f"{base.name}__{prof}")
            results[prof] = run_sweep(
                cfg, progress=lambda i, m, t, k=k, p=prof: self._progress(k * m + i, n * m, f"{p}: {t}"))
        self.results = results
        self._show(plotting.comparison_figure(results), f"comparison_{base.name}", base)
        self.status.value = "&nbsp;done"

    def run_site(self):
        """Button 'Run single site': one synapse position in detail (traces at the soma, synapse and Ca²⁺)."""
        cfg = self.config(sites_um=[self.w["site_um"].value])
        res = run_sweep(cfg, progress=self._progress)
        if res.summary.empty:
            raise ValueError("The site is beyond the tip of the selected morphologies.")
        self._show(plotting.site_figure(res), f"site_{cfg.name}_{self.w['site_um'].value:g}um", cfg)
        cols = ["morphology", "site_actual_um", "peak_soma_mV", "peak_syn_mV", "attenuation",
                "dcai_syn_uM", "n_spikes_soma"]
        with self.out:
            display(res.summary[cols].round(3))

    def export(self):
        """Button 'Export figure': save the last figure (PDF/SVG/PNG) and its Config in figures/."""
        if self.fig is None:
            raise ValueError("Run something first.")
        paths = plotting.save_figure(self.fig, self.fig_stem)
        self.last_config.to_json(FIGURES_DIR / f"{self.fig_stem}.config.json")
        self.status.value = "&nbsp;saved " + ", ".join(p.name for p in paths) + " in figures/"
        for p in paths:
            if p.suffix == ".pdf":
                _download(p)

    def save_config(self):
        """Button 'Save config': write the current Config to configs/<name>.json (reproducible with scripts/run.py)."""
        cfg = self.config()
        path = CONFIG_DIR / f"{cfg.name}.json"
        cfg.to_json(path)
        self.status.value = (f"&nbsp;saved configs/{path.name} &mdash; reproduce with "
                             f"<code>python scripts/run.py configs/{path.name}</code>")
        _download(path)


def explorer(cfg: Config | None = None) -> Explorer:
    """Create the Explorer, display it in the notebook and return it (to inspect results afterwards)."""
    ex = Explorer(cfg)
    display(ex.ui)
    return ex
