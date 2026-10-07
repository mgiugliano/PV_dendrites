"""Human L2/3 PV+ interneuron (HL23PV) of Yao et al. (2022), in Python.

Faithful translation of models/NeuronTemplate.hoc (init, geom_nseg,
delete_axon_BPO) and models/biophys_HL23PV.hoc of the original release
(Zenodo doi:10.5281/zenodo.5771000). The only additions are optional and
confined to the target dendrite: finer segmentation along it, and removal of
the side branches leaving it.
"""
from __future__ import annotations

from neuron import h

from . import morphology as morph
from ._paths import ensure_mechanisms

# models/biophys_HL23PV.hoc ------------------------------------------------------
ALL = dict(Ra=100, cm=2, e_pas=-83.92924122901199, g_pas=0.00011830111773572024,
           gbar_Ih=2.7671764064314368e-05)
ACTIVE_MECHS = ("NaTg", "Nap", "K_P", "K_T", "Kv3_1", "Im", "SK", "Ca_HVA", "Ca_LVA", "CaDynamics")
SOMATIC = dict(
    ek=-85, ena=50,
    gbar_NaTg=0.49958525078702043, vshiftm_NaTg=0, vshifth_NaTg=10, slopem_NaTg=9, slopeh_NaTg=6,
    gbar_Nap=0.008795461417521086, gbar_K_P=9.606092478937705e-06, gbar_K_T=0.0011701702607527396,
    gbar_Kv3_1=2.9921080101237565, gbar_Im=0.04215865946497755, gbar_SK=3.7265770903193036e-06,
    gbar_Ca_HVA=0.00017953651378188165, gbar_Ca_LVA=0.09250008555398015,
    gamma_CaDynamics=0.0005, decay_CaDynamics=531.0255920416845,
)
AXONAL = dict(
    ek=-85, ena=50,
    gbar_NaTg=0.10914576408883477, vshiftm_NaTg=0, vshifth_NaTg=10, slopem_NaTg=9, slopeh_NaTg=6,
    gbar_Nap=0.001200899579358837, gbar_K_P=0.6854776593761795, gbar_K_T=0.07603372775662909,
    gbar_Kv3_1=2.988867483754507, gbar_Im=0.029587905136596156, gbar_SK=0.5121938998281017,
    gbar_Ca_HVA=0.002961469262723619, gbar_Ca_LVA=5.9457835817342756e-05,
    gamma_CaDynamics=0.0005, decay_CaDynamics=163.03538024059918,
)
VSHIFT_KV3_1 = 0  # GLOBAL in Kv3_1.mod


def _set(sec, params):
    for name, value in params.items():
        setattr(sec, name, value)


class PVCell:
    """One HL23PV cell.

    Parameters
    ----------
    morphology : 'short' | 'long'
        'short' is the original HL23PV.swc; 'long' has the target dendrite
        grown to 400 um (see scripts/make_morphology.py).
    prune_side_branches : remove the subtrees that leave the target path.
    max_seg_len_um : maximum segment length on the target path (None: original rule).
    """

    def __init__(self, morphology="short", prune_side_branches=False, max_seg_len_um=2.0,
                 swc_path=None, tip_xyz=None):
        ensure_mechanisms()
        self.label = morphology
        self.soma, self.dend, self.apic, self.axon = [], [], [], []

        if swc_path is None:
            meta = morph.load_target_meta()[morphology]
            swc_path, tip_xyz = morph.morphology_file(morphology), meta["tip_xyz"]

        self._load_morphology(swc_path)
        self.geom_nseg()

        self.target_path, self.pruned = [], []
        if tip_xyz is not None:
            tip = morph.find_tip_section(self, tip_xyz)
            self.target_path = morph.path_to_soma(self, tip)
            if prune_side_branches:
                self._prune(morph.side_branches(self, self.target_path))
            if max_seg_len_um:
                for sec in self.target_path:
                    sec.nseg = max(sec.nseg, 2 * int(sec.L / max_seg_len_um / 2) + 1)

        self.delete_axon_BPO()
        self.biophys()

    def __str__(self):
        return f"PVCell_{self.label}"

    # --- NeuronTemplate.hoc ---------------------------------------------------------
    def _load_morphology(self, swc_path):
        reader = h.Import3d_SWC_read()
        reader.quiet = 1
        reader.input(str(swc_path))
        h.Import3d_GUI(reader, 0).instantiate(self)
        self._rebuild_lists()

    def _rebuild_lists(self):
        self.all, self.somatic, self.basal = h.SectionList(), h.SectionList(), h.SectionList()
        self.apical, self.axonal = h.SectionList(), h.SectionList()
        for group, lst in ((self.soma, self.somatic), (self.dend, self.basal),
                           (self.apic, self.apical), (self.axon, self.axonal)):
            for sec in group:
                lst.append(sec)
                self.all.append(sec)

    def geom_nseg(self):
        self.soma[0](0.5).area()  # make sure diam reflects 3d points
        for sec in self.all:
            sec.nseg = 1 + 2 * int(sec.L / 40)

    def delete_axon_BPO(self):
        if len(self.axon) == 0:
            d1 = d2 = 1
        elif len(self.axon) == 1:
            d1 = d2 = self.axon[0].diam
        else:
            d1 = d2 = self.axon[0].diam
            for sec in self.axon:
                if h.distance(self.soma[0](0.5), sec(0.5)) > 60:
                    d2 = sec.diam
                    break
        for sec in self.axon:
            h.delete_section(sec=sec)
        self.axon = [h.Section(name=f"axon[{i}]", cell=self) for i in range(2)]
        for sec, d in zip(self.axon, (d1, d2)):
            sec.L, sec.nseg, sec.diam = 30, 1, d
        self.axon[0].connect(self.soma[0](1))
        self.axon[1].connect(self.axon[0](1))
        self._rebuild_lists()

    # --- biophys_HL23PV.hoc ---------------------------------------------------------
    def biophys(self):
        for sec in self.all:
            sec.insert("pas")
            sec.insert("Ih")
            _set(sec, ALL)
        for lst, params in ((self.somatic, SOMATIC), (self.axonal, AXONAL)):
            for sec in lst:
                for mech in ACTIVE_MECHS:
                    sec.insert(mech)
                _set(sec, params)
        h.vshift_Kv3_1 = VSHIFT_KV3_1

    # --- optional modification of the target dendrite ----------------------------------
    def _prune(self, roots):
        doomed = [s for r in roots for s in morph.subtree(r)]
        self.pruned = [s.name() for s in doomed]
        keep = [s for s in self.dend if all(s != d for d in doomed)]  # before deleting
        for sec in doomed:
            h.delete_section(sec=sec)
        self.dend = keep
        self._rebuild_lists()

    # --- convenience ---------------------------------------------------------------
    def distance(self, seg) -> float:
        return morph.soma_distance(self, seg)

    @property
    def tip_distance(self) -> float:
        return self.distance(self.target_path[-1](1))

    def path_segments(self):
        """(segment, distance) for every segment of the target path, soma to tip."""
        return [(seg, self.distance(seg)) for sec in self.target_path for seg in sec]
