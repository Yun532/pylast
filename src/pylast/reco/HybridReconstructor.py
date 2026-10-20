"""Stereo-first union with exact-one-image mono, never a 36-times image sum."""
from __future__ import annotations

from ..helper import ReconstructedEnergy
from .MonoReconstructor import empty_geometry


class HybridReconstructor:
    def __init__(self, subarray, stereo_processor, mono_reconstructor,
                 stereo_geometry_name="HillasReconstructor", stereo_energy_name="EnergyRegressor",
                 geometry_name="HybridDirection", energy_name="HybridEnergy",
                 stereo_particle_name="ParticleClassifier"):
        self.subarray = subarray
        self.stereo_processor = stereo_processor
        self.mono = mono_reconstructor
        self.stereo_geometry_name = stereo_geometry_name
        self.stereo_energy_name = stereo_energy_name
        self.geometry_name = geometry_name
        self.energy_name = energy_name
        self.stereo_particle_name = stereo_particle_name
        if len({geometry_name, energy_name, mono_reconstructor.geometry_name,
                mono_reconstructor.energy_name, stereo_geometry_name, stereo_energy_name}) != 6:
            raise ValueError("Stereo/mono/hybrid ROOT tree names must all be different")
        self.last_result = None

    def __call__(self, event, run_stereo=True):
        if run_stereo:
            self.stereo_processor(event)
        rows = self.mono.event_table(event)
        guard = self.mono.guard_mask(rows)
        candidates = rows.loc[guard, "telescope_id"].astype(int).tolist()
        dl2 = event.ensure_dl2()
        geometry = dl2.geometry.get(self.stereo_geometry_name)
        energy = dl2.energy.get(self.stereo_energy_name)
        particle = getattr(dl2, "particle", {}).get(self.stereo_particle_name)
        sidecar = {"run_id": int(event.run_id), "event_id": int(event.event_id),
                   "n_guard_tel": len(candidates), "guard_telescope_ids": candidates,
                   "branch": "rejected", "accepted": False, "mono": None,
                   "stereo_hadroness": float(particle.hadroness) if particle is not None else float("nan"),
                   "stereo_pid_valid": bool(particle is not None and particle.is_valid),
                   "cross_telescope_transfer": "exploratory"}
        if len(candidates) >= 2 and geometry is not None and geometry.is_valid:
            # Native add_* copies values, leaving original stereo entries unchanged.
            dl2.add_geometry(self.geometry_name, geometry)
            if energy is not None:
                dl2.add_energy(self.energy_name, energy)
            else:
                dl2.add_energy(self.energy_name, ReconstructedEnergy())
            sidecar["branch"] = "stereo"
            sidecar["accepted"] = True
        elif len(candidates) == 1:
            mono = self.mono(event, telescope_id=candidates[0], store=False, pid_mode="exact_one")
            self.mono.store_result(event, mono, self.geometry_name, self.energy_name, selected=True)
            sidecar["mono"] = mono
            sidecar["branch"] = "mono_exact_one"
            sidecar["accepted"] = bool(mono.accepted.iloc[0])
        else:
            # >=2 rejected/failed stereo images are not rescued by selecting a lucky image.
            dl2.add_geometry(self.geometry_name, empty_geometry())
            dl2.add_energy(self.energy_name, ReconstructedEnergy())
        self.last_result = sidecar
        return sidecar
