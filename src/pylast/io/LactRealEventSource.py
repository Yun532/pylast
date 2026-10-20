"""Measured LACT ROOT input with the same event API as other pylast sources."""

import json
from pathlib import Path

import numpy as np

from ..helper import LactRealEventSource as _NativeLactRealEventSource


class LactRealEventSource:
    """Read raw ``waveforms`` or preprocessed ``eventShow`` data.

    ``mapping_file`` is the measured camera mapping CSV. ``analysis_file``
    optionally pairs raw waveforms with official eventShow signal summaries,
    retaining the original preprocessing (including offline ML pulse labels).
    Without it, raw data use the server's independent traditional L1 extractor.
    Pointing arguments are degrees. Waveform sample timing is unknown unless
    ``sampling_interval_ns`` is supplied in ``config_str``; Rabbit's 8 ns ticks
    are a separate clock and are never substituted for that sampling interval.
    """

    def __init__(self, filename, mapping_file, max_events=-1, *, config_str=None,
                 analysis_file=None, pointing_file=None,
                 pointing_alt=None, pointing_az=None):
        config = json.loads(config_str or "{}") if isinstance(config_str, str) else dict(config_str or {})
        if analysis_file is not None:
            config["analysis_file"] = str(analysis_file)
        if (pointing_alt is None) != (pointing_az is None):
            raise ValueError("pointing_alt and pointing_az must be supplied together (degrees)")
        if pointing_file is not None and pointing_alt is not None:
            raise ValueError("Supply either pointing_file or explicit pointing_alt/pointing_az")
        self._source = _NativeLactRealEventSource(str(filename), str(mapping_file),
                                                max_events, json.dumps(config))
        self._sampling_interval_ns = float(config.get("sampling_interval_ns", 0.0))
        self._pointing = None if pointing_alt is None else (float(pointing_alt), float(pointing_az))
        self._pointing_table = self._read_pointing(pointing_file) if pointing_file else None
        self.preprocessing = ("paired eventShow" if config.get("analysis_file") else
                              "traditional L1" if self.input_format == "waveforms" else "eventShow")

    def __getattr__(self, name):
        return getattr(self._source, name)

    def __len__(self):
        return len(self._source)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        event = self._source[index]
        pointing = self._pointing
        if self._pointing_table is not None:
            times, azimuth, altitude = self._pointing_table
            if not times[0] <= event.time_mjd <= times[-1]:
                raise ValueError("Event time is outside pointing log coverage")
            az = float(np.rad2deg(np.interp(event.time_mjd, times, azimuth)) % 360.0)
            alt = float(np.interp(event.time_mjd, times, altitude))
            pointing = (alt, az)
        if pointing is not None:
            self._source.set_pointing(event, *pointing)
        return event

    @staticmethod
    def _read_pointing(filename):
        import pandas as pd

        table = pd.read_csv(Path(filename), sep=r"\s+|,", engine="python")
        table.columns = table.columns.str.strip()

        def column(*names):
            for name in names:
                if name in table:
                    return table[name].to_numpy(dtype=float)
            raise ValueError(f"Missing pointing column; expected one of {names}")

        time = column("MJD", "mjd")
        azimuth = column("方位当前角度", "az", "az_deg", "Azimuth", "azimuth")
        altitude = column("俯仰当前角度", "el", "el_deg", "Elevation", "elevation", "alt", "alt_deg")
        if len(time) < 2 or not np.all(np.isfinite([time, azimuth, altitude])):
            raise ValueError("Pointing log requires at least two finite time/angle rows")
        order = np.argsort(time, kind="stable")
        time, azimuth, altitude = time[order], azimuth[order], altitude[order]
        if np.any(np.diff(time) <= 0) or np.any(np.abs(altitude) > 90):
            raise ValueError("Pointing times must be unique and altitudes within [-90,90]")
        return time, np.unwrap(np.deg2rad(azimuth)), altitude

    def get_triggered_tels(self, event_or_id):
        return (self.telescope_id,)

    def get_waveform_timing(self, event_or_id, telescope_id):
        if int(telescope_id) != self.telescope_id or self._sampling_interval_ns <= 0:
            return {}
        event = event_or_id if hasattr(event_or_id, "c1") else None
        if event is None or event.c1 is None:
            return {}
        count = event.c1.tels[self.telescope_id].waveform[0].shape[1]
        axis = np.arange(count, dtype=float) * self._sampling_interval_ns
        return {"time_centers_ns": axis, "time_reference": "waveform start"}


# Original prototype naming remains usable by existing notebooks.
LACT1EventSource = LactRealEventSource
