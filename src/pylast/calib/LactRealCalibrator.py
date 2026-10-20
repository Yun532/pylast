"""Measured LACT gain selection and LED area calibration."""

import json

from ..helper import LactRealCalibrator as _NativeLactRealCalibrator


class LactRealCalibrator:
    def __init__(self, calibration_file=None, *, config_str=None):
        config = config_str if isinstance(config_str, str) else json.dumps(config_str or {})
        self._calibrator = _NativeLactRealCalibrator(
            "" if calibration_file is None else str(calibration_file), config)

    def __call__(self, event):
        self._calibrator(event)

    def load_calibration_file(self, filename):
        self._calibrator.load_calibration_file(str(filename))

    @property
    def has_calibration(self):
        return self._calibrator.has_calibration


LACT1Calibrator = LactRealCalibrator
