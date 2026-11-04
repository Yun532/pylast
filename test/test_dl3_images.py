"""Image export checks without native pylast, model assets, or Astropy."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "pylast_dl3_images_tested", Path(__file__).resolve().parents[1] / "src/pylast/io/dl3_images.py")
images = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(images)


def camera(size=200):
    return NS(image=np.array([100., 100.]), mask=np.array([True, True]),
              image_parameters=NS(
                  hillas=NS(intensity=size, length=.02, width=.01, x=.03, y=-.04,
                            r=.05, psi=.2, phi=-.9, skewness=-.3, kurtosis=2.7),
                  leakage=NS(pixels_width_1=.1, pixels_width_2=.2,
                             intensity_width_1=.05, intensity_width_2=.15),
                  concentration=NS(concentration_cog=.2, concentration_core=.7, concentration_pixel=.3),
                  morphology=NS(n_pixels=6, n_islands=2, n_small_islands=1,
                                n_medium_islands=1, n_large_islands=0),
                  intensity=NS(intensity_max=100, intensity_mean=50, intensity_std=20,
                               intensity_skewness=.5, intensity_kurtosis=-.2)))


def event(cameras=None):
    return NS(event_id=17, run_id=5, dl1=NS(tels={0: camera()} if cameras is None else cameras),
              dl2=NS(geometry={"direction": NS(telescopes=[2], is_valid=False)},
                     energy={"energy": NS(telescopes=[0, 2], energy_valid=False)}))


def test_multitel_predictions_match_keys_and_preserve_participation_when_rejected():
    original = event({2: camera(300), 0: camera(200)})
    prediction = [dict(event_id=17, run_id=5, telescope_id=2, valid_image=True,
                       image_cut_pass=True, reconstruction_valid=True, quality_pass=False,
                       pass_pid=True, accepted=False, disp_reco_deg=-.7,
                       quality_proxy_log_error=-.8, quality_threshold_smooth=-.9,
                       pid_gamma_score=.9, pid_threshold_gamma_score=.7,
                       group="36tel_z20", branch="fixed"),
                  dict(event_id=17, telescope_id=9, disp_reco_deg=99, accepted=True)]
    rows = images.event_to_dl3_image_rows(original, prediction, obs_id=123,
                                          geometry_name="direction", energy_name="energy")
    assert [row["TEL_ID"] for row in rows] == [0, 2]
    zero, two = rows
    assert (two["OBS_ID"], two["EVENT_ID"]) == (123, 17)
    assert two["HILLAS_INTENSITY"] == 300 and zero["HILLAS_INTENSITY"] == 200
    assert two["HILLAS_LENGTH"] == pytest.approx(np.degrees(.02))
    assert two["HILLAS_PSI"] == pytest.approx(np.degrees(.2))
    assert two["HILLAS_KURTOSIS"] == 2.7 and two["INTENSITY_KURTOSIS"] == -.2
    assert two["LEAKAGE_INTENSITY_WIDTH_1"] == .05 and two["N_SMALL_ISLANDS"] == 1
    assert two["DISP_RECO"] == -.7 and two["QUALITY_PROXY"] == -.8
    assert two["MODEL_GROUP"] == "36tel_z20" and two["MODEL_BRANCH"] == "fixed"
    assert two["USED_IN_DIRECTION"] == two["USED_IN_ENERGY"] == 1
    assert two["ACCEPTED"] == two["QUALITY_PASS"] == 0 and two["PID_PASS"] == 1
    assert zero["USED_IN_DIRECTION"] == 0 and zero["USED_IN_ENERGY"] == 1
    assert zero["ACCEPTED"] == -1 and np.isnan(zero["DISP_RECO"]) and zero["MODEL_GROUP"] == ""
    np.testing.assert_array_equal(original.dl1.tels[2].image, [100, 100])
    np.testing.assert_array_equal(original.dl1.tels[2].mask, [True, True])
    assert not any("TRIGGER" in name for name in two)


def test_unknown_missing_parameters_and_empty_schema():
    measured = event({0: camera(), 1: NS(image_parameters=None), 2: camera(0)})
    measured.dl1.tels[0].image_parameters.intensity.intensity_std = np.inf
    rows = images.event_to_dl3_image_rows(measured, obs_id=123)
    assert rows[0]["HAS_PARAMETERS"] == 1 and np.isnan(rows[0]["INTENSITY_STD"])
    assert rows[0]["IMAGE_VALID"] == rows[0]["USED_IN_DIRECTION"] == rows[0]["USED_IN_ENERGY"] == -1
    for row in rows[1:]:
        assert row["HAS_PARAMETERS"] == 0
        assert np.isnan(row["HILLAS_INTENSITY"]) and np.isnan(row["N_PIXELS"])
    assert images.event_to_dl3_image_rows(event({}), obs_id=123) == []
    assert images.event_to_dl3_image_rows(NS(event_id=17, dl1=None), obs_id=123) == []
    assert set(images.IMAGE_DTYPES) == set(images.IMAGE_UNITS) == set(images.IMAGE_DESCRIPTIONS) == set(rows[0])
    assert images.IMAGE_UNITS["HILLAS_LENGTH"] == "deg" and images.IMAGE_UNITS["HILLAS_INTENSITY"] == "ct"
    assert "photoelectrons" in images.IMAGE_DESCRIPTIONS["HILLAS_INTENSITY"]
    assert "not a measured error" in images.IMAGE_DESCRIPTIONS["QUALITY_PROXY"]
    empty = np.empty(0, dtype=list(images.IMAGE_DTYPES.items()))
    assert empty.dtype["MODEL_GROUP"].kind == "U"
    assert not any(name.startswith("EXTRA_") or "SCALE_RATIO" in name for name in rows[0])


@pytest.mark.parametrize("prediction", [
    {"event_id": 18, "telescope_id": 0},
    {"event_id": 17, "run_id": 6, "telescope_id": 0},
    [{"event_id": 17, "telescope_id": 0}, {"event_id": 17, "telescope_id": 0}],
    {"event_id": 17},
    {"event_id": 17, "telescope_id": .5},
    {"event_id": 17, "telescope_id": 0, "accepted": "False"},
])
def test_ambiguous_prediction_identity_and_flags_are_rejected(prediction):
    with pytest.raises(ValueError):
        images.event_to_dl3_image_rows(event(), prediction, obs_id=123)


def test_series_and_dataframe_predictions_follow_the_same_telescope_mapping():
    pd = pytest.importorskip("pandas")
    record = dict(event_id=17, run_id=5, telescope_id=0, valid_image=True,
                  accepted=np.nan, group="36tel_z20", branch="fixed", disp_reco_deg=.8)
    for prediction in (record, pd.Series(record), pd.DataFrame([record])):
        row = images.event_to_dl3_image_rows(event(), prediction, obs_id=123)[0]
        assert row["IMAGE_VALID"] == 1 and row["ACCEPTED"] == -1
        assert row["DISP_RECO"] == .8 and row["MODEL_GROUP"] == "36tel_z20"
