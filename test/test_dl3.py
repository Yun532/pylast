"""Small event-list checks independent of native pyLAST and model assets."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS, ModuleType

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.utils import iers

_io_path = Path(__file__).resolve().parents[1] / "src/pylast/io"
_package = ModuleType("_pylast_dl3_tests")
_package.__path__ = [str(_io_path)]
sys.modules[_package.__name__] = _package
_spec = importlib.util.spec_from_file_location("_pylast_dl3_tests.dl3", _io_path / "dl3.py")
dl3 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = dl3
_spec.loader.exec_module(dl3)

OBS_ID = 20260215031346
GEOMETRY, ENERGY = "OriginalSMonoDirection", "OriginalSMonoEnergy"
RABBIT_SECONDS, RABBIT_TICK = 1771096465, 95829135


@pytest.fixture(autouse=True)
def offline_coordinates():
    with iers.conf.set_temp("auto_download", False), iers.conf.set_temp("auto_max_age", None):
        yield


def native_event(event_id=7, *, seconds=RABBIT_SECONDS, tick=RABBIT_TICK):
    geometry = NS(alt=1.21, az=-0.7, is_valid=True, telescopes=[0],
                  core_x=np.nan, core_y=np.nan, hmax=np.nan, xmax=np.nan)
    energy = NS(estimate_energy=2.5, energy_valid=True, telescopes=[0])
    return NS(event_id=event_id, run_id=0, rabbit_time=seconds, rabbit_tick=tick,
              time_mjd=40587 + (seconds - 37 + tick * 8e-9) / 86400, is_led=False,
              pointing=NS(array_altitude=1.2, array_azimuth=0.3),
              dl2=NS(geometry={GEOMETRY: geometry}, energy={ENERGY: energy}))


def row(event_id=7, *, accepted=True, seconds=RABBIT_SECONDS, tick=RABBIT_TICK):
    return dl3.event_to_dl3_row(
        native_event(event_id, seconds=seconds, tick=tick),
        {"event_id": event_id, "accepted": accepted}, obs_id=OBS_ID,
        geometry_name=GEOMETRY, energy_name=ENERGY)


def epoch():
    return Time(RABBIT_SECONDS - 37, RABBIT_TICK * 8e-9, format="unix", scale="utc")


def observation():
    return dict(location=EarthLocation.from_geodetic(100 * u.deg, 30 * u.deg, 4400 * u.m),
                gti=epoch() + np.array([[-1, 1]]) * u.s, livetime=1.5,
                observation_mode="POINTING", reference_time=epoch() - 60 * u.s)


def fits_times(hdu):
    header = hdu.header
    reference = Time(header["MJDREFI"], header["MJDREFF"], format="mjd", scale="tt")
    return reference + TimeDelta(np.asarray(hdu.data["TIME"]), format="sec")


@pytest.mark.parametrize("prediction_type", ["dict", "series"])
def test_row_uses_native_tev_and_keeps_rejection_separate(prediction_type):
    event = native_event()
    prediction = dict(event_id=7, accepted=False, image_cut_pass=True,
                      reconstruction_valid=True, quality_pass=True, pass_pid=False,
                      energy_reco_GeV=2500.0)
    if prediction_type == "series":
        prediction = pytest.importorskip("pandas").Series(prediction)
    result = dl3.event_to_dl3_row(event, prediction, obs_id=OBS_ID,
                                  geometry_name=GEOMETRY, energy_name=ENERGY)
    assert result["OBS_ID"] == OBS_ID and result["EVENT_ID"] == 7
    assert result["ENERGY"] == pytest.approx(2.5)  # Native DL2 is already TeV.
    assert result["ALT"] == pytest.approx(np.degrees(1.21))
    assert result["AZ"] == pytest.approx(np.degrees(-0.7) % 360)
    assert result["POINTING_ALT"] == pytest.approx(np.degrees(1.2))
    assert result["POINTING_AZ"] == pytest.approx(np.degrees(0.3))
    assert result["GEOMETRY_VALID"] and result["ENERGY_VALID"]
    assert not result["ACCEPTED"] and result["PID_PASS"] == 0
    assert result["QUALITY_PASS"] == result["IMAGE_CUT_PASS"] == result["RECONSTRUCTION_VALID"] == 1
    assert result["TIME_MJD"] == pytest.approx(event.time_mjd, rel=0, abs=1e-11)
    assert (result["RABBIT_TIME"], result["RABBIT_TICK"]) == (RABBIT_SECONDS, RABBIT_TICK)
    minimal = row()
    assert minimal["QUALITY_PASS"] == minimal["PID_PASS"] == minimal["IMAGE_CUT_PASS"] == -1


def test_bad_prediction_identity_and_duplicate_events_are_rejected(tmp_path):
    for prediction in ({"event_id": 8, "accepted": True}, {"accepted": True}, {"event_id": 7}):
        with pytest.raises((ValueError, KeyError)):
            dl3.event_to_dl3_row(native_event(), prediction, obs_id=OBS_ID,
                                 geometry_name=GEOMETRY, energy_name=ENERGY)
    with pytest.raises(ValueError):
        dl3.write_dl3_events([row(), row()], tmp_path / "duplicate.fits", obs_id=OBS_ID, preview=True)


def test_nan_acceptance_is_rejected_by_helper_and_external_row_writer(tmp_path):
    with pytest.raises(ValueError):
        dl3.event_to_dl3_row(native_event(), {"event_id": 7, "accepted": np.nan},
                             obs_id=OBS_ID, geometry_name=GEOMETRY, energy_name=ENERGY)
    external = row()
    external["ACCEPTED"] = np.nan
    with pytest.raises(ValueError):
        dl3.write_dl3_events([external], tmp_path / "nan_selection.fits", obs_id=OBS_ID, preview=True)


def test_preview_does_not_invent_coordinates_or_gti(tmp_path):
    rows = [row(), row(8, accepted=False, tick=RABBIT_TICK + 1)]
    with pytest.raises(ValueError):
        dl3.write_dl3_events(rows, tmp_path / "strict.fits", obs_id=OBS_ID)
    path = dl3.write_dl3_events(rows, tmp_path / "preview.fits", obs_id=OBS_ID,
                                preview=True, accepted_only=False)
    assert path == tmp_path / "preview.fits" and isinstance(path, Path)
    with fits.open(path) as hdus:
        assert [hdu.name for hdu in hdus] == ["PRIMARY", "EVENTS", "POINTING", "SCHEMA", "PROVENANCE"]
        events = hdus["EVENTS"]
        assert len(events.data) == 2 and events.data["ACCEPTED"].tolist() == [True, False]
        assert np.isnan(events.data["RA"]).all() and np.isnan(events.data["DEC"]).all()
        assert events.header["TIMESYS"] == "TT"
        assert events.header["DL3STAT"] == "EVENTS_ONLY" and not events.header["IRFAVAIL"]
        assert events.header["MISSING"]
        assert events.data["RECONSTRUCTION_VALID"].tolist() == [-1, -1]
        assert events.columns["RECONSTRUCTION_VALID"].format == "I"
        assert "LIVETIME" not in events.header and "ONTIME" not in events.header


def test_tt_time_keeps_an_eight_ns_rabbit_tick(tmp_path):
    path = dl3.write_dl3_events(
        [row(), row(8, tick=RABBIT_TICK + 1)], tmp_path / "ticks.fits", obs_id=OBS_ID,
        preview=True, reference_time=epoch() - 60 * u.s)
    with fits.open(path) as hdus:
        events = hdus["EVENTS"]
        assert events.columns["TIME"].unit == "s"
        assert events.header["MJDTSYS"] == "UTC" and events.header["RBTSYS"] == "TAI"
        assert events.header["RBTTICK"] == 8e-9
        assert events.data["TIME"][1] - events.data["TIME"][0] == pytest.approx(8e-9, abs=5e-11)
        np.testing.assert_array_equal(events.data["RABBIT_TIME"], [RABBIT_SECONDS] * 2)
        np.testing.assert_array_equal(events.data["RABBIT_TICK"], [RABBIT_TICK, RABBIT_TICK + 1])
        expected = epoch() + np.array([0, 8e-9]) * u.s
        np.testing.assert_allclose((fits_times(events) - expected.tt).to_value(u.s), 0, atol=5e-10)


def test_pointing_to_icrs_roundtrip_and_real_exposure(tmp_path):
    metadata = observation()
    path = dl3.write_dl3_events([row()], tmp_path / "located.fits", obs_id=OBS_ID, **metadata)
    with fits.open(path) as hdus:
        events = hdus["EVENTS"]
        sky = SkyCoord(ra=events.data["RA"] * u.deg, dec=events.data["DEC"] * u.deg, frame="icrs")
        recovered = sky.transform_to(AltAz(obstime=fits_times(events), location=metadata["location"], pressure=0 * u.hPa))
        original = SkyCoord(alt=events.data["ALT"] * u.deg, az=events.data["AZ"] * u.deg,
                            frame=AltAz(obstime=fits_times(events), location=metadata["location"], pressure=0 * u.hPa))
        assert np.max(recovered.separation(original).to_value(u.deg)) < 1e-6
        assert [hdu.name for hdu in hdus] == ["PRIMARY", "EVENTS", "POINTING", "GTI", "SCHEMA", "PROVENANCE"]
        assert events.header["ONTIME"] == pytest.approx(2, abs=1e-7)
        assert events.header["LIVETIME"] == pytest.approx(1.5)
        assert events.header["DEADC"] == pytest.approx(0.75, abs=1e-7)
        assert not events.header["IRFAVAIL"] and events.header["DL3STAT"] == "EVENTS_ONLY"


def test_tracking_pointing_mean_uses_each_rows_own_time(tmp_path):
    location = observation()["location"]
    fixed = SkyCoord(ra=42 * u.deg, dec=75 * u.deg, frame="icrs")
    offsets = np.array([0, 7200, 14400])  # Four hours of tracking one fixed sky direction.
    horizontal = fixed.transform_to(AltAz(obstime=epoch() + offsets * u.s,
                                          location=location, pressure=0 * u.hPa))
    rows = []
    for index, seconds in enumerate(offsets):
        item = row(7 + index, seconds=RABBIT_SECONDS + int(seconds))
        item.update(ALT=float(horizontal.alt[index].deg), AZ=float(horizontal.az[index].deg),
                    POINTING_ALT=float(horizontal.alt[index].deg), POINTING_AZ=float(horizontal.az[index].deg))
        rows.append(item)
    path = dl3.write_dl3_events(rows, tmp_path / "tracking.fits", obs_id=OBS_ID,
                                location=location, reference_time=epoch(), preview=True)
    with fits.open(path) as hdus:
        header = hdus["EVENTS"].header
        mean = SkyCoord(ra=header["RA_PNT"] * u.deg, dec=header["DEC_PNT"] * u.deg, frame="icrs")
        assert mean.separation(fixed).to_value(u.deg) < 1e-6


def test_fits_and_root_share_filtered_rows_units_and_metadata(tmp_path):
    uproot = pytest.importorskip("uproot")
    rows = [row(), row(8, accepted=False), row(9, tick=RABBIT_TICK + 1)]
    metadata = observation()
    provenance = {"model": "test-only", "input": "measured.root",
                  "site": {"approximate": True, "name": "test-only site", "source": "synthetic fixture"}}
    paths = [dl3.write_dl3_events(rows, tmp_path / ("events" + suffix), obs_id=OBS_ID,
                                  provenance=provenance, **metadata) for suffix in (".fits", ".root")]
    with fits.open(paths[0]) as hdus, uproot.open(paths[1]) as root:
        assert root["EVENTS"].classname == "TTree"
        arrays = root["EVENTS"].arrays(library="np")
        np.testing.assert_array_equal(arrays["EVENT_ID"], [7, 9])
        for name in ("EVENT_ID", "OBS_ID", "TIME", "RA", "DEC", "ALT", "AZ", "ENERGY",
                     "ACCEPTED", "GEOMETRY_VALID", "ENERGY_VALID", "RABBIT_TIME", "RABBIT_TICK"):
            np.testing.assert_allclose(arrays[name], hdus["EVENTS"].data[name], rtol=0, atol=0, equal_nan=True)
        saved = json.loads(str(root["metadata"]))
        assert saved["header"]["SITEAPP"] is True and hdus["EVENTS"].header["SITEAPP"]
        assert saved["header"]["SITENAME"] == hdus["EVENTS"].header["SITENAME"] == "test-only site"
        assert saved["header"]["SITESRC"] == hdus["EVENTS"].header["SITESRC"] == "synthetic fixture"
        for name, unit in {"ENERGY": "TeV", "TIME": "s", "RABBIT_TIME": "s", "ALT": "deg", "AZ": "deg", "RA": "deg", "DEC": "deg"}.items():
            assert saved["units"][name] == hdus["EVENTS"].columns[name].unit == unit
        for key in ("TIMESYS", "MJDREFI", "MJDREFF", "MJDTSYS", "RBTSYS", "RBTTICK",
                    "IRFAVAIL", "DL3STAT", "ONTIME", "LIVETIME", "DEADC"):
            assert saved["header"][key] == hdus["EVENTS"].header[key]
        assert all(saved["provenance"][key] == value for key, value in provenance.items())


def test_gti_coverage_is_checked_after_selection(tmp_path):
    rows = [row(), row(8, accepted=False, seconds=RABBIT_SECONDS + 10)]
    metadata = observation()
    dl3.write_dl3_events(rows, tmp_path / "selected.fits", obs_id=OBS_ID, **metadata)
    with pytest.raises(ValueError):
        dl3.write_dl3_events(rows, tmp_path / "all.fits", obs_id=OBS_ID, accepted_only=False, **metadata)
    overlap = dict(metadata, gti=epoch() + np.array([[-1, 1], [0, 2]]) * u.s)
    with pytest.raises(ValueError):
        dl3.write_dl3_events([row()], tmp_path / "overlap.fits", obs_id=OBS_ID, **overlap)
    missing_mode = {key: value for key, value in metadata.items() if key != "observation_mode"}
    with pytest.raises(ValueError):
        dl3.write_dl3_events([row()], tmp_path / "mode.fits", obs_id=OBS_ID, **missing_mode)


@pytest.mark.parametrize("suffix", [".fits", ".root"])
def test_empty_accepted_selection_preserves_a_writable_schema(tmp_path, suffix):
    path = dl3.write_dl3_events([row(accepted=False)], tmp_path / ("empty" + suffix),
                                obs_id=OBS_ID, preview=True, image_rows=[])
    if suffix == ".fits":
        with fits.open(path) as hdus:
            assert len(hdus["EVENTS"].data) == 0 and "GTI" not in hdus
            assert hdus["EVENTS"].header["NINPUT"] == 1
            assert hdus["EVENTS"].header["NSELECT"] == 0
            assert {"TIME", "ENERGY", "ACCEPTED"}.issubset(hdus["EVENTS"].columns.names)
            assert len(hdus["IMAGE_PARAMETERS"].data) == 0
            assert len(hdus["POINTING"].data) == 1
    else:
        with pytest.importorskip("uproot").open(path) as root:
            assert root["EVENTS"].num_entries == 0
            assert {"TIME", "ENERGY", "ACCEPTED"}.issubset(root["EVENTS"].keys())
            header = json.loads(str(root["metadata"]))["header"]
            assert header["NINPUT"] == 1 and header["NSELECT"] == 0
            assert root["IMAGE_PARAMETERS"].num_entries == 0
            assert root["POINTING"].num_entries == 1


def test_release_tables_strings_selection_and_metadata_roundtrip(tmp_path):
    uproot = pytest.importorskip("uproot")
    from _pylast_dl3_tests.dl3_external import default_external_row
    rows = [row(), row(8, accepted=False), row(9, tick=RABBIT_TICK + 1)]
    for item in rows:
        external = default_external_row()
        for detector, offset in (("KM2A", .1), ("WCDA", -.1)):
            external.update({f"{detector}_MATCH_STATE": "MATCHED",
                             f"{detector}_IDENTITY_VERIFIED": True,
                             f"{detector}_DIRECTION_VALID": True,
                             f"{detector}_CORE_VALID": True,
                             f"{detector}_ALT": item["ALT"] + offset,
                             f"{detector}_AZ": item["AZ"],
                             f"{detector}_CORE_EAST": offset * 100,
                             f"{detector}_CORE_NORTH": offset * 200})
        item.update(external)
    image_rows = [dict(OBS_ID=OBS_ID, EVENT_ID=eid, TEL_ID=tel, HILLAS_INTENSITY=400.,
                       MODEL_GROUP="test group", IMAGE_VALID=True)
                  for eid, tel in ((7, 0), (7, 1), (8, 0), (9, 2))]
    paths = [dl3.write_dl3_events(rows, tmp_path / ("release" + suffix), obs_id=OBS_ID,
                                  image_rows=image_rows, provenance={"model": "synthetic"}, **observation())
             for suffix in (".fits", ".root")]
    with fits.open(paths[0], checksum=True) as hdus, uproot.open(paths[1]) as root:
        assert hdus["EVENTS"].data["EVENT_ID"].tolist() == [7, 9]
        assert hdus["POINTING"].data["EVENT_ID"].tolist() == [7, 8, 9]
        assert hdus["IMAGE_PARAMETERS"].data["EVENT_ID"].tolist() == [7, 7, 9]
        for hdu in hdus[1:]:
            assert root[hdu.name].classname == "TTree"
            actual = root[hdu.name].arrays(library="np")
            for name in hdu.columns.names:
                expected = np.asarray(hdu.data[name])
                if expected.dtype.kind in "SUO":
                    np.testing.assert_array_equal(np.asarray(actual[name]).astype(str), expected.astype(str))
                else:
                    np.testing.assert_allclose(actual[name], expected, rtol=0, atol=0, equal_nan=True,
                                               err_msg=f"{hdu.name}.{name}")
        metadata = json.loads(str(root["metadata"]))
        assert json.loads(hdus["PROVENANCE"].data["JSON"][0]) == metadata
        assert metadata["schema_version"] == "1.0"
        assert metadata["tables"]["IMAGE_PARAMETERS"]["units"]["HILLAS_INTENSITY"] == "ct"
        assert hdus["POINTING"].header["GEOLON"] == hdus["EVENTS"].header["GEOLON"]
        assert hdus["EVENTS"].header["EXTTIME"] == "LACT_EVENT"
        assert not hdus["EVENTS"].header["EXTUSED"]
        np.testing.assert_allclose(hdus["EVENTS"].data["ANGSEP_LACT_KM2A"], .1, atol=1e-10)
        np.testing.assert_allclose(hdus["EVENTS"].data["CORESEP_KM2A_WCDA"], np.hypot(20, 40), atol=1e-12)
        external_sky = SkyCoord(ra=hdus["EVENTS"].data["KM2A_RA"] * u.deg,
                                dec=hdus["EVENTS"].data["KM2A_DEC"] * u.deg, frame="icrs")
        recovered = external_sky.transform_to(AltAz(obstime=fits_times(hdus["EVENTS"]),
                    location=observation()["location"], pressure=0 * u.hPa))
        np.testing.assert_allclose(recovered.alt.deg, hdus["EVENTS"].data["KM2A_ALT"], atol=1e-8)


@pytest.mark.parametrize("bad_images", [
    [dict(OBS_ID=OBS_ID, EVENT_ID=42, TEL_ID=0)],
    [dict(OBS_ID=OBS_ID, EVENT_ID=7, TEL_ID=.5)],
    [dict(OBS_ID=OBS_ID, EVENT_ID=7, TEL_ID=0)] * 2,
])
def test_foreign_fractional_or_duplicate_image_keys_are_rejected(tmp_path, bad_images):
    with pytest.raises(ValueError):
        dl3.write_dl3_events([row()], tmp_path / "bad_image.fits", obs_id=OBS_ID,
                             preview=True, image_rows=bad_images)


@pytest.mark.parametrize("livetime", [-1, np.nan, 3])
def test_invalid_livetime_is_rejected(tmp_path, livetime):
    metadata = dict(observation(), livetime=livetime)
    with pytest.raises(ValueError):
        dl3.write_dl3_events([row()], tmp_path / "bad_live.fits", obs_id=OBS_ID, **metadata)
