"""Scalar sidecar identity/state checks without native bindings or model files."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

uproot = pytest.importorskip("uproot")
_spec = importlib.util.spec_from_file_location(
    "dl3_external_tested", Path(__file__).resolve().parents[1] / "src/pylast/io/dl3_external.py")
external = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(external)

SECONDS, TICK = 1771096465, 95829135


def event(event_id=7, seconds=SECONDS, tick=TICK):
    return SimpleNamespace(event_id=event_id, rabbit_time=seconds, rabbit_tick=tick)


def sidecar(tmp_path, detector, *, event_ids=(7,), omit=(), **columns):
    n = len(event_ids)
    data = dict(event=np.asarray(event_ids, dtype=np.int32),
                RabbitTime=np.full(n, SECONDS, dtype=np.uint32),
                Rabbittime=np.full(n, TICK, dtype=np.uint32))
    if detector == "KM2A":
        data.update({"Rec/rec_theta": np.full(n, .3), "Rec/rec_phi": np.full(n, .4),
                     "Rec/rec_x": np.full(n, 10.), "Rec/rec_y": np.full(n, 20.),
                     "mkm2a": np.ones(n, dtype=np.int32), "deltaTkm2a": np.full(n, 10005439.),
                     "Event/mjd": np.full(n, 61086.1345), "Rec/mjd": np.full(n, 61086.1346),
                     "Event/ev_n": np.full(n, 101, dtype=np.int32),
                     "Rec/ev_n": np.full(n, 102, dtype=np.int32), "Erec_km2a": np.full(n, 12.)})
    else:
        data.update(theta=np.full(n, .3), phi=np.full(n, .4), xc=np.full(n, 10.), yc=np.full(n, 20.),
                    match_success=np.ones(n, dtype=np.int32), Nmatch=np.ones(n, dtype=np.int32),
                    mwcda=np.ones(n, dtype=np.int32), deltaTwcda=np.full(n, 9999972.),
                    mjd=np.full(n, 61086.1347), mjdns4=np.full(n, 12345, dtype=np.int32),
                    matched_wcda_entry=np.full(n, 43, dtype=np.int32), nhit=np.full(n, 250, dtype=np.int32))
    data.update({key: np.broadcast_to(value, (n,)).copy() for key, value in columns.items()})
    for key in omit:
        data.pop(key, None)
    path = tmp_path / f"{detector}.root"
    tree_name = "event" if detector == "KM2A" else "event_wcda"
    with uproot.recreate(path) as root:
        root.mktree(tree_name, {key: value.dtype for key, value in data.items()})
        root[tree_name].extend(data)
    return path


def test_disabled_schema_is_stable_and_provenance_json():
    source = external.DL3ExternalSource()
    row = source.rows_for_event(event())
    assert set(row) == set(external.EXTERNAL_DTYPES) == set(external.EXTERNAL_DESCRIPTIONS)
    for name, dtype in external.EXTERNAL_DTYPES.items():
        assert np.asarray(row[name], dtype=dtype).shape == ()
    assert row["KM2A_MATCH_STATE"] == row["WCDA_MATCH_STATE"] == "NOT_REQUESTED"
    assert row["KM2A_SOURCE_ENTRY"] == -1 and row["WCDA_N_RECORDS"] == 0
    assert "KM2A_DELTA_T_RAW" not in external.EXTERNAL_UNITS
    assert "KM2A_EREC_KM2A_RAW" not in external.EXTERNAL_UNITS
    assert json.loads(json.dumps(source.provenance))["sources"]["WCDA"]["enabled"] is False
    assert source.provenance["coordinates"]["ground_frame"] == "LHAASO_ARRAY_MAP"
    row["KM2A_MATCH_STATE"] = "changed"
    assert source.rows_for_event(event())["KM2A_MATCH_STATE"] == "NOT_REQUESTED"


def test_no_record_is_distinct_from_unmatched_padding(tmp_path):
    path = sidecar(tmp_path, "WCDA", match_success=0, Nmatch=0, mwcda=0,
                   RabbitTime=0, Rabbittime=0, theta=-9999., phi=-9999., xc=-9999., yc=-9999.)
    source = external.DL3ExternalSource(wcda_file=path)
    assert source.rows_for_event(event(8))["WCDA_MATCH_STATE"] == "NO_RECORD"
    row = source.rows_for_event(event())
    assert row["WCDA_MATCH_STATE"] == "UNMATCHED" and row["WCDA_SOURCE_ENTRY"] == 0
    assert row["WCDA_N_RECORDS"] == 1 and not row["WCDA_IDENTITY_VERIFIED"]
    assert row["WCDA_THETA_RAW"] == -9999 and np.isnan(row["WCDA_ALT"])
    assert not row["WCDA_DIRECTION_VALID"] and not row["WCDA_CORE_VALID"]


def test_matched_raw_times_and_known_coordinate_conventions(tmp_path):
    km = sidecar(tmp_path, "KM2A")
    wc = sidecar(tmp_path, "WCDA")
    source = external.DL3ExternalSource(km, wc)
    row = source.rows_for_event(event())
    for detector in ("KM2A", "WCDA"):
        assert row[detector + "_MATCH_STATE"] == "MATCHED"
        assert row[detector + "_IDENTITY_VERIFIED"]
        assert row[detector + "_LACT_RABBIT_TIME"] == SECONDS
        assert row[detector + "_LACT_RABBIT_TICK"] == TICK
        assert row[detector + "_DIRECTION_VALID"] and row[detector + "_CORE_VALID"]
        assert row[detector + "_ALT"] == pytest.approx(90 - np.degrees(.3))
    assert row["KM2A_AZ"] == pytest.approx(np.degrees(-.4 - np.pi) % 360 + .61)
    assert row["WCDA_AZ"] == pytest.approx((90 - 29.442 - np.degrees(.4)) % 360)
    assert row["KM2A_CORE_EAST"] == -20.
    assert row["KM2A_CORE_NORTH"] == 10.
    assert row["WCDA_CORE_EAST"] == pytest.approx(10 * np.cos(np.pi/6) - 20 * np.sin(np.pi/6) - 7)
    assert row["WCDA_CORE_NORTH"] == pytest.approx(10 * np.sin(np.pi/6) + 20 * np.cos(np.pi/6) + 7.97)
    assert row["KM2A_EVENT_MJD_RAW"] == 61086.1345
    assert row["KM2A_REC_MJD_RAW"] == 61086.1346
    assert row["KM2A_EVENT_ID_RAW"] == 101 and row["KM2A_REC_EVENT_ID_RAW"] == 102
    assert row["WCDA_MJD_RAW"] == 61086.1347 and row["WCDA_MJDNS4_RAW"] == 12345
    assert row["KM2A_DELTA_T_RAW"] == 10005439 and row["WCDA_DELTA_T_RAW"] == 9999972
    assert row["KM2A_EREC_KM2A_RAW"] == 12 and np.isnan(row["WCDA_ENERGY_RAW"])
    assert source.provenance["sources"]["KM2A"]["tree"] == "event"
    assert source.provenance["coordinates"]["WCDA"]["core_rotation_deg"] == 30


def test_km2a_array_map_cardinals_and_upstream_direction_vectors(tmp_path):
    native_xy = [(100., 0.), (0., 100.), (-100., 0.), (0., -100.)]
    map_xy = [(0., 100.), (-100., 0.), (0., -100.), (100., 0.)]
    phi_deg = np.array([0., 90., 180., 270.])
    theta = np.radians([20., 35., 50., 65.])
    path = sidecar(tmp_path, "KM2A", event_ids=(7, 8, 9, 10), **{
        "Rec/rec_x": np.asarray(native_xy)[:, 0], "Rec/rec_y": np.asarray(native_xy)[:, 1],
        "Rec/rec_theta": theta, "Rec/rec_phi": np.radians(phi_deg)})
    source = external.DL3ExternalSource(km2a_file=path)
    upstream_phi = np.radians(phi_deg - .61 + 270.)
    upstream_vectors = np.column_stack((np.sin(theta) * np.cos(upstream_phi),
                                        np.sin(theta) * np.sin(upstream_phi), np.cos(theta)))
    for index, expected_map in enumerate(map_xy):
        row = source.rows_for_event(event(7 + index))
        np.testing.assert_allclose([row["KM2A_CORE_EAST"], row["KM2A_CORE_NORTH"]], expected_map, atol=1e-12)
        az, alt = np.radians([row["KM2A_AZ"], row["KM2A_ALT"]])
        vector = [np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)]
        np.testing.assert_allclose(vector, upstream_vectors[index], rtol=0, atol=1e-14)
        assert 0 <= row["KM2A_AZ"] < 360


def test_upstream_resolved_multiplicity_and_unresolved_count(tmp_path):
    path = sidecar(tmp_path, "WCDA", Nmatch=3)
    row = external.DL3ExternalSource(wcda_file=path).rows_for_event(event())
    assert row["WCDA_MATCH_STATE"] == "MATCHED" and row["WCDA_MULTI_CANDIDATE"]
    assert row["WCDA_MATCH_SUCCESS_RAW"] == 1 and row["WCDA_NMATCH_RAW"] == 3
    path = sidecar(tmp_path, "WCDA", omit=("match_success",), Nmatch=2)
    row = external.DL3ExternalSource(wcda_file=path).rows_for_event(event())
    assert row["WCDA_MATCH_STATE"] == "AMBIGUOUS" and row["WCDA_MULTI_CANDIDATE"]
    assert row["WCDA_NMATCH_RAW"] == 2 and np.isnan(row["WCDA_ALT"])


def test_duplicate_selected_rows_are_never_arbitrarily_chosen(tmp_path):
    path = sidecar(tmp_path, "KM2A", event_ids=(7, 7), **{"Rec/rec_x": [10., 99.]})
    row = external.DL3ExternalSource(km2a_file=path).rows_for_event(event())
    assert row["KM2A_MATCH_STATE"] == "AMBIGUOUS" and row["KM2A_N_RECORDS"] == 2
    assert row["KM2A_SOURCE_ENTRY"] == -1 and row["KM2A_MULTI_CANDIDATE"]
    assert row["KM2A_IDENTITY_VERIFIED"] and np.isnan(row["KM2A_CORE_X_RAW"])
    assert np.isnan(row["KM2A_CORE_EAST"]) and not row["KM2A_DIRECTION_VALID"]


def test_one_explicit_selected_row_among_unmatched_rows(tmp_path):
    path = sidecar(tmp_path, "WCDA", event_ids=(7, 7), match_success=[0, 1], Nmatch=[0, 2],
                   RabbitTime=[0, SECONDS], Rabbittime=[0, TICK], xc=[-9999., 123.])
    row = external.DL3ExternalSource(wcda_file=path).rows_for_event(event())
    assert row["WCDA_MATCH_STATE"] == "MATCHED" and row["WCDA_SOURCE_ENTRY"] == 1
    assert row["WCDA_IDENTITY_VERIFIED"]
    assert row["WCDA_CORE_X_RAW"] == 123 and row["WCDA_CORE_VALID"]


def test_match_is_independent_of_reconstruction_validity(tmp_path):
    path = sidecar(tmp_path, "KM2A", **{"Rec/rec_theta": -10., "Rec/rec_x": -10000., "Rec/rec_y": -10000.})
    row = external.DL3ExternalSource(km2a_file=path).rows_for_event(event())
    assert row["KM2A_MATCH_STATE"] == "MATCHED" and row["KM2A_IDENTITY_VERIFIED"]
    assert row["KM2A_THETA_RAW"] == -10 and row["KM2A_CORE_X_RAW"] == -10000
    assert not row["KM2A_DIRECTION_VALID"] and not row["KM2A_CORE_VALID"]
    assert np.isnan(row["KM2A_ALT"]) and np.isnan(row["KM2A_CORE_EAST"])


@pytest.mark.parametrize("changes", [{"Rabbittime": TICK + 1}, {"RabbitTime": SECONDS + 1},
                                     {"RabbitTime": 0, "Rabbittime": 0}])
def test_selected_timestamp_must_verify_exactly(tmp_path, changes):
    path = sidecar(tmp_path, "WCDA", **changes)
    with pytest.raises(ValueError, match="Rabbit"):
        external.DL3ExternalSource(wcda_file=path).rows_for_event(event())


def test_missing_selected_clock_and_legitimate_unmatched_wrong_clock(tmp_path):
    path = sidecar(tmp_path, "KM2A", omit=("Rabbittime",))
    with pytest.raises(ValueError, match="Cannot verify"):
        external.DL3ExternalSource(km2a_file=path).rows_for_event(event())
    path = sidecar(tmp_path, "WCDA", match_success=0, Nmatch=0, RabbitTime=SECONDS + 1)
    with pytest.raises(ValueError, match="Wrong WCDA sidecar"):
        external.DL3ExternalSource(wcda_file=path).rows_for_event(event())


def test_io_and_schema_failures_do_not_become_no_record(tmp_path):
    with pytest.raises(FileNotFoundError):
        external.DL3ExternalSource(km2a_file=tmp_path / "absent.root")
    path = tmp_path / "wrong.root"
    with uproot.recreate(path) as root:
        root.mktree("unrelated", {"value": "float64"})
    with pytest.raises(ValueError, match="Expected one WCDA"):
        external.DL3ExternalSource(wcda_file=path)
