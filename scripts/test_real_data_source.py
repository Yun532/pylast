"""Small ROOT fixtures checking measured-reader and calibration boundaries.

Run against the newly built runtime, for example with PYTHONPATH set before
``python -m pytest -q scripts/test_real_data_source.py``. PyROOT writes the same
std::vector branch types as production eventShow and LED files.
"""
from array import array
import csv
import json

import numpy as np
import pytest
import ROOT

from pylast import helper
from pylast.calib import LactRealCalibrator
from pylast.io import DataWriter, LactRealEventSource, RootEventSource


UNIX_TIME = 1_700_000_000


def mapping_file(path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["map_number", "s_number", "s_x", "s_y", "fee_id"])
        for pixel in range(1, 10):
            writer.writerow([pixel, pixel, ((pixel - 1) % 3 - 1) * 25.4,
                             ((pixel - 1) // 3 - 1) * 25.4, 1])
    return path


def summary_file(path, events, led_flags=None):
    """Each event is (id, UTC seconds, Rabbit ticks, channel tuples)."""
    output = ROOT.TFile(str(path), "RECREATE")
    tree = ROOT.TTree("eventShow", "measured summary fixture")
    event_id, seconds, ticks = array("i", [0]), array("I", [0]), array("I", [0])
    tree.Branch("event", event_id, "event/I")
    tree.Branch("RabbitTime", seconds, "RabbitTime/i")
    tree.Branch("Rabbittime", ticks, "Rabbittime/i")
    led_flag = array("b", [0])
    if led_flags is not None:
        assert len(led_flags) == len(events)
        tree.Branch("LED_event", led_flag, "LED_event/O")
    types = {"map_number": "int", "HG_LG": "bool", "base": "float",
             "amp": "float", "area": "float", "PeakPos": "short"}
    vectors = {name: ROOT.std.vector(kind)() for name, kind in types.items()}
    for name, vector in vectors.items():
        tree.Branch(name, vector)
    for index, (number, utc, tick, channels) in enumerate(events):
        event_id[0], seconds[0], ticks[0] = number, utc + 37, tick
        if led_flags is not None:
            led_flag[0] = led_flags[index]
        for vector in vectors.values():
            vector.clear()
        for channel in channels:
            for vector, value in zip(vectors.values(), channel):
                vector.push_back(value)
        tree.Fill()
    tree.Write()
    output.Close()
    return path


def led_file(path, *, time_offset_s=37):
    output = ROOT.TFile(str(path), "RECREATE")
    tree = ROOT.TTree("led_cal", "time-dependent area factors")
    timestamp = array("d", [0])
    high, low = ROOT.std.vector("double")(), ROOT.std.vector("double")()
    tree.Branch("event_time", timestamp, "event_time/D")
    tree.Branch("GainFactor_AreaH", high)
    tree.Branch("GainFactor_AreaL", low)
    for offset, high_factor, low_factor in ((0, 2.0, 4.0), (20, 5.0, 10.0)):
        # Production LED event_time uses the raw Rabbit clock, before TAI->UTC.
        timestamp[0] = UNIX_TIME + offset + time_offset_s
        high.clear()
        low.clear()
        for pixel in range(9):
            high.push_back(0.0 if pixel == 3 else np.nan if pixel == 4 else high_factor)
            low.push_back(0.0 if pixel in (3, 4, 5) else low_factor)
        tree.Fill()
    tree.Write()
    output.Close()
    return path


def raw_file(path, rows):
    """Rows are (event, UTC seconds, ticks, FEE, channel, HG, amplitude)."""
    output = ROOT.TFile(str(path), "RECREATE")
    tree = ROOT.TTree("waveforms", "raw waveform fixture")
    buffers = {
        "event": (array("i", [0]), "I"),
        "RabbitTime": (array("I", [0]), "i"),
        "Rabbittime": (array("I", [0]), "i"),
        "FEE_ID": (array("H", [0]), "s"),
        "channel": (array("B", [0]), "b"),
        "HG_LG": (array("b", [0]), "O"),
    }
    for name, (buffer, kind) in buffers.items():
        tree.Branch(name, buffer, f"{name}/{kind}")
    waveform = array("f", [0] * 256)
    tree.Branch("waveform", waveform, "waveform[256]/F")
    for number, utc, tick, fee, channel, high, amplitude in rows:
        values = (number, utc + 37, tick, fee, channel, high)
        for (buffer, _), value in zip(buffers.values(), values):
            buffer[0] = value
        for sample in range(256):
            waveform[sample] = 2.0 + (amplitude if 80 <= sample < 85 else 0.0)
        tree.Fill()
    tree.Write()
    output.Close()
    return path


def test_nominal_gains_unknown_timing_and_native_iteration(tmp_path):
    mapping = mapping_file(tmp_path / "mapping.csv")
    channels = [(1, True, 2.0, 25.0, 100.0, 7), (1, False, 3.0, 10.0, 40.0, 9),
                (2, True, 2.0, 250.0, 200.0, 7), (2, False, 3.0, 40.0, 80.0, 9),
                (3, False, 3.0, 3.0, 12.0, 10), (4, True, 2.0, 7.5, 30.0, 11)]
    path = summary_file(tmp_path / "summary.root",
                        [(number, UNIX_TIME + number - 101, 0, channels) for number in (101, 102, 103)])
    for source_type in (helper.LactRealEventSource, LactRealEventSource):
        empty = source_type(str(path), str(mapping), max_events=0)
        assert len(empty) == 0
        assert list(empty) == []
        source = source_type(str(path), str(mapping))
        assert [event.event_id for event in source] == [101, 102, 103]
    source = LactRealEventSource(path, mapping)
    geometry = source.subarray.tels[0].camera.geometry
    # Pixels 2/4 are asymmetric: mm hardware (s_x,s_y) -> (-s_y,-s_x) metres.
    # The renderer displays canonical (pix_y,pix_x), reflecting both hardware axes.
    np.testing.assert_allclose(geometry.pix_x[[1, 3]], [0.0254, 0.0])
    np.testing.assert_allclose(geometry.pix_y[[1, 3]], [0.0, 0.0254])
    event = source[0]
    LactRealCalibrator()(event)
    np.testing.assert_allclose(event.dl0.tels[0].image,
                               [10.0, 408.0, 30.6, 3.0, 0, 0, 0, 0, 0])
    assert event.r1 is None
    assert np.isnan(event.dl0.tels[0].peak_time).all()
    assert event.pointing is None
    assert event.is_led is None
    assert event.time_mjd == pytest.approx(40587.0 + UNIX_TIME / 86400.0, rel=0, abs=1e-10)
    timed = LactRealEventSource(path, mapping, config_str={"sampling_interval_ns": 2.0})[0]
    LactRealCalibrator()(timed)
    np.testing.assert_array_equal(timed.dl0.tels[0].peak_time[:4], [14, 18, 20, 22])


def test_led_nearest_time_missing_gain_and_bad_factors(tmp_path):
    mapping = mapping_file(tmp_path / "mapping.csv")
    channels = [(1, True, 2.0, 25.0, 100.0, 7), (1, False, 3.0, 10.0, 40.0, 9),
                (2, True, 2.0, 250.0, 200.0, 7), (2, False, 3.0, 40.0, 80.0, 9),
                (3, False, 3.0, 3.0, 12.0, 10)]
    channels += [(pixel, high, 2.0, amp, area, 7)
                 for pixel in (4, 5, 6) for high, amp, area in
                 ((True, 25.0, 100.0), (False, 40.0, 80.0))]
    path = summary_file(tmp_path / "summary.root", [(201, UNIX_TIME + 1, 0, channels),
                        (202, UNIX_TIME + 19, 0, channels), (203, UNIX_TIME + 30, 0, channels)])
    source = LactRealEventSource(path, mapping)
    calibrator = LactRealCalibrator(led_file(tmp_path / "led.root"))
    for index, expected in enumerate(([50, 20, 3, 0, 0, 50, 0, 0, 0],
                                       [20, 8, 1.2, 0, 0, 20, 0, 0, 0])):
        event = source[index]
        calibrator(event)
        np.testing.assert_allclose(event.dl0.tels[0].image, expected)
        assert np.isfinite(event.dl0.tels[0].image).all()
    with pytest.raises(RuntimeError, match="outside LED calibration coverage"):
        calibrator(source[2])
    utc_calibrator = LactRealCalibrator(
        led_file(tmp_path / "led_utc.root", time_offset_s=0),
        config_str={"led_time_offset_s": 0})
    utc_event = source[0]
    utc_calibrator(utc_event)
    np.testing.assert_allclose(utc_event.dl0.tels[0].image, [50, 20, 3, 0, 0, 50, 0, 0, 0])


def test_duplicate_nan_summary_and_invalid_mapping_are_rejected(tmp_path):
    mapping = mapping_file(tmp_path / "mapping.csv")
    channels = [(1, True, 2.0, 25.0, 100.0, 7)]
    good = summary_file(tmp_path / "good.root", [(301, UNIX_TIME, 0, channels)])
    duplicate = summary_file(tmp_path / "duplicate.root", [(301, UNIX_TIME, 0,
        [(1, True, np.nan, 25.0, 100.0, 7), *channels])])
    source = LactRealEventSource(duplicate, mapping)
    with pytest.raises(RuntimeError, match="duplicate pixel/gain"):
        source[0]
    missing = tmp_path / "missing_column.csv"
    missing.write_text("map_number,s_number,s_x,s_y\n1,1,0,0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="(?i)(column|header)"):
        LactRealEventSource(good, missing)
    bad_mapping = tmp_path / "duplicate_mapping.csv"
    bad_mapping.write_text(mapping.read_text(encoding="utf-8") + "1,1,0,0,1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate pixel/electronics"):
        LactRealEventSource(good, bad_mapping)


def test_raw_nominal_units_led_charge_and_matching_boundaries(tmp_path):
    mapping = mapping_file(tmp_path / "mapping.csv")
    rows = [(401, UNIX_TIME + 1, 0, 1, 1, True, 10.0),
            (401, UNIX_TIME + 1, 0, 1, 1, False, 1.0)]
    path = raw_file(tmp_path / "raw.root", rows)
    source = LactRealEventSource(path, mapping)
    event = source[0]
    LactRealCalibrator()(event)
    assert event.dl0.tels[0].image[0] == pytest.approx(4.0)
    assert event.r1 is None  # Peak-only conversion must not manufacture charge-calibrated R1.
    assert event.c1.tels[0].waveform[0].shape == (9, 256)
    led_event = source[0]
    LactRealCalibrator(led_file(tmp_path / "led.root"))(led_event)
    assert led_event.dl0.tels[0].image[0] == pytest.approx(25.0)
    assert led_event.r1.tels[0].waveform[0].sum() == pytest.approx(25.0)
    for name, bad_rows, error in (
        ("clock", [rows[0], (401, UNIX_TIME + 2, 0, 1, 1, False, 1.0)], "Rabbit timestamp"),
        ("channel", [(401, UNIX_TIME + 1, 0, 1, 10, True, 10.0)], "Unmapped FEE/channel"),
    ):
        invalid = LactRealEventSource(raw_file(tmp_path / f"{name}.root", bad_rows), mapping)
        with pytest.raises(RuntimeError, match=error):
            invalid[0]
    sidecar = summary_file(tmp_path / "sidecar.root", [(401, UNIX_TIME + 1, 1,
        [(1, True, 2.0, 10.0, 50.0, 80), (1, False, 2.0, 1.0, 5.0, 80)])])
    paired = LactRealEventSource(path, mapping, analysis_file=sidecar)
    with pytest.raises(RuntimeError, match="analysis_file disagrees"):
        paired[0]


def test_interleaved_raw_events_preserve_channels_gains_and_clock_checks(tmp_path):
    mapping = mapping_file(tmp_path / "mapping.csv")
    # Builder-style channel/gain blocks: each event reappears throughout the file.
    rows = [(501 + index, UNIX_TIME + index, 0, 1, channel, high,
             30.0 * index + 5.0 * channel + (0 if high else 1))
            for channel in (1, 2) for high in (True, False) for index in (0, 1)]
    path = raw_file(tmp_path / "interleaved.root", rows)
    for source_type in (helper.LactRealEventSource, LactRealEventSource):
        source = source_type(str(path), str(mapping))
        events = list(source)
        assert len(source) == len(events) == 2
        assert [event.event_id for event in events] == [501, 502]
        for index, event in enumerate(events):
            camera = event.c1.tels[0]
            expected = np.array([[5, 10], [6, 11]]) + 30 * index
            np.testing.assert_allclose(camera.peak[:, :2], expected)
            assert camera.valid[:, :2].all()
            assert not camera.valid[:, 2:].any()
            for gain in (0, 1):
                np.testing.assert_allclose(camera.waveform[gain][:2, 80], expected[gain] + 2)
            LactRealCalibrator()(event)
            np.testing.assert_allclose(event.dl0.tels[0].image[:2],
                                       [2, 4] if index == 0 else [14, 418.2])
    bad_rows = list(rows)
    number, utc, tick, fee, channel, high, amp = bad_rows[-1]
    bad_rows[-1] = number, utc + 1, tick, fee, channel, high, amp
    invalid = LactRealEventSource(raw_file(tmp_path / "interleaved_bad_clock.root", bad_rows), mapping)
    with pytest.raises(RuntimeError, match="Rabbit timestamp"):
        invalid[1]


def test_led_flag_source_roundtrip_and_legacy_unknown(tmp_path):
    mapping = mapping_file(tmp_path / "mapping.csv")
    channels = [(1, True, 2.0, 10.0, 50.0, 80)]
    flagged_path = summary_file(tmp_path / "flagged.root",
        [(601, UNIX_TIME + 1, 0, channels), (602, UNIX_TIME + 2, 0, channels)],
        led_flags=[True, False])
    flagged = LactRealEventSource(flagged_path, mapping)
    unknown = LactRealEventSource(summary_file(tmp_path / "unknown.root",
        [(603, UNIX_TIME + 3, 0, channels)]), mapping)
    events = [flagged[0], flagged[1], unknown[0]]
    assert [event.is_led for event in events] == [True, False, None]
    raw_path = raw_file(tmp_path / "raw.root", [(601, UNIX_TIME + 1, 0, 1, 1, True, 10.0)])
    assert LactRealEventSource(raw_path, mapping)[0].is_led is None
    paired = LactRealEventSource(raw_path, mapping, analysis_file=flagged_path)
    assert paired[0].is_led is True

    output = tmp_path / "roundtrip.root"
    writer = DataWriter(flagged._source, str(output), config_str=json.dumps({
        "write_dl0": True, "write_dl1": False, "write_dl2": False,
        "write_simulation_shower": False, "write_simulated_camera": False,
        "write_simulation_config": False,
    }))
    calibrator = LactRealCalibrator()
    for event in events:
        calibrator(event)
        writer(event)
    writer.close()
    # The native iterator must keep even a temporary source alive until the end.
    assert [(event.event_id, event.is_led) for event in RootEventSource(str(output))] == [
        (601, True), (602, False), (603, None)]
    stored = ROOT.TFile(str(output), "READ")
    tree = stored.Get("events/event_index")
    assert tree.GetLeaf("is_led").GetTypeName() == "Char_t"
    values = []
    for index in range(tree.GetEntries()):
        tree.GetEntry(index)
        values.append(int(tree.GetLeaf("is_led").GetValue()))
    assert values == [1, 0, -1]
    stored.Close()

    # A pre-metadata pylast index must retain unknown, never imply a false flag.
    legacy_path = tmp_path / "legacy.root"
    stored = ROOT.TFile(str(output), "READ")
    legacy = ROOT.TFile(str(legacy_path), "RECREATE")

    def copy_directory(source, destination):
        # Copy latest cycles to a different file; never clone a tree onto itself.
        for name in {key.GetName() for key in source.GetListOfKeys()}:
            item = source.Get(name)
            if item.InheritsFrom("TDirectory"):
                copy_directory(item, destination.mkdir(name))
            else:
                destination.cd()
                if item.InheritsFrom("TTree"):
                    if name == "event_index":
                        item.SetBranchStatus("is_led", False)
                    copied = item.CloneTree(-1, "fast")
                    copied.Write(name)
                else:
                    item.Write(name)

    copy_directory(stored, legacy)
    legacy.Close()
    stored.Close()
    legacy_source = RootEventSource(str(legacy_path))
    assert [(event.event_id, event.is_led) for event in legacy_source] == [
        (601, None), (602, None), (603, None)]


def test_measured_dl1_mono_dl2_roundtrip_without_model_assets(tmp_path):
    from pylast.image import ImageProcessor
    from pylast.reco import MonoReconstructor
    from pylast.reco.MonoReconstructor import disp_to_sky

    mapping = mapping_file(tmp_path / "mapping.csv")
    channels = [(pixel, True, 2.0, amplitude, amplitude * 5, 80)
                for pixel, amplitude in enumerate((10, 20, 10, 15, 30, 25, 10, 35, 20), 1)]
    path = summary_file(tmp_path / "measured.root",
                        [(701, UNIX_TIME, 17, channels)], led_flags=[False])
    source = LactRealEventSource(path, mapping, pointing_alt=70, pointing_az=23)
    event = source[0]
    assert event.simulation is None and event.dl2 is None
    LactRealCalibrator()(event)
    ImageProcessor(source.subarray, config_str=json.dumps({"Tailcuts_cleaner": {
        "picture_thresh": 2, "boundary_thresh": 1, "min_number_picture_neighbors": 1,
    }}))(event)

    # Exercise the measured-image adapter and native storage without private weights.
    mono = MonoReconstructor.__new__(MonoReconstructor)
    mono.zenith_deg, mono.use_fake_hillas = 20, False
    mono.geometry_name, mono.energy_name = "MonoDISPReconstructor", "MonoEnergyRegressor"
    rows = mono.event_table(event)
    assert rows.event_id.tolist() == [701] and rows.telescope_id.tolist() == [0]
    assert mono.valid_images(rows).tolist() == [True]
    assert rows.hillas_intensity.iloc[0] == pytest.approx(event.dl1.tels[0].image_parameters.hillas.intensity)
    assert rows.n_pixels.iloc[0] == 9
    result = rows.assign(valid_image=True, image_cut_pass=True, reconstruction_valid=True,
                         accepted=True, disp_reco_deg=0.7, energy_reco_GeV=2500.0)
    expected_alt, expected_az = disp_to_sky(
        rows.nominal_x.iloc[0], rows.nominal_y.iloc[0], rows.nominal_psi.iloc[0], 0.7,
        event.pointing.array_azimuth, event.pointing.array_altitude)
    mono.store_result(event, result)
    dl2 = event.ensure_dl2()
    assert event.ensure_dl2() is dl2
    assert set(dl2.geometry) == {mono.geometry_name} and set(dl2.energy) == {mono.energy_name}

    output = tmp_path / "mono_roundtrip.root"
    writer = DataWriter(source._source, str(output), config_str=json.dumps({
        "write_dl0": True, "write_dl1": True, "write_dl1_image": True, "write_dl2": True,
        "write_pointing": True, "write_simulation_shower": False,
        "write_simulated_camera": False, "write_simulation_config": False,
    }))
    writer(event)
    writer.close()
    restored_source = RootEventSource(str(output))
    restored = restored_source[0]
    assert restored.event_id == event.event_id and restored.simulation is None
    assert restored.is_led is False
    np.testing.assert_allclose(restored.dl1.tels[0].image, event.dl1.tels[0].image)
    for stored_event in (event, restored):
        geometry = stored_event.dl2.geometry[mono.geometry_name]
        energy = stored_event.dl2.energy[mono.energy_name]
        assert geometry.is_valid and energy.energy_valid
        np.testing.assert_allclose([geometry.alt, geometry.az], [expected_alt, expected_az], rtol=0, atol=1e-12)
        assert energy.estimate_energy == pytest.approx(2.5)
        assert list(geometry.telescopes) == list(energy.telescopes) == [0]
        assert np.isnan([geometry.core_x, geometry.core_y, geometry.hmax, geometry.xmax]).all()
