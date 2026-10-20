"""Real-event plots share the current camera renderer and reject mismatched IDs."""
from pathlib import Path
from types import SimpleNamespace as NS
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Polygon
import numpy as np
import pytest
try:
    import uproot
except ImportError:
    uproot = None

from pylast.visualize import (
    EventVisualizer, load_km2a_event_record, load_wcda_event_record,
    list_matched_event_ids, plot_km2a_event, plot_wcda_event,
    plot_clean_images, plot_gathered_images, plot_image_detail,
)
from pylast.visualize.event_visualizer import read_event_data, incident_point_on_camera
from pylast.visualize.real_data import _load_table, _records


def example():
    x, y = np.meshgrid(np.linspace(-0.3, 0.3, 7), np.linspace(-0.3, 0.3, 7))
    image = 40 * np.exp(-((x.ravel() - 0.08)**2 + y.ravel()**2) / 0.02)
    geometry = NS(pix_x=x.ravel(), pix_y=y.ravel(), pix_area=np.full(x.size, 0.01))
    telescope = NS(optics=NS(equivalent_focal_length=8), camera=NS(geometry=geometry))
    source = NS(subarray=NS(tel_positions={0: [30, 40, 0]}, tels={0: telescope}))
    hillas = NS(x=0.01, y=0, length=0.008, width=0.003, psi=0.1, intensity=float(image.sum()))
    event = NS(event_id=7, pointing=NS(array_altitude=1.1, array_azimuth=0.2),
               dl0=NS(tels={0: NS(image=image.copy())}),
               dl1=NS(tels={0: NS(image=image.copy(), mask=image > 5,
                                image_parameters=NS(hillas=hillas))}))
    records = {name: dict(event_id=7, core_east_m=100, core_north_m=150,
                         azimuth_rad=0.21 + i * 0.01, altitude_rad=1.11)
               for i, name in enumerate(("KM2A", "WCDA"))}
    return EventVisualizer(source), event, records


def test_current_camera_style_and_contents_survive_external_overlays():
    vis, event, records = example()
    image = event.dl1.tels[0].image.copy()
    plain, axes = vis.plot_event(event, image_level="dl1", show_hillas=True, show=False)
    base = np.asarray(axes[1].collections[0].get_array()).copy()
    figure, overlay_axes = vis.plot_event(event, image_level="dl1", show_hillas=True,
                                         external_reconstructions=records, show=False)
    ax = overlay_axes[1]
    np.testing.assert_array_equal(base, np.asarray(ax.collections[0].get_array()))
    np.testing.assert_array_equal(event.dl1.tels[0].image, image)
    assert ax.get_xlabel() == axes[1].get_xlabel()
    assert ax.get_xlim() == axes[1].get_xlim()
    assert len(ax.patches) == len(axes[1].patches)  # Existing Hillas ellipse retained.
    assert {"KM2A direction", "WCDA direction", "KM2A SDP", "WCDA SDP"}.issubset(ax.get_legend_handles_labels()[1])
    for detector, record in records.items():
        _, _, cx, cy = incident_point_on_camera(record["azimuth_rad"], np.pi / 2 - record["altitude_rad"],
                                                0.2, np.pi / 2 - 1.1, focal_length=800)
        point = next(line for line in ax.lines if line.get_label() == detector + " direction")
        np.testing.assert_allclose(point.get_xdata(), [cy])
        np.testing.assert_allclose(point.get_ydata(), [cx])
    plt.close(plain)
    plt.close(figure)


def test_external_legend_does_not_cover_hillas_parameter_box():
    vis, event, records = example()
    fig, ax = vis.plot_image_detail(event, tel_id=1, external_reconstructions=records, show=False)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    assert not ax._pylast_parameter_box.get_window_extent(renderer).overlaps(
        ax.get_legend().get_window_extent(renderer))
    plt.close(fig)


def test_invalid_external_direction_is_reported_without_a_fake_overlay():
    vis, event, records = example()
    records["KM2A"].update(azimuth_rad=np.nan, altitude_rad=np.nan,
                           core_east_m=np.nan, core_north_m=np.nan)
    fig, axes = vis.plot_event(event, image_level="dl1", external_reconstructions=records, show=False)
    labels = axes[1].get_legend_handles_labels()[1]
    assert "KM2A direction" not in labels and "KM2A SDP" not in labels
    assert "KM2A direction unavailable (invalid reconstruction)" in labels
    assert "WCDA direction" in labels
    plt.close(fig)


@pytest.mark.parametrize("is_led", [True, False, None])
def test_led_label_preserves_unknown_status(is_led):
    vis, event, _ = example()
    event.is_led = is_led
    data = read_event_data(event, vis.tel_geoms, image_level="dl1")
    assert data.is_led is is_led
    assert ("LED calibration event" in vis._event_text(data, 1)) == (is_led is True)
    fig, ax = vis.plot_image_detail(event, tel_id=1, show=False)
    assert ("(LED)" in ax.get_title()) == (is_led is True)
    plt.close(fig)


@pytest.mark.parametrize("method,kwargs", [
    ("plot_event", {"image_level": "dl1", "show_hillas": True}),
    ("plot_gathered_event", {"image_level": "dl1"}),
    ("plot_image_detail", {"tel_id": 1}),
    ("plot_event_cores", {}),
    ("plot_event_sdp_planes", {}),
    ("plot_event_sdp_planes_3d", {}),
])
def test_plot_without_simulation_omits_truth(method, kwargs):
    vis, event, _ = example()
    data = read_event_data(event, vis.tel_geoms, image_level="dl1")
    assert np.isnan(data.energy) and np.isnan(data.core_x)
    figure, axes = getattr(vis, method)(event, show=False, **kwargs)
    for ax in np.asarray(axes).ravel():
        assert not any("True" in label or "Truth" in label for label in ax.get_legend_handles_labels()[1])
        assert not any("nan" in text.get_text().lower() for text in ax.texts)
    plt.close(figure)


def test_interactive_sdp_without_truth_keeps_reconstruction():
    vis, event, _ = example()
    event.dl2 = NS(geometry={"HillasReconstructor": NS(is_valid=True, core_x=100, core_y=-20, alt=1.1, az=0.2)})
    fig = vis.plot_event_sdp_planes_3d_interactive(event, image_level="dl1")
    assert any(trace.name == "Reco core" for trace in fig.data)
    assert not any("True" in trace.name or "Truth" in trace.name for trace in fig.data)


@pytest.mark.parametrize("absolute_time", [False, True])
def test_unknown_waveform_interval_uses_sample_index(absolute_time):
    vis, event, _ = example()
    event.r1 = NS(tels={0: NS(waveform=np.array([[1., 2., 3.], [2., 3., 4.]]))})
    vis.source.get_waveform_timing = lambda event, tel_id: {}
    vis.source.get_trigger_timing = lambda event: {0: {"trigger_first_time_ns": 99}}
    fig, ax = vis.plot_pe_time_series(event, absolute_time=absolute_time, show=False)
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 1, 2])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [3, 5, 7])
    assert "sample index" in ax.get_xlabel().lower() and "ns" not in ax.get_xlabel()
    assert len(ax.lines) == 1  # No ns trigger marker on an unknown sample axis.
    plt.close(fig)


def test_quicklooks_forward_external_records_and_validate_event_id():
    vis, event, records = example()
    for function, kwargs in ((plot_clean_images, {}), (plot_gathered_images, {"image_type": "clean"}),
                             (plot_image_detail, {"tel_id": 1})):
        result = function(event, visualizer=vis, external_reconstructions=records, show=False, **kwargs)
        axes = np.asarray(result["axes"]).ravel()
        assert any("WCDA direction" in axis.get_legend_handles_labels()[1] for axis in axes)
        plt.close(result["figure"])
    records["WCDA"]["event_id"] = 8
    with pytest.raises(ValueError, match="expected 7"):
        vis.plot_event(event, image_level="dl1", external_reconstructions=records, show=False)
    plt.close("all")


@pytest.mark.parametrize("method,kwargs", [
    ("plot_event", {"image_level": "dl1"}), ("plot_event_cores", {}),
])
@pytest.mark.parametrize("field", ["rabbit_time", "rabbit_tick"])
def test_same_id_with_wrong_lact_timestamp_rejects_sidecar(method, kwargs, field):
    vis, event, records = example()
    event.rabbit_time, event.rabbit_tick = 1771112345, 12345678
    for record in records.values():
        record.update(rabbit_time=event.rabbit_time, rabbit_tick=event.rabbit_tick)
    records["WCDA"][field] += 1
    try:
        with pytest.raises(ValueError, match="Wrong WCDA sidecar.*Rabbit timestamp"):
            getattr(vis, method)(event, external_reconstructions=records, show=False, **kwargs)
    finally:
        plt.close("all")


def test_timestamp_check_accepts_exact_legacy_and_simulation_records():
    _, event, records = example()
    event.rabbit_time, event.rabbit_tick = 1771112345, 12345678
    records["KM2A"].update(rabbit_time=event.rabbit_time, rabbit_tick=event.rabbit_tick, mjd=61000)
    # WCDA's old scalar-only schema has no LACT Rabbit timestamp and stays usable.
    assert len(list(_records(records, event.event_id, event=event))) == 2
    records["KM2A"]["rabbit_tick"] += 1
    # Native simulation events expose an unset (zero) Rabbit time.
    event.rabbit_time = 0
    assert len(list(_records(records, event.event_id, event=event))) == 2


@pytest.mark.parametrize("flag", ["match_success", "Nmatch", "mwcda"])
@pytest.mark.skipif(uproot is None, reason="Matched-sidecar tests require uproot")
def test_wcda_match_schemas_exclude_unmatched_and_ambiguous_entries(tmp_path, flag):
    path = tmp_path / (flag + ".root")
    values = {"event": np.array([7, 8, 9], dtype=np.int32), flag: np.array([1, 0, 2], dtype=np.int32)}
    values.update({name: np.ones(3) for name in ("theta", "phi", "mjd", "xc", "yc")})
    with uproot.recreate(path) as root:
        root.mktree("event_wcda", values)
    record = load_wcda_event_record(path, 7)
    assert record["event_id"] == 7 and record["match_success"] == 1
    assert load_wcda_event_record(path, 8) is None
    assert load_wcda_event_record(path, 9) is None
    assert list_matched_event_ids(wcda_root_file=path)["WCDA"] == [7]


@pytest.mark.parametrize("stem,km2a_count,wcda_count", [
    ("20260215064901", 3, 24), ("20260215064950", 4, 26),
])
@pytest.mark.skipif(uproot is None, reason="Matched-sidecar tests require uproot")
def test_actual_matched_sidecars(stem, km2a_count, wcda_count):
    default = Path(__file__).resolve().parents[2] / "real_data_plot" / "reference_data" / "20260214"
    directory = Path(os.environ.get("PYLAST_REAL_DATA_REFERENCE", default))
    km2a = directory / "km2a" / f"analysis_build_{stem}.rec.root"
    wcda = directory / "wcda" / f"analysis_build_{stem}.merge.root"
    if not km2a.exists() or not wcda.exists():
        pytest.skip("Reference real-data sidecars are not installed")
    ids = list_matched_event_ids(km2a_root_file=km2a, wcda_root_file=wcda)
    assert len(ids["KM2A"]) == km2a_count
    assert len(ids["WCDA"]) == wcda_count
    analysis = directory / "analysis" / f"analysis_build_{stem}.root"
    with uproot.open(analysis) as root:
        timestamps = root["eventShow"].arrays(["event", "RabbitTime", "Rabbittime"], library="np")
    for path, values, loader in ((km2a, ids["KM2A"], load_km2a_event_record),
                                 (wcda, ids["WCDA"], load_wcda_event_record)):
        for event_id in values:
            record = loader(path, event_id)
            if record["detector"] == "KM2A" and event_id == 352455:
                assert record["rec_theta"] == -10 and record["rec_x"] == -10000
                assert not record["direction_valid"] and not record["core_valid"]
                assert np.all(np.isnan([record[key] for key in ("core_east_m", "core_north_m", "azimuth_rad", "altitude_rad")]))
            else:
                assert record["direction_valid"] and record["core_valid"]
                assert np.all(np.isfinite([record[key] for key in ("core_east_m", "core_north_m", "azimuth_rad", "altitude_rad")]))
            index, = np.flatnonzero(timestamps["event"] == event_id)
            event = NS(event_id=event_id, rabbit_time=int(timestamps["RabbitTime"][index]),
                       rabbit_tick=int(timestamps["Rabbittime"][index]))
            assert len(list(_records({record["detector"]: record}, event_id, event=event))) == 1
    figure, axes = plot_km2a_event(km2a, 352455 if stem == "20260215064950" else ids["KM2A"][0], show=False)
    for ax in axes:
        assert {"KM2A ED (5216)", "KM2A MD (1188)", "WCDA outer 310 x 270 m"}.issubset(
            ax.get_legend_handles_labels()[1])
        assert sum(isinstance(patch, Polygon) for patch in ax.patches) == 4
        assert any(collection.get_array() is not None for collection in ax.collections)
    assert axes[0].get_xlim() == axes[1].get_xlim()
    assert axes[0].get_ylim() == axes[1].get_ylim()
    if stem == "20260215064950":
        assert any("invalid reconstructed direction" in text.get_text() for ax in axes for text in ax.texts)
    figure.canvas.draw()
    plt.close(figure)
    if stem == "20260215064901":
        visualizer, event, _ = example()
        event.event_id = ids["KM2A"][0]
        core_figure, core_ax = visualizer.plot_event_cores(event, show=False)
        figure, axes = plot_km2a_event(km2a, event.event_id, event=event, visualizer=visualizer, show=False)
        for ax in axes:
            assert {"T1", "Telescope"}.issubset(text.get_text() for text in ax.texts)
            assert ax.get_xlim() == core_ax.get_xlim()
            assert ax.get_ylim() == core_ax.get_ylim()
        figure.canvas.draw()
        plt.close(figure)
        plt.close(core_figure)
    figure, ax = plot_wcda_event(wcda, ids["WCDA"][0], show=False)
    assert any(text.get_text() == "WCDA direction" for text in ax.texts)
    figure.canvas.draw()
    plt.close(figure)
    _load_table.cache_clear()
