import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

import pylast.visualize.event_visualizer as event_visualizer_module
from pylast.helper import convert_to_fov
from pylast.visualize.event_quicklook import hillas_parameter_rows
from pylast.visualize.event_visualizer import (
    EventData,
    EventVisualizer,
    HillasParameters,
    TelescopeGeometry,
    _camera_to_plot_xy,
    incident_point_on_camera,
)


def test_peak_timing_colors_only_current_five_triggered_telescopes(monkeypatch):
    selected = [1, 4, 8, 13, 21]
    visualizer = EventVisualizer.__new__(EventVisualizer)
    visualizer._verts_cache = {}
    visualizer._extent_cache = {}
    visualizer._transparent_plasma = None
    visualizer.enable_secondary_axes = False
    visualizer.outline_pixels = True
    visualizer.edge_color = (0, 0, 0, 0.5)
    visualizer.edge_linewidth = 0.2
    visualizer.tel_geoms = {
        tel_id: TelescopeGeometry(
            tel_id=tel_id,
            pos_x=float(tel_id % 6) * 20.0,
            pos_y=float(tel_id // 6) * 20.0,
            focal_length=800.0,
            pix_x=np.array([]),
            pix_y=np.array([]),
            pix_size=np.array([]),
        )
        for tel_id in range(29)
    }
    timing = {
        tel_id: {
            "time_first_ns": float(tel_id),
            "time_peak_ns": float(tel_id) + 5.0,
        }
        for tel_id in visualizer.tel_geoms
    }
    visualizer.source = SimpleNamespace(
        get_observation_timing=lambda event: timing,
        subarray=SimpleNamespace(tel_positions={}),
    )
    event = SimpleNamespace(
        event_id=93500000,
        simulation=SimpleNamespace(triggered_tels=selected),
    )
    event_data = EventData(
        event_id=93500000,
        energy=1.0,
        core_x=0.0,
        core_y=0.0,
        zenith_deg=20.0,
        azimuth_deg=0.0,
        x_max=400.0,
        first_interaction_height=20000.0,
        image_by_tel={},
        image_sum_by_tel={tel_id: float(tel_id + 1) for tel_id in range(29)},
        active_tels=np.arange(29, dtype=int),
    )
    monkeypatch.setattr(
        event_visualizer_module,
        "read_event_data",
        lambda event, tel_geoms, image_level: event_data,
    )

    figure, axes = visualizer.plot_trigger_timing(
        event,
        image_level="dl0",
        show_lhaaso_background=False,
        annotate=False,
        time_field="peak",
        show=False,
    )

    collections = axes[0].collections
    assert len(collections[0].get_offsets()) == 29
    assert len(collections[1].get_offsets()) == 5
    assert len(collections[2].get_offsets()) == 5
    plt.close(figure)


def test_fake_clean_plot_excludes_triggered_camera_rejected_by_cleaning():
    visualizer = EventVisualizer.__new__(EventVisualizer)
    visualizer._verts_cache = {}
    visualizer._extent_cache = {}
    visualizer._transparent_plasma = None
    visualizer.enable_secondary_axes = False
    visualizer.outline_pixels = True
    visualizer.edge_color = (0, 0, 0, 0.5)
    visualizer.edge_linewidth = 0.2
    visualizer.tel_geoms = {
        tel_id: TelescopeGeometry(
            tel_id=tel_id,
            pos_x=float(tel_id),
            pos_y=0.0,
            focal_length=800.0,
            pix_x=np.array([0.0, 1.0]),
            pix_y=np.array([0.0, 0.0]),
            pix_size=np.array([1.0, 1.0]),
        )
        for tel_id in (0, 1)
    }
    valid_hillas = SimpleNamespace(
        length=0.01, width=0.005, psi=0.0, x=0.0, y=0.0
    )
    cameras = {
        0: SimpleNamespace(
            fake_image=np.array([60.0, 0.0]),
            fake_image_mask=np.array([True, False]),
            image_parameters=SimpleNamespace(hillas=valid_hillas),
        ),
        1: SimpleNamespace(
            fake_image=np.array([80.0, 20.0]),
            fake_image_mask=np.array([], dtype=bool),
            image_parameters=SimpleNamespace(
                hillas=SimpleNamespace(
                    length=0.0, width=0.0, psi=0.0, x=0.0, y=0.0
                )
            ),
        ),
    }
    event = SimpleNamespace(
        event_id=1,
        simulation=SimpleNamespace(
            tels=cameras,
            triggered_tels=[0, 1],
            shower=SimpleNamespace(
                energy=1.0,
                core_x=0.0,
                core_y=0.0,
                alt=np.pi / 2,
                az=0.0,
                x_max=400.0,
                h_first_int=20000.0,
            ),
        ),
    )
    visualizer.source = SimpleNamespace(
        get_triggered_tels=lambda event: [0, 1]
    )

    rows = hillas_parameter_rows(
        event, image_level="simulation_fake_clean"
    )
    assert [row["tel_id"] for row in rows] == [0]

    figure, axes = visualizer.plot_event(
        event,
        image_level="simulation_fake_clean",
        show_hillas=True,
        include_non_triggered=False,
        show=False,
    )

    visible_titles = [axis.texts[0].get_text() for axis in axes[1:] if axis.texts]
    assert visible_titles == ["Telescope 1"]
    plt.close(figure)


def test_camera_to_plot_uses_original_pylast_axis_order():
    camera_x = np.array([1.0, 2.0])
    camera_y = np.array([-3.0, -4.0])

    plot_x, plot_y = _camera_to_plot_xy(camera_x, camera_y)

    np.testing.assert_array_equal(plot_x, camera_y)
    np.testing.assert_array_equal(plot_y, camera_x)


def test_incident_point_matches_canonical_pylast_fov_interface():
    source_alt = np.deg2rad(35.0)
    source_az = np.deg2rad(5.0)
    pointing_alt = np.deg2rad(30.0)
    pointing_az = 0.0

    expected_x, expected_y = convert_to_fov(
        source_alt,
        source_az,
        pointing_alt,
        pointing_az,
    )
    camera_x, camera_y, focal_x, focal_y = incident_point_on_camera(
        source_azimuth_rad=source_az,
        source_zenith_rad=np.pi / 2.0 - source_alt,
        telescope_azimuth_rad=pointing_az,
        telescope_zenith_rad=np.pi / 2.0 - pointing_alt,
        focal_length=800.0,
    )

    assert camera_x == expected_x
    assert camera_y == expected_y
    assert focal_x == 800.0 * expected_x
    assert focal_y == 800.0 * expected_y


def test_hillas_overlay_uses_same_axis_swap_as_pixels():
    visualizer = EventVisualizer.__new__(EventVisualizer)
    hillas = HillasParameters(
        length=10.0,
        width=2.0,
        psi=30.0,
        cog_x=3.0,
        cog_y=-4.0,
        focal_length=800.0,
    )
    figure, axis = plt.subplots()

    visualizer._draw_hillas_ellipse(axis, hillas)

    ellipse = axis.patches[0]
    assert ellipse.center == (-4.0, 3.0)
    assert ellipse.angle == 60.0
    plt.close(figure)


def test_truth_sdp_camera_line_uses_mc_direction_and_core():
    visualizer = EventVisualizer.__new__(EventVisualizer)
    visualizer.tel_geoms = {
        0: TelescopeGeometry(
            tel_id=0,
            pos_x=20.0,
            pos_y=10.0,
            focal_length=800.0,
            pix_x=np.array([]),
            pix_y=np.array([]),
            pix_size=np.array([]),
        )
    }
    event = SimpleNamespace(
        pointing=SimpleNamespace(
            array_azimuth=np.deg2rad(0.0),
            array_altitude=np.deg2rad(30.0),
        ),
        simulation=SimpleNamespace(
            shower=SimpleNamespace(
                az=np.deg2rad(5.0),
                alt=np.deg2rad(35.0),
                core_x=100.0,
                core_y=-50.0,
            )
        ),
    )
    figure, axis = plt.subplots()
    axis.set_xlim(-300.0, 300.0)
    axis.set_ylim(-300.0, 300.0)

    assert visualizer._draw_truth_sdp_line(axis, event, 0)

    line = axis.lines[0]
    assert line.get_color() == "magenta"
    assert line.get_linestyle() == "-"
    _, _, camera_x, camera_y = incident_point_on_camera(
        source_azimuth_rad=event.simulation.shower.az,
        source_zenith_rad=np.pi / 2.0 - event.simulation.shower.alt,
        telescope_azimuth_rad=event.pointing.array_azimuth,
        telescope_zenith_rad=np.pi / 2.0 - event.pointing.array_altitude,
        focal_length=800.0,
    )
    expected_x, expected_y = _camera_to_plot_xy(camera_x, camera_y)
    assert np.isclose(np.mean(line.get_xdata()), expected_x, rtol=0.0, atol=1e-12)
    assert np.isclose(np.mean(line.get_ydata()), expected_y, rtol=0.0, atol=1e-12)
    plt.close(figure)


def test_truth_sdp_camera_line_skips_missing_core():
    visualizer = EventVisualizer.__new__(EventVisualizer)
    visualizer.tel_geoms = {
        0: TelescopeGeometry(
            tel_id=0,
            pos_x=0.0,
            pos_y=0.0,
            focal_length=800.0,
            pix_x=np.array([]),
            pix_y=np.array([]),
            pix_size=np.array([]),
        )
    }
    event = SimpleNamespace(
        pointing=SimpleNamespace(array_azimuth=0.0, array_altitude=1.0),
        simulation=SimpleNamespace(
            shower=SimpleNamespace(az=0.0, alt=1.0, core_x=np.nan, core_y=np.nan)
        ),
    )
    figure, axis = plt.subplots()

    assert not visualizer._draw_truth_sdp_line(axis, event, 0)
    assert not axis.lines
    plt.close(figure)
