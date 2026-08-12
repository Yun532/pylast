import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

from pylast.helper import convert_to_fov
from pylast.visualize.event_visualizer import (
    EventVisualizer,
    HillasParameters,
    TelescopeGeometry,
    _camera_to_plot_xy,
    incident_point_on_camera,
)


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
