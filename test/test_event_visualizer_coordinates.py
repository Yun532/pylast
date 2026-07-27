import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pylast.helper import convert_to_fov
from pylast.visualize.event_visualizer import (
    EventVisualizer,
    HillasParameters,
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
