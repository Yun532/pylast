"""Geometry, mask provenance and empty-image checks for single-image Hillas plots."""
from types import SimpleNamespace as NS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pytest

from pylast.visualize import EventVisualizer, plot_image_detail, plot_hillas_circle


def example(psi=-0.7):
    x, y = np.meshgrid(np.linspace(-0.3, 0.3, 9), np.linspace(-0.3, 0.3, 9))
    geometry = NS(pix_x=x.ravel(), pix_y=y.ravel(), pix_area=np.full(x.size, 0.075**2))
    telescope = NS(optics=NS(equivalent_focal_length=8.0), camera=NS(geometry=geometry))
    source = NS(subarray=NS(tel_positions={0: [0, 0, 0]}, tels={0: telescope}))
    image = np.exp(-((x.ravel() - 0.10)**2 + (y.ravel() + 0.08)**2) / 0.015) * 30
    mask = image > 7.5
    h = NS(x=0.0125, y=-0.01, length=0.007, width=0.002, psi=psi,
           intensity=float(image[mask].sum()), r=np.hypot(0.0125, 0.01),
           phi=np.arctan2(-0.01, 0.0125), skewness=0.12, kurtosis=2.8)
    camera = NS(image=image, mask=mask, image_parameters=NS(hillas=h))
    event = NS(event_id=42, dl1=NS(tels={0: camera}), dl0=NS(tels={0: NS(image=image)}),
               simulation=NS(tels={0: NS(true_image=image, fake_image=image,
                                          fake_image_mask=mask, image_parameters=NS(hillas=h))}))
    return EventVisualizer(source), event, h


@pytest.mark.parametrize("psi", [-1.4, -0.7, 0, 0.8, np.pi / 2])
def test_detail_geometry_and_inputs(psi):
    vis, event, params = example(psi)
    image = event.dl1.tels[0].image.copy()
    mask = event.dl1.tels[0].mask.copy()
    result = plot_image_detail(event, visualizer=vis, tel_id=0, zoom=True, show=False)
    fig, ax = result["figure"], result["axes"]
    ellipse = ax.patches[0]
    np.testing.assert_allclose(ellipse.center, [-8, 10])
    assert ellipse.width == pytest.approx(2 * 800 * params.length)
    assert ellipse.height == pytest.approx(2 * 800 * params.width)
    assert ellipse.angle == pytest.approx(90 - np.degrees(psi))
    centroid = next(line for line in ax.lines if line.get_marker() == "o")
    assert centroid.get_markersize() == 7
    np.testing.assert_allclose([centroid.get_xdata()[0], centroid.get_ydata()[0]], ellipse.center)
    # The measurement endpoints use one sigma, not the full ellipse diameter.
    arrows = [t for t in ax.texts if getattr(t, "arrow_patch", None) is not None and t.get_text() == ""]
    lengths = [np.linalg.norm(np.array(t.xy) - np.array(t.xyann)) for t in arrows]
    np.testing.assert_allclose(lengths, [800 * params.length, 800 * params.width])
    np.testing.assert_array_equal(np.ma.getmaskarray(ax.collections[0].get_array()), ~mask)
    np.testing.assert_array_equal(event.dl1.tels[0].image, image)
    np.testing.assert_array_equal(event.dl1.tels[0].mask, mask)
    FigureCanvasAgg(fig).draw()
    plt.close(fig)


def test_empty_invalid_and_wrong_inputs():
    vis, event, params = example()
    event.dl1.tels[0].mask[:] = False
    fig, ax = vis.plot_image_detail(event, 0, zoom=True, show=False)
    assert len(ax.patches) == 0
    assert "unavailable" in ax._pylast_parameter_box.txt.get_text()
    FigureCanvasAgg(fig).draw()
    params.length = np.nan
    event.dl1.tels[0].mask[:] = True
    fig, ax = vis.plot_image_detail(event, 0, show=False)
    assert len(ax.patches) == 0
    FigureCanvasAgg(fig).draw()
    with pytest.raises(ValueError, match="Unknown tel_id"):
        vis.plot_image_detail(event, 99, show=False)
    event.dl1.tels[0].mask = np.ones(2, bool)
    with pytest.raises(ValueError):
        vis.plot_image_detail(event, 0, show=False)


def test_fake_source_and_centered_image():
    vis, event, params = example()
    params.x = params.y = params.r = 0
    params.phi = 0
    event.dl1 = None
    fig, ax = vis.plot_image_detail(event, 0, image_level="simulation_fake_clean", cmap="gray_r", show=False)
    assert len(ax.patches) == 1
    assert "Intensity:" in ax._pylast_parameter_box.txt.get_text()
    FigureCanvasAgg(fig).draw()
    fig, ax = vis.plot_image_detail(event, 0, image_level="simulation", show=False)
    assert len(ax.patches) == 0  # Never silently use fake parameters on a truth image.


def test_color_scale_and_legacy_outline():
    vis, _, _ = example()
    for data in [np.zeros(3), np.array([0, 0.2, 0.8]), np.array([np.nan, 0, 5]), np.array([-1, 0, 3])]:
        norm, cmap = vis._image_norm(data)
        assert norm.vmax > norm.vmin
        fig, ax = plt.subplots()
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        FigureCanvasAgg(fig).draw()
        plt.close(fig)
    norm, cmap = vis._image_norm(np.array([0, 0.2, 0.8]))
    background = matplotlib.colors.to_rgba("#f0f1f3")
    assert tuple(cmap(norm(0))) == background
    assert tuple(cmap(norm(0.2))) != background
    norm, cmap = vis._image_norm(np.array([0, 0.2, 500]))
    np.testing.assert_array_equal(cmap(norm(np.array([0.0, 0.2, 500.0])))[0], background)
    fig, ax = plt.subplots()
    assert plot_hillas_circle(ax, 0.01, -0.02, 0.005, 0.002, -0.7) is ax
    assert ax.patches[0].width == pytest.approx(np.degrees(0.005) * 2)
    plt.close(fig)


def test_default_full_camera_and_parameter_box_inside_axes():
    vis, event, _ = example()
    result = plot_image_detail(event, visualizer=vis, tel_id=0, show=False)
    fig, ax = result["figure"], result["axes"]
    xlim, ylim = vis._extent_cache[0]
    np.testing.assert_allclose(ax.get_xlim(), xlim)
    np.testing.assert_allclose(ax.get_ylim(), ylim)
    assert len(fig.axes) == 2  # camera and colorbar, no external parameter panel
    FigureCanvasAgg(fig).draw()
    box = ax._pylast_parameter_box
    bounds = box.get_window_extent(fig.canvas.get_renderer())
    assert ax.bbox.contains(bounds.x0, bounds.y0)
    assert ax.bbox.contains(bounds.x1, bounds.y1)
    assert "Length (1 sigma)" in box.txt.get_text()
    assert "Image:" not in box.txt.get_text()
    assert "Hillas:" not in box.txt.get_text()
    assert "Ellipse axes" not in box.txt.get_text()
    assert ax.get_title() == "LACT image event_id=42"
    assert ax.title.get_fontweight() == "normal"
    assert ax.title.get_fontsize() >= ax.xaxis.label.get_fontsize()
    assert ax.title.get_fontsize() >= ax.yaxis.label.get_fontsize()
    plt.close(fig)


def test_truth_metadata_and_named_overlay_lines():
    from pylast.visualize.event_visualizer import _camera_legend

    vis, event, params = example()
    event.simulation.shower = NS(energy=12.5, core_x=100.0, core_y=-200.0,
                                 alt=1.2, az=0.1)
    event.pointing = NS(array_altitude=1.2, array_azimuth=0.0)
    event.dl2 = NS(geometry={"HillasReconstructor": NS(
        is_valid=True, alt=1.19, az=0.11, core_x=110.0, core_y=-205.0)})
    fig, ax = vis.plot_image_detail(event, 0, show=False)
    parameters = ax._pylast_parameter_box.txt.get_text()
    assert parameters.startswith("Telescope ID: 0\nPointing (array):\nZen: 21.25 deg   Az: 0.00 deg")
    assert "True" not in parameters
    text = ax._pylast_truth_box.txt.get_text()
    assert "True energy: 12.50 TeV" in text
    assert "True core east: 200.0 m" in text
    assert "True core north: 100.0 m" in text
    assert "True zenith: 21.25 deg" in text
    assert "True azimuth: 5.73 deg" in text
    FigureCanvasAgg(fig).draw()
    renderer = fig.canvas.get_renderer()
    assert not ax._pylast_parameter_box.get_window_extent(renderer).overlaps(
        ax._pylast_truth_box.get_window_extent(renderer))
    assert all(t.get_bbox_patch() is None for t in ax.texts)
    assert vis._draw_truth_sdp_line(ax, event, 0)
    assert vis._draw_reco_sdp_line(ax, event, 0)
    vis._draw_truth_sdp_line(ax, event, 0)  # gathered views must deduplicate
    _camera_legend(ax)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Hillas major axis", "True SDP", "Reconstructed SDP"]
    colors = {line.get_label(): line.get_color() for line in ax.lines}
    assert colors["Hillas major axis"] == "r"
    assert colors["True SDP"] == "magenta"
    assert colors["Reconstructed SDP"] == "#2166ac"
    plt.close(fig)


@pytest.mark.parametrize("shower", [None, NS(energy=np.nan, core_x=None, core_y=np.nan, alt=None, az=np.nan)])
def test_missing_truth_is_omitted(shower):
    vis, event, _ = example()
    event.simulation = None if shower is None else NS(shower=shower)
    event.pointing = NS(array_altitude=1.2, array_azimuth=0.0)
    fig, ax = vis.plot_image_detail(event, 0, show_ideal_position=True, show=False)
    assert not hasattr(ax, "_pylast_truth_box")
    assert "Pointing (array)" in ax._pylast_parameter_box.txt.get_text()
    assert "True direction" not in ax.get_legend_handles_labels()[1]
    FigureCanvasAgg(fig).draw()
    plt.close(fig)


def test_shared_camera_style_and_distinct_annotations():
    from pylast.visualize import plot_camera_image

    vis, event, _ = example()
    fig, ax = vis.plot_image_detail(event, 0, show=False)
    colors = {t.get_text(): t.get_color() for t in ax.texts}
    assert colors["length"] != colors[r"$\psi$"]
    legacy = plot_camera_image(np.array([0, 1]), np.array([0, 1]), 0.1, np.array([0, 1]))
    for camera_axis in (ax, legacy):
        assert camera_axis.xaxis.label.get_fontsize() == 14
        assert camera_axis.yaxis.label.get_fontsize() == 14
        colorbar_axis = camera_axis.figure.axes[-1]
        assert colorbar_axis.get_ylabel() == "Charge [p.e.]"
        assert colorbar_axis.yaxis.label.get_fontsize() == 14
        assert camera_axis.get_xticklabels()[0].get_fontsize() == 12
    np.testing.assert_allclose(legacy.patches[0].get_facecolor(), matplotlib.colors.to_rgba("#f0f1f3"))
    fig2, gathered = plt.subplots()
    vis._draw_camera_frame(gathered, vis.tel_geoms[0])
    np.testing.assert_allclose(gathered.collections[0].get_facecolors()[0], legacy.patches[0].get_facecolor())
    assert vis._transparent_zero_cmap().get_bad()[-1] == 0
    plt.close("all")


@pytest.mark.parametrize("sign_x,sign_y", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
def test_detail_boxes_avoid_signal_pixel_footprints(sign_x, sign_y):
    vis, event, params = example()
    geom = vis.tel_geoms[0]
    camera = event.dl1.tels[0]
    camera.mask = (sign_x * geom.pix_x > 15) & (sign_y * geom.pix_y > 15)
    camera.image = np.where(camera.mask, 100.0, 0.0)
    params.x, params.y = sign_x * 0.03, sign_y * 0.03
    event.simulation.shower = NS(energy=10, core_x=100, core_y=200, alt=1.2, az=0.1)
    event.pointing = NS(array_altitude=1.2, array_azimuth=0)
    fig, ax = vis.plot_image_detail(event, 0, show=False)
    FigureCanvasAgg(fig).draw()
    vertices = vis._vertices_for(geom)[camera.mask]
    points = ax.transData.transform(vertices.reshape(-1, 2)).reshape(vertices.shape)
    lower, upper = points.min(axis=1), points.max(axis=1)
    bounds = [box.get_window_extent() for box in (ax._pylast_parameter_box, ax._pylast_truth_box)]
    assert not bounds[0].overlaps(bounds[1])
    for box in bounds:
        covered = ((lower[:, 0] < box.x1) & (upper[:, 0] > box.x0)
                   & (lower[:, 1] < box.y1) & (upper[:, 1] > box.y0))
        assert not np.any(covered)
        assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
    plt.close(fig)
