# Single-image Hillas detail

```python
from pylast.visualize import plot_image_detail

# Process or load the event's images and Hillas parameters first.
result = plot_image_detail(
    event,
    source=source,
    tel_id=1,  # Telescope 1, matching the number shown in camera overviews
    image_level="dl1",
    zoom=False,  # full camera (the default)
    output_path="image_detail.png",
)
fig, ax = result["figure"], result["axes"]
```

`EventVisualizer(source).plot_image_detail(event, tel_id=1, ...)` returns
`(figure, camera_axis)`. The quicklook function returns the same dictionary
fields as the other quicklook helpers. `show=False` supports headless rendering.
The default `zoom=False` shows the full camera. All parameters appear in a box
inside the camera axes. After layout, the rendered box sizes and complete pixel
footprints are checked at four candidate corners, with a 6 pt clearance.
Both boxes are positioned jointly: first avoid overlapping each other, then
minimize covered absolute charge, then avoid annotation text when charge scores
tie. This placement is computed when plotting; subsequent interactive pan/zoom
does not reposition the boxes. If all corners contain signal, some obstruction
may remain. `zoom=True` is an optional cropped view. `cmap="gray_r"` provides a grayscale background.
The title uses the normal plot font at 16 pt (above the 14 pt axis labels) and reports the event
ID. The parameter box starts with `Telescope {tel_id}`, followed by pointing
zenith and azimuth in degrees. The current Python binding exposes only array
pointing, so this is explicitly labeled `Pointing (array)`.
True energy, east/north core position, and true zenith/azimuth appear in a
separate box in another low-charge corner. Each value is included only when
available and finite; if there are no valid truth values, no truth box is drawn.
Both boxes use the usual plot font at 10 pt, a translucent white background and
no visible border. Image-layer/source
headings and explanatory footer lines are omitted.
`show_ideal_position=True` and `show_reco_position=True` reuse existing position
overlays. The default reconstructor is `HillasReconstructor`.

## Telescope numbering

All plot labels use the existing LACT display convention, **actual data key + 1**:
camera overview, detail, gathered cameras, array/timing views, and static and
interactive SDP plots. For example, data key `0` is displayed as `Telescope 1`
or `T1`; key `7` is displayed as `Telescope 8`, even if other keys are absent.
Both the public `plot_image_detail` function and the `EventVisualizer` method
accept this **one-based display number**: `tel_id=1` selects `Telescope 1`,
`tel_id=2` selects `Telescope 2`, and `tel_id=8` selects data key 7.
Zero, negative and non-integer selectors are rejected. Numbers are never inferred
from which telescopes triggered or from their position in a list.

Event containers and returned data (`hillas_parameter_rows()[i]["tel_id"]`,
`hillas_telescope_ids`) still contain actual data keys. When selecting from
these records, call `plot_image_detail(..., tel_id=row["tel_id"] + 1)`.
The existing selectors of other plotting APIs (`tel_ids`, `telescope_ids`)
retain their data-key convention; this change applies to the detail selector.

## Data and coordinates

- DL1 displays apply the stored DL1 mask and read its stored Hillas parameters.
- `simulation_fake_clean` applies `fake_image_mask` and uses the simulation
  camera's stored Hillas parameters. For LACT truth-image processing with the
  native NSB injection, this is the matching image/parameter pair.
- `simulation_fake` shows the unmasked fake image with those same parameters.
- `dl0` and `simulation` can display raw/truth images with DL1 parameters. If DL1 parameters
  are absent, no ellipse is drawn. Plotting never runs cleaning or Hillas fitting.
- Quicklook aliases (`clean`, `raw`, `fake_clean`, etc.) follow the existing API.
- Empty masks and nonfinite/degenerate Hillas parameters produce an image with
  an unavailable-parameter note. Missing images/telescope IDs raise `ValueError`.

The plot retains horizontal camera Y and vertical camera X. Primary axes are
centimeters, with the existing focal-length-based angular secondary axes.
Parameter values are converted directly from stored radians to degrees. The
ellipse retains the existing focal-plane small-angle scaling by focal length.
Length and width are one-sigma spreads: dimension marks are offset outside the
ellipse, with faint extension lines referring to the centroid and semi-axis
endpoint. The dimension labels follow their respective axes; numeric values
remain in the parameter box. Full ellipse axes are twice those values.
Length is blue, width green, and psi ochre so length and psi remain distinct.
Detail annotations use 12–16 pt text with bold weight, 2–2.2 pt dimension
and angle lines, and a 2.8 pt ellipse. The parameter boxes stay at 10 pt.
The detail COG marker is 7 pt; overview/legacy centroid markers retain 3 pt.
Psi and phi are measured from camera +X toward +Y (upward toward right in the
display). Negative angles retain their sign, including phi near -180 degrees.

## Existing camera plots

Existing public function names and return types remain unchanged. The original
red ellipse, red centroid and red dashed major axis are retained. The legacy
`plot_hillas_circle` shares the ellipse renderer. The legacy camera palette
matches `plasma`. `plot_clean_images` / `plot_event_cameras` / `plot_gathered_images`
now label the actual overlays: red dashed = Hillas major axis, magenta solid =
True SDP, blue dashed = Reconstructed SDP. True/reconstructed direction markers
also have labels. Unavailable overlays are omitted and gathered legends are
deduplicated across telescopes.
Charge colorbars use continuous scaling with readable ticks, preserve positive
sub-photoelectron values, and handle zero/constant/nonfinite images. Gathered
images keep zero pixels transparent over a shared light-gray camera frame.
All camera plots use `#f0f1f3` for empty/masked pixels. Camera axes, angular
secondary axes and colorbars share 14 pt labels and 12 pt ticks, preserving the
configured Matplotlib font family. Colorbar labels read `Charge [p.e.]`.
Colorbar spacing accounts for secondary axes.

## Verification

Run `python -m pytest -q test/test_image_detail.py test/test_event_visualizer_coordinates.py`
in an environment with pylast's native extension and matplotlib available.
Checks cover coordinate swaps, positive/negative orientations, one-sigma
dimension lengths, cleaning masks, fake-image provenance, missing/invalid
parameters, unchanged input arrays, zero/background scaling and legacy calls.

The workspace `pylast_image_detail_20260918/cog/` contains a LACT simulated event example,
its saved image/mask/Hillas values, the before-change renderer, reproduction
script and PNG/PDF outputs. Before and after use exactly the same in-memory
image, mask and parameters. The configured native NSB generator is stochastic;
the saved NPZ and JSON preserve the exact displayed realization.
The full-view comparisons call the public `plot_clean_images` and
`plot_gathered_images` functions with the old/new EventVisualizer implementations,
including the notebook's truth/reconstructed position and SDP overlays. Neither output
is cropped. `public_api_paths.json` records the actual imported source files.
