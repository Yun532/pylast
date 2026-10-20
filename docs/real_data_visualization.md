# Real-data event displays

Run the real-data source, calibrator and the normal `ImageProcessor` first.
All plots below consume that same in-memory event. They do not recalibrate,
clean, or reconstruct it again. Camera rendering, Hillas annotations, colour
scales, telescope numbering and axis conventions remain those of the existing
`EventVisualizer` and quicklook functions.

```python
from pylast.visualize import (
    EventVisualizer,
    load_km2a_event_record,
    load_wcda_event_record,
    plot_raw_images,
    plot_clean_images,
    plot_image_detail,
    plot_gathered_images,
    plot_event_cores,
)

# `event` is already calibrated and processed; `source` supplies its geometry.
visualizer = EventVisualizer(source)
external = {
    "KM2A": load_km2a_event_record(km2a_root_file, event.event_id),
    "WCDA": load_wcda_event_record(wcda_root_file, event.event_id),
}

plot_raw_images(event, visualizer=visualizer, external_reconstructions=external)
plot_clean_images(event, visualizer=visualizer, external_reconstructions=external)
plot_image_detail(event, tel_id=1, visualizer=visualizer,
                  external_reconstructions=external)
plot_gathered_images(event, image_type="clean", visualizer=visualizer,
                     external_reconstructions=external)
plot_event_cores(event, visualizer=visualizer, external_reconstructions=external)

# Available with raw waveforms and LED area calibration (R1).
# eventShow-only input and nominal peak calibration do not produce R1.
# An unknown sampling interval is displayed as a sample index, not nanoseconds.
visualizer.plot_pe_time_series(event)
```

For a single-detector overlay, pass only its dictionary entry. A missing
record (`None`) is omitted. A record for a different LACT event ID raises an
error rather than drawing the wrong event. Real events have no MC shower;
truth energy, truth core, truth direction and truth SDP are omitted. Existing
DL2 reconstruction displays remain available, including the 3D SDP views.

When the event and sidecar supply the raw LACT Rabbit timestamp, joint plots
also require exact agreement of the seconds and 8 ns tick. This rejects a
different build file with the same event ID. The external detector's MJD is
retained as metadata and is not used for this identity check.

## Matched sidecar files

These helpers read existing LACT-to-KM2A/WCDA matches. They do not search for
time coincidences. Pass the sidecar belonging to the LACT build file explicitly;
event IDs need not be unique across different build files.

```python
from pylast.visualize import list_matched_event_ids, plot_km2a_event, plot_wcda_event

ids = list_matched_event_ids(km2a_root_file=km2a_root_file,
                             wcda_root_file=wcda_root_file)
print(ids["KM2A"], ids["WCDA"], ids["both"])

plot_km2a_event(km2a_root_file, ids["KM2A"][0])
plot_wcda_event(wcda_root_file, ids["WCDA"][0])

# Include this LACT event's actual telescope positions and pointing.
plot_km2a_event(km2a_root_file, event.event_id,
                event=event, visualizer=visualizer)
```

KM2A reads the LACT `event` branch, `Rec/rec_x`, `Rec/rec_y`,
`Rec/rec_theta`, `Rec/rec_phi` and available scalar reconstruction metadata.
When present, `mkm2a == 1` selects unique matches. WCDA supports both the old
`match_success == 1` format and the `event_wcda` tree with `Nmatch`
or `mwcda`. An explicit `match_success` flag takes priority and may represent
an upstream resolved multiple match. When it is absent, only multiplicity 1
is accepted. Unmatched sentinel rows are excluded. Duplicate successful IDs in a sidecar raise
an ambiguity error. Raw match flags and timing differences remain in the
returned record for inspection.

Records include `event_id`, `detector`, `source_path`, `entry_index`,
`core_east_m`, `core_north_m`, `azimuth_rad`, `altitude_rad` and available
source fields. Source energies are retained as supplied, without assuming
whether an energy column is logarithmic or which unit it uses.

A successful time match does not imply a valid detector reconstruction.
Non-finite directions or zenith angles outside [0, pi] are unavailable;
the supplied core sentinels -10000/-9999 are also unavailable. Canonical
display fields are NaN in these cases while the original fields remain in
the record. The plots report the unavailable reconstruction and retain
usable hit maps and other detectors' results.

`plot_km2a_event` draws the ED/MD hit charges using the bundled detector
position tables and split `Event/HitsE` / `Event/HitsM` branches. Both panels
reuse the core view's complete ED/MD and WCDA pond background, East/North axes,
compass, core marker and arrival arrow. They show selected hit counts, summed
charge and maximum charge/detector ID. Optional `event` and `visualizer`
arguments add the actual LACT telescope positions and pointing and align the
view limits with the core plot; the event ID and available Rabbit timestamp
must agree with the sidecar. The quicklook CLI supplies these arguments.
Reconstruction angles are labelled zenith and geographic azimuth, in degrees.
It accepts
`ed_pos_file` and `md_pos_file` overrides. The hit selection follows the
reference display (`status >= 2`, `pe >= 1`). Missing hit branches raise a
clear error. `plot_wcda_event` draws the matched core and direction on the
existing array background; the current WCDA sidecars contain reconstruction
scalars and do not contain a cell hit map.

## Coordinates and checks

KM2A directions use the reference 0.61-degree offset; WCDA directions use
29.442 degrees. WCDA core positions reuse `lhaaso_background.wcda_xy_to_en`
(30-degree rotation and the established centre translation). KM2A native
core coordinates are converted to East/North using the 0.61-degree offset.
Bundled detector positions already use the array plot axes and are reused
unchanged. Camera directions and SDPs use the current pylast projector and
the current camera Y/X display order, not the older standalone script's
independent flip/rotation convention.

`test/test_real_data_visualize.py` checks matching schemas, event identity,
camera projection and unchanged image/Hillas content. It also renders the
two 2026-02-14 sample sidecars when available at
`../real_data_plot/reference_data/20260214`, or at the directory specified by
`PYLAST_REAL_DATA_REFERENCE`. The existing image-detail and coordinate tests
provide the simulation plotting regression checks.

## LED status and a measured joint-event example

`event.is_led` is `True` for a recorded LED event, `False` for a recorded
non-LED event, and `None` when the input does not identify its class. Plots
label known LED events explicitly. Raw waveform files without a paired
summary do not provide this classification; an unknown status must not be
treated as non-LED. Select `event.is_led is False` when excluding LED events.

The supplied second summary's event 352455 is non-LED and has accepted KM2A
and WCDA time matches. Its WCDA reconstruction is usable; its KM2A direction
and core are invalid, while its KM2A hits remain available. From the repository root, using the isolated validation
runtime, the complete CLI example is:

```bash
../real_data_plot/validation/run_validation.sh scripts/real_data_quicklook.py \
  --input ../real_data_plot/reference_data/20260214/analysis/analysis_build_20260215064950.root \
  --mapping-file ../real_data_plot/mapping-v6.csv \
  --calibration-file ../real_data_plot/reference_data/20260214/led/LC01_analysis_build_20260214212003_led_cal.root \
  --pointing-file ../real_data_plot/reference_data/20260214/pointing/2026021415_encoder_data.txt \
  --km2a-file ../real_data_plot/reference_data/20260214/km2a/analysis_build_20260215064950.rec.root \
  --wcda-file ../real_data_plot/reference_data/20260214/wcda/analysis_build_20260215064950.merge.root \
  --event-id 352455 --output-dir ../real_data_plot/validation/quicklook
```

It writes the existing raw, clean, detailed Hillas and array/core views,
plus separate KM2A and WCDA views. Event 352012 in the first summary is an
LED example; its nearly uniform full-camera illumination is not an air-shower
image. LED *calibration* can be applied to both event classes and is separate
from this event classification.

For event 352455, the WCDA direction is about 3.239 degrees from the image's
Hillas axis. Independently recomputing the legacy hardware view gives the
same distance: the current camera view rotates both pixels and projected
directions by 180 degrees together. These sidecars provide accepted time
matches; their presence does not establish agreement with the image or
validate the encoder's pointing calibration. The display retains the supplied
coordinates and does not adjust them to fit the image.
