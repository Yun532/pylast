#!/usr/bin/env python3
import statistics
from ..helper import register_exe
import sys
import argparse
from pathlib import Path


def _real_shower_config(config):
    """Use measured DL1 parameters with the existing reconstructors."""
    import copy

    result = copy.deepcopy(config)
    shower = result.get("ShowerProcessor", result)
    names = shower.get("GeometryReconstructionTypes", ["HillasReconstructor"])
    supported = {"HillasReconstructor", "EnergyRegressor", "ParticleClassifier"}
    unsupported = set(names) - supported
    if unsupported:
        raise ValueError("Real data does not support reconstructors: " + ", ".join(sorted(unsupported)))
    for name in names:
        shower.setdefault(name, {})["use_fake_hillas"] = False
    return result


def _plot_requested(args):
    return bool(getattr(args, "plot_dir", None))


def _make_event_plotter(source, args):
    if not _plot_requested(args):
        return None, None

    from pylast.visualize import EventVisualizer

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    visualizer = EventVisualizer(
        source,
        enable_secondary_axes=not args.plot_no_secondary_axes,
        outline_pixels=not args.plot_no_pixel_edges,
    )
    return visualizer, plot_dir


def _maybe_plot_event(visualizer, plot_dir, event, event_index, input_index, plot_count, args):
    if visualizer is None:
        return False
    if args.plot_max is not None and plot_count >= args.plot_max:
        return False
    if args.plot_every <= 0:
        return False
    if event_index % args.plot_every != 0:
        return False

    event_id = getattr(event, "event_id", event_index)
    output_path = plot_dir / f"input{input_index:03d}_event{event_id}_{args.plot_kind}_{args.plot_level}.png"

    if args.plot_kind == "telescopes":
        visualizer.plot_telescopes(
            event,
            output_path=str(output_path),
            image_level=args.plot_level,
            show=False,
        )
    elif args.plot_kind == "gathered":
        visualizer.plot_gathered_event(
            event,
            output_path=str(output_path),
            image_level=args.plot_level,
            show_hillas=args.plot_hillas,
            show_ideal_position=args.plot_ideal_position,
            show=False,
        )
    else:
        visualizer.plot_event(
            event,
            output_path=str(output_path),
            image_level=args.plot_level,
            show_hillas=args.plot_hillas,
            show_ideal_position=args.plot_ideal_position,
            show=False,
        )
    return True

def enhanced_hillas_reco(args):
    """
    Enhanced Hillas reconstruction implementation.
    Add your custom implementation here.
    
    Args:
        args: Parsed command line arguments
    """
    print("Running enhanced Hillas reconstruction...")
    
    # Handle multiple input and output files
    input_files = args.input if isinstance(args.input, list) else [args.input]
    output_files = args.output if isinstance(args.output, list) else [args.output]
    
    # Assert that the number of input and output files match
    assert len(input_files) == len(output_files), f"Number of input files ({len(input_files)}) must match number of output files ({len(output_files)})"
    
    print(f"Processing {len(input_files)} file pairs:")
    for i, (input_file, output_file) in enumerate(zip(input_files, output_files)):
        print(f"  {i+1}: {input_file} -> {output_file}")
    
    from pylast.io import SimtelEventSource, LactEventSource, DataWriter
    from pylast.reco import ShowerProcessor
    from pylast.calib import Calibrator
    from pylast.image import ImageProcessor
    import json
    from pylast.helper import Statistics
    
    with open(args.config, encoding="utf-8") as config_file:
        config = json.load(config_file)
    is_real = args.input_format == "real"
    shower_config = config.get("shower_processor", {})
    if is_real:
        if not args.mapping_file:
            raise ValueError("Real data requires --mapping-file")
        if _plot_requested(args) and args.plot_level == "simulation":
            raise ValueError("Real data plots require --plot-level dl0 or dl1")
        from pylast.io import LactRealEventSource
        from pylast.calib import LactRealCalibrator

        shower_config = _real_shower_config(shower_config)
    
    # Process each input-output file pair
    for input_index, (input_fname, output_fname) in enumerate(zip(input_files, output_files)):
        print(f"Processing: {input_fname} -> {output_fname}")
        tels = [int(tel_id) for tel_id in args.subarray.split(',')] if args.subarray else []
        if is_real:
            if tels:
                raise ValueError("Real data telescope selection is defined by --mapping-file")
            source = LactRealEventSource(
                input_fname, args.mapping_file, max_events=-1,
                config_str=json.dumps(config.get("source", {})),
                pointing_file=args.pointing_file,
                pointing_alt=args.pointing_alt, pointing_az=args.pointing_az,
                analysis_file=args.analysis_file,
            )
            calibrator = LactRealCalibrator(
                args.calibration_file, config_str=json.dumps(config.get("calibrator", {})))
        else:
            source_type = LactEventSource if args.input_format == "lact" else SimtelEventSource
            source = source_type(input_fname, -1, tels)
            calibrator = (None if args.input_format == "lact" else
                          Calibrator(source.subarray, config_str=json.dumps(config.get("calibrator", {}))))
        if args.atmosphere_profile:
            source.load_atmosphere_model(args.atmosphere_profile)
        image_processor = ImageProcessor(source.subarray, config_str=json.dumps(config.get("image_processor", {})))
        shower_processor = ShowerProcessor(source.subarray, config_str=json.dumps(shower_config))
        writer_config = dict(config.get("data_writer", {}))
        if is_real:
            writer_config.get("DataWriter", writer_config).update(
                write_simulation_shower=False, write_simulated_camera=False,
                write_simulated_camera_image=False, write_simulation_config=False,
                write_pointing=True)
        if args.atmosphere_profile or (args.input_format == "lact" and source.atmosphere_model is not None):
            writer_config["write_atmosphere_model"] = True
        data_writer = DataWriter(getattr(source, "_source", source), output_fname,
                                 config_str=json.dumps(writer_config))
        visualizer, plot_dir = _make_event_plotter(source, args)
        
        plot_count = 0
        reported_missing_pointing = False
        for event_index, event in enumerate(source):
            if calibrator is not None:
                calibrator(event)
            image_processor(event)
            if not is_real or event.pointing is not None:
                shower_processor(event)
            elif not reported_missing_pointing:
                print("No telescope pointing: saving calibrated images without shower reconstruction")
                reported_missing_pointing = True
            if _maybe_plot_event(visualizer, plot_dir, event, event_index, input_index, plot_count, args):
                plot_count += 1
            data_writer(event)

        if not is_real:
            data_writer.write_all_simulation_shower(source.shower_array)
        data_writer.close()
        del source, data_writer, calibrator, shower_processor, image_processor
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='Hillas reconstruction tool')
    parser.add_argument('--enhanced', action='store_true', 
                       help='Use enhanced Hillas reconstruction implementation')
    parser.add_argument('--input-format', choices=['simtel', 'lact', 'real'],
                        help='Input format: simtel, LACTsim ROOT images, or measured LACT ROOT data')
    parser.add_argument('--mapping-file', help='Pixel mapping CSV for measured data')
    parser.add_argument('--calibration-file', help='Calibration constants for measured data')
    parser.add_argument('--analysis-file',
                        help='Matched official eventShow ML pulse summary for raw waveform data')
    parser.add_argument('--pointing-file', help='Telescope pointing table for measured data')
    parser.add_argument('--pointing-alt', type=float, help='Measured telescope altitude in degrees')
    parser.add_argument('--pointing-az', type=float, help='Measured telescope azimuth in degrees')
    parser.add_argument('--atmosphere-profile',
                        help='Matched four-column atmosphere profile: km, g/cm3, g/cm2, n-1')
    
    # Add common arguments that the hillas_reco executable typically uses
    parser.add_argument('-i', '--input', required=True, action='append',
                       help='Input file path (can be specified multiple times)')
    parser.add_argument('-o', '--output', required=True, action='append',
                       help='Output file path (can be specified multiple times)')
    parser.add_argument('-c', '--config', 
                       help='Configuration file path')
    parser.add_argument('-s', '--subarray',
                        help='Specify telescopes to use (comma-separated list, e.g., "1,2,3,4")')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--plot-dir',
                       help='Directory for event-check plots. Enables the Python enhanced processing loop.')
    parser.add_argument('--plot-every', type=int, default=1,
                       help='Plot every N processed events when --plot-dir is set (default: 1)')
    parser.add_argument('--plot-max', type=int,
                       help='Maximum number of plots to write per input file')
    parser.add_argument('--plot-level', choices=['simulation', 'dl0', 'dl1'], default='dl1',
                       help='Image level to plot after processing (default: dl1)')
    parser.add_argument('--plot-kind', choices=['event', 'gathered', 'telescopes'], default='event',
                       help='Plot type to write (default: event)')
    parser.add_argument('--plot-hillas', action='store_true',
                       help='Overlay Hillas ellipses when plotting camera images')
    parser.add_argument('--plot-ideal-position', action='store_true',
                       help='Overlay the true source position projected to the camera when pointing is available')
    parser.add_argument('--plot-no-secondary-axes', action='store_true',
                       help='Disable camera degree secondary axes for faster plotting')
    parser.add_argument('--plot-no-pixel-edges', action='store_true',
                       help='Disable pixel outlines for faster plotting')
    
    # Parse all arguments
    args = parser.parse_args()
    
    if args.enhanced or _plot_requested(args) or args.input_format or args.atmosphere_profile:
        return enhanced_hillas_reco(args)
    else:
        # Use the default executable - pass all args except --enhanced
        original_argv = sys.argv.copy()
        if '--enhanced' in sys.argv:
            sys.argv.remove('--enhanced')
        try:
            return register_exe("hillas_reco")
        finally:
            # Restore original argv
            sys.argv = original_argv

if __name__ == "__main__":
    sys.exit(main())
