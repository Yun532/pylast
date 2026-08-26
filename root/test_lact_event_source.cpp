#include "LactEventSource.hh"
#include "Calibration.hh"

#include "TFile.h"
#include "TTree.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class WaveformCase {
    Complete,
    NoTree,
    MissingTriggered,
    MismatchedArrays,
    MeasuredMv,
    MeasuredMvMissingCalibration,
};

void writeFixture(const std::filesystem::path& path, WaveformCase waveform_case)
{
    TFile file(path.string().c_str(), "RECREATE");

    std::string schema_name = "lact_event_root";
    std::string profile = "test";
    int schema_version = 1;
    int run_id = 7;
    TTree config("config", "config");
    config.Branch("schema_name", &schema_name);
    config.Branch("schema_version", &schema_version);
    config.Branch("profile", &profile);
    config.Branch("run_id", &run_id);
    config.Fill();
    config.Write();

    int pixel_id = 1;
    double x_m = 0.03;
    double y_m = -0.04;
    double size_m = 0.05;
    int shape_code = 1;
    TTree camera("camera_pixels", "camera_pixels");
    camera.Branch("pixel_id", &pixel_id);
    camera.Branch("x_m", &x_m);
    camera.Branch("y_m", &y_m);
    camera.Branch("size_m", &size_m);
    camera.Branch("shape_code", &shape_code);
    camera.Fill();
    camera.Write();

    std::string optics_name = "LACT";
    int num_mirrors = 1;
    double mirror_area_m2 = 1.0;
    double equivalent_focal_length_m = 1.0;
    double effective_focal_length_m = 1.0;
    TTree optics("optics", "optics");
    optics.Branch("name", &optics_name);
    optics.Branch("num_mirrors", &num_mirrors);
    optics.Branch("mirror_area_m2", &mirror_area_m2);
    optics.Branch("equivalent_focal_length_m", &equivalent_focal_length_m);
    optics.Branch("effective_focal_length_m", &effective_focal_length_m);
    optics.Fill();
    optics.Write();

    int telescope_id = 0;
    std::string telescope_name = "LACT-0";
    double telescope_x = 0.0;
    double telescope_y = 0.0;
    double telescope_z = 0.0;
    double pointing_az_deg = 0.0;
    double pointing_el_deg = 70.0;
    TTree telescopes("telescopes", "telescopes");
    telescopes.Branch("telescope_id", &telescope_id);
    telescopes.Branch("name", &telescope_name);
    telescopes.Branch("array_x_north_m", &telescope_x);
    telescopes.Branch("array_y_west_m", &telescope_y);
    telescopes.Branch("array_z_up_m", &telescope_z);
    telescopes.Branch("pointing_az_deg", &pointing_az_deg);
    telescopes.Branch("pointing_el_deg", &pointing_el_deg);
    telescopes.Fill();
    telescopes.Write();

    long long corsika_event_id = 100;
    int shower_event_id = 1;
    int array_id = 2;
    double array_time_offset_ns = 3.5;
    double area_weight_m2 = 1250.0;
    bool has_explicit_area_weight = true;
    TTree corsika_events("corsika_events", "corsika_events");
    corsika_events.Branch("event_id", &corsika_event_id);
    corsika_events.Branch("shower_event_id", &shower_event_id);
    corsika_events.Branch("array_id", &array_id);
    corsika_events.Branch("array_time_offset_ns", &array_time_offset_ns);
    corsika_events.Branch("area_weight_m2", &area_weight_m2);
    corsika_events.Branch("has_explicit_area_weight",
                          &has_explicit_area_weight);
    corsika_events.Fill();
    corsika_event_id = 101;
    array_id = 3;
    corsika_events.Fill();
    corsika_events.Write();

    const bool measured_mv =
        waveform_case == WaveformCase::MeasuredMv ||
        waveform_case == WaveformCase::MeasuredMvMissingCalibration;
    int n_time_bins = measured_mv ? 6 : 2;
    double time_bin_width_ns = measured_mv ? 2.0 : 1.0;
    std::vector<double> time_centers_ns = measured_mv
        ? std::vector<double>{1.0, 3.0, 5.0, 7.0, 9.0, 11.0}
        : std::vector<double>{0.5, 1.5};
    std::string sample_unit = measured_mv ? "mV" : "";
    double single_pe_area_mv_ns =
        waveform_case == WaveformCase::MeasuredMv ? 16.0 : 0.0;
    std::string template_time_reference = measured_mv ? "peak" : "";
    std::vector<double> reference_pulse_time_ns = measured_mv
        ? std::vector<double>{-4.0, -2.0, 0.0, 2.0, 4.0, 6.0}
        : std::vector<double>{};
    std::vector<double> reference_pulse_amplitude = measured_mv
        ? std::vector<double>{0.0, 1.0, 4.0, 2.0, 1.0, 0.0}
        : std::vector<double>{};
    TTree waveform_config("waveform_config", "waveform_config");
    waveform_config.Branch("n_time_bins", &n_time_bins);
    waveform_config.Branch("time_bin_width_ns", &time_bin_width_ns);
    waveform_config.Branch("time_centers_ns", &time_centers_ns);
    waveform_config.Branch("sample_unit", &sample_unit);
    waveform_config.Branch("single_pe_area_mv_ns", &single_pe_area_mv_ns);
    waveform_config.Branch("template_time_reference",
                           &template_time_reference);
    waveform_config.Branch("reference_pulse_time_ns",
                           &reference_pulse_time_ns);
    waveform_config.Branch("reference_pulse_amplitude",
                           &reference_pulse_amplitude);
    waveform_config.Fill();
    waveform_config.Write();

    long long event_id = 100;
    bool triggered = true;
    int n_pixels_camera = 1;
    std::vector<int> observation_pixels = {1};
    std::vector<float> image_pe = {4.25f};
    std::vector<float> image_cherenkov_pe = {5.25f};
    std::vector<float> image_time_peak_ns = {0.5f};
    double reference_time_ns = -1.0;
    double time_first_ns = -1.0;
    double time_mean_ns = 0.25;
    double time_rms_ns = 0.75;
    double time_peak_ns = 0.5;
    double trigger_time_ns = 12.5;
    double trigger_first_time_ns = 11.5;
    double trigger_max_multiplicity_time_ns = 12.5;
    double geometric_delay_ns = -2.0;
    double coincidence_time_ns = 10.5;
    double impact_parameter_m = 37.25;
    TTree observations("observations", "observations");
    observations.Branch("event_id", &event_id);
    observations.Branch("telescope_id", &telescope_id);
    observations.Branch("triggered", &triggered);
    observations.Branch("n_pixels_camera", &n_pixels_camera);
    observations.Branch("pixel_id", &observation_pixels);
    observations.Branch("image_pe", &image_pe);
    observations.Branch("image_cherenkov_pe", &image_cherenkov_pe);
    observations.Branch("image_time_peak_ns", &image_time_peak_ns);
    observations.Branch("reference_time_ns", &reference_time_ns);
    observations.Branch("time_first_ns", &time_first_ns);
    observations.Branch("time_mean_ns", &time_mean_ns);
    observations.Branch("time_rms_ns", &time_rms_ns);
    observations.Branch("time_peak_ns", &time_peak_ns);
    observations.Branch("trigger_time_ns", &trigger_time_ns);
    observations.Branch("trigger_first_time_ns", &trigger_first_time_ns);
    observations.Branch("trigger_max_multiplicity_time_ns",
                        &trigger_max_multiplicity_time_ns);
    observations.Branch("geometric_delay_ns", &geometric_delay_ns);
    observations.Branch("coincidence_time_ns", &coincidence_time_ns);
    observations.Branch("impact_parameter_m", &impact_parameter_m);
    observations.Fill();
    event_id = 101;
    triggered = false;
    image_pe = {0.0f};
    image_cherenkov_pe = {0.0f};
    observations.Fill();
    observations.Write();

    if (waveform_case != WaveformCase::NoTree) {
        event_id = 100;
        std::vector<int> waveform_pixels = measured_mv
            ? std::vector<int>{1, 1, 1, 1, 1, 1}
            : std::vector<int>{1};
        std::vector<unsigned short> time_bin = measured_mv
            ? std::vector<unsigned short>{0, 1, 2, 3, 4, 5}
            : std::vector<unsigned short>{0};
        std::vector<unsigned int> time_bin_u32(
            time_bin.begin(), time_bin.end());
        std::vector<float> values = measured_mv
            ? std::vector<float>{0.0f, 1.0f, 4.0f, 2.0f, 1.0f, 0.0f}
            : std::vector<float>{4.25f};
        TTree waveforms("waveforms", "waveforms");
        waveforms.Branch("event_id", &event_id);
        waveforms.Branch("telescope_id", &telescope_id);
        waveforms.Branch("n_pixels_camera", &n_pixels_camera);
        waveforms.Branch("n_time_bins", &n_time_bins);
        waveforms.Branch("pixel_id", &waveform_pixels);
        waveforms.Branch("time_bin", &time_bin);
        if (waveform_case == WaveformCase::Complete) {
            waveforms.Branch("time_bin_u32", &time_bin_u32);
        }
        waveforms.Branch(measured_mv ? "sample_value" : "pe", &values);
        if (waveform_case == WaveformCase::Complete || measured_mv) {
            waveforms.Fill();
            if (waveform_case == WaveformCase::Complete) {
                event_id = 101;
                values = {1.5f};
                waveforms.Fill();
            }
        } else if (waveform_case == WaveformCase::MismatchedArrays) {
            time_bin.clear();
            waveforms.Fill();
        }
        waveforms.Write();
    }

    file.Close();
}

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void requireThrows(Function&& function, const std::string& expected_text)
{
    try {
        function();
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find(expected_text) != std::string::npos,
                "unexpected error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected error containing: " + expected_text);
}

} // namespace

int main()
{
    const auto base = std::filesystem::temp_directory_path();
    const auto complete = base / "pylast_lact_complete.root";
    const auto no_waveforms = base / "pylast_lact_no_waveforms.root";
    const auto missing = base / "pylast_lact_missing_waveform.root";
    const auto mismatched = base / "pylast_lact_mismatched_waveform.root";
    const auto measured = base / "pylast_lact_measured_mv.root";
    const auto missing_calibration =
        base / "pylast_lact_measured_mv_missing_calibration.root";

    try {
        SimulatedCamera default_camera;
        require(default_camera.true_image_sum == 0,
                "default simulated-camera image sum must be zero");
        require(std::isnan(default_camera.impact_parameter),
                "default simulated-camera impact must be unavailable");
        require(std::isnan(default_camera.time_range_10_90),
                "default simulated-camera time range must be unavailable");

        writeFixture(complete, WaveformCase::Complete);
        writeFixture(no_waveforms, WaveformCase::NoTree);
        writeFixture(missing, WaveformCase::MissingTriggered);
        writeFixture(mismatched, WaveformCase::MismatchedArrays);
        writeFixture(measured, WaveformCase::MeasuredMv);
        writeFixture(missing_calibration,
                     WaveformCase::MeasuredMvMissingCalibration);

        LactEventSource complete_source(complete.string());
        require(complete_source.event_count() == 2, "complete fixture event count");
        auto first = complete_source.get_event(0);
        auto second = complete_source.get_event(1);
        require(first.event_id == 100, "first indexed event id");
        require(second.event_id == 101, "second indexed event id");
        require(first.simulation.has_value(), "first simulation container");
        require(first.simulation->shower_event_id == 1 &&
                    first.simulation->array_id == 2 &&
                    std::abs(first.simulation->array_time_offset_ns - 3.5) <
                        1.0e-12 &&
                    std::abs(first.simulation->area_weight_m2 - 1250.0) <
                        1.0e-12 &&
                    first.simulation->has_explicit_area_weight,
                "CORSIKA array identity and area weight must reach the event");
        require(std::abs(first.simulation->tels.at(0)->impact_parameter - 37.25) <
                    1.0e-12,
                "observation impact parameter must reach SimulatedCamera");
        require(first.simulation->tels.at(0)->true_image_sum ==
                    first.simulation->tels.at(0)->true_image.sum(),
                "true-image total must equal the stored per-pixel values");
        require(std::abs(first.simulation->tels.at(0)->true_image_pe[0] - 5.25) <
                    1.0e-12 &&
                    std::abs(first.simulation->tels.at(0)->true_image_pe.sum() -
                             5.25) < 1.0e-12,
                "fractional expectation truth must remain available in p.e.");
        require(first.simulation->triggered_tels == std::vector<int>{0},
                "native triggered telescope list");
        require(second.simulation->triggered_tels.empty(),
                "empty trigger list must remain valid");
        require(second.r1->tels.empty(),
                "default source must not expose untriggered readout");
        const auto trigger_timing = complete_source.get_trigger_timing(100);
        require(trigger_timing.size() == 1 && trigger_timing.count(0) == 1,
                "native trigger timing row");
        require(std::abs(trigger_timing.at(0).trigger_time_ns - 12.5) <
                    1.0e-12 &&
                std::abs(trigger_timing.at(0).trigger_first_time_ns - 11.5) <
                    1.0e-12 &&
                std::abs(trigger_timing.at(0).geometric_delay_ns + 2.0) <
                    1.0e-12 &&
                std::abs(trigger_timing.at(0).coincidence_time_ns - 10.5) <
                    1.0e-12 &&
                trigger_timing.at(0).trigger_diagnostics_available,
                "native trigger timing values");
        require(complete_source.get_trigger_timing(101).empty(),
                "non-triggered event must not expose trigger timing");
        const auto observation_timing =
            complete_source.get_observation_timing(100);
        require(observation_timing.size() == 1 &&
                    std::abs(observation_timing.at(0).time_first_ns + 1.0) <
                        1.0e-12 &&
                    std::abs(observation_timing.at(0).time_peak_ns - 0.5) <
                        1.0e-12 &&
                    std::abs(observation_timing.at(0).geometric_delay_ns + 2.0) <
                        1.0e-12,
                "native observation timing values");
        require(complete_source.subarray.has_value(),
                "LACT subarray must be available");
        const auto& geometry = complete_source.subarray->tels.at(0)
                                   .camera_description.camera_geometry;
        require(std::abs(geometry.pix_x[0] - 0.04) < 1.0e-12,
                "LACT v must map to negative pylast pix_x");
        require(std::abs(geometry.pix_y[0] + 0.03) < 1.0e-12,
                "LACT u must map to negative pylast pix_y");
        Calibrator calibrator(*complete_source.subarray);
        calibrator(first);
        require(first.dl0.has_value(),
                "default calibrator must accept LACT p.e. proxy waveforms");
        require(std::abs(first.dl0->tels.at(0)->image[0] - 4.25) < 1.0e-12,
                "LACT proxy waveform must not receive pulse-shape correction");

        LactEventSource readout_source(
            complete.string(), -1, {}, false, true);
        auto untriggered_readout = readout_source.get_event(1);
        require(untriggered_readout.simulation->triggered_tels.empty(),
                "readout switch must not change trigger truth");
        require(untriggered_readout.r1->tels.count(0) == 1 &&
                    std::abs(untriggered_readout.r1->tels.at(0)->waveform.sum() -
                             1.5) < 1.0e-12,
                "readout switch must expose saved untriggered waveform");
        require(readout_source.get_readout_tels(101) == std::vector<int>{0},
                "readout telescope list must be trigger independent");
        require(std::abs(readout_source.get_raw_waveform(101, 0).sum() - 1.5) <
                    1.0e-12,
                "raw waveform accessor must preserve stored samples");

        LactEventSource filtered_source(complete.string(), -1, {1});
        require(filtered_source.event_count() == 0,
                "allowed telescope filter must exclude telescope 0 events");

        LactEventSource no_waveform_source(no_waveforms.string());
        auto no_waveform_event = no_waveform_source.get_event(0);
        require(no_waveform_event.dl0.has_value(),
                "file without waveforms must use integrated images");
        require(std::abs(no_waveform_event.dl0->tels.at(0)->image[0] - 4.25) <
                    1.0e-12,
                "image_pe must map to no-waveform DL0 image");
        require(no_waveform_event.simulation->tels.at(0)->true_image[0] == 5,
                "image_cherenkov_pe must map to simulation true_image");
        LactEventSource no_waveform_readout_source(
            no_waveforms.string(), -1, {}, false, true);
        auto untriggered_image = no_waveform_readout_source.get_event(1);
        require(untriggered_image.dl0->tels.count(0) == 1,
                "readout switch must expose saved untriggered image");

        requireThrows([&]() { LactEventSource source(missing.string()); },
                      "missing waveform");
        requireThrows([&]() { LactEventSource source(mismatched.string()); },
                      "inconsistent LACT ROOT waveform arrays");

        LactEventSource measured_source(measured.string());
        const auto& readout = measured_source.subarray->tels.at(0)
                                  .camera_description.camera_readout;
        require(readout.reference_pulse_shape.rows() == 1 &&
                    readout.reference_pulse_shape.cols() == 6,
                "measured reference pulse must reach CameraReadout");
        require(std::abs(measured_source.get_raw_waveform(100, 0).sum() - 8.0) <
                    1.0e-12,
                "raw mV accessor must precede p.e. calibration");

        LactEventSource baseline_source(
            measured.string(), -1, {}, false, false, 2);
        auto baseline_event = baseline_source.get_event(0);
        require(std::abs(baseline_event.r1->tels.at(0)->waveform.sum() - 0.625) <
                    1.0e-12,
                "optional pre-window baseline must be subtracted before R1");

        auto measured_full_event = measured_source.get_event(0);
        require(std::abs(
                    measured_full_event.r1->tels.at(0)->waveform.sum() - 1.0) <
                    1.0e-12,
                "source must normalize measured mV samples to p.e. charge");
        Calibrator full_calibrator(
            *measured_source.subarray,
            std::string(
                R"({"image_extractor_type":"FullWaveFormExtractor"})"));
        full_calibrator(measured_full_event);
        require(std::abs(
                    measured_full_event.dl0->tels.at(0)->image[0] - 1.0) <
                    1.0e-12,
                "full mV waveform integration must close to one p.e.");

        auto measured_local_event = measured_source.get_event(0);
        Calibrator local_calibrator(
            *measured_source.subarray,
            std::string(
                R"json({"image_extractor_type":"LocalPeakExtractor","LocalPeakExtractor":{"window_width":3,"window_shift":1,"apply_correction":true}})json"));
        local_calibrator(measured_local_event);
        require(std::abs(
                    measured_local_event.dl0->tels.at(0)->image[0] - 1.0) <
                    1.0e-12,
                "local mV integration and pulse-tail correction must close");

        requireThrows(
            [&]() { LactEventSource source(missing_calibration.string()); },
            "single_pe_area_mv_ns");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    std::filesystem::remove(complete);
    std::filesystem::remove(no_waveforms);
    std::filesystem::remove(missing);
    std::filesystem::remove(mismatched);
    std::filesystem::remove(measured);
    std::filesystem::remove(missing_calibration);
    return 0;
}
