/**
 * @file LactEventSource.hh
 * @brief EventSource adapter for LACT_sim lact_event_root_v1 files.
 */

#pragma once

#include "ArrayEvent.hh"
#include "EventSource.hh"
#include "SimulatedShowerArray.hh"
#include "SubarrayDescription.hh"

#include "TFile.h"

#include <Eigen/Dense>

#include <array>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

class LactEventSource : public EventSource
{
public:
    struct TriggerTimingRow {
        double trigger_time_ns = std::numeric_limits<double>::quiet_NaN();
        double trigger_first_time_ns = std::numeric_limits<double>::quiet_NaN();
        double trigger_max_multiplicity_time_ns =
            std::numeric_limits<double>::quiet_NaN();
        double geometric_delay_ns = std::numeric_limits<double>::quiet_NaN();
        double coincidence_time_ns = std::numeric_limits<double>::quiet_NaN();
        bool trigger_diagnostics_available = false;
    };

    struct ObservationTimingRow {
        double reference_time_ns = std::numeric_limits<double>::quiet_NaN();
        double time_first_ns = std::numeric_limits<double>::quiet_NaN();
        double time_mean_ns = std::numeric_limits<double>::quiet_NaN();
        double time_rms_ns = std::numeric_limits<double>::quiet_NaN();
        double time_peak_ns = std::numeric_limits<double>::quiet_NaN();
        double geometric_delay_ns = std::numeric_limits<double>::quiet_NaN();
    };

    LactEventSource(const std::string& filename,
                    int64_t max_events = -1,
                    std::vector<int> subarray = {},
                    bool load_simulated_showers = false,
                    bool read_untriggered = false,
                    int baseline_samples = 0);
    ~LactEventSource() override;

    void open_file() override;
    void init_metaparam() override;
    void init_atmosphere_model() override;
    void init_subarray() override;
    void init_simulation_config() override;
    void load_all_simulated_showers() override;
    ArrayEvent get_event() override;
    ArrayEvent get_event(int index) override;
    bool is_finished() override;
    const SimulatedShowerArray& get_shower_array()
    {
        if (!shower_array.has_value()) {
            load_all_simulated_showers();
        }
        return *shower_array;
    }
    std::size_t event_count() const { return event_order.size(); }
    std::map<int, TriggerTimingRow> get_trigger_timing(long long event_id) const;
    std::map<int, ObservationTimingRow>
    get_observation_timing(long long event_id) const;
    std::vector<int> get_readout_tels(long long event_id) const;
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor>
    get_raw_waveform(long long event_id, int telescope_id) const;

private:
    struct CameraPixelRow {
        int pixel_id = 0;
        double x_m = 0.0;
        double y_m = 0.0;
        double size_m = 0.0;
        int shape_code = 0;
    };

    struct OpticsRow {
        std::string name = "LACT";
        int num_mirrors = 0;
        double mirror_area_m2 = 0.0;
        double equivalent_focal_length_m = 0.0;
        double effective_focal_length_m = 0.0;
    };

    struct TelescopeRow {
        int telescope_id = 0;
        std::string name = "LACT";
        std::array<double, 3> position = {0.0, 0.0, 0.0};
        double pointing_az_deg = 0.0;
        double pointing_el_deg = 90.0;
    };

    struct CorsikaEventRow {
        long long event_id = 0;
        int shower_event_id = 0;
        int array_id = 0;
        double array_time_offset_ns =
            std::numeric_limits<double>::quiet_NaN();
        double area_weight_m2 = std::numeric_limits<double>::quiet_NaN();
        bool has_explicit_area_weight = false;
        int run_id = 0;
        int primary_type = 0;
        double energy_gev = 0.0;
        double altitude_deg = 0.0;
        double azimuth_north_to_east_deg = 0.0;
        double core_x_north_m = 0.0;
        double core_y_west_m = 0.0;
        double h_first_int_m = 0.0;
        double x_max_g_cm2 = 0.0;
        double h_max_m = 0.0;
        double starting_grammage_g_cm2 = 0.0;
    };

    struct ObservationRow {
        long long event_id = 0;
        int telescope_id = 0;
        bool triggered = false;
        int n_pixels_camera = 0;
        double impact_parameter_m =
            std::numeric_limits<double>::quiet_NaN();
        std::vector<int> pixel_id;
        std::vector<float> image_pe;
        std::vector<float> image_cherenkov_pe;
        std::vector<float> image_time_peak_ns;
        ObservationTimingRow observation_timing;
        TriggerTimingRow trigger_timing;
    };

    struct WaveformRow {
        long long event_id = 0;
        int telescope_id = 0;
        int n_pixels_camera = 0;
        int n_time_bins = 0;
        std::vector<int> pixel_id;
        std::vector<unsigned int> time_bin;
        std::vector<float> sample_value;
    };

    struct WaveformConfig {
        bool available = false;
        int n_time_bins = 1;
        double time_bin_width_ns = 1.0;
        std::string sample_unit;
        double single_pe_area_mv_ns = 0.0;
        std::string template_time_reference;
        std::vector<double> time_centers_ns;
        std::vector<double> reference_pulse_time_ns;
        std::vector<double> reference_pulse_amplitude;
    };

    void load_schema();
    void load_camera_pixels();
    void load_optics();
    void load_telescopes();
    void load_corsika_events();
    void load_observations();
    void load_waveforms();
    void validate_waveforms() const;
    void build_event_order();
    bool keep_tel(int tel_id) const;
    int pixel_index(int pixel_id) const;
    Eigen::VectorXd dense_image(const ObservationRow& obs) const;
    Eigen::VectorXd dense_cherenkov_image(const ObservationRow& obs) const;
    Eigen::VectorXd dense_peak_time(const ObservationRow& obs) const;
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor>
    dense_waveform(const ObservationRow& obs) const;
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor>
    dense_raw_waveform(const ObservationRow& obs) const;

    std::unique_ptr<TFile> file;
    std::string schema_name;
    int schema_version = 0;
    std::string profile;
    int run_id = 0;
    std::vector<CameraPixelRow> camera_pixels;
    std::unordered_map<int, int> pixel_id_to_index;
    OpticsRow optics;
    std::vector<TelescopeRow> telescopes;
    std::unordered_map<long long, CorsikaEventRow> corsika_by_event;
    std::vector<ObservationRow> observations;
    std::map<std::pair<long long, int>, std::size_t> observation_index;
    std::unordered_map<long long, std::vector<std::size_t>> observation_indices_by_event;
    std::map<std::pair<long long, int>, WaveformRow> waveforms;
    WaveformConfig waveform_config;
    bool has_waveform_tree = false;
    bool has_trigger_first_time = false;
    bool has_trigger_max_multiplicity_time = false;
    bool has_geometric_delay = false;
    bool has_coincidence_time = false;
    bool read_untriggered = false;
    int baseline_samples = 0;
    std::vector<long long> event_order;
};
