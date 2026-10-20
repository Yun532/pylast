#pragma once

#include "EventSource.hh"
#include "nlohmann_json/json.hpp"
#include "TFile.h"
#include "TTree.h"
#include <map>
#include <memory>

// Native reader for measured eventShow summaries and raw waveforms TTrees.
class LactRealEventSource : public EventSource {
public:
    LactRealEventSource(const std::string& filename,
                        const std::string& mapping_file,
                        int64_t max_events = -1,
                        const std::string& config_str = "{}");
    ~LactRealEventSource() override = default;
    void load_all_simulated_showers() override {}
    void set_pointing(ArrayEvent& event, double altitude_deg,
                      double azimuth_deg) const;
    std::string input_format() const { return raw_waveforms ? "waveforms" : "eventShow"; }
    int telescope_id() const { return tel_id; }

protected:
    void open_file() override;
    void init_subarray() override;
    void init_metaparam() override {}
    void init_atmosphere_model() override {}
    void init_simulation_config() override {}
    bool is_finished() override { return current_event_index >= max_events; }
    ArrayEvent get_event() override {
        return is_finished() ? ArrayEvent{} : get_event(current_event_index);
    }
    ArrayEvent get_event(int index) override;

private:
    std::string mapping_file;
    nlohmann::json config;
    std::unique_ptr<TFile> file;
    std::unique_ptr<TFile> analysis_file;
    TTree* tree = nullptr;
    TTree* summary_tree = nullptr;
    std::map<int, Long64_t> summary_entries;
    bool raw_waveforms = false;
    int tel_id = 0;
    int n_pixels = 0;
    int n_samples = 0;
    double sampling_interval_ns = 0.0;
    std::map<std::pair<int, int>, int> channel_to_pixel;
    std::vector<std::vector<Long64_t>> event_rows;
    void read_summaries(Long64_t entry, CalibCamera& camera, ArrayEvent& event);
    void read_waveforms(const std::vector<Long64_t>& rows,
                        CalibCamera& camera, ArrayEvent& event);
    void set_time(ArrayEvent& event, unsigned int seconds,
                  unsigned int ticks) const;
};
