#include "LactRealEventSource.hh"
#include "LactWaveformExtractor.hh"
#include "TLeaf.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"
#include "TTreeReaderValue.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <fstream>
#include <sstream>

namespace {
constexpr double missing_value = std::numeric_limits<double>::quiet_NaN();

void require_branch(TTree* tree, const char* name) {
    if (!tree->GetBranch(name))
        throw std::runtime_error(std::string("Missing measured-data branch: ") + name);
}

void checked_entry(TTreeReader& reader, Long64_t entry) {
    if (reader.SetEntry(entry) != TTreeReader::kEntryValid)
        throw std::runtime_error("Cannot read measured-data entry " + std::to_string(entry));
}

std::vector<std::string> csv_fields(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') { field += '"'; ++i; }
            else quoted = !quoted;
        } else if (c == ',' && !quoted) { fields.push_back(field); field.clear(); }
        else if (c != '\r') field += c;
    }
    if (quoted) throw std::runtime_error("Unterminated quote in mapping CSV");
    fields.push_back(field);
    return fields;
}

}

LactRealEventSource::LactRealEventSource(const std::string& filename,
        const std::string& mapping_file, int64_t requested_events,
        const std::string& config_str)
    : EventSource(filename), mapping_file(mapping_file),
      config(nlohmann::json::parse(config_str)) {
    if (!config.is_object() || requested_events < -1)
        throw std::invalid_argument("Source config must be an object; max_events must be >= -1");
    max_events = requested_events;
    load_simulated_showers = false;
    tel_id = config.value("tel_id", 0);
    // Rabbit ticks are 8 ns; waveform sampling is independent and must be
    // specified by the acquisition configuration. Zero means unknown.
    sampling_interval_ns = config.value("sampling_interval_ns", 0.0);
    if (tel_id < 0 || !std::isfinite(sampling_interval_ns) || sampling_interval_ns < 0)
        throw std::invalid_argument("tel_id and sampling_interval_ns must be nonnegative");
    initialize();
}

void LactRealEventSource::open_file() {
    const auto path = input_filename.rfind("/eos/", 0) == 0
        ? "root://eos01.ihep.ac.cn/" + input_filename : input_filename;
    file.reset(TFile::Open(path.c_str(), "READ"));
    if (!file || file->IsZombie())
        throw std::runtime_error("Cannot open measured ROOT: " + input_filename);
    tree = dynamic_cast<TTree*>(file->Get("eventShow"));
    if (!tree) {
        tree = dynamic_cast<TTree*>(file->Get("waveforms"));
        raw_waveforms = true;
    }
    if (!tree)
        throw std::runtime_error("Measured ROOT requires eventShow or waveforms TTree");
    for (const auto* name : {"event", "RabbitTime", "Rabbittime", "HG_LG"})
        require_branch(tree, name);
    if (raw_waveforms) {
        for (const auto* name : {"waveform", "FEE_ID", "channel"})
            require_branch(tree, name);
        auto* leaf = tree->GetLeaf("waveform");
        if (!leaf || leaf->GetLenStatic() <= 0)
            throw std::runtime_error("waveform must be a fixed-size float array");
        n_samples = leaf->GetLenStatic();
        TTreeReader reader(tree);
        TTreeReaderValue<int> id(reader, "event");
        std::map<int, std::size_t> event_indices;
        while (reader.Next()) {
            const auto row = reader.GetCurrentEntry();
            const auto [entry, inserted] = event_indices.emplace(*id, event_rows.size());
            if (inserted) event_rows.emplace_back();
            // The measured builder orders rows by FEE and then by event;
            // channels of a single event can be interleaved across the file.
            event_rows[entry->second].push_back(row);
        }
    } else {
        for (const auto* name : {"map_number", "base", "amp", "area"})
            require_branch(tree, name);
        for (Long64_t row = 0; row < tree->GetEntries(); ++row)
            event_rows.push_back({row});
    }
    summary_tree = raw_waveforms ? nullptr : tree;
    const std::string analysis_path = config.value("analysis_file", std::string{});
    if (!analysis_path.empty()) {
        if (!raw_waveforms) throw std::invalid_argument("analysis_file is only used with raw waveforms");
        const auto path = analysis_path.rfind("/eos/", 0) == 0
            ? "root://eos01.ihep.ac.cn/" + analysis_path : analysis_path;
        analysis_file.reset(TFile::Open(path.c_str(), "READ"));
        if (!analysis_file || analysis_file->IsZombie())
            throw std::runtime_error("Cannot open paired analysis_file: " + analysis_path);
        summary_tree = dynamic_cast<TTree*>(analysis_file->Get("eventShow"));
        if (!summary_tree) throw std::runtime_error("analysis_file requires eventShow TTree");
        TTreeReader reader(summary_tree);
        TTreeReaderValue<int> id(reader, "event");
        while (reader.Next()) {
            if (!summary_entries.emplace(*id, reader.GetCurrentEntry()).second)
                throw std::runtime_error("Duplicate event number in paired analysis_file");
        }
    }
    const auto count = static_cast<int64_t>(event_rows.size());
    max_events = max_events < 0 ? count : std::min(max_events, count);
}

void LactRealEventSource::init_subarray() {
    std::ifstream input(mapping_file);
    if (!input) throw std::runtime_error("Cannot open measured camera mapping: " + mapping_file);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("Empty mapping CSV");
    if (line.rfind("\xef\xbb\xbf", 0) == 0) line.erase(0, 3);
    const auto header = csv_fields(line);
    std::map<std::string, std::size_t> columns;
    for (std::size_t index = 0; index < header.size(); ++index) columns[header[index]] = index;
    for (const auto* name : {"map_number", "s_number", "s_x", "s_y", "fee_id"})
        if (!columns.contains(name)) throw std::runtime_error(std::string("Missing mapping column: ") + name);
    std::map<int, std::pair<double, double>> positions;
    int map_number, serial, fee;
    double sx, sy;
    while (std::getline(input, line)) {
        if (line.empty() || line == "\r") continue;
        const auto fields = csv_fields(line);
        if (fields.size() != header.size()) throw std::runtime_error("Inconsistent mapping CSV column count");
        map_number = std::stoi(fields.at(columns.at("map_number")));
        serial = std::stoi(fields.at(columns.at("s_number")));
        fee = std::stoi(fields.at(columns.at("fee_id")));
        sx = std::stod(fields.at(columns.at("s_x")));
        sy = std::stod(fields.at(columns.at("s_y")));
        const int channel = (serial - 1) % 16 + 1;
        if (map_number < 1 || serial < 1 || fee < 0 ||
                !std::isfinite(sx) || !std::isfinite(sy) ||
                !positions.emplace(map_number, std::make_pair(sx, sy)).second ||
                !channel_to_pixel.emplace(std::make_pair(fee, channel), map_number - 1).second)
            throw std::runtime_error("Invalid or duplicate pixel/electronics entry in mapping CSV");
    }
    n_pixels = static_cast<int>(positions.size());
    if (n_pixels < 9 || positions.begin()->first != 1 || positions.rbegin()->first != n_pixels)
        throw std::runtime_error("Mapping must contain contiguous pixel numbers 1..N (N>=9)");
    Eigen::VectorXd x(n_pixels), y(n_pixels);
    for (const auto& [pixel, xy] : positions) {
        // Hardware s_x is horizontal and s_y is upward on the focal plane.
        // Convert once to pylast sky-offset coordinates, as LactEventSource does.
        x(pixel - 1) = -xy.second / 1000.0;
        y(pixel - 1) = -xy.first / 1000.0;
    }
    const double focal = config.value("focal_length_m", 8.0);
    const double width = config.value("pixel_width_m", 0.0254);
    if (!std::isfinite(focal) || focal <= 0 || !std::isfinite(width) || width <= 0)
        throw std::invalid_argument("focal_length_m and pixel_width_m must be positive");
    CameraGeometry geometry("LACT1", n_pixels, x, y,
        Eigen::VectorXd::Constant(n_pixels, width * width),
        Eigen::VectorXi::Constant(n_pixels, 2), 0.0);
    geometry.pix_x_fov = geometry.pix_x / focal;
    geometry.pix_y_fov = geometry.pix_y / focal;
    geometry.pix_width_fov = geometry.pix_width / focal;
    CameraReadout readout{};
    readout.camera_name = "LACT1";
    readout.n_pixels = n_pixels;
    readout.n_samples = n_samples;
    readout.n_channels = 2;
    readout.sampling_rate = sampling_interval_ns > 0 ? 1.0 / sampling_interval_ns : 0.0;
    readout.reference_pulse_sample_width = sampling_interval_ns;
    TelescopeDescription telescope;
    telescope.tel_name = "LACT1";
    telescope.camera_description = {"LACT1", std::move(geometry), std::move(readout)};
    telescope.optics_description = {"LACT1", 1, config.value("mirror_area_m2", 1.0), focal, focal};
    // Prototype position used by the original KM2A/WCDA plotter, converted
    // from KM2A (-144.645,-28.7345) to pylast ground (North,West,Up).
    const auto location = config.value("telescope_position_m",
        std::vector<double>{-144.942718773876, -27.192936393245507, 0});
    if (location.size() != 3 || !std::all_of(location.begin(), location.end(),
                                           [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("telescope_position_m must contain three finite coordinates");
    subarray.emplace();
    subarray->reference_position = {0, 0, 0};
    subarray->add_telescope(tel_id, std::move(telescope), {location[0], location[1], location[2]});
}

void LactRealEventSource::set_time(ArrayEvent& event, unsigned int seconds,
                                  unsigned int ticks) const {
    if (ticks >= 125000000u)
        throw std::runtime_error("Rabbit fractional time exceeds one second (8 ns ticks)");
    event.rabbit_time = seconds;
    event.rabbit_tick = ticks;
    const auto utc_seconds = static_cast<int64_t>(seconds) - config.value("tai_minus_utc_s", 37);
    event.time_mjd = 40587.0 + utc_seconds / 86400.0 + ticks * 8e-9 / 86400.0;
}

void LactRealEventSource::set_pointing(ArrayEvent& event, double altitude,
                                      double azimuth) const {
    if (!std::isfinite(altitude) || altitude < -90 || altitude > 90 || !std::isfinite(azimuth))
        throw std::invalid_argument("Pointing requires finite altitude [-90,90] and azimuth in degrees");
    const double az = std::fmod(std::fmod(azimuth, 360.0) + 360.0, 360.0) * M_PI / 180.0;
    const double alt = altitude * M_PI / 180.0;
    event.pointing.emplace();
    event.pointing->set_array_pointing(az, alt);
    event.pointing->add_tel(tel_id, az, alt);
}

ArrayEvent LactRealEventSource::get_event(int index) {
    if (index < 0 || index >= max_events) throw std::out_of_range("Measured event index out of range");
    ArrayEvent event;
    event.run_id = config.value("run_id", 0);
    CalibCamera camera;
    camera.base = camera.peak = camera.area = camera.peak_sample = camera.peak_time =
        CalibCamera::Matrix::Constant(2, n_pixels, missing_value);
    camera.valid = decltype(camera.valid)::Constant(2, n_pixels, false);
    camera.sampling_interval_ns = sampling_interval_ns;
    const auto& rows = event_rows.at(index);
    if (raw_waveforms) {
        read_waveforms(rows, camera, event);
        if (summary_tree) {
            const auto found = summary_entries.find(event.event_id);
            if (found == summary_entries.end())
                throw std::runtime_error("Raw event is absent from paired analysis_file");
            const auto original_id = event.event_id;
            const auto original_led = event.is_led;
            const auto seconds = event.rabbit_time, ticks = event.rabbit_tick;
            const auto raw_valid = camera.valid;
            camera.valid.setConstant(false);
            read_summaries(found->second, camera, event);
            if (event.event_id != original_id || event.rabbit_time != seconds || event.rabbit_tick != ticks ||
                (original_led && event.is_led && original_led != event.is_led) ||
                (camera.valid != raw_valid).any())
                throw std::runtime_error("Paired analysis_file disagrees on event/time/LED flag/channel coverage");
        }
    } else read_summaries(rows.front(), camera, event);
    event.c1.emplace();
    event.c1->add_tel(tel_id, std::move(camera));
    return event;
}

void LactRealEventSource::read_summaries(Long64_t entry, CalibCamera& camera, ArrayEvent& event) {
    TTreeReader reader(summary_tree);
    TTreeReaderValue<int> id(reader, "event");
    TTreeReaderValue<unsigned int> seconds(reader, "RabbitTime"), ticks(reader, "Rabbittime");
    TTreeReaderArray<int> pixels(reader, "map_number");
    TTreeReaderArray<bool> gains(reader, "HG_LG");
    TTreeReaderArray<float> bases(reader, "base"), peaks(reader, "amp"), areas(reader, "area");
    std::unique_ptr<TTreeReaderArray<short>> peak_samples;
    if (summary_tree->GetBranch("PeakPos")) peak_samples = std::make_unique<TTreeReaderArray<short>>(reader, "PeakPos");
    std::unique_ptr<TTreeReaderValue<bool>> led;
    if (summary_tree->GetBranch("LED_event")) led = std::make_unique<TTreeReaderValue<bool>>(reader, "LED_event");
    checked_entry(reader, entry);
    event.event_id = *id;
    if (led) event.is_led = **led;
    set_time(event, *seconds, *ticks);
    const auto n = pixels.GetSize();
    auto seen = decltype(camera.valid)::Constant(2, n_pixels, false).eval();
    if (gains.GetSize() != n || bases.GetSize() != n || peaks.GetSize() != n ||
            areas.GetSize() != n || (peak_samples && peak_samples->GetSize() != n))
        throw std::runtime_error("eventShow channel vectors have inconsistent lengths");
    for (std::size_t row = 0; row < n; ++row) {
        const int pixel = pixels[row] - 1, gain = gains[row] ? 0 : 1;
        if (pixel < 0 || pixel >= n_pixels || seen(gain, pixel))
            throw std::runtime_error("Invalid or duplicate pixel/gain in eventShow");
        seen(gain, pixel) = true;
        camera.base(gain, pixel) = bases[row];
        camera.peak(gain, pixel) = peaks[row];
        camera.area(gain, pixel) = areas[row];
        if (peak_samples) {
            camera.peak_sample(gain, pixel) = (*peak_samples)[row];
            camera.peak_time(gain, pixel) = sampling_interval_ns > 0
                ? (*peak_samples)[row] * sampling_interval_ns : missing_value;
        }
        camera.valid(gain, pixel) = std::isfinite(bases[row]) && std::isfinite(peaks[row]) && std::isfinite(areas[row]);
    }
}

void LactRealEventSource::read_waveforms(const std::vector<Long64_t>& rows,
                                        CalibCamera& camera, ArrayEvent& event) {
    for (auto& waveform : camera.waveform)
        waveform = CalibCamera::Matrix::Constant(n_pixels, n_samples, missing_value);
    TTreeReader reader(tree);
    TTreeReaderValue<int> id(reader, "event");
    TTreeReaderValue<unsigned int> seconds(reader, "RabbitTime"), ticks(reader, "Rabbittime");
    TTreeReaderValue<unsigned short> fee(reader, "FEE_ID");
    TTreeReaderValue<unsigned char> channel(reader, "channel");
    TTreeReaderValue<bool> high_gain(reader, "HG_LG");
    TTreeReaderArray<float> values(reader, "waveform");
    std::unique_ptr<TTreeReaderValue<bool>> led;
    if (tree->GetBranch("LED_event")) led = std::make_unique<TTreeReaderValue<bool>>(reader, "LED_event");
    std::vector<float> waveform(n_samples);
    for (auto row : rows) {
        checked_entry(reader, row);
        if (row == rows.front()) { event.event_id = *id; set_time(event, *seconds, *ticks); }
        if (led) {
            if (event.is_led && event.is_led.value() != **led)
                throw std::runtime_error("Waveform rows disagree on LED_event flag");
            event.is_led = **led;
        }
        if (event.event_id != *id || event.rabbit_time != *seconds || event.rabbit_tick != *ticks)
            throw std::runtime_error("Waveform rows disagree on event number or Rabbit timestamp");
        const auto found = channel_to_pixel.find({*fee, *channel});
        if (found == channel_to_pixel.end())
            throw std::runtime_error("Unmapped FEE/channel in measured waveforms");
        const int pixel = found->second, gain = *high_gain ? 0 : 1;
        if (camera.valid(gain, pixel) || static_cast<int>(values.GetSize()) != n_samples)
            throw std::runtime_error("Duplicate channel/gain or inconsistent waveform length");
        for (int sample = 0; sample < n_samples; ++sample) {
            if (!std::isfinite(values[sample])) throw std::runtime_error("Non-finite measured waveform sample");
            waveform[sample] = values[sample];
            camera.waveform[gain](pixel, sample) = values[sample];
        }
        if (!summary_tree) {
            const auto result = lact_real::extract_waveform(waveform);
            camera.base(gain, pixel) = result.base;
            camera.peak(gain, pixel) = result.amplitude;
            camera.area(gain, pixel) = result.area;
            camera.peak_sample(gain, pixel) = result.peak_sample;
            camera.peak_time(gain, pixel) = sampling_interval_ns > 0
                ? result.peak_sample * sampling_interval_ns : missing_value;
        }
        camera.valid(gain, pixel) = true;
    }
}
