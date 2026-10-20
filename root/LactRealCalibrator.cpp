#include "LactRealCalibrator.hh"
#include "TFile.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"
#include "TTreeReaderValue.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

LactRealCalibrator::LactRealCalibrator(const std::string& filename,
                                     const std::string& config_str) {
    const auto config = nlohmann::json::parse(config_str);
    if (!config.is_object()) throw std::invalid_argument("Calibrator config must be an object");
    gain_threshold = config.value("gain_threshold_mv", 36.0);
    high_gain_mv_per_pe = config.value("high_gain_mv_per_pe", 2.5);
    low_gain_pe_per_mv = config.value("low_gain_pe_per_mv", 10.2);
    // The measured LED event_time retains Rabbit TAI seconds, just like the
    // event builder. Convert both clocks to UTC before selecting a row.
    // A calibration file already expressed in UTC must configure zero.
    led_time_offset_s = config.value("led_time_offset_s", -37.0);
    if (!std::isfinite(gain_threshold) || gain_threshold <= 0 ||
        !std::isfinite(high_gain_mv_per_pe) || high_gain_mv_per_pe <= 0 ||
        !std::isfinite(low_gain_pe_per_mv) || low_gain_pe_per_mv <= 0)
        throw std::invalid_argument("Gain threshold and nominal conversion factors must be positive");
    if (!std::isfinite(led_time_offset_s))
        throw std::invalid_argument("led_time_offset_s must be finite");
    if (!filename.empty()) load_calibration_file(filename);
    else spdlog::warn("Measured calibration uses nominal peak-to-p.e. conversion; no LED area calibration loaded");
}

void LactRealCalibrator::load_calibration_file(const std::string& filename) {
    const auto path = filename.rfind("/eos/", 0) == 0
        ? "root://eos01.ihep.ac.cn/" + filename : filename;
    std::unique_ptr<TFile> file(TFile::Open(path.c_str(), "READ"));
    if (!file || file->IsZombie()) throw std::runtime_error("Cannot open LED calibration: " + filename);
    auto* tree = dynamic_cast<TTree*>(file->Get("led_cal"));
    if (!tree) throw std::runtime_error("LED calibration requires led_cal TTree");
    for (const auto* branch : {"event_time", "GainFactor_AreaH", "GainFactor_AreaL"})
        if (!tree->GetBranch(branch)) throw std::runtime_error(std::string("Missing LED branch: ") + branch);
    TTreeReader reader(tree);
    TTreeReaderValue<double> time(reader, "event_time");
    TTreeReaderArray<double> high(reader, "GainFactor_AreaH"), low(reader, "GainFactor_AreaL");
    std::vector<double> new_times;
    std::vector<Eigen::VectorXd> new_high, new_low;
    while (reader.Next()) {
        if (!std::isfinite(*time) || high.GetSize() == 0 || high.GetSize() != low.GetSize() ||
            (!new_high.empty() && high.GetSize() != static_cast<std::size_t>(new_high.front().size())))
            throw std::runtime_error("Invalid LED time or gain-factor dimensions");
        const double mjd = 40587.0 + (*time + led_time_offset_s) / 86400.0;
        if (!new_times.empty() && mjd < new_times.back())
            throw std::runtime_error("LED calibration times must be sorted");
        new_times.push_back(mjd);
        Eigen::VectorXd h(high.GetSize()), l(low.GetSize());
        for (std::size_t pixel = 0; pixel < high.GetSize(); ++pixel) {
            h(pixel) = high[pixel]; l(pixel) = low[pixel];
        }
        new_high.push_back(std::move(h)); new_low.push_back(std::move(l));
    }
    if (new_times.empty()) throw std::runtime_error("LED calibration contains no entries");
    times.swap(new_times); high_factors.swap(new_high); low_factors.swap(new_low);
}

std::size_t LactRealCalibrator::calibration_index(double mjd) const {
    if (!std::isfinite(mjd) || mjd < times.front() || mjd > times.back())
        throw std::runtime_error("Event time is outside LED calibration coverage");
    const auto it = std::lower_bound(times.begin(), times.end(), mjd);
    if (it == times.begin()) return 0;
    if (it == times.end()) return times.size() - 1;
    const auto hi = static_cast<std::size_t>(it - times.begin());
    return mjd - times[hi - 1] <= times[hi] - mjd ? hi - 1 : hi;
}

void LactRealCalibrator::operator()(ArrayEvent& event) const {
    if (!event.c1) throw std::runtime_error("LactRealCalibrator requires measured c1 data");
    const bool led = has_calibration();
    const std::size_t index = led ? calibration_index(event.time_mjd) : 0;
    event.dl0.emplace();
    event.r1.reset();
    for (auto& [tel_id, pointer] : event.c1->tels) {
        auto& camera = *pointer;
        const auto count = camera.peak.cols();
        if (led && high_factors[index].size() != count)
            throw std::runtime_error("LED calibration pixel count does not match mapping");
        camera.pe = CalibCamera::Matrix::Zero(2, count);
        camera.count_to_pe = CalibCamera::Matrix::Zero(2, count);
        DL0Camera output;
        output.image = Eigen::VectorXd::Zero(count);
        output.peak_time = Eigen::VectorXd::Constant(count, std::numeric_limits<double>::quiet_NaN());
        R1Camera waveform;
        // Nominal peak-to-p.e. factors do not calibrate integrated sample
        // charge. Keep raw samples in c1 until LED area factors are available.
        const bool have_waveforms = led && camera.waveform[0].cols() > 0;
        if (have_waveforms) {
            waveform.n_pixels = count;
            waveform.n_samples = camera.waveform[0].cols();
            waveform.waveform = decltype(waveform.waveform)::Zero(count, waveform.n_samples);
            waveform.gain_selection = Eigen::VectorXi::Constant(count, -1);
        }
        for (int pixel = 0; pixel < count; ++pixel) {
            bool usable[2];
            for (int gain = 0; gain < 2; ++gain) {
                const double factor = led ? (gain == 0 ? high_factors[index](pixel) : low_factors[index](pixel))
                    : (gain == 0 ? high_gain_mv_per_pe : 1.0 / low_gain_pe_per_mv);
                usable[gain] = camera.valid(gain, pixel) && std::isfinite(factor) && factor > 0;
                if (usable[gain]) {
                    camera.count_to_pe(gain, pixel) = 1.0 / factor;
                    camera.pe(gain, pixel) = (led ? camera.area(gain, pixel) : camera.peak(gain, pixel)) / factor;
                }
            }
            int gain = camera.peak(1, pixel) > gain_threshold ? 1 : 0;
            if (!usable[gain]) gain = 1 - gain;
            if (!usable[gain]) continue; // Missing/bad channels remain masked by zero charge.
            output.image(pixel) = camera.pe(gain, pixel);
            output.peak_time(pixel) = camera.peak_time(gain, pixel);
            if (have_waveforms) {
                waveform.gain_selection(pixel) = gain;
                waveform.waveform.row(pixel) =
                    (camera.waveform[gain].row(pixel).array() - camera.base(gain, pixel)) * camera.count_to_pe(gain, pixel);
            }
        }
        if (have_waveforms) {
            if (!event.r1) event.r1.emplace();
            event.r1->add_tel(tel_id, std::move(waveform));
        }
        event.dl0->add_tel(tel_id, std::move(output));
    }
}
