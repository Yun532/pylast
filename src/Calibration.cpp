#include "Calibration.hh"
#include "Configurable.hh"
#include "DL0Event.hh"
#include <cmath>
#include <stdexcept>

Eigen::VectorXi select_gain_channel_by_threshold(const std::array<Eigen::Matrix<uint16_t, -1, -1, Eigen::RowMajor>, 2>& waveform, const double threshold)
{
    // Matrix is (n_pixels, n_samples)
    // Vector returned is (n_pixels)
    if(waveform[1].isZero())
    {
        return Eigen::VectorXi::Zero(waveform[0].rows());
    }
    else 
    {
        // If the high gain channel exceeds the threshold, select the low gain channel
        Eigen::VectorXi gain_selector = Eigen::VectorXi::Zero(waveform[0].rows());
        for(int i = 0; i < waveform[0].rows(); i++)
        {
            if((waveform[0].row(i).array() > threshold).any())
            {
                gain_selector(i) = 1;
            }
        }
        return gain_selector;
    }
}

json Calibrator::get_default_config()
{
    std::string default_config = R"(
    {
        "image_extractor_type": "LocalPeakExtractor"
    }
    )";
    json base_config = Configurable::from_string(default_config);
    base_config["LocalPeakExtractor"] = LocalPeakExtractor::get_default_config();
    return base_config;
}

void Calibrator::configure(const json &config)
{
    try {
        const json& cfg = config.contains("Calibrator") ? config.at("Calibrator") : config;
        image_extractor_type = cfg["image_extractor_type"];
        if(image_extractor_type == "LocalPeakExtractor")
        {
            image_extractor = ImageExtractorFactory::create<LocalPeakExtractor>(subarray, cfg);
        }
        else if(image_extractor_type == "FullWaveFormExtractor")
        {
            image_extractor = ImageExtractorFactory::create<FullWaveFormExtractor>(subarray);
        }
        else {
            throw std::runtime_error("Unknow ImageExtractor type: " + image_extractor_type);
        }
    }
    catch(const std::exception& e) {
        throw std::runtime_error("Error configuring Calibrator: " + std::string(e.what()));
    }
}
void Calibrator::operator()(ArrayEvent& event)
{
    if(!event.dl0)
    {
        event.dl0 = DL0Event();
    }
    for(const auto& [tel_id, r1_camera]: event.r1->tels)
    {
        auto [charge, peak_time] = (*image_extractor)(r1_camera->waveform, r1_camera->gain_selection, tel_id);
        charge *= waveform_sum_to_pe_scale(tel_id);
        event.dl0->add_tel(tel_id, DL0Camera{.image = std::move(charge), .peak_time=std::move(peak_time)});
    }
}

double Calibrator::waveform_sum_to_pe_scale(int tel_id) const
{
    const auto& readout =
        subarray.tels.at(tel_id).camera_description.camera_readout;
    const auto& unit = readout.waveform_sample_unit;
    if (unit.empty() || unit == "pe" || unit == "pe_per_sample" ||
        unit == "fired_pe_per_sample" || unit == "pe_charge_per_sample") {
        return 1.0;
    }
    if (unit != "mV") {
        throw std::runtime_error(
            "Calibrator does not know how to convert waveform sample unit '" +
            unit + "' to p.e.");
    }
    if (!std::isfinite(readout.sampling_rate) ||
        readout.sampling_rate <= 0.0) {
        throw std::runtime_error(
            "mV waveform calibration requires a positive sampling rate");
    }
    if (!std::isfinite(readout.single_pe_area_mv_ns) ||
        readout.single_pe_area_mv_ns <= 0.0) {
        throw std::runtime_error(
            "mV waveform calibration requires single_pe_area_mv_ns > 0");
    }
    const double sample_width_ns = 1.0 / readout.sampling_rate;
    return sample_width_ns / readout.single_pe_area_mv_ns;
}
