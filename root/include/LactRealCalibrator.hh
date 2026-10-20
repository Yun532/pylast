#pragma once

#include "ArrayEvent.hh"
#include "nlohmann_json/json.hpp"
#include <string>
#include <vector>

class LactRealCalibrator {
public:
    LactRealCalibrator(const std::string& calibration_file = "",
                       const std::string& config_str = "{}");
    void load_calibration_file(const std::string& filename);
    void operator()(ArrayEvent& event) const;
    bool has_calibration() const { return !times.empty(); }

private:
    double gain_threshold = 36.0;
    double high_gain_mv_per_pe = 2.5;
    double low_gain_pe_per_mv = 10.2;
    double led_time_offset_s = -37.0;
    std::vector<double> times;
    std::vector<Eigen::VectorXd> high_factors, low_factors;
    std::size_t calibration_index(double mjd) const;
};
