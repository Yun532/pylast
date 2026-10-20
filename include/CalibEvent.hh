#pragma once

#include "BaseTelContainer.hh"
#include <Eigen/Dense>
#include <array>

// Measured LACT signals before conversion to photoelectrons. Gain 0 is HG,
// gain 1 is LG; columns are ordered by the one-based mapping CSV pixel number.
class CalibCamera {
public:
    using Matrix = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;
    Matrix base, peak, area, peak_sample, peak_time, pe, count_to_pe;
    Eigen::Array<bool, -1, -1, Eigen::RowMajor> valid;
    std::array<Matrix, 2> waveform;
    double sampling_interval_ns = 0.0;
};

class CalibEvent : public BaseTelContainer<CalibCamera> {};
