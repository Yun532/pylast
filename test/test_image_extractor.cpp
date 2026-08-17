#include "ImageExtractor.hh"

#include <cmath>
#include <iostream>

int main()
{
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> waveform(2, 4);
    waveform << 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 3.0, 0.0;
    const Eigen::VectorXi peak_index =
        Eigen::VectorXi::Zero(2);
    const Eigen::VectorXi window_width = Eigen::VectorXi::Constant(2, 4);
    const Eigen::VectorXi window_shift = Eigen::VectorXi::Zero(2);

    const auto [charge, peak_time] = extract_around_peak(
        waveform, peak_index, window_width, window_shift, 0.5);

    if (charge[0] != 0.0 || !std::isnan(peak_time[0])) {
        std::cerr << "empty waveform pixel must have an invalid peak time\n";
        return 1;
    }
    if (std::abs(charge[1] - 4.0) > 1.0e-12 ||
        std::abs(peak_time[1] - 3.5) > 1.0e-12) {
        std::cerr << "positive waveform weighted time is incorrect\n";
        return 1;
    }
    return 0;
}
