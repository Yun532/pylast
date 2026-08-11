/**
 * @file CameraReadout.hh
 * @author Zach Peng (zhipzhang@mail.ustc.edu.cn)
 * @brief Camera readout information
 * @version 0.1
 * @date 2024-12-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once

#include <string>
#include "Eigen/Dense"
using std::string;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class CameraReadout
{
public:
    /** @brief Name of the camera */
    string camera_name;
    /** @brief Sampling rate of the waveform [Hz] */
    double sampling_rate;
    /** @brief The amount of time corresponding to each sample in reference_pulse_shape [ns] */
    double reference_pulse_sample_width;
    /** @brief Number of gain channels */
    int n_channels;
    /** @brief Number of pixels */
    int n_pixels;
    /** @brief Number of waveform samples for normal events */
    int n_samples;
    /** @brief Expected pulse shape for a signal in the waveform. 2 dimensional, first dimension is gain channel */
    Eigen::MatrixXd reference_pulse_shape;

    /** @brief Unit of R1 waveform samples. Empty keeps legacy p.e.-sample semantics. */
    string waveform_sample_unit;
    /** @brief Area of one measured single-p.e. pulse [mV ns], for mV samples. */
    double single_pe_area_mv_ns = 0.0;
    /** @brief Time convention used by the stored reference pulse. */
    string template_time_reference;


    const string print() const;
};
