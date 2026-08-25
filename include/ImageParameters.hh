/**
 * @file ImageParameters.hh
 * @author Zach Peng (zhipzhang@mail.ustc.edu.cn)
 * @brief  class to describe image parameters
 * @version 0.1
 * @date 2024-12-06
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include <optional>
class HillasParameter {
public:
  double length = 0.0;      // [rad]
  double width = 0.0;       // [rad]
  double psi = 0.0;         // [rad] orientation of the major axis
  double x = 0.0;           // [rad]
  double y = 0.0;           // [rad]
  double skewness = 0.0;    // [ ]
  double kurtosis = 0.0;    // [ ]
  double intensity = 0.0;   // [p.e.]
  double r = 0.0;           // [rad]  distance from the center of the camera
  double phi = 0.0;         // [rad]  angle from the center of the camera
  double scale_ratio = 0.0; // [] ratio of second level clean to first level clean
};

class TwoGaussianFitResult {
public:
  bool converged = false;
  int status = 0;
  double amplitude = 0.0;
  double mean_x = 0.0;
  double mean_y = 0.0;
  double length = 0.0;
  double width = 0.0;
  double psi = 0.0;
  double fit_size = 0.0;
  double chi2 = 0.0;
  bool use_gaussian_fit = false;
  double beta_err = 0.0;
  double cog_err = 0.0;
  double miss = 0.0;
  double disp = 0.0;
};
class LeakageParameter {
public:
  double pixels_width_1 = 0.0;
  double pixels_width_2 = 0.0;
  double intensity_width_1 = 0.0;
  double intensity_width_2 = 0.0;
};
class ConcentrationParameter {
public:
  double concentration_cog = 0.0;   // one pixel diameter from the cog
  double concentration_core = 0.0;  // all_pixels inside the hillas ellipse,
                              // transformed to the hillas ellipse
  double concentration_pixel = 0.0; // brightest pixel
};

class MorphologyParameter {
public:
  int n_pixels = 0;
  int n_islands = 0;
  int n_small_islands = 0;
  int n_medium_islands = 0;
  int n_large_islands = 0;
};

class IntensityParameter {
public:
  double intensity_max = 0.0;
  double intensity_mean = 0.0;
  double intensity_std = 0.0;
  double intensity_skewness = 0.0;
  double intensity_kurtosis = 0.0;
};
class ExtraParameters {
public:
  double miss = 0;
  double disp = 0;
  double theta = 0;
  double true_psi = 0;
  double cog_err = 0;
  double beta_err = 0;
};
class ImageParameters {
public:
  HillasParameter hillas{};
  TwoGaussianFitResult two_gaussian_fit{};
  LeakageParameter leakage{};
  ConcentrationParameter concentration{};
  MorphologyParameter morphology{};
  IntensityParameter intensity{};
  ExtraParameters extra{};
};
