#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lact_real {

struct WaveformSummary {
  double base;
  double amplitude;
  double area;
  double peak_sample;
};

namespace waveform_detail {

inline int local_gradient_peak(const std::vector<float> &values) {
  std::vector<float> gradients(256);
  for (int i = 0; i < 256; ++i)
    gradients[i] = std::abs(values[i] - values[(i + 255) % 256]);
  std::nth_element(gradients.begin(), gradients.begin() + 128, gradients.end());
  const float threshold = 3.6f * gradients[128];
  std::vector<float> remaining = values;
  for (int attempt = 0; attempt < 3; ++attempt) {
    const auto maximum = std::max_element(remaining.begin(), remaining.end());
    const int index = static_cast<int>(maximum - remaining.begin());
    if (*maximum == -std::numeric_limits<float>::infinity())
      return static_cast<int>(std::max_element(values.begin(), values.end()) -
                              values.begin());
    const float left = std::abs(*maximum - remaining[(index + 255) % 256]);
    const float right = std::abs(*maximum - remaining[(index + 1) % 256]);
    if (!(left > threshold && right > threshold))
      return index;
    remaining[index] = -std::numeric_limits<float>::infinity();
  }
  return static_cast<int>(
      std::max_element(remaining.begin(), remaining.end()) - remaining.begin());
}

inline int peak(const std::vector<float> &values) {
  const int index = static_cast<int>(
      std::max_element(values.begin(), values.end()) - values.begin());
  float sampled_gradients[16];
  for (int i = 0; i < 16; ++i) {
    const int sample = i * 16;
    sampled_gradients[i] =
        std::abs(values[sample] - values[(sample + 255) % 256]);
  }
  std::sort(sampled_gradients, sampled_gradients + 16);
  const float threshold = 5.0f * sampled_gradients[8];
  const float left = std::abs(values[index] - values[(index + 255) % 256]);
  const float right = std::abs(values[index] - values[(index + 1) % 256]);
  return (left > threshold && right > threshold)
             ? local_gradient_peak(values)
             : index;
}

} // namespace waveform_detail

// Traditional independent extraction from the server's
// /eos/lhaaso/decode/lact/Decode_code/code_2/wave_analyzer_L1_rush.C
// (snapshot in real_data_plot/reference_code/decode). Preserve its float
// accumulation, pulse gate, spike rejection, and circular integration order.
// Input/base/amplitude are mV; area is mV * sample, peak_sample is an index.
// This applies no ML labels or smoothing. Official ML summaries must instead
// supply their eventShow base/amp/area/PeakPos values.
inline WaveformSummary extract_waveform(const std::vector<float> &values) {
  if (values.size() != 256)
    throw std::invalid_argument("LACT waveforms must contain 256 samples");
  if (!std::all_of(values.begin(), values.end(),
                   [](float value) { return std::isfinite(value); }))
    throw std::invalid_argument("LACT waveform samples must be finite");

  float minimum_score = 1e30f;
  float baseline = 0.0f;
  float baseline_rms = 0.0f;
  for (int region = 0; region < 8; ++region) {
    const int start = region * 32;
    float sum = 0.0f;
    for (int j = start; j < start + 32; ++j)
      sum += values[j];
    const float mean = sum / 32;
    float squared_sum = 0.0f;
    for (int j = start; j < start + 32; ++j) {
      const float difference = values[j] - mean;
      squared_sum += difference * difference;
    }
    const float rms = std::sqrt(squared_sum / 32);
    const float score = mean + 3 * rms;
    if (score < minimum_score) {
      minimum_score = score;
      baseline = mean;
      baseline_rms = rms;
    }
  }

  std::vector<float> corrected(256);
  for (int j = 0; j < 256; ++j)
    corrected[j] = values[j] - baseline;

  const float detection_threshold = 10.0f * baseline_rms;
  int consecutive = 0;
  bool detected = false;
  for (float value : corrected) {
    consecutive = value > detection_threshold ? consecutive + 1 : 0;
    if (consecutive >= 5) {
      detected = true;
      break;
    }
  }
  // The reference labels interior/edge pulses separately but computes the same
  // metrics for both; that distinction does not affect this four-field result.
  if (!detected)
    return {baseline, 0.0, 0.0, 0.0};

  const int peak = waveform_detail::peak(corrected);
  const float amplitude = corrected[peak];
  const float integration_threshold = static_cast<float>(0.1 * amplitude);
  int start = -1;
  int end = -1;
  bool start_wrapped = false;
  bool end_wrapped = false;
  for (int j = peak; j >= 0; --j) {
    if (corrected[j] < integration_threshold) {
      start = j;
      break;
    }
  }
  if (start == -1) {
    start_wrapped = true;
    for (int j = 255; j > peak; --j) {
      if (corrected[j] < integration_threshold) {
        start = j;
        break;
      }
    }
    if (start == -1)
      start = peak;
  }
  for (int j = peak; j < 256; ++j) {
    if (corrected[j] < integration_threshold) {
      end = j;
      break;
    }
  }
  if (end == -1) {
    end_wrapped = true;
    for (int j = 0; j < peak; ++j) {
      if (corrected[j] < integration_threshold) {
        end = j;
        break;
      }
    }
    if (end == -1)
      end = peak;
  }

  float area = 0.0f;
  const auto accumulate = [&](int first, int last) {
    for (int j = first; j <= last; ++j)
      area += corrected[j];
  };
  if (!start_wrapped && !end_wrapped) {
    if (start <= end) {
      accumulate(start, end);
    } else {
      accumulate(start, 255);
      accumulate(0, end);
    }
  } else if (start_wrapped && !end_wrapped) {
    accumulate(0, end);
    accumulate(start, 255);
  } else if (!start_wrapped && end_wrapped) {
    accumulate(start, 255);
    accumulate(0, end);
  }
  return {baseline, amplitude, area, static_cast<double>(peak)};
}

} // namespace lact_real
