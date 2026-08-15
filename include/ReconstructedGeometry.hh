/**
 * @file ReconstructedGeometry.hh
 * @author Zach Peng (zhipzhang@mail.ustc.edu.cn)
 * @brief  store the information of the reconstructed geometry
 * @version 0.1
 * @date 2025-03-04
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once
#include <limits>
#include <vector>
class ReconstructedGeometry
{
    public:
        bool is_valid = false;
        double alt = std::numeric_limits<double>::quiet_NaN();
        double alt_uncertainty = std::numeric_limits<double>::quiet_NaN();
        double az = std::numeric_limits<double>::quiet_NaN();
        double az_uncertainty = std::numeric_limits<double>::quiet_NaN();
        double direction_error = std::numeric_limits<double>::quiet_NaN();
        double core_x = std::numeric_limits<double>::quiet_NaN();
        double core_y = std::numeric_limits<double>::quiet_NaN();
        double core_pos_error = std::numeric_limits<double>::quiet_NaN();
        double tilted_core_x = std::numeric_limits<double>::quiet_NaN();
        double tilted_core_y = std::numeric_limits<double>::quiet_NaN();
        double tilted_core_uncertainty_x = std::numeric_limits<double>::quiet_NaN();
        double tilted_core_uncertainty_y = std::numeric_limits<double>::quiet_NaN();
        double hmax = std::numeric_limits<double>::quiet_NaN();
        double xmax = std::numeric_limits<double>::quiet_NaN();
        std::vector<int> telescopes;
        
};

class ReconstructedEnergy
{
    public:
        bool energy_valid = false;
        double estimate_energy = 0;
        double estimate_energy_std = 0;
        std::vector<int> telescopes;
};

class ReconstructedParticle
{
    public:
        double hadroness = 0;
        double mrsl = -999;
        double mrsw = -999;
        bool is_valid = false;
        std::vector<int> telescopes;
};
