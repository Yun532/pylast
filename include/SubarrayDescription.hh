/**
 * @file SubarrayDescription.hh
 * @author Zach Peng
 * @brief Description of the subarray, this class describe the list of telescopes and their positions
 * @version 0.1
 * @date 2024-11-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once


#include "CameraDescription.hh"
#include "OpticsDescription.hh"
#include <unordered_map>
using telescope_id_t = int;
class TelescopeDescription
{
public:
    /** @brief Name of the telescope */
    string tel_name = "LACT";
    /** @brief Description of the camera */
    CameraDescription camera_description;
    /** @brief Description of the optics */
    OpticsDescription optics_description;

    TelescopeDescription() = default;
    TelescopeDescription(CameraDescription camera_description, OpticsDescription optics_description);
    TelescopeDescription(TelescopeDescription&& other) noexcept = default;
    TelescopeDescription(const TelescopeDescription& other) = default;
    TelescopeDescription& operator=(TelescopeDescription&& other) noexcept = default;
    TelescopeDescription& operator=(const TelescopeDescription& other) = default;
    ~TelescopeDescription() = default;
    const string print() const;
};
class SubarrayDescription
{
public:
    SubarrayDescription() = default;
    ~SubarrayDescription() = default;

    /** @brief Descriptions of the telescopes */
    std::unordered_map<telescope_id_t, TelescopeDescription> tel_descriptions;
    /** @brief Positions of the telescopes */
    std::unordered_map<telescope_id_t, std::array<double, 3>> tel_positions;
    /** @brief Reference position of the subarray */
    std::array<double, 3> reference_position ;
    void add_telescope(const telescope_id_t tel_id,  TelescopeDescription&& tel_description, const std::array<double, 3>& tel_position);
    const string print() const;
};