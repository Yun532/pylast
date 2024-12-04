#include "OpticsDescription.hh"
#include "spdlog/fmt/fmt.h"
OpticsDescription::OpticsDescription(string optics_name, int num_mirrors, double mirror_area, double equivalent_focal_length, double effective_focal_length):
    optics_name(optics_name), num_mirrors(num_mirrors), mirror_area(mirror_area), equivalent_focal_length(equivalent_focal_length), effective_focal_length(effective_focal_length)
{
}
const string OpticsDescription::print() const
{
    return fmt::format(
        "OpticsDescription(\n"
        "    optics_name: {}\n"
        "    num_mirrors: {}\n" 
        "    mirror_area: {:.3f} m²\n"
        "    equivalent_focal_length: {:.3f} m\n"
        "    effective_focal_length: {:.3f} m\n"
        ")",
        optics_name, num_mirrors, mirror_area, 
        equivalent_focal_length, effective_focal_length
    );
}
