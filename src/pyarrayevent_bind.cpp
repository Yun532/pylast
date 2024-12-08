#include "nanobind/nanobind.h"
#include "ArrayEvent.hh"

namespace nb = nanobind;

void bind_array_event(nb::module_ &m) {
    nb::class_<ArrayEvent>(m, "ArrayEvent")
        .def_ro("simulation", &ArrayEvent::simulated_event);
    nb::class_<SimulatedEvent>(m, "SimulatedEvent")
        .def_ro("shower", &SimulatedEvent::shower)
        .def_ro("cameras", &SimulatedEvent::cameras);
    nb::class_<SimulatedCamera>(m, "SimulatedCamera")
        .def_ro("true_image_sum", &SimulatedCamera::true_image_sum)
        .def_ro("true_image", &SimulatedCamera::true_image)
        .def_ro("impact_parameter", &SimulatedCamera::impact_parameter);
    nb::class_<TelImpactParameter>(m, "TelImpactParameter")
        .def_ro("impact_distance", &TelImpactParameter::impact_parameter)
        .def_ro("impact_distance_error", &TelImpactParameter::impact_parameter_error);
    nb::class_<SimulatedShower>(m, "shower")
        .def_ro("alt", &SimulatedShower::alt)
        .def_ro("az", &SimulatedShower::az)
        .def_ro("core_x", &SimulatedShower::core_x)
        .def_ro("core_y", &SimulatedShower::core_y)
        .def_ro("energy", &SimulatedShower::energy)
        .def_ro("h_first_int", &SimulatedShower::h_first_int)
        .def_ro("x_max", &SimulatedShower::x_max)
        .def_ro("starting_grammage", &SimulatedShower::starting_grammage)
        .def_ro("shower_primary_id", &SimulatedShower::shower_primary_id);
}