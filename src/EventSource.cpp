#include "EventSource.hh"
#include <algorithm>

void EventSource::load_atmosphere_model(const std::string& filename)
{
    if (atmosphere_from_input)
        throw std::runtime_error("Input already contains an atmosphere profile; external profile is forbidden");
    TableAtmosphereModel profile(filename);
    atmosphere_model = profile;
    TableAtmosphereModel::global_instance() = profile;
}

bool EventSource::is_subarray_selected(int tel_id) const
{
    if(allowed_tels.empty()) return true;
    return std::find(allowed_tels.begin(), allowed_tels.end(), tel_id) != allowed_tels.end();
}
