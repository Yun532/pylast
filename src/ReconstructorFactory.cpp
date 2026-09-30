#include "ReconstructorFactory.hh"

ReconstructorFactory& ReconstructorFactory::instance()
{
    // Keep one registry across the core shared library and Python extensions,
    // whose hidden visibility otherwise creates a separate inline singleton.
    static ReconstructorFactory factory;
    return factory;
}

void ReconstructorFactory::register_reconstructor(const std::string& type, const std::function<std::unique_ptr<Reconstructor>(const SubarrayDescription& subarray, const json& config)>& creator)
{
    creators_[type] = creator;
}
bool ReconstructorFactory::is_registered(const std::string& type) const
{
    return creators_.contains(type);
}
std::unique_ptr<Reconstructor> ReconstructorFactory::create(const std::string& type, const SubarrayDescription& subarray, const json& config)
{
    auto it = creators_.find(type);
    if(it == creators_.end())
    {
        throw std::runtime_error("Unknown reconstructor type: " + type);
    }
    return it->second(subarray, config);
}
