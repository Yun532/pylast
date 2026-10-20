#include <stdexcept>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "SimtelEventSource.hh"
#include <filesystem>

namespace {
class CountingEventSource : public EventSource {
public:
    CountingEventSource(int available, int limit) : available(available) {
        max_events = limit;
        load_simulated_showers = false;
    }
    int reads = 0;
    bool has_current_event() const { return current_event.has_value(); }
    void load_all_simulated_showers() override {}

protected:
    bool is_finished() override { return current_event_index >= available; }
    ArrayEvent get_event() override {
        ++reads;
        if (is_finished()) throw std::runtime_error("read beyond available events");
        ArrayEvent event;
        event.event_id = current_event_index;
        return event;
    }
    void open_file() override {}
    void init_metaparam() override {}
    void init_atmosphere_model() override {}
    void init_simulation_config() override {}
    void init_subarray() override {}

private:
    int available;
};
}

TEST_CASE("EventSource iterator end never reads and empty begin stays empty")
{
    for (int available : {0, 3}) {
        CountingEventSource source(available, 0);
        auto end = source.end(); // Mimic make_iterator evaluating end first.
        CHECK(source.reads == 0);
        CHECK_FALSE(source.has_current_event());
        auto begin = source.begin();
        CHECK(source.reads == 0);
        CHECK(begin == end);
    }
    CountingEventSource exhausted(0, -1);
    CHECK(exhausted.begin() == exhausted.end());
    CHECK(exhausted.reads == 0);
}

TEST_CASE("EventSource iterator reads the first event once and advances to the end")
{
    for (int limit : {3, -1}) {
        CountingEventSource source(3, limit);
        auto end = source.end();
        CHECK(source.reads == 0);
        CHECK_FALSE(source.has_current_event());
        auto current = source.begin();
        for (int index = 0; index < 3; ++index) {
            REQUIRE(current != end);
            CHECK((*current).event_id == index);
            CHECK(source.reads == index + 1);
            ++current;
        }
        CHECK(current == end);
        CHECK(source.reads == 3);
    }
}

TEST_CASE("SimtelEventSourceInitialize")
{
    SUBCASE("FileNotExist")
    {
        CHECK_THROWS_AS(SimtelEventSource("not_exist.simtel.zst"), std::runtime_error);
    }
    auto test_directory = std::filesystem::path(__FILE__);
    auto test_file = test_directory.parent_path() / "test_data" / "lact_prod0_simtel_particle_gamma_energy_1000.0_1000.0_zenith_0.0_azimuth_0.0_run_1_event_0.zst";
    SUBCASE("test_initialize")
    {
        SimtelEventSource simtel_event_source(test_file.string());
        CHECK(simtel_event_source.is_stream == true);
        CHECK(simtel_event_source.metaparam.has_value() == true);
        CHECK(simtel_event_source.atmosphere_model.has_value() == true);
        CHECK(simtel_event_source.simulation_config.has_value() == true);

        //  test file have 16 telescopes.
        CHECK(simtel_event_source.subarray->tels.size() == 16);
        CHECK(simtel_event_source.subarray->tel_positions.size()  == 16);
    }
    SUBCASE("test_max_events")
    {
        SimtelEventSource simtel_event_source(test_file.string(), 10);
        CHECK(simtel_event_source.max_events == 10);
        CHECK(simtel_event_source.is_stream == true);
        
        int all_event = 0;
        for(const auto& event: simtel_event_source) {
            all_event++;
        }
        CHECK(all_event == 10);
        SUBCASE("test_random_access")
        {
            CHECK_THROWS_AS(simtel_event_source[4], std::runtime_error);
            CHECK_THROWS_AS(simtel_event_source[11], std::out_of_range);
        }
    }
    SUBCASE("test_subarray")
    {
        SimtelEventSource simtel_event_source(test_file.string(), -1, {1, 2, 3});
        CHECK(simtel_event_source.allowed_tels.size() == 3);
        CHECK(simtel_event_source.subarray->tels.size() == 3);
        CHECK(simtel_event_source.subarray->tel_positions.size() == 3);
    }
    SUBCASE("test_load_simulated_shower_true")
    {
        SimtelEventSource simtel_event_source(test_file.string(), -1, {}, true);
        CHECK(simtel_event_source.shower_array.has_value() == true);
        CHECK(simtel_event_source.shower_array->size() > 0);
    }
    SUBCASE("test_load_simulated_shower_false")
    {
        SimtelEventSource simtel_event_source(test_file.string(), 10, {}, false);
        CHECK(simtel_event_source.shower_array.has_value() == false);
        for(const auto& event: simtel_event_source) {
            // do no things
        }
        CHECK(simtel_event_source.get_shower_array().size() > 0);
    }
    
}
