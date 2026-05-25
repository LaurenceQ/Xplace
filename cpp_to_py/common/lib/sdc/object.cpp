#include "object.h"

namespace gt::sdc {

Object parse_port(const std::string& line) {
    if (line.find("all_inputs") != std::string::npos) {
        return AllInputs{};
    }

    if (line.find("all_outputs") != std::string::npos) {
        return AllOutputs{};
    }

    if (line.find("all_clocks") != std::string::npos) {
        return AllClocks{};
    }

    const static std::regex ws_re("\\s+|\\n+|\\t+");

    auto strip_braces = [](std::string token) {
        while (token.size() >= 2 && token.front() == '{' && token.back() == '}') {
            token = token.substr(1, token.size() - 2);
        }
        return token;
    };

    const std::string get_pins_prefix = "__get_pins__";
    if (line.rfind(get_pins_prefix, 0) == 0) {
        auto pins_line = line.substr(get_pins_prefix.size());
        auto itr = std::sregex_token_iterator(pins_line.begin(), pins_line.end(), ws_re, -1);
        auto end = std::sregex_token_iterator();

        GetPins get_pins;
        for (; itr != end; ++itr) {
            auto pin = strip_braces(itr->str());
            if (!pin.empty()) {
                get_pins.pins.push_back(std::move(pin));
            }
        }
        return get_pins;
    }

    const std::string get_clocks_prefix = "__get_clocks__";
    if (line.rfind(get_clocks_prefix, 0) == 0) {
        auto clocks_line = line.substr(get_clocks_prefix.size());
        auto itr = std::sregex_token_iterator(clocks_line.begin(), clocks_line.end(), ws_re, -1);
        auto end = std::sregex_token_iterator();

        GetClocks get_clocks;
        for (; itr != end; ++itr) {
            auto clock = strip_braces(itr->str());
            if (!clock.empty()) {
                get_clocks.clocks.push_back(std::move(clock));
            }
        }
        return get_clocks;
    }

    auto itr = std::sregex_token_iterator(line.begin(), line.end(), ws_re, -1);
    auto end = std::sregex_token_iterator();
    // auto num = std::distance(itr, end);

    GetPorts get_ports;

    for (; itr != end; ++itr) {
        auto port = strip_braces(itr->str());
        if (!port.empty()) {
            get_ports.ports.push_back(std::move(port));
        }
    }

    return get_ports;
}

};  // namespace gt::sdc
