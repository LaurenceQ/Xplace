#include "gputimer/core/openroad/OpenroadRcInternal.h"
#include "common/utils/utils.h"

namespace gt {
namespace openroad_rc {

std::string normalized_spef_name(std::string name)
{
    validate_token(name);
    return name;
}

bool spef_digits_only(const std::string& value)
{
    if (value.empty()) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
    });
}

void add_name_alias(std::unordered_map<std::string, int>& map,
                           const std::string& name,
                           int value)
{
    if (name.empty()) {
        return;
    }
    map.emplace(name, value);
    map.emplace(normalized_spef_name(name), value);
}

std::string replace_char(std::string name, char from, char to)
{
    std::replace(name.begin(), name.end(), from, to);
    return name;
}

std::string replace_last_char(std::string name, char from, char to)
{
    const std::size_t pos = name.rfind(from);
    if (pos != std::string::npos) {
        name[pos] = to;
    }
    return name;
}

void add_gr_name_alias(std::unordered_map<std::string, int>& map,
                              const std::string& name,
                              int value)
{
    add_name_alias(map, name, value);
    add_name_alias(map, replace_char(name, '/', ':'), value);
    add_name_alias(map, replace_char(name, ':', '/'), value);
    add_name_alias(map, replace_last_char(name, '/', ':'), value);
    add_name_alias(map, replace_last_char(name, ':', '/'), value);
}

std::vector<std::string> split_tsv(const std::string& line)
{
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (begin <= line.size()) {
        std::size_t end = line.find('\t', begin);
        if (end == std::string::npos) {
            fields.emplace_back(line.substr(begin));
            break;
        }
        fields.emplace_back(line.substr(begin, end - begin));
        begin = end + 1;
    }
    return fields;
}

bool parse_int_field(const std::string& value, int& out)
{
    try {
        std::size_t consumed = 0;
        out = std::stoi(value, &consumed);
        return consumed == value.size();
    } catch (...) {
        return false;
    }
}

const char* skip_route_ws(const char* ptr, const char* end)
{
    while (ptr < end && std::isspace(static_cast<unsigned char>(*ptr))) {
        ++ptr;
    }
    return ptr;
}

bool parse_int_token(const char*& ptr, const char* end, int& out)
{
    ptr = skip_route_ws(ptr, end);
    if (ptr >= end) {
        return false;
    }
    bool negative = false;
    if (*ptr == '+' || *ptr == '-') {
        negative = *ptr == '-';
        ++ptr;
    }
    if (ptr >= end || !std::isdigit(static_cast<unsigned char>(*ptr))) {
        return false;
    }
    long long value = 0;
    while (ptr < end && std::isdigit(static_cast<unsigned char>(*ptr))) {
        value = value * 10 + static_cast<int>(*ptr - '0');
        if ((!negative && value > std::numeric_limits<int>::max()) ||
            (negative && -value < std::numeric_limits<int>::min())) {
            return false;
        }
        ++ptr;
    }
    out = static_cast<int>(negative ? -value : value);
    return true;
}

bool parse_token_range(const char*& ptr,
                              const char* end,
                              const char*& begin,
                              const char*& finish)
{
    ptr = skip_route_ws(ptr, end);
    begin = ptr;
    while (ptr < end && !std::isspace(static_cast<unsigned char>(*ptr))) {
        ++ptr;
    }
    finish = ptr;
    return begin < finish;
}

bool route_rest_is_ws(const char* ptr, const char* end)
{
    return skip_route_ws(ptr, end) == end;
}

bool parse_route_segment_row(const std::string& line,
                                    int& x1,
                                    int& y1,
                                    const char*& layer1_begin,
                                    const char*& layer1_end,
                                    int& x2,
                                    int& y2,
                                    const char*& layer2_begin,
                                    const char*& layer2_end)
{
    const char* ptr = line.data();
    const char* end = ptr + line.size();
    if (!parse_int_token(ptr, end, x1) ||
        !parse_int_token(ptr, end, y1) ||
        !parse_token_range(ptr, end, layer1_begin, layer1_end) ||
        !parse_int_token(ptr, end, x2) ||
        !parse_int_token(ptr, end, y2) ||
        !parse_token_range(ptr, end, layer2_begin, layer2_end)) {
        return false;
    }
    return route_rest_is_ws(ptr, end);
}

bool parse_float_field(const std::string& value, float& out)
{
    try {
        std::size_t consumed = 0;
        out = std::stof(value, &consumed);
        return consumed == value.size();
    } catch (...) {
        return false;
    }
}

std::vector<std::string> split_whitespace(const std::string& line)
{
    std::vector<std::string> fields;
    std::istringstream stream(line);
    std::string field;
    while (stream >> field) {
        fields.emplace_back(field);
    }
    return fields;
}

std::string lowercase_string(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

int trailing_integer(const std::string& value)
{
    std::size_t pos = value.size();
    while (pos > 0 && std::isdigit(static_cast<unsigned char>(value[pos - 1]))) {
        --pos;
    }
    if (pos == value.size()) {
        return -1;
    }
    int parsed = -1;
    return parse_int_field(value.substr(pos), parsed) ? parsed : -1;
}

int trailing_integer_token(const char* begin, const char* end)
{
    const char* digits = end;
    while (digits > begin && std::isdigit(static_cast<unsigned char>(*(digits - 1)))) {
        --digits;
    }
    if (digits == end) {
        return -1;
    }
    const char* ptr = digits;
    int value = -1;
    return parse_int_token(ptr, end, value) && ptr == end ? value : -1;
}


int resolve_route_net_token(const std::unordered_map<std::string_view, int>& net_name_to_index,
                            const char* begin,
                            const char* end)
{
    auto find_net = [&](std::string_view alias) {
        auto iter = net_name_to_index.find(alias);
        return iter == net_name_to_index.end() ? -1 : iter->second;
    };
    const std::string_view token(begin, static_cast<std::size_t>(end - begin));
    int net_idx = find_net(token);
    if (net_idx >= 0) {
        return net_idx;
    }
    const std::string name(begin, end);
    const std::string normalized = normalized_spef_name(name);
    if (normalized != name) {
        net_idx = find_net(std::string_view(normalized));
        if (net_idx >= 0) {
            return net_idx;
        }
    }
    const std::array<std::string, 4> aliases = {
        replace_char(name, '/', ':'),
        replace_char(name, ':', '/'),
        replace_last_char(name, '/', ':'),
        replace_last_char(name, ':', '/'),
    };
    for (const std::string& alias : aliases) {
        if (alias == name || alias == normalized) {
            continue;
        }
        net_idx = find_net(std::string_view(alias));
        if (net_idx >= 0) {
            return net_idx;
        }
    }
    return -1;
}

int resolve_route_layer_token(const std::unordered_map<std::string, int>& layer_name_to_level,
                              const char* begin,
                              const char* end)
{
    const int trailing = trailing_integer_token(begin, end);
    if (trailing > 0) {
        return trailing;
    }
    std::string lower(begin, end);
    lower = lowercase_string(std::move(lower));
    auto iter = layer_name_to_level.find(lower);
    if (iter != layer_name_to_level.end()) {
        return iter->second;
    }
    return -1;
}

}  // namespace openroad_rc
}  // namespace gt
