#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <vector>

#include "common/db/BsRouteInfo.h"
#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "common/db/DatabaseClass.h"
#include "common/db/DesignRule.h"
#include "common/db/GCellGrid.h"
#include "common/db/Geometry.h"
#include "common/db/Layer.h"
#include "common/db/Net.h"
#include "common/db/Pin.h"
#include "common/db/Region.h"
#include "common/db/Row.h"
#include "common/db/SNet.h"
#include "common/db/Site.h"
#include "common/db/SiteMap.h"
#include "common/db/Via.h"

// tcl have another definition of EXTERN
#undef EXTERN

#include "def58/include/defiUtil.hpp"
#include "def58/include/defrReader.hpp"
#include "lef58/include/lefrReader.hpp"

using namespace db;

namespace {

bool defProfileEnabled()
{
    const char* def_profile = std::getenv("XPLACE_DEF_PROFILE");
    const char* io_profile = std::getenv("XPLACE_IO_PROFILE");
    const char* value = (def_profile != nullptr) ? def_profile : io_profile;
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' ||
             value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

struct DefReadProfile {
    bool enabled = false;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point last;
    const char* phase = "setup";
    long long rows = 0;
    long long tracks = 0;
    long long gcells = 0;
    long long vias = 0;
    long long ndrs = 0;
    long long components = 0;
    long long pins = 0;
    long long blockages = 0;
    long long snets = 0;
    long long nets = 0;
    long long net_connections = 0;
    long long regions = 0;

    void begin()
    {
        enabled = defProfileEnabled();
        start = std::chrono::steady_clock::now();
        last = start;
        phase = "setup";
        rows = tracks = gcells = vias = ndrs = components = pins = 0;
        blockages = snets = nets = net_connections = regions = 0;
    }

    void switchPhase(const char* next)
    {
        if (!enabled) {
            phase = next;
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - last).count();
        const double total = std::chrono::duration<double>(now - start).count();
        std::fprintf(stdout,
                     "[XPLACE_DEF_PROFILE] phase=%s elapsed=%.3f total=%.3f\n",
                     phase, elapsed, total);
        std::fflush(stdout);
        last = now;
        phase = next;
    }

    void finish(const char* status)
    {
        if (!enabled) {
            return;
        }
        switchPhase(phase);
        const auto now = std::chrono::steady_clock::now();
        const double total = std::chrono::duration<double>(now - start).count();
        std::fprintf(stdout,
                     "[XPLACE_DEF_PROFILE] phase=summary status=%s total=%.3f rows=%lld tracks=%lld gcells=%lld vias=%lld ndrs=%lld components=%lld pins=%lld blockages=%lld snets=%lld nets=%lld net_connections=%lld regions=%lld\n",
                     status, total, rows, tracks, gcells, vias, ndrs,
                     components, pins, blockages, snets, nets, net_connections, regions);
        std::fflush(stdout);
    }
};

DefReadProfile g_def_profile;


bool defBufferProfileEnabled()
{
    const char* value = std::getenv("XPLACE_DEF_BUFFER_PROFILE");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' ||
             value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

struct DefMappedFile {
    int fd = -1;
    const char* data = nullptr;
    std::size_t size = 0;

    explicit DefMappedFile(const std::string& file)
    {
        fd = ::open(file.c_str(), O_RDONLY);
        if (fd < 0) {
            std::fprintf(stderr,
                         "[XPLACE_DEF_BUFFER_PROFILE] open failed file=%s error=%s\n",
                         file.c_str(), std::strerror(errno));
            return;
        }

        struct stat st;
        if (::fstat(fd, &st) != 0) {
            std::fprintf(stderr,
                         "[XPLACE_DEF_BUFFER_PROFILE] stat failed file=%s error=%s\n",
                         file.c_str(), std::strerror(errno));
            ::close(fd);
            fd = -1;
            return;
        }
        if (st.st_size <= 0) {
            return;
        }
        size = static_cast<std::size_t>(st.st_size);
        void* mapped = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            std::fprintf(stderr,
                         "[XPLACE_DEF_BUFFER_PROFILE] mmap failed file=%s size=%zu error=%s\n",
                         file.c_str(), size, std::strerror(errno));
            data = nullptr;
            size = 0;
            return;
        }
        data = static_cast<const char*>(mapped);
    }

    DefMappedFile(const DefMappedFile&) = delete;
    DefMappedFile& operator=(const DefMappedFile&) = delete;

    ~DefMappedFile()
    {
        if (data != nullptr) {
            ::munmap(const_cast<char*>(data), size);
        }
        if (fd >= 0) {
            ::close(fd);
        }
    }
};

struct DefBufferSection {
    const char* name = "";
    long long declared_count = -1;
    std::size_t header_begin = 0;
    std::size_t body_begin = 0;
    std::size_t end_begin = 0;
    std::size_t end_after = 0;
    long long object_lines = 0;
    long long connection_lines = 0;
    bool found = false;
};

const char* defSkipWs(const char* p, const char* end)
{
    while (p < end && std::isspace(static_cast<unsigned char>(*p))) {
        ++p;
    }
    return p;
}

bool defTokenEquals(const char* begin, const char* end, const char* token)
{
    const std::size_t token_len = std::strlen(token);
    return static_cast<std::size_t>(end - begin) == token_len &&
           std::memcmp(begin, token, token_len) == 0;
}

bool defLineStartsWithToken(const char* line_begin, const char* line_end, const char* token)
{
    const char* p = defSkipWs(line_begin, line_end);
    const char* q = p;
    while (q < line_end && !std::isspace(static_cast<unsigned char>(*q))) {
        ++q;
    }
    return defTokenEquals(p, q, token);
}

bool defParseLongLongToken(const char*& p, const char* end, long long& value)
{
    p = defSkipWs(p, end);
    if (p == end || !std::isdigit(static_cast<unsigned char>(*p))) {
        return false;
    }
    long long result = 0;
    while (p < end && std::isdigit(static_cast<unsigned char>(*p))) {
        result = result * 10 + (*p - '0');
        ++p;
    }
    value = result;
    return true;
}

std::size_t defNextLineOffset(const char* data, std::size_t size, std::size_t offset)
{
    if (offset >= size) {
        return size;
    }
    const void* newline = std::memchr(data + offset, '\n', size - offset);
    if (newline == nullptr) {
        return size;
    }
    return static_cast<const char*>(newline) - data + 1;
}

std::size_t defAlignLineStart(const char* data, std::size_t size, std::size_t offset)
{
    if (offset == 0 || offset >= size || data[offset - 1] == '\n') {
        return offset;
    }
    return defNextLineOffset(data, size, offset);
}

void defFindSection(const char* data,
                    std::size_t size,
                    const char* section_name,
                    DefBufferSection& section)
{
    section.name = section_name;
    bool in_section = false;
    for (std::size_t line_begin_offset = 0; line_begin_offset < size;) {
        const std::size_t line_end_offset = [&]() {
            const void* newline = std::memchr(data + line_begin_offset, '\n', size - line_begin_offset);
            return newline == nullptr ? size : static_cast<const char*>(newline) - data;
        }();
        const std::size_t next_line_offset = line_end_offset < size ? line_end_offset + 1 : line_end_offset;
        const char* line_begin = data + line_begin_offset;
        const char* line_end = data + line_end_offset;

        if (!in_section) {
            const char* p = defSkipWs(line_begin, line_end);
            const char* q = p;
            while (q < line_end && !std::isspace(static_cast<unsigned char>(*q))) {
                ++q;
            }
            if (defTokenEquals(p, q, section_name)) {
                p = q;
                long long declared = -1;
                if (defParseLongLongToken(p, line_end, declared)) {
                    section.declared_count = declared;
                }
                section.header_begin = line_begin_offset;
                section.body_begin = next_line_offset;
                in_section = true;
            }
        } else if (defLineStartsWithToken(line_begin, line_end, "END")) {
            const char* p = defSkipWs(line_begin, line_end);
            while (p < line_end && !std::isspace(static_cast<unsigned char>(*p))) {
                ++p;
            }
            const char* q = defSkipWs(p, line_end);
            const char* r = q;
            while (r < line_end && !std::isspace(static_cast<unsigned char>(*r))) {
                ++r;
            }
            if (defTokenEquals(q, r, section_name)) {
                section.end_begin = line_begin_offset;
                section.end_after = next_line_offset;
                section.found = true;
                return;
            }
        }
        line_begin_offset = next_line_offset;
    }
}

void defCountSectionObjects(const char* data,
                            const DefBufferSection& section,
                            int num_threads,
                            long long& object_lines,
                            long long& connection_lines)
{
    object_lines = 0;
    connection_lines = 0;
    if (!section.found || section.body_begin >= section.end_begin) {
        return;
    }
    num_threads = std::max(1, num_threads);
    std::vector<long long> object_counts(num_threads, 0);
    std::vector<long long> connection_counts(num_threads, 0);
    const std::size_t begin = section.body_begin;
    const std::size_t end = section.end_begin;
    const std::size_t size = end - begin;

    auto scan_chunk = [&](int tid) {
        const std::size_t chunk_begin =
            begin + (size * static_cast<std::size_t>(tid)) / static_cast<std::size_t>(num_threads);
        const std::size_t chunk_end =
            begin + (size * static_cast<std::size_t>(tid + 1)) / static_cast<std::size_t>(num_threads);
        const std::size_t scan_begin = defAlignLineStart(data, end, chunk_begin);
        const std::size_t scan_end = (tid + 1 == num_threads) ? end : defAlignLineStart(data, end, chunk_end);
        long long local_objects = 0;
        long long local_connections = 0;
        for (std::size_t line_begin_offset = scan_begin; line_begin_offset < scan_end;) {
            const void* newline = std::memchr(data + line_begin_offset, '\n', scan_end - line_begin_offset);
            const std::size_t line_end_offset = newline == nullptr ? scan_end : static_cast<const char*>(newline) - data;
            const std::size_t next_line_offset = line_end_offset < scan_end ? line_end_offset + 1 : line_end_offset;
            const char* line_begin = data + line_begin_offset;
            const char* line_end = data + line_end_offset;
            const char* p = defSkipWs(line_begin, line_end);
            if (p < line_end && *p == '-') {
                ++local_objects;
            }
            if (p < line_end && *p == '(') {
                ++local_connections;
            }
            line_begin_offset = next_line_offset;
        }
        object_counts[tid] = local_objects;
        connection_counts[tid] = local_connections;
    };

    std::vector<std::thread> workers;
    workers.reserve(num_threads > 1 ? static_cast<std::size_t>(num_threads - 1) : 0);
    for (int tid = 1; tid < num_threads; ++tid) {
        workers.emplace_back(scan_chunk, tid);
    }
    scan_chunk(0);
    for (std::thread& worker : workers) {
        worker.join();
    }

    for (int tid = 0; tid < num_threads; ++tid) {
        object_lines += object_counts[tid];
        connection_lines += connection_counts[tid];
    }
}

void profileDefBufferScan(const std::string& file, int requested_threads)
{
    if (!defBufferProfileEnabled()) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    auto seconds_since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    };

    DefMappedFile mapped(file);
    if (mapped.data == nullptr || mapped.size == 0) {
        return;
    }
    std::fprintf(stdout,
                 "[XPLACE_DEF_BUFFER_PROFILE] phase=mmap elapsed=%.3f size=%zu\n",
                 seconds_since(start), mapped.size);
    std::fflush(stdout);

    std::vector<DefBufferSection> sections(4);
    defFindSection(mapped.data, mapped.size, "COMPONENTS", sections[0]);
    defFindSection(mapped.data, mapped.size, "PINS", sections[1]);
    defFindSection(mapped.data, mapped.size, "SPECIALNETS", sections[2]);
    defFindSection(mapped.data, mapped.size, "NETS", sections[3]);
    std::fprintf(stdout,
                 "[XPLACE_DEF_BUFFER_PROFILE] phase=find_sections elapsed=%.3f\n",
                 seconds_since(start));
    std::fflush(stdout);

    int num_threads = std::max(1, requested_threads);
    if (const char* env_threads = std::getenv("XPLACE_DEF_BUFFER_THREADS")) {
        const int value = std::atoi(env_threads);
        if (value > 0) {
            num_threads = value;
        }
    }
    for (DefBufferSection& section : sections) {
        defCountSectionObjects(mapped.data, section, num_threads,
                               section.object_lines, section.connection_lines);
        std::fprintf(stdout,
                     "[XPLACE_DEF_BUFFER_PROFILE] section=%s found=%d declared=%lld object_lines=%lld connection_lines=%lld bytes=%zu body_begin=%zu end_begin=%zu threads=%d elapsed=%.3f\n",
                     section.name,
                     section.found ? 1 : 0,
                     section.declared_count,
                     section.object_lines,
                     section.connection_lines,
                     section.found ? section.end_begin - section.body_begin : 0,
                     section.body_begin,
                     section.end_begin,
                     num_threads,
                     seconds_since(start));
        std::fflush(stdout);
    }
}

}  // namespace

bool isFlipX(int orient) {
    switch (orient) {
        case 0:
            return false;  // N
        case 2:
            return true;  // S
        case 4:
            return true;  // FN
        case 6:
            return false;  // FS
    }
    return false;
}

bool isFlipY(int orient) {
    switch (orient) {
        case 0:
            return false;  // N
        case 2:
            return true;  // S
        case 4:
            return false;  // FN
        case 6:
            return true;  // FS
    }
    return false;
}

string getRowOrient(bool flipX, bool flipY) {
    if (flipX) {
        if (flipY) {
            return "S";
        } else {
            return "FN";
        }
    } else {
        if (flipY) {
            return "FS";
        } else {
            return "N";
        }
    }
}

string getOrient(int orient) {
    // 0:N, 1:W, 2:S, 3:E, 4:FN, 5:FW, 6:FS, 7:FE, -1:NONE
    switch (orient) {
        case 0:
            return "N";
        case 1:
            return "W";
        case 2:
            return "S";
        case 3:
            return "E";
        case 4:
            return "FN";
        case 5:
            return "FW";
        case 6:
            return "FS";
        case 7:
            return "FE";
        case -1:
            return "NONE";
        default:
            return "N";
    }
}

#define DIR_UP 1
#define DIR_DOWN 2
#define DIR_LEFT 4
#define DIR_RIGHT 8

unsigned char pointDir(const int lx, const int ly, const int hx, const int hy) {
    if ((lx == hx) == (ly == hy)) {
        return 0;
    }
    if (lx < hx) {
        return DIR_RIGHT;
    }
    if (lx > hx) {
        return DIR_LEFT;
    }
    if (ly < hy) {
        return DIR_UP;
    }
    return DIR_DOWN;
}

class Point {
public:
    int x, y;
    unsigned char outdir = 0;

    Point(const int x = 0, const int y = 0, const unsigned char dir = 0) : x(x), y(y), outdir(dir) {}

    inline bool operator<(const Point& T) const {
        if (y < T.y) {
            return true;
        }
        if (y > T.y) {
            return false;
        }
        if (x < T.x) {
            return true;
        }
        if (x > T.x) {
            return false;
        }
        if ((outdir | DIR_DOWN) && T.outdir | DIR_UP) {
            return true;
        }
        if ((outdir | DIR_UP) && T.outdir | DIR_DOWN) {
            return false;
        }
        if ((outdir | DIR_LEFT) && T.outdir | DIR_RIGHT) {
            return true;
        }
        return false;
    }

    inline bool operator==(const Point& T) const { return x == T.x && y == T.y; }
    inline bool operator!=(const Point& T) const { return !(*this == T); }
};

int readLefUnits(lefrCallbackType_e c, lefiUnits* unit, lefiUserData ud);
int readLefProp(lefrCallbackType_e c, lefiProp* prop, lefiUserData ud);
int readLefLayer(lefrCallbackType_e c, lefiLayer* leflayer, lefiUserData ud);
int readLefVia(lefrCallbackType_e c, lefiVia* lvia, lefiUserData ud);
int readLefMacroBegin(lefrCallbackType_e c, const char* macroName, lefiUserData ud);
int readLefObs(lefrCallbackType_e c, lefiObstruction* obs, lefiUserData ud);
int readLefPin(lefrCallbackType_e c, lefiPin* pin, lefiUserData ud);
int readLefSite(lefrCallbackType_e c, lefiSite* site, lefiUserData ud);
int readLefMacro(lefrCallbackType_e c, lefiMacro* macro, lefiUserData ud);
int readLefMacroEnd(lefrCallbackType_e c, const char* macroName, lefiUserData ud);

int readDefUnits(defrCallbackType_e c, double d, defiUserData ud);
int readDefVersion(defrCallbackType_e c, double d, defiUserData ud);
int readDefDesign(defrCallbackType_e c, const char* name, defiUserData ud);
int readDefDieArea(defrCallbackType_e c, defiBox* dbox, defiUserData ud);
int readDefRow(defrCallbackType_e c, defiRow* drow, defiUserData ud);
int readDefTrack(defrCallbackType_e c, defiTrack* dtrack, defiUserData ud);
int readDefGcellGrid(defrCallbackType_e c, defiGcellGrid* dgrid, defiUserData ud);
int readDefViaStart(defrCallbackType_e c, int num, defiUserData ud);
int readDefVia(defrCallbackType_e c, defiVia* dvia, defiUserData ud);
int readDefNdr(defrCallbackType_e c, defiNonDefault* nd, defiUserData ud);
int readDefComponentStart(defrCallbackType_e c, int num, defiUserData ud);
int readDefComponent(defrCallbackType_e c, defiComponent* co, defiUserData ud);
int readDefPinStart(defrCallbackType_e c, int num, defiUserData ud);
int readDefPin(defrCallbackType_e c, defiPin* dpin, defiUserData ud);
int readDefBlockageStart(defrCallbackType_e c, int num, defiUserData ud);
int readDefBlockage(defrCallbackType_e c, defiBlockage* dblk, defiUserData ud);
int readDefSNetStart(defrCallbackType_e c, int num, defiUserData ud);
int readDefSNet(defrCallbackType_e c, defiNet* dnet, defiUserData ud);
int readDefNetStart(defrCallbackType_e c, int num, defiUserData ud);
int readDefNet(defrCallbackType_e c, defiNet* dnet, defiUserData ud);
int readDefRegionStart(defrCallbackType_e c, int num, defiUserData ud);
int readDefRegion(defrCallbackType_e c, defiRegion* dreg, defiUserData ud);
int readDefGroupName(defrCallbackType_e c, const char* cl, defiUserData ud);
int readDefGroupMember(defrCallbackType_e c, const char* cl, defiUserData ud);
int readDefGroup(defrCallbackType_e c, defiGroup* dgp, defiUserData ud);
inline void fastCopy(char* t, const char* s, size_t n);


namespace {

bool fastDefComponentsEnabled()
{
    const char* value = std::getenv("XPLACE_FAST_DEF_COMPONENTS");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' ||
             value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

struct FastDefToken {
    const char* begin = nullptr;
    const char* end = nullptr;

    bool empty() const { return begin == end; }

    bool equals(const char* text) const
    {
        return defTokenEquals(begin, end, text);
    }

    std::string str() const
    {
        return std::string(begin, end);
    }
};

struct FastDefObjectRange {
    std::size_t begin = 0;
    std::size_t end = 0;
};

struct FastDefComponent {
    FastDefToken name;
    FastDefToken type;
    int x = INT_MIN;
    int y = INT_MIN;
    int orient = -1;
    char placement = 'u';  // u: unplaced, p: placed, f: fixed
};

const char* fastDefSkipWsAndComments(const char* p, const char* end)
{
    while (p < end) {
        while (p < end && std::isspace(static_cast<unsigned char>(*p))) {
            ++p;
        }
        if (p < end && *p == '#') {
            while (p < end && *p != '\n') {
                ++p;
            }
            continue;
        }
        break;
    }
    return p;
}

bool fastDefNextToken(const char*& p, const char* end, FastDefToken& token)
{
    p = fastDefSkipWsAndComments(p, end);
    if (p >= end) {
        token = FastDefToken{end, end};
        return false;
    }
    const char ch = *p;
    if (ch == '-' && p + 1 < end && std::isdigit(static_cast<unsigned char>(p[1]))) {
        const char* begin = p++;
        while (p < end && std::isdigit(static_cast<unsigned char>(*p))) {
            ++p;
        }
        token = FastDefToken{begin, p};
        return true;
    }
    if (ch == '(' || ch == ')' || ch == '+' || ch == '-' || ch == ';') {
        token = FastDefToken{p, p + 1};
        ++p;
        return true;
    }
    const char* begin = p;
    while (p < end) {
        const char c = *p;
        if (std::isspace(static_cast<unsigned char>(c)) ||
            c == '(' || c == ')' || c == '+' || c == '-' || c == ';') {
            break;
        }
        ++p;
    }
    token = FastDefToken{begin, p};
    return true;
}

bool fastDefParseIntToken(const FastDefToken& token, int& value)
{
    if (token.empty()) {
        return false;
    }
    const char* p = token.begin;
    bool negative = false;
    if (p < token.end && *p == '-') {
        negative = true;
        ++p;
    }
    if (p == token.end || !std::isdigit(static_cast<unsigned char>(*p))) {
        return false;
    }
    long long result = 0;
    while (p < token.end && std::isdigit(static_cast<unsigned char>(*p))) {
        result = result * 10 + (*p - '0');
        ++p;
    }
    if (p != token.end) {
        return false;
    }
    value = static_cast<int>(negative ? -result : result);
    return true;
}

int fastDefOrient(const FastDefToken& token)
{
    if (token.equals("N")) return 0;
    if (token.equals("W")) return 1;
    if (token.equals("S")) return 2;
    if (token.equals("E")) return 3;
    if (token.equals("FN")) return 4;
    if (token.equals("FW")) return 5;
    if (token.equals("FS")) return 6;
    if (token.equals("FE")) return 7;
    if (token.equals("NONE")) return -1;
    return 0;
}

void fastDefCollectObjectRanges(const char* data,
                                const DefBufferSection& section,
                                int num_threads,
                                std::vector<FastDefObjectRange>& ranges)
{
    ranges.clear();
    if (!section.found || section.body_begin >= section.end_begin) {
        return;
    }
    num_threads = std::max(1, num_threads);
    const std::size_t begin = section.body_begin;
    const std::size_t end = section.end_begin;
    const std::size_t size = end - begin;
    std::vector<std::vector<std::size_t>> starts_by_thread(num_threads);

    auto scan_chunk = [&](int tid) {
        const std::size_t chunk_begin =
            begin + (size * static_cast<std::size_t>(tid)) / static_cast<std::size_t>(num_threads);
        const std::size_t chunk_end =
            begin + (size * static_cast<std::size_t>(tid + 1)) / static_cast<std::size_t>(num_threads);
        const std::size_t scan_begin = defAlignLineStart(data, end, chunk_begin);
        const std::size_t scan_end = (tid + 1 == num_threads) ? end : defAlignLineStart(data, end, chunk_end);
        std::vector<std::size_t>& starts = starts_by_thread[tid];
        starts.reserve(static_cast<std::size_t>(section.declared_count > 0
                           ? section.declared_count / num_threads + 1
                           : 1024));
        for (std::size_t line_begin_offset = scan_begin; line_begin_offset < scan_end;) {
            const void* newline = std::memchr(data + line_begin_offset, '\n', scan_end - line_begin_offset);
            const std::size_t line_end_offset = newline == nullptr ? scan_end : static_cast<const char*>(newline) - data;
            const std::size_t next_line_offset = line_end_offset < scan_end ? line_end_offset + 1 : line_end_offset;
            const char* line_begin = data + line_begin_offset;
            const char* line_end = data + line_end_offset;
            const char* first = defSkipWs(line_begin, line_end);
            if (first < line_end && *first == '-') {
                starts.push_back(static_cast<std::size_t>(first - data));
            }
            line_begin_offset = next_line_offset;
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(num_threads > 1 ? static_cast<std::size_t>(num_threads - 1) : 0);
    for (int tid = 1; tid < num_threads; ++tid) {
        workers.emplace_back(scan_chunk, tid);
    }
    scan_chunk(0);
    for (std::thread& worker : workers) {
        worker.join();
    }

    std::size_t total = 0;
    for (const auto& starts : starts_by_thread) {
        total += starts.size();
    }
    std::vector<std::size_t> starts;
    starts.reserve(total);
    for (const auto& local : starts_by_thread) {
        starts.insert(starts.end(), local.begin(), local.end());
    }
    starts_by_thread.clear();
    std::sort(starts.begin(), starts.end());

    ranges.resize(starts.size());
    for (std::size_t i = 0; i < starts.size(); ++i) {
        ranges[i].begin = starts[i];
        ranges[i].end = (i + 1 < starts.size()) ? starts[i + 1] : section.end_begin;
    }
}

bool fastDefParseComponent(const char* data,
                           const FastDefObjectRange& range,
                           FastDefComponent& component)
{
    const char* p = data + range.begin;
    const char* end = data + range.end;
    FastDefToken token;
    if (!fastDefNextToken(p, end, token) || !token.equals("-")) {
        return false;
    }
    if (!fastDefNextToken(p, end, component.name) || component.name.empty()) {
        return false;
    }
    if (!fastDefNextToken(p, end, component.type) || component.type.empty()) {
        return false;
    }

    while (fastDefNextToken(p, end, token)) {
        if (token.equals(";")) {
            return true;
        }
        if (token.equals("UNPLACED")) {
            component.placement = 'u';
            continue;
        }
        if (!token.equals("PLACED") && !token.equals("FIXED")) {
            continue;
        }
        component.placement = token.equals("FIXED") ? 'f' : 'p';
        FastDefToken open;
        FastDefToken x_token;
        FastDefToken y_token;
        FastDefToken close;
        FastDefToken orient;
        if (!fastDefNextToken(p, end, open) || !open.equals("(") ||
            !fastDefNextToken(p, end, x_token) ||
            !fastDefNextToken(p, end, y_token) ||
            !fastDefNextToken(p, end, close) || !close.equals(")") ||
            !fastDefNextToken(p, end, orient)) {
            return false;
        }
        if (!fastDefParseIntToken(x_token, component.x) ||
            !fastDefParseIntToken(y_token, component.y)) {
            return false;
        }
        component.orient = fastDefOrient(orient);
    }
    return false;
}

bool fastDefParseComponents(const char* data,
                            const DefBufferSection& section,
                            int num_threads,
                            std::vector<FastDefComponent>& components)
{
    std::vector<FastDefObjectRange> ranges;
    fastDefCollectObjectRanges(data, section, num_threads, ranges);
    if (section.declared_count >= 0 &&
        ranges.size() != static_cast<std::size_t>(section.declared_count)) {
        std::fprintf(stderr,
                     "[XPLACE_FAST_DEF_COMPONENTS] component count mismatch declared=%lld found=%zu\n",
                     section.declared_count, ranges.size());
        return false;
    }

    components.resize(ranges.size());
    num_threads = std::max(1, num_threads);
    std::vector<int> ok(num_threads, 1);
    auto parse_chunk = [&](int tid) {
        const std::size_t begin = (ranges.size() * static_cast<std::size_t>(tid)) /
                                  static_cast<std::size_t>(num_threads);
        const std::size_t end = (ranges.size() * static_cast<std::size_t>(tid + 1)) /
                                static_cast<std::size_t>(num_threads);
        for (std::size_t i = begin; i < end; ++i) {
            if (!fastDefParseComponent(data, ranges[i], components[i])) {
                ok[tid] = 0;
                return;
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(num_threads > 1 ? static_cast<std::size_t>(num_threads - 1) : 0);
    for (int tid = 1; tid < num_threads; ++tid) {
        workers.emplace_back(parse_chunk, tid);
    }
    parse_chunk(0);
    for (std::thread& worker : workers) {
        worker.join();
    }
    return std::all_of(ok.begin(), ok.end(), [](int value) { return value != 0; });
}

void fastDefMaterializeComponents(Database& db,
                                  const std::vector<FastDefComponent>& components,
                                  int num_threads)
{
    const std::size_t base = db.cells.size();
    const std::size_t count = components.size();
    db.cells.resize(base + count, nullptr);
    db.name_cells.reserve(base + count);
    num_threads = std::max(1, num_threads);

    auto build_chunk = [&](int tid) {
        const std::size_t begin = (count * static_cast<std::size_t>(tid)) /
                                  static_cast<std::size_t>(num_threads);
        const std::size_t end = (count * static_cast<std::size_t>(tid + 1)) /
                                static_cast<std::size_t>(num_threads);
        for (std::size_t i = begin; i < end; ++i) {
            const FastDefComponent& component = components[i];
            std::string type_name(component.type.begin, component.type.end);
            CellType* celltype = db.getCellType(type_name);
            std::string cell_name(component.name.begin, component.name.end);
            validate_token(cell_name);

            Cell* cell = new Cell(cell_name, celltype);
            if (component.placement == 'u') {
                cell->fixed(false);
                cell->unplace();
            } else {
                cell->place(component.x, component.y, component.orient);
                if (component.placement == 'f') {
                    cell->fixed(true);
                } else if (celltype && (celltype->cls == "CORE" || celltype->cls == "BLOCK")) {
                    cell->fixed(false);
                } else {
                    cell->fixed(true);
                }
            }
            db.cells[base + i] = cell;
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(num_threads > 1 ? static_cast<std::size_t>(num_threads - 1) : 0);
    for (int tid = 1; tid < num_threads; ++tid) {
        workers.emplace_back(build_chunk, tid);
    }
    build_chunk(0);
    for (std::thread& worker : workers) {
        worker.join();
    }

    for (std::size_t i = 0; i < count; ++i) {
        Cell* cell = db.cells[base + i];
        if (!cell) {
            continue;
        }
        auto inserted = db.name_cells.emplace(cell->name(), cell);
        if (!inserted.second) {
            logger.warning("cell re-defined: %s", cell->name().c_str());
        }
    }
}


bool fastDefNetsEnabled()
{
    const char* value = std::getenv("XPLACE_FAST_DEF_NETS");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' ||
             value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

struct FastDefConnection {
    FastDefToken inst;
    FastDefToken pin;
    bool is_iopin = false;
};

struct FastDefNet {
    FastDefToken name;
    std::vector<FastDefConnection> connections;
    char type = 's';
    bool skip = false;
};

char fastDefNetUseType(const FastDefToken& token)
{
    if (token.equals("POWER")) return 'p';
    if (token.equals("GROUND")) return 'g';
    if (token.equals("SIGNAL")) return 's';
    if (token.equals("CLOCK")) return 'c';
    return 's';
}

bool fastDefParseNet(const char* data,
                     const FastDefObjectRange& range,
                     FastDefNet& net)
{
    const char* p = data + range.begin;
    const char* end = data + range.end;
    FastDefToken token;
    if (!fastDefNextToken(p, end, token) || !token.equals("-")) {
        return false;
    }
    if (!fastDefNextToken(p, end, net.name) || net.name.empty()) {
        return false;
    }

    bool in_options = false;
    while (fastDefNextToken(p, end, token)) {
        if (token.equals(";")) {
            net.skip = net.connections.empty();
            return true;
        }
        if (token.equals("+")) {
            in_options = true;
            FastDefToken keyword;
            if (!fastDefNextToken(p, end, keyword)) {
                return false;
            }
            if (keyword.equals("USE")) {
                FastDefToken use;
                if (!fastDefNextToken(p, end, use)) {
                    return false;
                }
                net.type = fastDefNetUseType(use);
            }
            continue;
        }
        if (in_options || !token.equals("(")) {
            continue;
        }
        FastDefToken inst;
        FastDefToken pin;
        FastDefToken close;
        if (!fastDefNextToken(p, end, inst) ||
            !fastDefNextToken(p, end, pin) ||
            !fastDefNextToken(p, end, close) || !close.equals(")")) {
            return false;
        }
        net.connections.push_back(FastDefConnection{inst, pin, inst.equals("PIN")});
    }
    return false;
}

bool fastDefParseNets(const char* data,
                      const DefBufferSection& section,
                      int num_threads,
                      std::vector<FastDefNet>& nets)
{
    std::vector<FastDefObjectRange> ranges;
    fastDefCollectObjectRanges(data, section, num_threads, ranges);
    if (section.declared_count >= 0 &&
        ranges.size() != static_cast<std::size_t>(section.declared_count)) {
        std::fprintf(stderr,
                     "[XPLACE_FAST_DEF_NETS] net count mismatch declared=%lld found=%zu\n",
                     section.declared_count, ranges.size());
        return false;
    }

    nets.resize(ranges.size());
    num_threads = std::max(1, num_threads);
    std::vector<int> ok(num_threads, 1);
    auto parse_chunk = [&](int tid) {
        const std::size_t begin = (ranges.size() * static_cast<std::size_t>(tid)) /
                                  static_cast<std::size_t>(num_threads);
        const std::size_t end = (ranges.size() * static_cast<std::size_t>(tid + 1)) /
                                static_cast<std::size_t>(num_threads);
        for (std::size_t i = begin; i < end; ++i) {
            if (!fastDefParseNet(data, ranges[i], nets[i])) {
                ok[tid] = 0;
                return;
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(num_threads > 1 ? static_cast<std::size_t>(num_threads - 1) : 0);
    for (int tid = 1; tid < num_threads; ++tid) {
        workers.emplace_back(parse_chunk, tid);
    }
    parse_chunk(0);
    for (std::thread& worker : workers) {
        worker.join();
    }
    return std::all_of(ok.begin(), ok.end(), [](int value) { return value != 0; });
}

void fastDefMaterializeNets(Database& db,
                            const std::vector<FastDefNet>& parsed_nets,
                            int num_threads)
{
    std::vector<int> out_index(parsed_nets.size(), -1);
    std::size_t valid_count = 0;
    for (std::size_t i = 0; i < parsed_nets.size(); ++i) {
        const FastDefNet& parsed = parsed_nets[i];
        if (parsed.skip || parsed.connections.empty()) {
            continue;
        }
        std::string net_name(parsed.name.begin, parsed.name.end);
        validate_token(net_name);
        if (net_name == "VDD" || net_name == "VSS") {
            continue;
        }
        out_index[i] = static_cast<int>(valid_count++);
    }

    const std::size_t base = db.nets.size();
    db.nets.resize(base + valid_count, nullptr);
    db.name_nets.reserve(base + valid_count);
    num_threads = std::max(1, num_threads);

    auto build_chunk = [&](int tid) {
        const std::size_t begin = (parsed_nets.size() * static_cast<std::size_t>(tid)) /
                                  static_cast<std::size_t>(num_threads);
        const std::size_t end = (parsed_nets.size() * static_cast<std::size_t>(tid + 1)) /
                                static_cast<std::size_t>(num_threads);
        for (std::size_t i = begin; i < end; ++i) {
            const int local_out = out_index[i];
            if (local_out < 0) {
                continue;
            }
            const FastDefNet& parsed = parsed_nets[i];
            std::string net_name(parsed.name.begin, parsed.name.end);
            validate_token(net_name);
            Net* net = new Net(net_name, nullptr);
            net->_type = parsed.type;
            if (parsed.connections.size() > 4) {
                net->pins.reserve(parsed.connections.size());
            }

            for (const FastDefConnection& connection : parsed.connections) {
                Pin* pin = nullptr;
                if (connection.is_iopin) {
                    std::string iopin_name(connection.pin.begin, connection.pin.end);
                    IOPin* iopin = db.getIOPin(iopin_name);
                    if (!iopin) {
                        continue;
                    }
                    pin = iopin->pin;
                    iopin->is_connected.store(true, std::memory_order_relaxed);
                } else {
                    std::string cell_name(connection.inst.begin, connection.inst.end);
                    validate_token(cell_name);
                    Cell* cell = db.getCell(cell_name);
                    if (!cell) {
                        continue;
                    }
                    std::string pin_name(connection.pin.begin, connection.pin.end);
                    pin = cell->pin(pin_name.c_str());
                    if (!pin) {
                        continue;
                    }
                    cell->is_connected.store(true, std::memory_order_relaxed);
                }
                pin->net = net;
                pin->is_connected.store(true, std::memory_order_relaxed);
                net->pins.push_back(pin);
            }
            db.nets[base + static_cast<std::size_t>(local_out)] = net;
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(num_threads > 1 ? static_cast<std::size_t>(num_threads - 1) : 0);
    for (int tid = 1; tid < num_threads; ++tid) {
        workers.emplace_back(build_chunk, tid);
    }
    build_chunk(0);
    for (std::thread& worker : workers) {
        worker.join();
    }

    for (std::size_t i = 0; i < valid_count; ++i) {
        Net* net = db.nets[base + i];
        if (!net) {
            continue;
        }
        auto inserted = db.name_nets.emplace(net->name, net);
        if (!inserted.second) {
            logger.warning("Net re-defined: %s", net->name.c_str());
        }
    }
}


struct FastDefStripRange {
    std::size_t begin = 0;
    std::size_t end = 0;
    const char* name = "";
};

void fastDefAddStripRange(std::vector<FastDefStripRange>& ranges,
                          const DefBufferSection& section)
{
    if (!section.found || section.end_after <= section.header_begin) {
        return;
    }
    ranges.push_back(FastDefStripRange{section.header_begin, section.end_after, section.name});
}

std::vector<char> fastDefBuildDefrStubBuffer(const char* data,
                                             std::size_t size,
                                             std::vector<FastDefStripRange> ranges)
{
    std::sort(ranges.begin(), ranges.end(), [](const FastDefStripRange& lhs,
                                               const FastDefStripRange& rhs) {
        return lhs.begin < rhs.begin;
    });

    std::size_t removed_bytes = 0;
    std::size_t replacement_bytes = 0;
    for (const FastDefStripRange& range : ranges) {
        if (range.end <= range.begin || range.end > size) {
            continue;
        }
        removed_bytes += range.end - range.begin;
        replacement_bytes += std::strlen(range.name) * 2 + 16;
    }

    std::vector<char> buffer;
    buffer.reserve(size - removed_bytes + replacement_bytes + 1);
    std::size_t cursor = 0;
    for (const FastDefStripRange& range : ranges) {
        if (range.end <= range.begin || range.end > size || range.begin < cursor) {
            continue;
        }
        buffer.insert(buffer.end(), data + cursor, data + range.begin);
        char replacement[128];
        const int n = std::snprintf(replacement,
                                    sizeof(replacement),
                                    "%s 0 ;\nEND %s\n",
                                    range.name,
                                    range.name);
        if (n > 0) {
            buffer.insert(buffer.end(), replacement, replacement + n);
        }
        cursor = range.end;
    }
    buffer.insert(buffer.end(), data + cursor, data + size);
    return buffer;
}

bool readDEFWithFastComponents(Database& db, const std::string& file)
{
    if (!fastDefComponentsEnabled() ||
        !setting.SkipDefNetWires ||
        !setting.SkipDefBlockages ||
        setting.EnablePG ||
        setting.EnableFence) {
        return false;
    }

    const auto start = std::chrono::steady_clock::now();
    auto seconds_since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    };
    auto last_phase = start;

    DefMappedFile mapped(file);
    if (mapped.data == nullptr || mapped.size == 0) {
        return false;
    }
    DefBufferSection components_section;
    defFindSection(mapped.data, mapped.size, "COMPONENTS", components_section);
    if (!components_section.found) {
        std::fprintf(stderr, "[XPLACE_FAST_DEF_COMPONENTS] COMPONENTS section not found; fallback\n");
        return false;
    }

    int num_threads = std::max(1, setting.numThreads);
    if (const char* env_threads = std::getenv("XPLACE_FAST_DEF_THREADS")) {
        const int value = std::atoi(env_threads);
        if (value > 0) {
            num_threads = value;
        }
    }

    std::vector<FastDefComponent> components;
    if (!fastDefParseComponents(mapped.data, components_section, num_threads, components)) {
        std::fprintf(stderr, "[XPLACE_FAST_DEF_COMPONENTS] parse failed; fallback\n");
        return false;
    }
    auto now = std::chrono::steady_clock::now();
    std::fprintf(stdout,
                 "[XPLACE_FAST_DEF_COMPONENTS] phase=parse_components elapsed=%.3f total=%.3f components=%zu threads=%d\n",
                 seconds_since(last_phase), seconds_since(start), components.size(), num_threads);
    std::fflush(stdout);
    last_phase = now;

    const bool use_fast_nets = fastDefNetsEnabled();
    std::vector<FastDefNet> parsed_nets;
    DefBufferSection nets_section;
    if (use_fast_nets) {
        defFindSection(mapped.data, mapped.size, "NETS", nets_section);
        if (!nets_section.found || !fastDefParseNets(mapped.data, nets_section, num_threads, parsed_nets)) {
            std::fprintf(stderr, "[XPLACE_FAST_DEF_NETS] parse failed; fallback\n");
            return false;
        }
        now = std::chrono::steady_clock::now();
        std::fprintf(stdout,
                     "[XPLACE_FAST_DEF_NETS] phase=parse_nets elapsed=%.3f total=%.3f nets=%zu threads=%d\n",
                     seconds_since(last_phase), seconds_since(start), parsed_nets.size(), num_threads);
        std::fflush(stdout);
        last_phase = now;
    }

    fastDefMaterializeComponents(db, components, num_threads);
    components.clear();
    components.shrink_to_fit();
    now = std::chrono::steady_clock::now();
    std::fprintf(stdout,
                 "[XPLACE_FAST_DEF_COMPONENTS] phase=materialize_components elapsed=%.3f total=%.3f cells=%zu\n",
                 seconds_since(last_phase), seconds_since(start), db.cells.size());
    std::fflush(stdout);
    last_phase = now;

    std::vector<char> defr_buffer;
    FILE* fp = nullptr;
    if (use_fast_nets) {
        std::vector<FastDefStripRange> strip_ranges;
        fastDefAddStripRange(strip_ranges, components_section);
        fastDefAddStripRange(strip_ranges, nets_section);
        defr_buffer = fastDefBuildDefrStubBuffer(mapped.data, mapped.size, std::move(strip_ranges));
        now = std::chrono::steady_clock::now();
        std::fprintf(stdout,
                     "[XPLACE_FAST_DEF_NETS] phase=build_defr_stub elapsed=%.3f total=%.3f bytes=%zu original_bytes=%zu\n",
                     seconds_since(last_phase), seconds_since(start), defr_buffer.size(), mapped.size);
        std::fflush(stdout);
        last_phase = now;
        fp = fmemopen(defr_buffer.data(), defr_buffer.size(), "r");
        if (!fp) {
            logger.error("Unable to open stripped DEF buffer: %s", std::strerror(errno));
            return false;
        }
    } else if (!(fp = fopen(file.c_str(), "r"))) {
        logger.error("Unable to open DEF file: %s", file.c_str());
        return false;
    }

    g_def_profile.begin();

    defrSetDesignCbk(readDefDesign);
    defrSetUnitsCbk(readDefUnits);
    defrSetVersionCbk(readDefVersion);
    defrSetDieAreaCbk(readDefDieArea);
    defrSetRowCbk(readDefRow);
    defrSetTrackCbk(readDefTrack);
    defrSetGcellGridCbk(readDefGcellGrid);
    defrSetViaStartCbk(readDefViaStart);
    defrSetViaCbk(readDefVia);
    defrSetNonDefaultCbk(readDefNdr);
    defrSetComponentStartCbk(readDefComponentStart);
    defrSetStartPinsCbk(readDefPinStart);
    defrSetPinCbk(readDefPin);
    defrSetNetStartCbk(readDefNetStart);
    if (!use_fast_nets) {
        defrSetNetCbk(readDefNet);
    }

    defrInit();
    defrReset();
    const int res = defrRead(fp, file.c_str(), (void*)&db, 1);
    if (res) {
        logger.error("Error in reading DEF");
        g_def_profile.finish("fast_components_error");
        defrReleaseNResetMemory();
        defrUnsetCallbacks();
        fclose(fp);
        return false;
    }
    g_def_profile.finish(use_fast_nets ? "fast_components_nets_defr_ok" : "fast_components_ok");
    defrReleaseNResetMemory();
    defrUnsetCallbacks();
    fclose(fp);

    if (use_fast_nets) {
        fastDefMaterializeNets(db, parsed_nets, num_threads);
        parsed_nets.clear();
        parsed_nets.shrink_to_fit();
        now = std::chrono::steady_clock::now();
        std::fprintf(stdout,
                     "[XPLACE_FAST_DEF_NETS] phase=materialize_nets elapsed=%.3f total=%.3f nets=%zu\n",
                     seconds_since(last_phase), seconds_since(start), db.nets.size());
        std::fflush(stdout);
        last_phase = now;
    }

    std::fprintf(stdout,
                 "[XPLACE_FAST_DEF_COMPONENTS] phase=total elapsed=%.3f\n",
                 seconds_since(start));
    std::fflush(stdout);
    return true;
}

}  // namespace

bool Database::readLEF(const std::string& file) {
    FILE* fp;
    if (!(fp = fopen(file.c_str(), "r"))) {
        logger.error("Unable to open LEF file: %s", file.c_str());
        return false;
    }

#ifndef NDEBUG
    logger.info("reading %s", file.c_str());
#endif

    lefrSetUnitsCbk(readLefUnits);
    lefrSetPropCbk(readLefProp);
    lefrSetLayerCbk(readLefLayer);
    lefrSetViaCbk(readLefVia);
    lefrSetMacroCbk(readLefMacro);
    lefrSetMacroBeginCbk(readLefMacroBegin);
    lefrSetObstructionCbk(readLefObs);
    lefrSetPinCbk(readLefPin);
    lefrSetSiteCbk(readLefSite);
    lefrSetMacroEndCbk(readLefMacroEnd);
    lefrInit();
    lefrReset();
    int res = lefrRead(fp, file.c_str(), (void*)this);
    if (res) {
        logger.error("Error in reading LEF");
        return false;
    }
    lefrReleaseNResetMemory();
    //  lefrUnsetCallbacks();
    lefrUnsetLayerCbk();
    lefrUnsetNonDefaultCbk();
    lefrUnsetViaCbk();
    fclose(fp);
    return true;
}

bool Database::readDEF(const std::string& file) {
    FILE* fp;
    if (!(fp = fopen(file.c_str(), "r"))) {
        logger.error("Unable to open DEF file: %s", file.c_str());
        return false;
    }

#ifndef NDEBUG
    logger.info("reading %s", file.c_str());
#endif

    profileDefBufferScan(file, setting.numThreads);
    if (readDEFWithFastComponents(*this, file)) {
        return true;
    }

    g_def_profile.begin();

    defrSetDesignCbk(readDefDesign);
    defrSetUnitsCbk(readDefUnits);
    defrSetVersionCbk(readDefVersion);
    defrSetDieAreaCbk(readDefDieArea);
    defrSetRowCbk(readDefRow);
    defrSetTrackCbk(readDefTrack);
    defrSetGcellGridCbk(readDefGcellGrid);
    defrSetViaStartCbk(readDefViaStart);
    defrSetViaCbk(readDefVia);
    defrSetNonDefaultCbk(readDefNdr);
    defrSetComponentStartCbk(readDefComponentStart);
    defrSetComponentCbk(readDefComponent);
    defrSetStartPinsCbk(readDefPinStart);
    defrSetPinCbk(readDefPin);
    if (!setting.SkipDefBlockages) {
        defrSetBlockageStartCbk(readDefBlockageStart);
        defrSetBlockageCbk(readDefBlockage);
    }

    if (setting.EnablePG) {
        defrSetSNetStartCbk(readDefSNetStart);
        defrSetSNetCbk(readDefSNet);
        //  defrSetSNetWireCbk(readDefSnetwire);
    }
    defrSetNetStartCbk(readDefNetStart);
    defrSetNetCbk(readDefNet);
    if (!setting.SkipDefNetWires) {
        // augment nets with path data
        defrSetAddPathToNet();
    }

    if (setting.EnableFence) {
        defrSetRegionStartCbk(readDefRegionStart);
        defrSetRegionCbk(readDefRegion);
        //  defrSetGroupNameCbk(readDefGroupName);
        defrSetGroupMemberCbk(readDefGroupMember);
        defrSetGroupCbk(readDefGroup);
    }

    defrInit();
    defrReset();
    int res = defrRead(fp, file.c_str(), (void*)this, 1);
    if (res) {
        logger.error("Error in reading DEF");
        g_def_profile.finish("error");
        defrReleaseNResetMemory();
        defrUnsetCallbacks();
        fclose(fp);
        return false;
    }
    g_def_profile.finish("ok");
    defrReleaseNResetMemory();
    defrUnsetCallbacks();
    fclose(fp);
    return true;
}

bool Database::readDEFPG(const std::string& file) {
    //  shape of pre-routed PG mesh is not supported by the reader
    string buffer;
    std::ifstream ifs(file.c_str());
    if (!ifs.good()) {
        logger.error("Unable to open DEF PG file: %s", file.c_str());
        return false;
    }

#ifndef NDEBUG
    logger.info("reading %s", file.c_str());
#endif

    Database* db = this;
    int n_snets = 0;

    //  ignore lines until the keyword
    do {
        ifs >> buffer;
        if (buffer == "SPECIALNETS") {
            ifs >> n_snets >> buffer;
            break;
        }
    } while (!ifs.eof());

    for (int i = 0; i < n_snets; i++) {
        //  SNet* snet = NULL;
        do {
            ifs >> buffer;
            if (buffer == "-") {
                string snetname;
                ifs >> snetname;
                // snet = db->addSNet(snetname);
            }
            if (buffer == "ROUTED" || buffer == "NEW") {
                string layername;
                int wirewidth;
                string vianame;
                string shape;
                char direction;
                int fx, fy;
                //  int tx, ty;
                string tx_str, ty_str;
                ifs >> layername >> wirewidth;
                if (wirewidth == 0) {
                    // via
                    ifs >> buffer >> buffer >> shape >> buffer >> fx >> fy >> buffer >> vianame;
                    ViaType* viatype = db->getViaType(vianame);
                    if (!viatype) {
                        logger.error("Via type is not defined: %s", vianame.c_str());
                        return false;
                    }
                    //  snet->addVia(viatype, fx, fy);
                } else {
                    // wire
                    ifs >> buffer >> buffer >> shape >> buffer >> fx >> fy >> buffer >> buffer >> tx_str >> ty_str >>
                        buffer;
                    if (tx_str == "*") {
                        //  tx = fx;
                    } else {
                        //  tx = atoi(tx_str.c_str());
                    }
                    if (ty_str == "*") {
                        //  ty = fy;
                    } else {
                        //  ty = atoi(ty_str.c_str());
                    }
                    //  if (tx == fx) {
                    //      direction = 'v';
                    //  } else {
                    //      direction = 'h';
                    //  }
                    Layer* layer = db->getLayer(layername);
                    if (!layer) {
                        logger.error("Layer is not defined: %s", layername.c_str());
                        return false;
                    }
                    //  int lx, ly, hx, hy;
                    if (direction == 'h') {
                        //  lx = min(fx, tx);
                        //  ly = fy - wirewidth / 2;
                        //  hx = max(fx, tx);
                        //  hy = ly + wirewidth;
                    } else {
                        //  lx = fx - wirewidth / 2;
                        //  ly = min(fy, ty);
                        //  hx = lx + wirewidth;
                        //  hy = max(fy, ty);
                    }
                    //  snet->addShape(*layer, lx, ly, hx, hy);
                    if ((layer->rIndex == 0 || layer->rIndex == 1) && direction == 'h') {
                        //  db->powerNet->addRail(snet, lx, hx, fy);
                    }
                }
            }
            if (buffer == "USE") {
                string type;
                ifs >> type;
                if (type == "POWER") {
                    //  snet->type = 'p';
                } else if (type == "GROUND") {
                    //  snet->type = 'g';
                } else {
                    logger.error("unknown use: %s", type.c_str());
                }
            }
            if (buffer == ";") {
                break;
            }
        } while (!ifs.eof());
    }
    ifs.close();
    return true;
}

string expand_name(const std::string& name) {
    // add '\' before '[' or ']'
    std::string result;
    for (char c : name) {
        if (c == '[' || c == ']') {
            result.push_back('\\');
        }
        result.push_back(c);
    }
    return result;
}

bool Database::writeComponents(std::ofstream& ofs) {
    int nCells = cells.size();
    ofs << "COMPONENTS " << nCells << " ;" << std::endl;
    // ofs << "COMPONENTS " << nCells << " ;" << std::endl;
    for (int i = 0; i < nCells; i++) {
        Cell* cell = cells[i];
#ifdef WRITE_BUFFER
        std::ostringstream oss;
#else
        std::ofstream& oss = ofs;
#endif
        oss << "   - " << cell->name() << " " << cell->ctype()->name << std::endl;
        // ofs << "   - " << cell->name() << " " << cell->ctype()->name <<
        // std::endl;
        if (cell->fixed()) {
            oss << "      + FIXED ( " << cell->lx() << " " << cell->ly() << " ) " << getOrient(cell->orient()) << " ;"
                << std::endl;
            // ofs << "      + FIXED ( " << cell->lx() << " " << cell->ly() << "
            // ) "
            //    << getOrient(cell->orient())
            //    << " ;" << std::endl;
        } else if (cell->placed()) {
            oss << "      + PLACED ( " << cell->lx() << " " << cell->ly() << " ) " << getOrient(cell->orient()) << " ;"
                << std::endl;
            // ofs << "      + PLACED ( " << cell->lx() << " " << cell->ly() <<
            // " ) "
            //    << getOrient(cell->orient())
            //    << " ;" << std::endl;
        } else {
            oss << "      + UNPLACED ;" << std::endl;
            // ofs << "      + UNPLACED ;" << std::endl;
        }
#ifdef WRITE_BUFFER
        string lines = oss.str();
        writeBuffer(ofs, lines);
#endif
    }
#ifdef WRITE_BUFFER
    writeBufferFlush(ofs);
#endif
    ofs << "END COMPONENTS\n\n";
    return true;
}

bool Database::writeNets(std::ofstream& ofs) {
    int nNets = nets.size();
    ofs << "NETS " << nNets << " ;" << std::endl;
    for (int i = 0; i < nNets; i++) {
        Net* net = nets[i];
        std::ofstream& oss = ofs;

        oss << "   - " << expand_name(net->name) << " \n";
        for (Pin* pin : net->pins) {
            if (pin->iopin)
                oss << " ( PIN "
                    << " " << pin->type->name() << " )";
            else
                oss << " ( " << expand_name(pin->cell->name()) << " " << pin->type->name() << " )";
        }

        const char use(net->_type);
        if (use == 'p') {
            oss << " + USE POWER ;\n";
        } else if (use == 'g') {
            oss << " + USE GROUND ;\n";
        } else if (use == 's') {
            oss << " + USE SIGNAL ;\n";
        } else if (use == 'c') {
            oss << " + USE CLOCK ;\n";
        } else {
            oss << ";\n";
        }
    }

    ofs << "END NETS\n\n";
    return true;
}

bool Database::writeICCAD2017(const string& inputDef, const string& outputDef) {
    std::ifstream ifs(inputDef.c_str());
    if (!ifs.good()) {
        logger.error("Unable to create/open DEF: %s", inputDef.c_str());
        return false;
    }

#ifndef NDEBUG
    logger.info("reading %s", inputDef.c_str());
#endif

    std::ofstream ofs(outputDef.c_str());
    if (!ofs.good()) {
        logger.error("Unable to create/open DEF: %s", outputDef.c_str());
        return false;
    }
    logger.info("writing %s", outputDef.c_str());

    string line;
    while (getline(ifs, line)) {
        std::istringstream iss(line);
        string s;
        if (!(iss >> s)) {
            ofs << line << std::endl;
            continue;
        } else if (s != "COMPONENTS") {
            ofs << line << std::endl;
            continue;
        }
        writeComponents(ofs);
        while (getline(ifs, line)) {
            std::istringstream iss(line);
            if (iss >> s && s == "END") {
                break;
            }
        }
        // process pair (a,b)
    }
    return true;
}

bool Database::writeICCAD2017(const string& outputDef) {
    std::ofstream ofs(outputDef.c_str(), std::ios::app);
    if (!ofs.good()) {
        logger.error("Unable to create/open DEF: %s", outputDef.c_str());
        return false;
    }
    logger.info("writing %s", outputDef.c_str());

    writeComponents(ofs);  // just replace the information of components, while
                           // others keep remain.

    ofs << "END DESIGN\n\n" << std::endl;

    ofs.close();
    return true;
}

bool Database::writeDEF(const string& file) {
    std::ofstream ofs(file.c_str());
    if (!ofs.good()) {
        logger.error("Unable to create/open DEF: %s", file.c_str());
        return false;
    }
    logger.info("writing %s", file.c_str());

    ofs << "VERSION 5.8 ;" << std::endl;
    ofs << "DIVIDERCHAR \"/\" ;" << std::endl;
    ofs << "BUSBITCHARS \"[]\" ;" << std::endl;
    ofs << "DESIGN " << designName << " ;" << std::endl;
    ofs << "UNITS DISTANCE MICRONS " << (int)DBU_Micron << " ; \n\n";
    ofs << "DIEAREA ( " << dieLX << ' ' << dieLY << " ) ( " << dieHX << ' ' << dieHY << " ) ;\n\n";

    for (Row* row : rows) {
        ofs << "ROW " << row->name() << ' ' << row->macro() << ' ' << row->x() << ' ' << row->y() << ' ';
        ofs << getRowOrient(false, row->flip());
        ofs << " DO " << row->xNum() << " BY " << row->yNum() << " STEP " << row->xStep() << ' ' << row->yStep()
            << " ;\n";
    }
    ofs << std::endl;

    for (Track* track : tracks) {
        ofs << "TRACKS " << track->macro() << ' ' << track->start << " DO " << track->num << " STEP " << track->step
            << " LAYER ";
        for (unsigned i = 0; i != track->numLayers(); ++i) {
            ofs << track->layer(i) << ' ';
        }
        ofs << ";\n";
    }
    ofs << std::endl;

    std::ostringstream ossv;
    unsigned nVias = 0;
    for (ViaType* via : viatypes) {
        if (!via->isDef()) {
            continue;
        }
        ++nVias;
        ossv << "\t- " << via->name;
        for (Geometry geo : via->rects) {
            ossv << "\n\t\t+ RECT " << geo.layer.name() << " ( " << geo.lx << ' ' << geo.ly << " ) ( " << geo.hx << ' '
                 << geo.hy << " )";
        }
        ossv << " ;\n";
    }
    ofs << "VIAS " << nVias << " ;\n";
    ofs << ossv.str();
    ofs << "END VIAS\n\n";

    ofs << "NONDEFAULTRULES " << ndrs.size() << " ;\n";
    for (const auto& [name, ndr] : ndrs) {
        //  const NDR* ndr = p.second;
        ofs << "\t- " << name;
        if (ndr->hardSpacing()) {
            ofs << "\n\t\t+ HARDSPACING";
        }
        for (const WireRule& rule : ndr->rules) {
            ofs << "\n\t\t+ LAYER " << rule.layer()->name() << " WIDTH " << rule.width;
            if (rule.space) {
                ofs << " SPACING " << rule.space;
            }
        }
        for (ViaType* via : ndr->vias) {
            ofs << "\n\t\t+ VIA " << via->name;
        }
        ofs << " ;\n";
    }
    ofs << "END NONDEFAULTRULES\n\n";

    std::ostringstream ossr;
    unsigned nRegions = 0;
    for (Region* region : regions) {
        if (region->name() == "default") {
            continue;
        }
        ++nRegions;
        ossr << "\t- " << region->name();
        for (const Rectangle& rect : region->rects) {
            ossr << "\t( " << rect.lx << ' ' << rect.ly << " ) ( " << rect.hx << ' ' << rect.hy << " )";
        }
        switch (region->type()) {
            case 'f':
                ossr << "\t+ TYPE FENCE ;\n";
                break;
            case 'g':
                ossr << "\t+ TYPE GUIDE ;\n";
                break;
            default:
                logger.error("region type not recognized: %c", region->type());
                ossr << " ;\n";
                break;
        }
    }
    ofs << "REGIONS " << nRegions << " ;\n";
    ofs << ossr.str();
    ofs << "END REGIONS\n\n";

    writeComponents(ofs);

    ofs << "PINS " << iopins.size() << " ;\n";
    for (IOPin* iopin : iopins) {
        ofs << "\t- " << iopin->name << " + NET " << iopin->netName();
        switch (iopin->type->direction()) {
            case 'f':
                ofs << "\n\n\t+ DIRECTION FEEDTHRU";
                break;
            case 'i':
                ofs << "\n\t\t+ DIRECTION OUTPUT";
                break;
            case 'o':
                ofs << "\n\t\t+ DIRECTION INPUT";
                break;
            case 'x':
                ofs << "\n\t\t+ DIRECTION INOUT";
                break;
            default:
                logger.error("iopin direction not recognized: %c", iopin->type->direction());
                break;
        }
        if (iopin->x != INT_MIN || iopin->y != INT_MIN) {
            ofs << "\n\t\t+ PLACED ( " << iopin->x << ' ' << iopin->y << " ) ";
        }
    }

    ofs << "END DESIGN" << std::endl;

    ofs.close();
    return true;
}

/***********************************/
/* .DEF File buffer Writing Scheme */
/***********************************/

/// a buffered writing scheme
bool Database::writeBuffer(std::ofstream& ofs, const string& line) {
    const char* b = line.c_str();
    size_t n = line.size();
    while (_bufferSize + n > _bufferCapacity)  // output exceeds the capacity
    {
        size_t nw = _bufferCapacity - _bufferSize;
        if (nw) {
            n -= nw;
            fastCopy(_buffer + _bufferSize, b, nw);
            b += nw;
        }

        ofs.write(_buffer,
                  _bufferCapacity);  // From _buffer write _bufferCapacity
                                     // charcters to ostream
        _bufferSize = 0;
    }

    if (n)  // wait until buffer is full then output
    {
        fastCopy(_buffer + _bufferSize, b, n);
        _bufferSize += n;
    }

    return true;
}

void Database::writeBufferFlush(std::ofstream& ofs) {
    if (_bufferSize)  // remember to write the rest content in the buffer
    {
        ofs.write(_buffer, _bufferSize);
        _bufferSize = 0;
    }
}

/// copy char* as unsigned long* and deal with boundary cases
inline void fastCopy(char* t, const char* s, size_t n) {
    if (n >= sizeof(unsigned long)) {
        unsigned long* tl = reinterpret_cast<unsigned long*>(t);
        const unsigned long* sl = reinterpret_cast<const unsigned long*>(s);

        while (n >= sizeof(unsigned long))  // copy char* as unsigned long*
        {
            *tl++ = *sl++;
            n -= sizeof(unsigned long);
        }

        t = reinterpret_cast<char*>(tl);
        s = reinterpret_cast<const char*>(sl);
    }

    while (n-- > 0)  // boundary cases
    {
        *t++ = *s++;
    }
}

/****************/
/* LEF Callback */
/****************/

int readLefUnits(lefrCallbackType_e c, lefiUnits* unit, lefiUserData ud) {
    Database* db = (Database*)ud;
    if (unit->lefiUnits::hasDatabase() && strcmp(unit->lefiUnits::databaseName(), "MICRONS") == 0) {
        db->LefConvertFactor = unit->lefiUnits::databaseNumber();
    }
    return 0;
}

//-----Property-----
int readLefProp(lefrCallbackType_e c, lefiProp* prop, lefiUserData ud) {
    Database* db = (Database*)ud;
    if (prop->lefiProp::hasString() && !strcmp(prop->lefiProp::propName(), "LEF58_CELLEDGESPACINGTABLE")) {
        std::stringstream sstable(prop->lefiProp::string());
        string buffer;

        while (!sstable.eof()) {
            sstable >> buffer;
            if (buffer == "EDGETYPE") {
                string type1, type2, type3;
                double microndist;
                sstable >> type1 >> type2 >> type3;
                if (type3 == "EXCEPTABUTTED") {
                    logger.warning("ignore EXCEPTABUTTED between %s and %s", type1.c_str(), type2.c_str());
                    sstable >> microndist;
                } else {
                    microndist = stod(type3);
                }
                int dbdist = 0;
                if (setting.EdgeSpacing) {
                    dbdist = (int)round(microndist * db->LefConvertFactor);
                }
                int type1_idx = db->edgetypes->getEdgeType(type1);
                int type2_idx = db->edgetypes->getEdgeType(type2);

                if (type1_idx < 0) {
                    type1_idx = db->edgetypes->types.size();
                    db->edgetypes->types.push_back(type1);
                }
                if (type2_idx < 0) {
                    type2_idx = db->edgetypes->types.size();
                    db->edgetypes->types.push_back(type2);
                }

                int numEdgeTypes = db->edgetypes->types.size();
                if (numEdgeTypes > (int)db->edgetypes->distTable.size()) {
                    db->edgetypes->distTable.resize(numEdgeTypes);
                }
                for (int i = 0; i < numEdgeTypes; i++) {
                    db->edgetypes->distTable[i].resize(numEdgeTypes, 0);
                }
                db->edgetypes->distTable[type1_idx][type2_idx] = dbdist;
                db->edgetypes->distTable[type2_idx][type1_idx] = dbdist;
            }
            if (buffer == ";") {
                break;
            }
        }
        // default edge type dist
        for (unsigned i = 0; i < db->edgetypes->types.size(); i++) {
            db->edgetypes->distTable[0][i] = 0;
            db->edgetypes->distTable[i][0] = 0;
        }
    }
    return 0;
}

//-----Layer-----
int readLefLayer(lefrCallbackType_e c, lefiLayer* leflayer, lefiUserData ud) {
    Database* db = (Database*)ud;
    const int convertFactor = db->LefConvertFactor;

    const string name(leflayer->name());
    char type = 'x';
    if (!strcmp(leflayer->type(), "ROUTING")) {
        type = 'r';
    } else if (!strcmp(leflayer->type(), "CUT")) {
        if (db->layers.empty()) {
            logger.warning("remove cut layer %s below the first metal layer", name.c_str());
            return 0;
        }
        type = 'c';
    } else if (!strcmp(leflayer->type(), "MASTERSLICE")) {
        logger.warning("Find MASTERSLICE layer %s.", name.c_str());
        type = 's';
    } else {
        return 0;
    }

    if (db->name_layers.find(name) != db->name_layers.end()) {
        logger.warning("layer type re-defined: %s", name.c_str());
        return 0;
    }

    if (type != 'r' && type != 'c') {
        logger.warning("unsupported layer type: %s", name.c_str());
        return 0;
    }
    Layer& layer = db->addLayer(name, type);

    switch (type) {
        case 'r':
            // routing layer
            if (leflayer->hasDirection()) {
                if (strcmp(leflayer->direction(), "HORIZONTAL") == 0) {
                    layer.direction = 'h';
                } else if (strcmp(leflayer->direction(), "VERTICAL") == 0) {
                    layer.direction = 'v';
                }
            } else {
                layer.direction = 'x';
            }

            if (leflayer->hasXYPitch()) {
                if (layer.direction == 'v') {
                    layer.pitch = lround(leflayer->pitchX() * convertFactor);
                } else {
                    layer.pitch = lround(leflayer->pitchY() * convertFactor);
                }
            }
            if (leflayer->hasPitch()) {
                layer.pitch = lround(leflayer->pitch() * convertFactor);
            }

            if (leflayer->hasXYOffset()) {
                if (layer.direction == 'v') {
                    layer.offset = lround(leflayer->offsetX() * convertFactor);
                } else {
                    layer.offset = lround(leflayer->offsetY() * convertFactor);
                }
            }
            if (leflayer->hasOffset()) {
                layer.offset = lround(leflayer->offset() * convertFactor);
            }

            if (leflayer->hasWidth()) {
                layer.width = lround(leflayer->width() * convertFactor);
            }

            if (leflayer->lefiLayer::hasArea()) {
                layer.area = lround(leflayer->lefiLayer::area() * convertFactor);
            }

            if (leflayer->lefiLayer::hasMinwidth()) {
                layer.minWidth = lround(leflayer->lefiLayer::minwidth() * convertFactor);
            }

            if (leflayer->lefiLayer::hasMaxwidth()) {
                layer.maxWidth = lround(leflayer->lefiLayer::maxwidth() * convertFactor);
            }

            if (layer.width > 0 && layer.pitch > 0) {
                layer.spacing = layer.pitch - layer.width;
            } else if (layer.width < 0 && layer.pitch > 0) {
                layer.width = layer.pitch / 2;
                layer.spacing = layer.pitch - layer.width;
            } else if (layer.pitch < 0 && layer.width > 0) {
                layer.pitch = layer.width * 2;
                layer.spacing = layer.width;
            }

            if (leflayer->lefiLayer::hasSpacingNumber()) {
                for (int i = 0; i < leflayer->lefiLayer::numSpacing(); i++) {
                    // spaceType: 0 -> minSpacing, 1 -> maxEOLSpace, 2 ->
                    // maxEOLSpaceParallelEdge
                    int spaceType = 0;
                    int spacing, width, within, parSpace, parWithin;
                    spacing = lround(leflayer->lefiLayer::spacing(i) * convertFactor);
                    if (leflayer->lefiLayer::hasSpacingEndOfLine(i)) {
                        width = lround(leflayer->lefiLayer::spacingEolWidth(i) * convertFactor);
                        within = lround(leflayer->lefiLayer::spacingEolWithin(i) * convertFactor);
                        spaceType++;
                        if (leflayer->lefiLayer::hasSpacingParellelEdge(i)) {
                            parSpace = lround(leflayer->lefiLayer::spacingParSpace(i) * convertFactor);
                            parWithin = lround(leflayer->lefiLayer::spacingParWithin(i) * convertFactor);
                            spaceType++;
                        }
                    }
                    switch (spaceType) {
                        case 0:
                            layer.spacing = spacing;
                            break;
                        case 1:
                            layer.maxEOLSpace = {spacing, width, within};
                            break;
                        case 2:
                            layer.maxEOLSpaceParallelEdge = {spacing, width, within, parSpace, parWithin};
                            break;
                        default:
                            layer.spacing = spacing;
                            break;
                    }
                }
            }

            for (int i = 0; i < leflayer->lefiLayer::numSpacingTable(); i++) {
                // TODO: we only support ParallelRunLength currently
                lefiSpacingTable* spTable = leflayer->lefiLayer::spacingTable(i);
                if (spTable->lefiSpacingTable::isParallel()) {
                    lefiParallel* parallel = spTable->lefiSpacingTable::parallel();
                    layer.parLength.resize(parallel->lefiParallel::numLength());
                    for (int j = 0; j < parallel->lefiParallel::numLength(); j++) {
                        layer.parLength[j] = lround(parallel->lefiParallel::length(j) * convertFactor);
                    }
                    layer.parWidth.resize(parallel->lefiParallel::numWidth());
                    layer.parWidthSpace.resize(parallel->lefiParallel::numWidth());
                    for (int j = 0; j < parallel->lefiParallel::numWidth(); j++) {
                        layer.parWidth[j] = lround(parallel->lefiParallel::width(j) * convertFactor);
                        layer.parWidthSpace[j].resize(parallel->lefiParallel::numLength());
                        for (int k = 0; k < parallel->lefiParallel::numLength(); k++) {
                            layer.parWidthSpace[j][k] =
                                lround(parallel->lefiParallel::widthSpacing(j, k) * convertFactor);
                        }
                    }
                }
            }

            return 0;
        case 'c':
            // cut (via) layer
            if (leflayer->hasSpacingNumber()) {
                switch (leflayer->numSpacing()) {
                    case 0:
                        logger.warning("layer has no spacing: %s", name.c_str());
                        return 0;
                    case 1:
                        layer.spacing = leflayer->spacing(0);
                        return 0;
                    default:
                        layer.spacing = leflayer->spacing(0);
                        logger.warning("layer has multiple spacing: %s", name.c_str());
                        return 0;
                }
            }
            return 0;
        case 's':
            return 0;
        default:
            return 0;
    }
}

//-----Via-----
int readLefVia(lefrCallbackType_e c, lefiVia* lvia, lefiUserData ud) {
    Database* db = (Database*)ud;
    int convertFactor = db->LefConvertFactor;

    string name(lvia->lefiVia::name());
    ViaType* via = db->addViaType(name, false);
    for (int i = 0; i < lvia->lefiVia::numLayers(); i++) {
        string layername(lvia->lefiVia::layerName(i));
        Layer* layer = db->getLayer(layername);
        if (!layer) {
            logger.error("layer not found: %s", layername.c_str());
        }
        for (int j = 0; j < lvia->lefiVia::numRects(i); ++j) {
            via->addRect(*layer,
                         lround(lvia->lefiVia::xl(i, j) * convertFactor),
                         lround(lvia->lefiVia::yl(i, j) * convertFactor),
                         lround(lvia->lefiVia::xh(i, j) * convertFactor),
                         lround(lvia->lefiVia::yh(i, j) * convertFactor));
        }
    }
    return 0;
}

//-----CellType-----
int readLefMacroBegin(lefrCallbackType_e c, const char* macroName, lefiUserData ud) {
    Database* db = (Database*)ud;

    string name(macroName);
    db->addCellType(name, db->celltypes.size());
    return 0;
}

int readLefObs(lefrCallbackType_e c, lefiObstruction* obs, lefiUserData ud) {
    Database* db = (Database*)ud;
    int convertFactor = db->LefConvertFactor;
    CellType* celltype = db->celltypes.back();  // get the last inserted celltype

    Layer* layer = nullptr;
    lefiGeometries* geom = obs->lefiObstruction::geometries();
    int nitem = geom->lefiGeometries::numItems();
    for (int i = 0; i < nitem; i++) {
        if (geom->lefiGeometries::itemType(i) == lefiGeomLayerE) {
            const string layername(geom->lefiGeometries::getLayer(i));
            layer = db->getLayer(layername);
        } else if (geom->lefiGeometries::itemType(i) == lefiGeomRectE && layer != NULL) {
            lefiGeomRect* rect = geom->lefiGeometries::getRect(i);
            celltype->addObs(*layer,
                             lround(rect->xl * convertFactor),
                             lround(rect->yl * convertFactor),
                             lround(rect->xh * convertFactor),
                             lround(rect->yh * convertFactor));
        }
    }
    return 0;
}

int readLefPin(lefrCallbackType_e c, lefiPin* pin, lefiUserData ud) {
    Database* db = (Database*)ud;
    int convertFactor = db->LefConvertFactor;
    CellType* celltype = db->celltypes.back();  // get the last inserted celltype

    const string name(pin->name());
    if (name == "VBB" || name == "VPP") {
        return 0;
    }

    char direction = 'x';
    char type = 's';
    if (pin->hasUse()) {
        const string use(pin->use());
        if (use == "ANALOG") {
            //  Pin is used for analog connectivity.
            type = 'a';
        } else if (use == "CLOCK") {
            //  Pin is used for clock net connectivity.
            type = 'c';
        } else if (use == "GROUND") {
            //  Pin is used for connectivity to the chip-
            //  level ground distribution network.
            type = 'g';
        } else if (use == "POWER") {
            //  Pin is used for connectivity to the chip-
            //  level power distribution network.
            type = 'p';
        } else if (use == "SIGNAL") {
            //  Pin is used for regular net connectivity.
        } else {
            logger.error("unknown use: %s.%s", celltype->name.c_str(), use.c_str());
        }
    }

    if (pin->hasDirection()) {
        const string sDir(pin->direction());
        if (sDir == "OUTPUT") {
            direction = 'o';
        } else if (sDir == "OUTPUT TRISTATE") {
            direction = 'o';
            logger.warning(
                "treat pin %s.%s direction %s as OUTPUT", celltype->name.c_str(), name.c_str(), sDir.c_str());
        } else if (sDir == "INPUT") {
            direction = 'i';
        } else if (sDir == "INOUT") {
            if (name != "VDD" && name != "vdd" && name != "VSS" && name != "vss") {
                logger.warning("unknown pin %s.%s direction: %s", celltype->name.c_str(), name.c_str(), sDir.c_str());
            }
        } else {
            logger.error("unknown pin %s.%s direction: %s", celltype->name.c_str(), name.c_str(), sDir.c_str());
        }
    }

    // switch (direction) {
    //     case 'i':
    //         rtimer.lLib.cells[celltype->libcell()].addIPin(name);
    //         break;
    //     case 'o':
    //         rtimer.lLib.cells[celltype->libcell()].addOPin(name);
    //         break;
    //     default:
    //         break;
    // }

    if (pin->hasTaperRule()) {
        logger.warning("pin %s has taper rule %s", name.c_str(), pin->taperRule());
    }

    PinType* pintype = celltype->addPin(name, direction, type);

    std::set<Point> nodes;
    Layer* layer = nullptr;
    lefiGeomRect* rect = nullptr;
    lefiGeomPolygon* polygon = nullptr;
    lefiGeometries* geom = pin->port(0);
    vector<std::pair<int, int>> poly;
    unsigned bj = 0;
    bool has45 = false;
    for (unsigned i = 0; i != (unsigned)geom->numItems(); ++i) {
        switch (geom->itemType(i)) {
            case lefiGeomUnknown:
                logger.warning("lefiGeomUnknown: %s.%s", celltype->name.c_str(), name.c_str());
                break;
            case lefiGeomLayerE:
                layer = db->getLayer(string(geom->getLayer(i)));
                assert(layer);
                break;
            case lefiGeomRectE:
                rect = geom->getRect(i);
                pintype->addShape(*layer,
                                  lround(rect->xl * convertFactor),
                                  lround(rect->yl * convertFactor),
                                  lround(rect->xh * convertFactor),
                                  lround(rect->yh * convertFactor));
                break;
            case lefiGeomPolygonE:
                polygon = geom->getPolygon(i);
                poly.clear();
                bj = 0;
                has45 = false;
                for (unsigned j = 0; static_cast<int>(j) != polygon->numPoints; ++j) {
                    poly.emplace_back(lround(polygon->x[j] * convertFactor), lround(polygon->y[j] * convertFactor));
                    if (polygon->y[j] < polygon->y[bj]) {
                        bj = j;
                    }
                    const unsigned jPost = (j + 1) % polygon->numPoints;
                    if (polygon->x[j] != polygon->x[jPost] && polygon->y[j] != polygon->y[jPost]) {
                        has45 = true;
                    }
                }
                if (has45) {
                    poly.clear();
                    const unsigned jPre = (bj + polygon->numPoints - 1) % polygon->numPoints;
                    const unsigned jPost = (bj + 1) % polygon->numPoints;
                    bool isClockwise = false;
                    if (polygon->x[jPre] > polygon->x[jPost]) {
                        isClockwise = true;
                    }
                    for (unsigned k = 0; static_cast<int>(k) != polygon->numPoints; ++k) {
                        poly.emplace_back(lround(polygon->x[k] * convertFactor), lround(polygon->y[k] * convertFactor));
                        const unsigned kPost = (k + 1) % polygon->numPoints;
                        if (polygon->x[k] == polygon->x[kPost] || polygon->y[k] == polygon->y[kPost]) {
                            continue;
                        }
                        if (isClockwise ^ (polygon->x[k] < polygon->x[kPost]) ^ (polygon->y[k] < polygon->y[kPost])) {
                            poly.emplace_back(lround(polygon->x[k] * convertFactor),
                                              lround(polygon->y[kPost] * convertFactor));
                        } else {
                            poly.emplace_back(lround(polygon->x[kPost] * convertFactor),
                                              lround(polygon->y[k] * convertFactor));
                        }
                    }
                }
                for (unsigned j = 0; j != poly.size(); ++j) {
                    unsigned jPre = (j + poly.size() - 1) % poly.size();
                    unsigned jPost = (j + 1) % poly.size();
                    unsigned char dir = pointDir(poly[j].first, poly[j].second, poly[jPre].first, poly[jPre].second);
                    dir |= pointDir(poly[j].first, poly[j].second, poly[jPost].first, poly[jPost].second);
                    if (dir == (DIR_DOWN | DIR_UP) || dir == (DIR_LEFT | DIR_RIGHT)) {
                        continue;
                    }
                    nodes.emplace(poly[j].first, poly[j].second, dir);
                }
                while (nodes.size()) {
                    std::set<Point>::iterator pk;
                    std::set<Point>::iterator pl;
                    std::set<Point>::iterator pm;
                    std::set<Point>::iterator pn;

                    while (nodes.size()) {
                        pk = nodes.begin();
                        pl = nodes.begin();
                        ++pl;

                        if (pk->x != pl->x) {
                            break;
                        }

                        nodes.erase(nodes.begin());
                        nodes.erase(nodes.begin());
                    }

                    if (nodes.empty()) {
                        break;
                    }

                    if (nodes.size() < 4) {
                        std::cout << "Error when partitioning rectangles." << std::endl;
                        std::cout << "Remaining points: ";
                        for (const Point& p : nodes) {
                            printf("(%d, %d), ", p.x, p.y);
                        }
                        std::cout << std::endl;
                        break;
                    }

                    for (pm = nodes.begin(); pm->y <= pk->y || pm->x < pk->x || pm->x >= pl->x; ++pm) {
                    }

                    for (pn = pm; pn->x < pl->x && pn->y <= pm->y; ++pn) {
                    }

                    pintype->addShape(*layer, pk->x, pk->y, pl->x, pm->y);

                    Point ul(pk->x, pm->y, DIR_DOWN | DIR_RIGHT);
                    Point ur(pl->x, pm->y, DIR_DOWN | DIR_LEFT);

                    // Insert or erase the upper-right point of the rectangle.
                    if (*pn == ur && pn->outdir != (DIR_UP | DIR_RIGHT)) {
                        nodes.erase(pn);
                    } else {
                        nodes.insert(ur);
                    }

                    // Insert or erase the upper-left point of the rectangle.
                    pm = nodes.find(ul);
                    if (pm == nodes.end()) {
                        nodes.insert(ul);
                    } else {
                        nodes.erase(pm);
                    }

                    nodes.erase(nodes.begin());
                    nodes.erase(nodes.begin());
                }
                break;
            default:
                logger.warning(
                    "unknown lefiGeomEnum %u: %s.%s", geom->itemType(i), celltype->name.c_str(), name.c_str());
                break;
        }
    }
    return 0;
}

int readLefSite(lefrCallbackType_e c, lefiSite* site, lefiUserData ud) {
    Database* db = reinterpret_cast<Database*>(ud);
    int convertFactor = db->LefConvertFactor;
    int width = lround(site->sizeX() * convertFactor);
    int height = lround(site->sizeY() * convertFactor);
    logger.info("site name: %s class: %s siteW: %d siteH: %d", site->name(), site->siteClass(), width, height);
    db->addSite(site->name(), site->siteClass(), width, height);
    return 0;
}

int readLefMacro(lefrCallbackType_e c, lefiMacro* macro, lefiUserData ud) {
    Database* db = reinterpret_cast<Database*>(ud);
    int convertFactor = db->LefConvertFactor;
    CellType* celltype = db->celltypes.back();

    celltype->width = round(macro->sizeX() * convertFactor);
    celltype->height = round(macro->sizeY() * convertFactor);

    if (macro->lefiMacro::hasClass()) {
        std::string clsname(macro->macroClass());
        if (clsname == "CORE") {
            celltype->cls = clsname;
            celltype->stdcell = true;
        } else if (clsname == "BLOCK") {
            celltype->cls = clsname;
        } else {
            celltype->cls = clsname;
            logger.warning("Class type is not defined: %s", celltype->cls.c_str());
        }
    } else {
        celltype->cls = "CORE";  // default value
    }

    if (macro->lefiMacro::hasOrigin()) {
        celltype->setOrigin(macro->originX() * convertFactor, macro->originY() * convertFactor);
    }

    if (macro->hasXSymmetry()) {
        celltype->setXSymmetry();
    }
    if (macro->hasYSymmetry()) {
        celltype->setYSymmetry();
    }
    if (macro->has90Symmetry()) {
        celltype->set90Symmetry();
    }

    if (macro->hasSiteName()) {
        celltype->siteName(string(macro->siteName()));
    }

    for (int i = 0; i < macro->numProperties(); ++i) {
        if (!strcmp(macro->propName(i), "LEF58_EDGETYPE")) {
            std::stringstream ssedgetype(macro->propValue(i));
            string buffer;
            while (!ssedgetype.eof()) {
                ssedgetype >> buffer;
                if (buffer == "EDGETYPE") {
                    string edgeside;
                    string edgetype;
                    ssedgetype >> edgeside >> edgetype;
                    if (edgeside == "LEFT") {
                        celltype->edgetypeL = db->edgetypes->getEdgeType(edgetype);
                    } else if (edgeside == "RIGHT") {
                        celltype->edgetypeR = db->edgetypes->getEdgeType(edgetype);
                    } else if (edgeside == "BOTTOM") {
                        static bool missBot = true;
                        if (missBot) {
                            logger.warning("unknown edge side: %s", edgeside.c_str());
                            missBot = false;
                        }
                    } else if (edgeside == "TOP") {
                        static bool missTop = true;
                        if (missTop) {
                            logger.warning("unknown edge side: %s", edgeside.c_str());
                            missTop = false;
                        }
                    } else {
                        logger.warning("unknown edge side: %s", edgeside.c_str());
                    }
                }
            }
        }
    }
    return 0;
}

int readLefMacroEnd(lefrCallbackType_e c, const char* macroName, lefiUserData ud) {
    //  TODO: sort by pin name.
    return 0;
}

/******************/
/*  DEF Callback  */
/******************/

//-----Unit-----
int readDefUnits(defrCallbackType_e c, double d, defiUserData ud) {
    reinterpret_cast<Database*>(ud)->DBU_Micron = d;
    return 0;
}

//-----Version-----
int readDefVersion(defrCallbackType_e c, double d, defiUserData ud) {
    reinterpret_cast<Database*>(ud)->version = d;
    return 0;
}

//-----Design-----
int readDefDesign(defrCallbackType_e c, const char* name, defiUserData ud) {
    reinterpret_cast<Database*>(ud)->designName = name;
    return 0;
}

//-----Die Area-----
int readDefDieArea(defrCallbackType_e c, defiBox* dbox, defiUserData ud) {
    Database* db = (Database*)ud;
    db->dieLX = dbox->xl();
    db->dieLY = dbox->yl();
    db->dieHX = dbox->xh();
    db->dieHY = dbox->yh();
    return 0;
}

//-----Row-----
int readDefRow(defrCallbackType_e c, defiRow* drow, defiUserData ud) {
    ++g_def_profile.rows;
    ((Database*)ud)
        ->addRow(string(drow->name()),
                 string(drow->macro()),
                 drow->x(),
                 drow->y(),
                 drow->xNum(),
                 drow->yNum(),
                 drow->orient(),
                 isFlipY(drow->orient()),
                 drow->xStep(),
                 drow->yStep());
    return 0;
}

//-----Track-----
int readDefTrack(defrCallbackType_e c, defiTrack* dtrack, defiUserData ud) {
    ++g_def_profile.tracks;
    Database* db = (Database*)ud;
    char direction = 'x';
    if (strcmp(dtrack->macro(), "X") == 0) {
        direction = 'v';
    } else if (strcmp(dtrack->macro(), "Y") == 0) {
        direction = 'h';
    }
    Track* track = db->addTrack(direction, dtrack->x(), dtrack->xNum(), dtrack->xStep());
    for (int i = 0; i < dtrack->numLayers(); i++) {
        const string layername(dtrack->layer(i));
        track->addLayer(layername);
    }
    for (int i = 0; i < dtrack->numLayers(); i++) {
        const string layername(dtrack->layer(i));
        Layer* layer = db->getLayer(layername);
        if (layer) {
            if (track->direction == 'x' || layer->direction == 'x') {
                layer->tracks.push_back(*track);
            } else if (layer->direction == track->direction) {
                layer->tracks.push_back(*track);
            } else if (layer->direction != track->direction) {
                layer->nonPreferDirTracks.push_back(*track);
            } else {
                logger.error("wrong definition of tracks for layer %s", layername.c_str());
            }
        } else {
            logger.error("layer name not found: %s", layername.c_str());
        }
    }
    return 0;
}

//-----GcellGrid-----
int readDefGcellGrid(defrCallbackType_e c, defiGcellGrid* dgrid, defiUserData ud) {
    ++g_def_profile.gcells;
    Database* db = (Database*)ud;
    const string macro(dgrid->macro());
    if (macro == "X") {
        db->gcellgrid->numX.emplace_back(dgrid->xNum());
        db->gcellgrid->stepX.emplace_back(dgrid->xStep());
        db->gcellgrid->startX.emplace_back(dgrid->x());
    } else if (macro == "Y") {
        db->gcellgrid->numY.emplace_back(dgrid->xNum());
        db->gcellgrid->stepY.emplace_back(dgrid->xStep());
        db->gcellgrid->startY.emplace_back(dgrid->x());
    }
    return 0;
}

//-----Via-----
int readDefViaStart(defrCallbackType_e c, int num, defiUserData ud) {
    g_def_profile.switchPhase("vias");
    reinterpret_cast<Database*>(ud)->reserveViaTypes(num);
    return 0;
}

int readDefVia(defrCallbackType_e c, defiVia* dvia, defiUserData ud) {
    ++g_def_profile.vias;
    Database* db = (Database*)ud;

    const string name(dvia->name());
    ViaType* via = db->getViaType(name);
    if (via) {
        via->isDef(true);
    } else {
        via = db->addViaType(name, true);
    }

    char* dvialayer = nullptr;
    int lx = 0;
    int ly = 0;
    int hx = 0;
    int hy = 0;
    for (int i = 0; i != dvia->numLayers(); ++i) {
        dvia->layer(i, &dvialayer, &lx, &ly, &hx, &hy);
        Layer* layer = db->getLayer(string(dvialayer));
        if (layer) {
            via->addRect(*layer, lx, ly, hx, hy);
        } else {
            logger.info("layer name not found: %s", dvialayer);
        }
    }

    if (dvia->defiVia::numPolygons()) {
        // TODO
    }

    if (dvia->defiVia::hasViaRule()) {
        ViaRule& rule = via->rule;
        char* vrn;                             // VIARULE Name
        int xs, ys;                            // CUTSIZE
        char *botlayer, *cutlayer, *toplayer;  // LAYERS
        int xcs, ycs;                          // CUTSPACING
        int xbe, ybe, xte, yte;                // ENCLOSURE
        int cr, cc;                            // ROWCOL
        int xo, yo;                            // ORIGIN
        int xbo, ybo, xto, yto;                // OFFSET

        (void)dvia->defiVia::viaRule(
            &vrn, &xs, &ys, &botlayer, &cutlayer, &toplayer, &xcs, &ycs, &xbe, &ybe, &xte, &yte);

        rule.hasViaRule = true;
        rule.name = string(vrn);

        rule.cutSize = std::make_pair(xs, ys);
        rule.cutSpacing = std::make_pair(xcs, ycs);
        rule.botEnclosure = std::make_pair(xbe, ybe);
        rule.topEnclosure = std::make_pair(xte, yte);
        rule.botLayer = db->getLayer(string(botlayer));
        rule.cutLayer = db->getLayer(string(cutlayer));
        rule.topLayer = db->getLayer(string(toplayer));

        if (dvia->defiVia::hasRowCol()) {
            (void)dvia->defiVia::rowCol(&cr, &cc);
            rule.numCutRows = cr;
            rule.numCutCols = cc;
        }

        if (dvia->defiVia::hasOrigin()) {
            (void)dvia->defiVia::origin(&xo, &yo);
            rule.originOffset = std::make_pair(xo, yo);
        }

        if (dvia->defiVia::hasOffset()) {
            (void)dvia->defiVia::offset(&xbo, &ybo, &xto, &yto);
            rule.botOffset = std::make_pair(xbo, ybo);
            rule.topOffset = std::make_pair(xto, yto);
        }

        if (dvia->defiVia::hasCutPattern()) {
            const string cutPattern(dvia->defiVia::cutPattern());
        }
    }

    return 0;
}

//-----NonDefaultRule-----

int readDefNdr(defrCallbackType_e c, defiNonDefault* nd, defiUserData ud) {
    ++g_def_profile.ndrs;
    Database* db = (Database*)ud;
    NDR* ndr = db->addNDR(string(nd->name()), nd->hasHardspacing());

    for (int i = 0; i < nd->numLayers(); i++) {
        int space = 0;
        if (nd->hasLayerSpacing(i)) {
            space = nd->layerSpacingVal(i);
        }
        ndr->rules.emplace_back(db->getLayer(string(nd->layerName(i))), nd->defiNonDefault::layerWidthVal(i), space);
    }
    for (int i = 0; i < nd->numVias(); i++) {
        string vianame(nd->viaName(i));
        ViaType* viatype = db->getViaType(vianame);
        if (!viatype) {
            logger.warning("NDR via type not found: %s", vianame.c_str());
        }
        ndr->vias.push_back(viatype);
    }
    return 0;
}

//-----Cell-----
int readDefComponentStart(defrCallbackType_e c, int num, defiUserData ud) {
    g_def_profile.switchPhase("components");
    reinterpret_cast<Database*>(ud)->reserveCells(num);
    return 0;
}

int readDefComponent(defrCallbackType_e c, defiComponent* co, defiUserData ud) {
    ++g_def_profile.components;
    Database* db = (Database*)ud;
    CellType* celltype = db->getCellType(co->name());

    string cellName(co->id());
    validate_token(cellName);
    Cell* cell = db->addCell(cellName, celltype);

    if (co->isUnplaced()) {
        cell->fixed(false);
        cell->unplace();
    } else if (co->isPlaced()) {
        cell->place(co->placementX(), co->placementY(), co->placementOrient());
        if (celltype->cls == "CORE" || celltype->cls == "BLOCK") {
            cell->fixed(false);
        } else {
            // Set all non-CORE yet non-BLOCK cells as fixed cells
            cell->fixed(true);
        }
        if (co->placementOrient() % 2 == 1) {
            // 0:N, 1:W, 2:S, 3:E, 4:FN, 5:FW, 6:FS, 7:FE, -1:NONE
            logger.warning("Cell [%s]'s placementOrient [%s] is not supported, CLASS: %s.",
                           cell->name().c_str(),
                           getOrient(co->placementOrient()).c_str(),
                           celltype->cls.c_str());
        }
    } else if (co->isFixed()) {
        cell->place(co->placementX(), co->placementY(), co->placementOrient());
        cell->fixed(true);
        if (co->placementOrient() % 2 == 1) {
            // 0:N, 1:W, 2:S, 3:E, 4:FN, 5:FW, 6:FS, 7:FE, -1:NONE
            logger.warning(
                "Fixed Cell [%s]'s placementOrient [%s] is not supported, "
                "CLASS: %s.",
                cell->name().c_str(),
                getOrient(co->placementOrient()).c_str(),
                celltype->cls.c_str());
        }
    }
    return 0;
}

//-----Pin-----
int readDefPinStart(defrCallbackType_e c, int num, defiUserData ud) {
    g_def_profile.switchPhase("pins");
    reinterpret_cast<Database*>(ud)->reserveIOPins(num);
    return 0;
}

int readDefPin(defrCallbackType_e c, defiPin* dpin, defiUserData ud) {
    ++g_def_profile.pins;
    Database* db = (Database*)ud;

    char direction = 'x';
    if (dpin->direction()) {
        if (!strcmp(dpin->direction(), "INPUT")) {
            // INPUT to the chip, output from external
            direction = 'o';
        } else if (!strcmp(dpin->direction(), "OUTPUT")) {
            // OUTPUT to the chip, input to external
            direction = 'i';
        } else {
            logger.warning("unknown pin signal direction: %s", dpin->direction());
        }
    } else {
        string pinName(dpin->pinName());
        logger.warning("Pin %s has no pin signal direction", pinName.c_str());
    }

    IOPin* iopin = db->addIOPin(string(dpin->pinName()), string(dpin->netName()), direction);

    if (dpin->hasPlacement()) {
        iopin->x = dpin->placementX();
        iopin->y = dpin->placementY();
        iopin->_orient = dpin->orient();
    }

    if (dpin->hasLayer()) {
        for (int i = 0; i < dpin->numLayer(); i++) {
            Layer* layer = db->getLayer(string(dpin->layer(i)));
            int lx, ly, hx, hy;
            dpin->bounds(i, &lx, &ly, &hx, &hy);
            iopin->type->addShape(*layer, lx, ly, hx, hy);
        }
    }

    if (dpin->hasPort()) {
        for (int j = 0; j < dpin->numPorts(); j++) {
            if (j > 1) {
                string pinName(dpin->pinName());
                logger.warning(
                    "DefPin %s has multiple ports. We currently only support "
                    "single port definition",
                    pinName.c_str());
                break;
            }
            defiPinPort* port = dpin->pinPort(j);

            if (port->hasPlacement()) {
                iopin->x = port->placementX();
                iopin->y = port->placementY();
                iopin->_orient = port->orient();
            }

            for (int i = 0; i < port->numLayer(); i++) {
                Layer* layer = db->getLayer(string(port->layer(i)));
                int lx, ly, hx, hy;
                port->bounds(i, &lx, &ly, &hx, &hy);
                iopin->type->addShape(*layer, lx, ly, hx, hy);
            }
        }
    }

    return 0;
}

//-----Blockages-----
int readDefBlockageStart(defrCallbackType_e c, int num, defiUserData ud) {
    g_def_profile.switchPhase("blockages");
    reinterpret_cast<Database*>(ud)->reserveBlockages(num);
    return 0;
}

int readDefBlockage(defrCallbackType_e c, defiBlockage* dblk, defiUserData ud) {
    ++g_def_profile.blockages;
    Database* db = (Database*)ud;

    if (dblk->hasLayer()) {
        string layername(dblk->layerName());
        Layer* layer = db->getLayer(layername);
        if (!layer) {
            logger.error("layer not found: %s", layername.c_str());
            return 1;
        }
        for (int i = 0; i < dblk->numRectangles(); ++i) {
            db->routeBlockages.emplace_back(*layer, dblk->xl(i), dblk->yl(i), dblk->xh(i), dblk->yh(i));
        }
    } else if (dblk->hasPlacement()) {
        for (int i = 0; i < dblk->defiBlockage::numRectangles(); ++i) {
            db->placeBlockages.emplace_back(dblk->xl(i), dblk->yl(i), dblk->xh(i), dblk->yh(i));
        }
    }
    return 0;
}

//-----Net-----
int readDefSNetStart(defrCallbackType_e c, int num, defiUserData ud) {
    g_def_profile.switchPhase("specialnets");
    reinterpret_cast<Database*>(ud)->reserveSNets(num);
    return 0;
}

int readDefSNet(defrCallbackType_e c, defiNet* dnet, defiUserData ud) {
    ++g_def_profile.snets;
    Database* db = (Database*)ud;

    SNet* snet = db->addSNet(dnet->name());

    if (dnet->hasUse()) {
        const string use(dnet->use());
        if (use == "POWER") {
            snet->type = 'p';
        } else if (use == "GROUND") {
            snet->type = 'g';
        } else {
            logger.error("unknown use: %s", use.c_str());
        }
    }

    int path = DEFIPATH_DONE;
    string layername = "";
    Layer* layer = nullptr;
    string vianame = "";
    ViaType* viatype = nullptr;
    int fx = 0;
    int fy = 0;
    int fz = 0;
    int tx = 0;
    int ty = 0;
    int tz = 0;
    int lx = 0;
    int ly = 0;
    int hx = 0;
    int hy = 0;
    int ext = 0;
    int nextExt = 0;
    int wirewidth = 0;
    for (unsigned i = 0; static_cast<int>(i) < dnet->numWires(); ++i) {
        const defiWire* dwire = dnet->wire(i);
        unsigned nNodes = 0;
        for (unsigned j = 0; static_cast<int>(j) < dwire->numPaths(); ++j) {
            const defiPath* dpath = dwire->path(j);
            dpath->initTraverse();
            while ((path = dpath->next()) != DEFIPATH_DONE) {
                switch (path) {
                    case DEFIPATH_LAYER:
                        layername = dpath->getLayer();
                        layer = db->getLayer(layername);
                        if (!layer) {
                            logger.error("Layer is not defined: %s", layername.c_str());
                            return false;
                        }
                        break;
                    case DEFIPATH_VIA:
                        vianame = dpath->getVia();
                        viatype = db->getViaType(vianame);
                        if (!viatype) {
                            logger.error("Via type is not defined: %s", vianame.c_str());
                            return false;
                        }
                        snet->addVia(viatype, fx, fy);
                        break;
                    case DEFIPATH_WIDTH:
                        wirewidth = dpath->getWidth();
                        nNodes = 0;
                        break;
                    case DEFIPATH_POINT:
                        if (nNodes) {
                            dpath->getPoint(&tx, &ty);
                            tz = -1;
                            ext = (fz == -1) ? 0 : fz;
                            nextExt = (tz == -1) ? 0 : tz;
                            if (fy == ty) {
                                lx = fx < tx ? fx - ext : tx - nextExt;
                                hx = fx < tx ? tx + nextExt : fx + ext;
                                ly = fy - wirewidth / 2;
                                hy = fy + wirewidth / 2;
                            } else if (fx == tx) {
                                lx = fx - wirewidth / 2;
                                hx = fx + wirewidth / 2;
                                ly = fy < ty ? fy - ext : ty - nextExt;
                                hy = fy < ty ? ty + nextExt : fy + ext;
                            } else {
                                // WARN
                            }
                            snet->addShape(*layer, lx, ly, hx, hy);
                            if ((layer->rIndex == 0 || layer->rIndex == 1) && fy == ty) {
                                db->powerNet->addRail(snet, lx, hx, fy);
                            }
                            fx = tx;
                            fy = ty;
                            fz = tz;
                        } else {
                            dpath->getPoint(&fx, &fy);
                            fz = -1;
                        }
                        ++nNodes;
                        break;
                    case DEFIPATH_FLUSHPOINT:
                        if (nNodes) {
                            dpath->getFlushPoint(&tx, &ty, &tz);
                            ext = (fz == -1) ? 0 : fz;
                            nextExt = (tz == -1) ? 0 : tz;
                            if (fy == ty) {
                                lx = fx < tx ? fx - ext : tx - nextExt;
                                hx = fx < tx ? tx + nextExt : fx + ext;
                                ly = fy - wirewidth / 2;
                                hy = fy + wirewidth / 2;
                            } else if (fx == tx) {
                                lx = fx - wirewidth / 2;
                                hx = fx + wirewidth / 2;
                                ly = fy < ty ? fy - ext : ty - nextExt;
                                hy = fy < ty ? ty + nextExt : fy + ext;
                            } else {
                                // WARN
                            }
                            snet->addShape(*layer, lx, ly, hx, hy);
                            if ((layer->rIndex == 0 || layer->rIndex == 1) && fy == ty) {
                                db->powerNet->addRail(snet, lx, hx, fy);
                            }
                            fx = tx;
                            fy = ty;
                            fz = tz;
                        } else {
                            dpath->getFlushPoint(&fx, &fy, &fz);
                        }
                        ++nNodes;
                        break;
                }
            }
        }
    }
    return 0;
}

int readDefNetStart(defrCallbackType_e c, int num, defiUserData ud) {
    g_def_profile.switchPhase("nets");
    reinterpret_cast<Database*>(ud)->reserveNets(num);
    return 0;
}

int readDefNet(defrCallbackType_e c, defiNet* dnet, defiUserData ud) {
    ++g_def_profile.nets;
    Database* db{reinterpret_cast<Database*>(ud)};
    NDR* ndr{nullptr};

    if (dnet->hasNonDefaultRule()) {
        string designrulename(dnet->nonDefaultRule());
        ndr = db->getNDR(designrulename);
        if (!ndr) {
            logger.warning("NDR rule is not defined: %s", designrulename.c_str());
        }
    }
    const unsigned num_connections = static_cast<unsigned>(dnet->numConnections());
    g_def_profile.net_connections += num_connections;
    if (num_connections == 0) {
        string netName(dnet->name());
        logger.warning("Net %s is 0-Pin net. Ignore.", netName.c_str());
        return 0;
    }

    string netName(dnet->name());
    validate_token(netName);
    // exclude VDD and VSS TODO:
    if (netName == "VDD" || netName == "VSS") {
        return 0;
    }
    
    Net* net = db->addNet(netName, ndr);
    // Avoid exact-reserving millions of small nets; high-fanout nets benefit most.
    if (num_connections > 4) {
        net->pins.reserve(num_connections);
    }

    if (dnet->hasUse()) {
        const string use(dnet->use());
        if (use == "POWER") {
            net->_type = 'p';
        } else if (use == "GROUND") {
            net->_type = 'g';
        } else if (use == "SIGNAL") {
            net->_type = 's';
        } else if (use == "CLOCK") {
            net->_type = 'c';
        } else {
            logger.error("unknown use: %s", use.c_str());
        }
    }

    for (unsigned i = 0; i != num_connections; ++i) {
        Pin* pin = nullptr;
        if (strcmp(dnet->instance(i), "PIN") == 0) {
            string iopinname(dnet->pin(i));
            IOPin* iopin = db->getIOPin(iopinname);
            if (!iopin) {
                logger.warning("IO pin is not defined: %s", iopinname.c_str());
            }
            pin = iopin->pin;
            if (pin->is_connected) {
                string netName(dnet->name());
                logger.warning("IO Pin is re-connected: %s %s", netName.c_str(), iopinname.c_str());
            }
            iopin->is_connected = true;
        } else {
            string cellname(dnet->instance(i));
            validate_token(cellname);
            const char* pinname = dnet->pin(i);
            Cell* cell = db->getCell(cellname);
            if (!cell) {
                logger.warning("Cell is not defined: %s", cellname.c_str());
            }
            pin = cell->pin(pinname);
            if (!pin) {
                string netName(dnet->name());
                logger.warning("Pin is not defined: %s %s %s", netName.c_str(), cellname.c_str(), pinname);
            }
            if (pin->is_connected) {
                string netName(dnet->name());
                logger.warning("Pin is re-connected: %s %s %s", netName.c_str(), cellname.c_str(), pinname);
            }
            cell->is_connected = true;
        }
        pin->net = net;
        pin->is_connected = true;
        net->addPin(pin);
    }

    if (setting.SkipDefNetWires) {
        return 0;
    }

    int path = DEFIPATH_DONE;
    const Layer* layer = nullptr;
    int fromx = 0;
    int fromy = 0;
    int fromz = 0;
    int tox = 0;
    int toy = 0;
    int toz = 0;
    unsigned nNodes = 0;
    for (unsigned i = 0; static_cast<int>(i) < dnet->numWires(); ++i) {
        const defiWire* dwire = dnet->wire(i);
        for (unsigned j = 0; static_cast<int>(j) < dwire->numPaths(); ++j) {
            const defiPath* dpath = dwire->path(j);
            dpath->initTraverse();
            while ((path = dpath->next()) != DEFIPATH_DONE) {
                switch (path) {
                    case DEFIPATH_LAYER:
                        nNodes = 0;
                        layer = db->getLayer(string(dpath->getLayer()));
                        break;
                    case DEFIPATH_VIA:
                        //  printf("%s ", dpath->getVia());
                        break;
                    case DEFIPATH_VIAROTATION:
                        break;
                    case DEFIPATH_WIDTH:
                        //  printf("%d ", dpath->getWidth());
                        break;
                    case DEFIPATH_POINT:
                        if (nNodes) {
                            dpath->getPoint(&tox, &toy);
                            toz = -1;
                            net->addWire(layer, fromx, fromy, fromz, tox, toy, toz);
                            fromx = tox;
                            fromy = toy;
                            fromz = toz;
                        } else {
                            dpath->getPoint(&fromx, &fromy);
                        }
                        ++nNodes;
                        break;
                    case DEFIPATH_FLUSHPOINT:
                        if (nNodes) {
                            dpath->getFlushPoint(&tox, &toy, &toz);
                            net->addWire(layer, fromx, fromy, fromz, tox, toy, toz);
                            fromx = tox;
                            fromy = toy;
                            fromz = toz;
                        } else {
                            dpath->getFlushPoint(&fromx, &fromy, &fromz);
                        }
                        ++nNodes;
                        break;
                    case DEFIPATH_TAPER:
                        //  printf("TAPER ");
                        break;
                    case DEFIPATH_TAPERRULE:
                        //  printf("TAPERRULE %s ", dpath->getTaperRule());
                        break;
                    case DEFIPATH_STYLE:
                        //  printf("STYLE %d ", dpath->getStyle());
                        break;
                }
            }
        }
    }

    return 0;
}

//-----Region-----
int readDefRegionStart(defrCallbackType_e c, int num, defiUserData ud) {
    g_def_profile.switchPhase("regions");
    reinterpret_cast<Database*>(ud)->reserveRegions(num);
    return 0;
}

int readDefRegion(defrCallbackType_e c, defiRegion* dreg, defiUserData ud) {
    ++g_def_profile.regions;
    Database* db = (Database*)ud;

    char type = 'f';
    if (dreg->hasType()) {
        if (!strcmp(dreg->type(), "FENCE")) {
            type = 'f';
        } else if (!strcmp(dreg->type(), "GUIDE")) {
            type = 'g';
        } else {
            logger.warning("Unknown region type: %s", dreg->type());
        }
    } else {
        logger.warning("Region is defined without type, use default region type = FENCE");
    }

    Region* region = db->addRegion(string(dreg->name()), type);

    for (unsigned i = 0; (int)i < dreg->numRectangles(); ++i) {
        region->addRect(dreg->xl(i), dreg->yl(i), dreg->xh(i), dreg->yh(i));
    }
    return 0;
}

//-----Group-----
// #define GROUP_MARKER 9999999 //first mark all group member cell with this
// marker, then replace the value in one scan
int readDefGroupName(defrCallbackType_e c, const char* cl, defiUserData ud) { return 0; }

int readDefGroupMember(defrCallbackType_e c, const char* cl, defiUserData ud) {
    ((Database*)ud)->regions[0]->members.emplace_back(cl);
    return 0;
}

int readDefGroup(defrCallbackType_e c, defiGroup* dgp, defiUserData ud) {
    Database* db = (Database*)ud;
    if (dgp->hasRegionName()) {
        string regionname(dgp->regionName());
        Region* region = db->getRegion(regionname);
        if (!region) {
            logger.warning("Region is not defined: %s", regionname.c_str());
            return 1;
        }
        region->members = db->regions[0]->members;
        db->regions[0]->members.clear();
    }
    return 0;
}
