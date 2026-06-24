#pragma once

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

inline bool xplace_env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' ||
             value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

class StageProfiler {
public:
    StageProfiler(const char* tag, bool enabled, FILE* stream = stdout)
        : tag_(tag),
          enabled_(enabled),
          stream_(stream),
          start_(std::chrono::steady_clock::now()),
          last_(start_) {}

    bool enabled() const { return enabled_; }

    double mark(const char* phase)
    {
        if (!enabled_) {
            return 0.0;
        }
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - last_).count();
        const double total = std::chrono::duration<double>(now - start_).count();
        std::fprintf(stream_, "[%s] phase=%s elapsed=%.3f total=%.3f\n",
                     tag_,
                     phase,
                     elapsed,
                     total);
        std::fflush(stream_);
        last_ = now;
        return elapsed;
    }

    double markSeconds(const char* phase)
    {
        if (!enabled_) {
            return 0.0;
        }
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - last_).count();
        const double total = std::chrono::duration<double>(now - start_).count();
        std::fprintf(stream_, "[%s] phase=%s elapsed=%.6f total=%.6f\n",
                     tag_,
                     phase,
                     elapsed,
                     total);
        std::fflush(stream_);
        last_ = now;
        return elapsed;
    }

    void markf(const char* phase, const char* fmt, ...)
    {
        if (!enabled_) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - last_).count();
        const double total = std::chrono::duration<double>(now - start_).count();
        std::fprintf(stream_, "[%s] phase=%s elapsed=%.3f total=%.3f",
                     tag_,
                     phase,
                     elapsed,
                     total);
        if (fmt != nullptr && fmt[0] != '\0') {
            std::fprintf(stream_, " ");
            va_list args;
            va_start(args, fmt);
            std::vfprintf(stream_, fmt, args);
            va_end(args);
        }
        std::fprintf(stream_, "\n");
        std::fflush(stream_);
        last_ = now;
    }

    void print(const char* fmt, ...)
    {
        if (!enabled_) {
            return;
        }
        va_list args;
        va_start(args, fmt);
        std::vfprintf(stream_, fmt, args);
        va_end(args);
        std::fflush(stream_);
    }

    double elapsedTotal() const
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }

private:
    const char* tag_;
    bool enabled_;
    FILE* stream_;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point last_;
};
