#pragma once

#include "common/StageProfiler.h"

#include <cstdarg>
#include <cstdio>

inline void xplace_env_logf(FILE* stream, const char* env_name, const char* fmt, ...)
{
    if (!xplace_env_enabled(env_name)) {
        return;
    }
    std::fprintf(stream, "[%s] ", env_name);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stream, fmt, args);
    va_end(args);
    std::fprintf(stream, "\n");
    std::fflush(stream);
}

inline void xplace_logf(FILE* stream, const char* tag, const char* fmt, ...)
{
    std::fprintf(stream, "[%s] ", tag);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stream, fmt, args);
    va_end(args);
    std::fprintf(stream, "\n");
    std::fflush(stream);
}

#define XPLACE_DEBUGF(env_name, fmt, ...) \
    xplace_env_logf(stderr, env_name, fmt, ##__VA_ARGS__)

#define XPLACE_PROFILEF(env_name, fmt, ...) \
    xplace_env_logf(stdout, env_name, fmt, ##__VA_ARGS__)

#define XPLACE_LOGF(tag, fmt, ...) \
    xplace_logf(stdout, tag, fmt, ##__VA_ARGS__)

#define XPLACE_ERRORF(tag, fmt, ...) \
    xplace_logf(stderr, tag, fmt, ##__VA_ARGS__)
