# Files
# NOTE: GLOB_RECURSE caches at configure time — re-run cmake after adding new .cpp files.
file(GLOB_RECURSE SRC_FILES_GT ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE SRC_FILES_GT_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# OpenMP
find_package(OpenMP REQUIRED)

# Flute MP
set(FLUTE_MP_INCLUDE_DIR ${PATH_THIRDPARTY_ROOT}/flute_mp)

# CUDA/CPP GGR Kernel
add_library(gt STATIC ${CMAKE_CURRENT_SOURCE_DIR}/../io_parser/gp/GPDatabase.cpp
                       ${SRC_FILES_GT}
                       ${SRC_FILES_GT_CUDA}
                       ${FLUTE_MP_INCLUDE_DIR}/flute.cpp
                       )

set_target_properties(gt PROPERTIES
                       CUDA_RESOLVE_DEVICE_SYMBOLS ON
                       CUDA_SEPARABLE_COMPILATION ON
                       POSITION_INDEPENDENT_CODE ON)

target_include_directories(gt PRIVATE ${PROJECT_SOURCE_DIR}/cpp_to_py ${TORCH_INCLUDE_DIRS} ${FLUTE_MP_INCLUDE_DIR})
target_link_libraries(gt PRIVATE torch ${TORCH_PYTHON_LIBRARY} xplace_common io_parser OpenMP::OpenMP_CXX stdc++fs pthread)
target_compile_options(gt PRIVATE -fPIC)
target_compile_options(gt PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")
option(DMP_LOAD_CROSSING_BISECTION "Use value-only bisection for DMP load crossing" OFF)
set(DMP_LOAD_BISECTION_ITERS "12" CACHE STRING "Fixed bisection iterations for DMP load crossing when enabled")
if(DMP_LOAD_CROSSING_BISECTION)
  target_compile_definitions(gt PRIVATE DMP_LOAD_CROSSING_BISECTION=1 DMP_LOAD_BISECTION_ITERS=${DMP_LOAD_BISECTION_ITERS})
else()
  target_compile_definitions(gt PRIVATE DMP_LOAD_CROSSING_BISECTION=0 DMP_LOAD_BISECTION_ITERS=${DMP_LOAD_BISECTION_ITERS})
endif()

pybind11_add_module(gputimer MODULE NO_EXTRAS PyBindCppMain.cpp)

target_include_directories(gputimer PRIVATE ${TORCH_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/cpp_to_py ${FLUTE_MP_INCLUDE_DIR})
target_link_libraries(gputimer PRIVATE torch ${TORCH_PYTHON_LIBRARY} xplace_common io_parser gt)
target_compile_definitions(gputimer PRIVATE 
                            TORCH_EXTENSION_NAME=gputimer
                            TORCH_VERSION_MAJOR=${TORCH_VERSION_MAJOR}
                            TORCH_VERSION_MINOR=${TORCH_VERSION_MINOR}
                            ENABLE_CUDA=${TORCH_ENABLE_CUDA})

install(TARGETS gputimer DESTINATION ${XPLACE_LIB_DIR})
