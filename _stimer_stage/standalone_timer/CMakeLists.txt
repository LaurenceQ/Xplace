cmake_minimum_required(VERSION 3.18)

project(standalone_timer LANGUAGES CXX CUDA)

set(XPLACE_VENDOR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xplace)
set(XPLACE_LEFDEF ${XPLACE_VENDOR}/thirdparty/lefdef)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

find_package(ZLIB REQUIRED)

add_library(stimer_core STATIC
  src/api/Timer.cpp
  src/api/TimerConfig.cpp
  src/db/DesignDB.cpp
  src/db/InputFiles.cpp
  src/db/PathResolver.cpp
  src/db/TimingFlatDB.cpp
  src/db/TimingGraph.cpp
  src/design/DefReader.cpp
  src/design/VerilogReader.cpp
  src/liberty/LibertyReader.cpp
  src/spef/SpefReader.cpp
  src/cuda/TimingCudaRuntime.cu
)

target_include_directories(stimer_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${XPLACE_VENDOR}/cpp_to_py
    ${XPLACE_VENDOR}/cpp_to_py/common/lib
    ${XPLACE_VENDOR}/cpp_to_py/common/lib/spef
    ${XPLACE_LEFDEF}
)

target_link_libraries(stimer_core
  PRIVATE
    ZLIB::ZLIB
    ${XPLACE_LEFDEF}/def58/lib/libdef.a
)

set_target_properties(stimer_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF
)

add_executable(stimer_run
  src/app/run_timer.cpp
)

target_link_libraries(stimer_run
  PRIVATE
    stimer_core
)

install(TARGETS stimer_core stimer_run
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/stimer
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

enable_testing()
add_test(NAME stimer_help COMMAND stimer_run --help)
add_test(NAME stimer_minimal_inputs
  COMMAND stimer_run
    --liberty ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/minimal.lib
    --def ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/minimal.def
    --spef ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/minimal.spef
    --sdc ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/minimal.sdc
    --mode dmp
    --cpu
)
add_test(NAME stimer_minimal_verilog
  COMMAND stimer_run
    --liberty ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/minimal.lib
    --verilog ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/minimal.v
    --spef ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/minimal.spef
    --sdc ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/minimal.sdc
    --mode dmp
    --cpu
)
add_test(NAME stimer_minimal_paths
  COMMAND stimer_run
    --platform ${CMAKE_CURRENT_SOURCE_DIR}/tests/data
    --design-dir ${CMAKE_CURRENT_SOURCE_DIR}/tests/data
    --design minimal
    --mode dmp
    --cpu
)
