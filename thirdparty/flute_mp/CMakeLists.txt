# cmake_minimum_required(VERSION 2.8.12)

# set(CMAKE_CXX_STANDARD 17)

# set(FLUTE_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "Directory where flute.h is located")

# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# project(flute)
# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE "Release" CACHE STRING
#     "Choose the type of build, options are: Debug Release."
#     FORCE)
# endif(NOT CMAKE_BUILD_TYPE)

# set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/flute.cpp)

# include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# add_library(${PROJECT_NAME} STATIC ${SOURCES})

# install(TARGETS ${PROJECT_NAME} DESTINATION thirdparty/${PROJECT_NAME})
# install(DIRECTORY lut.ICCAD2015 DESTINATION thirdparty/${PROJECT_NAME})


CMAKE_MINIMUM_REQUIRED (VERSION 3.1)
PROJECT(flute_mp)

SET(CMAKE_CXX_STANDARD 17)

set(FLUTE_MP_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "Directory where flute.h is located")

FILE(GLOB_RECURSE SRC_FILES_FLUTE_MP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

ADD_LIBRARY(flute_mp
  SHARED
  ${SRC_FILES_FLUTE_MP}
  )

TARGET_INCLUDE_DIRECTORIES(flute_mp PUBLIC ${FLUTE_MP_INCLUDE_DIR})
TARGET_COMPILE_OPTIONS(flute_mp PRIVATE -fPIC)

INSTALL(TARGETS flute_mp DESTINATION ${XPLACE_LIB_DIR})