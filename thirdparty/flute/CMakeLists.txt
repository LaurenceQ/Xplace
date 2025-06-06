################################################################################
## BSD 3-Clause License
##
## Copyright (c) 2018, Iowa State University All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
## * Redistributions of source code must retain the above copyright notice,
## this list of conditions and the following disclaimer.
##
## * Redistributions in binary form must reproduce the above copyright notice,
## this list of conditions and the following disclaimer in the documentation
## and/or other materials provided with the distribution.
##
## * Neither the name of the copyright holder nor the names of its contributors
## may be used to endorse or promote products derived from this software
## without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
## FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
## DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
## SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
## CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
## OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
## USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
################################################################################

CMAKE_MINIMUM_REQUIRED (VERSION 3.18)
PROJECT(flute)

SET(CMAKE_CXX_STANDARD 11)

set(FLUTE_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "Directory where flute.h is located")

FILE(GLOB_RECURSE SRC_FILES_FLUTE ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

ADD_LIBRARY(flute
  SHARED
  ${SRC_FILES_FLUTE}
  )

TARGET_INCLUDE_DIRECTORIES(flute PUBLIC ${FLUTE_INCLUDE_DIR})
TARGET_COMPILE_OPTIONS(flute PRIVATE -fPIC)

INSTALL(TARGETS flute DESTINATION ${XPLACE_LIB_DIR})