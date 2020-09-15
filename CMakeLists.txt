cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(traker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For profiler on Ubuntu
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# Build python package
option(BUILD_PYTHON_PACKAGE "build python package" OFF)

# GCC need to link against stdc++fs
if(MSVC)
set(STDCXXFS "")
else()
set(STDCXXFS "stdc++fs")
endif()

# .exe and .dll
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/thirdpart/libtorch)
add_subdirectory(${PROJECT_SOURCE_DIR}/thirdpart/pybind11)

add_subdirectory(tracking)
if(NOT BUILD_PYTHON_PACKAGE)
    message("-- Build demo")
    add_subdirectory(${PROJECT_SOURCE_DIR}/demo/c++)
endif()