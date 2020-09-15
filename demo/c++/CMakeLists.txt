find_package(OpenCV REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_subdirectory(detection)

aux_source_directory(. PROCESSING_SRCS)

add_executable(demo ${PROCESSING_SRCS})
target_link_libraries(demo PRIVATE pybind11::embed ${OpenCV_LIBS} detection tracking ${STDCXXFS} )
target_include_directories(demo PRIVATE "${PROJECT_BINARY_DIR}")