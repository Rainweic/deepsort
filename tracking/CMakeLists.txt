# Opencv
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

aux_source_directory(src TRACKING_SRCS)

if(BUILD_PYTHON_PACKAGE)
    message("-- Build python package")
    pybind11_add_module(tracking ${TRACKING_SRCS})
else()
    message("-- Build c++ dynamic library")
    add_library(tracking SHARED ${TRACKING_SRCS})
endif()


include(GenerateExportHeader)
GENERATE_EXPORT_HEADER(tracking)

target_link_libraries(tracking PUBLIC ${OpenCV_LIBS} PRIVATE ${TORCH_LIBRARIES})
target_include_directories(tracking
        PUBLIC include ${CMAKE_CURRENT_BINARY_DIR}
        PRIVATE src)