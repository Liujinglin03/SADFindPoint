cmake_minimum_required(VERSION 2.8)
project(SADFind)

find_package(OpenCV 3)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

add_library(${PROJECT_NAME} SHARED
src/Sadfind.cc
include/Sadfind.h
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}")

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
)

# Build examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Sadfindxy)


add_executable(Sadfindxy Examples/Sadfindxy.cc)
target_link_libraries(Sadfindxy ${PROJECT_NAME})
