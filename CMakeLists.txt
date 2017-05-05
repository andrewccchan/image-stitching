cmake_minimum_required(VERSION 2.8)
project(Stitching)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")
find_package(OpenCV REQUIRED)
aux_source_directory(src SRC)
add_executable(MSOP ${SRC})
target_link_libraries(MSOP ${OpenCV_LIBS})
