cmake_minimum_required(VERSION 3.0)
project(A)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_CXX_STANDARD 17)

aux_source_directory(src DIR_SRCS)

add_executable(A ${DIR_SRCS})

target_link_libraries(A  
    ${EIGEN3_LIBRARIES}
    ${OpenCV_LIBRARIES}
)
