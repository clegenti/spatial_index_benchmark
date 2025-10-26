cmake_minimum_required(VERSION 3.10)
project(BatchBubbleGrowerDemo)

set(CMAKE_CXX_STANDARD 17)
# Set as release
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx -mbmi2")

# Find the Eigen3 package
find_package(Eigen3 REQUIRED)



# Call the cmake file in the lib folder
add_subdirectory(lib)


add_executable(spatial_index_benchmark app/spatial_index_benchmark.cpp)
target_link_libraries(spatial_index_benchmark phtree ikdtree)
target_include_directories(spatial_index_benchmark PUBLIC lib include ${EIGEN3_INCLUDE_DIR})


# Enable openMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
