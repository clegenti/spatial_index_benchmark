
find_package(PCL 1.8 REQUIRED)

add_library(ikdtree STATIC ikd_tree.cpp)
target_link_libraries(ikdtree ${PCL_LIBRARIES})