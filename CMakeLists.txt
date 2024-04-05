cmake_minimum_required (VERSION 3.10)
PROJECT(C)

set(ROOT_PATH "${PROJECT_SOURCE_DIR}")
configure_file(
    "${PROJECT_SOURCE_DIR}/include/cmake_definition.h.in"
    "${PROJECT_BINARY_DIR}/include/cmake_definition.h"
)
set (CMAKE_CXX_STANDARD 17)
if (UNIX)
    set (IS_LINUX 1)
else (UNIX)
    set (IS_LINUX 0)
endif (UNIX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(Boost REQUIRED filesystem serialization )
find_package(Eigen3 CONFIG REQUIRED)
include( ${EIGEN3_USE_FILE} ) 
find_package(OpenCV CONFIG REQUIRED)
find_package(argparse CONFIG REQUIRED)
find_package(CGAL CONFIG REQUIRED)
find_package(tinyply CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


function(link_general arg)
    target_include_directories(${ARGV0} PRIVATE ${PROJ_INCLUDE_DIRS})
    target_include_directories(${ARGV0} PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(${ARGV0} general  ${OpenCV_LIBRARIES})
    target_link_libraries(${ARGV0} Boost::filesystem Boost::serialization)
    target_link_libraries(${ARGV0} general argparse::argparse)
    target_link_libraries(${ARGV0} general CGAL::CGAL)
    target_link_libraries(${ARGV0} general glog::glog)
    target_link_libraries(${ARGV0} general tinyply)
    target_link_libraries(${ARGV0} general tinyobjloader::tinyobjloader)

    IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
        target_link_libraries(${ARGV0}  general jsoncpp_object jsoncpp_static)
    ELSE()
        target_link_libraries(${ARGV0}  general jsoncpp_lib jsoncpp_object)
    ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")
endfunction()

FILE(GLOB LIBRARY_FILE "include/*.cpp" "include/*.h")

include_directories(${PROJECT_BINARY_DIR}/include/)

# configure_airsim()
add_library(C STATIC ${LIBRARY_FILE})
target_include_directories(C PUBLIC "include")
target_include_directories(C PUBLIC ${Boost_INCLUDE_DIRS})
target_include_directories(C PUBLIC ${ROOT_PATH}/external/tqdmcpp/)
target_include_directories(C PUBLIC ${ROOT_PATH}/external/tinycolormap/include)
target_include_directories(C PUBLIC /root/NVDNet/external/vcglib)
link_general(C)

set(BVH_DIR  "${ROOT_PATH}/external/bvh-distance-queries/")
add_subdirectory(${BVH_DIR} "bvh-distance-queries")
add_subdirectory(src)

