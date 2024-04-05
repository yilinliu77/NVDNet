cmake_minimum_required (VERSION 3.10)
PROJECT(prepare_data_3d)


FILE(GLOB SOURCE_FILE *.cpp *.h *.c lzf/*.cpp lzf/*.c lzf/*.h)

find_package(yaml-cpp CONFIG REQUIRED)

find_package(CUDAToolkit)
find_package(hdf5 CONFIG REQUIRED)

set(BVH_DIR  "../../../external/bvh-distance-queries/")
add_subdirectory(${BVH_DIR} "bvh-distance-queries")

add_executable(${PROJECT_NAME} ${SOURCE_FILE})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "GSP_Field")


if(IS_LINUX)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,stack-size=10000000")
else(IS_LINUX)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")
endif(IS_LINUX)

target_include_directories(${PROJECT_NAME} PUBLIC lzf)
target_include_directories(${PROJECT_NAME} PUBLIC ${BVH_DIR}/src)
target_include_directories(${PROJECT_NAME} PUBLIC ${z5}../../../include)
target_include_directories(${PROJECT_NAME} PUBLIC ../../../include)
target_include_directories(${PROJECT_NAME} PUBLIC ../../../external/libnpy/include)

target_link_libraries(${PROJECT_NAME} general bvh_distance_queries)
target_link_libraries(${PROJECT_NAME} general C)
link_general(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} general yaml-cpp)
target_link_libraries(${PROJECT_NAME} general CUDA::cudart)
target_link_libraries(${PROJECT_NAME} general CUDA::cufft)

if(IS_LINUX)
target_link_libraries(${PROJECT_NAME} general hdf5::hdf5-static hdf5::hdf5_hl-static hdf5::hdf5_cpp-static hdf5::hdf5_hl_cpp-static)
else(IS_LINUX)
target_link_libraries(${PROJECT_NAME} general hdf5::hdf5-shared hdf5::hdf5_hl-shared hdf5::hdf5_cpp-shared hdf5::hdf5_hl_cpp-shared)
endif(IS_LINUX)
target_link_libraries(${PROJECT_NAME} general TBB::tbb TBB::tbbmalloc)