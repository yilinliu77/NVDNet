cmake_minimum_required (VERSION 3.10)
PROJECT(prepare_udf)


FILE(GLOB SOURCE_FILE *.cpp *.h *.c lzf/*.cpp lzf/*.c lzf/*.h)

find_package(hdf5 CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)

find_package(CUDAToolkit)
set(BVH_DIR  "${ROOT_PATH}/external/bvh-distance-queries/")
#add_subdirectory(${BVH_DIR} "bvh-distance-queries")

add_executable(${PROJECT_NAME} ${SOURCE_FILE})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "GSP_Field")

if(IS_LINUX)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,stack-size=10000000")
else(IS_LINUX)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")
endif(IS_LINUX)

target_include_directories(${PROJECT_NAME} PUBLIC ../)
target_include_directories(${PROJECT_NAME} PUBLIC ${ROOT_PATH}/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${BVH_DIR}/src)
target_include_directories(${PROJECT_NAME} PUBLIC lzf)


target_link_libraries(${PROJECT_NAME} general bvh_distance_queries)
target_link_libraries(${PROJECT_NAME} general C)
link_general(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} general TBB::tbb TBB::tbbmalloc)
target_link_libraries(${PROJECT_NAME} general CUDA::cudart)
target_link_libraries(${PROJECT_NAME} general CUDA::cufft)

if(IS_LINUX)
target_link_libraries(${PROJECT_NAME} general hdf5::hdf5-static hdf5::hdf5_hl-static hdf5::hdf5_cpp-static hdf5::hdf5_hl_cpp-static)
else(IS_LINUX)
target_link_libraries(${PROJECT_NAME} general hdf5::hdf5-shared hdf5::hdf5_hl-shared hdf5::hdf5_cpp-shared hdf5::hdf5_hl_cpp-shared)
endif(IS_LINUX)