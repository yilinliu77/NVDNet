cmake_minimum_required (VERSION 3.10)
PROJECT(prepare_evaluate_gt)


FILE(GLOB SOURCE_FILE *.cpp *.h)

# OpenCascade
if (IS_LINUX)
    set(OpenCASCADE_DIR "${ROOT_PATH}/external/OCCT-7_7_2/build/install/lib/cmake/opencascade")
else (IS_LINUX)
    set(OpenCASCADE_DIR "${ROOT_PATH}/external/OCCT-7_7_2/build/install/cmake")
endif (IS_LINUX)
find_package(OpenCASCADE CONFIG REQUIRED)
FILE(GLOB OpenCASCADE_DLL ${OpenCASCADE_BINARY_DIR}/../bini/*.dll) # Only needed on windows
if(CMAKE_CONFIGURATION_TYPES)
    set(TARGET_OUTPUT_PATH ${PROJECT_BINARY_DIR}/$<CONFIG>)
else()
    set(TARGET_OUTPUT_PATH ${PROJECT_BINARY_DIR})
endif()

#find_package(yaml-cpp CONFIG REQUIRED)
find_package(hdf5 CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(ryml CONFIG REQUIRED)


find_package(CUDAToolkit)
set(BVH_DIR  "${ROOT_PATH}/external/bvh-distance-queries/")

add_executable(${PROJECT_NAME} ${SOURCE_FILE})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "GSP_Field")

target_include_directories(${PROJECT_NAME} PUBLIC ../)
target_include_directories(${PROJECT_NAME} PUBLIC ${ROOT_PATH}/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${BVH_DIR}/src)

target_link_libraries(${PROJECT_NAME} general bvh_distance_queries)
target_link_libraries(${PROJECT_NAME} general C)
link_general(${PROJECT_NAME})
#target_link_libraries(${PROJECT_NAME} general yaml-cpp)
target_link_libraries(${PROJECT_NAME} general ryml::ryml)
target_link_libraries(${PROJECT_NAME} general ${OpenCASCADE_LIBRARIES})
target_link_libraries(${PROJECT_NAME} general CUDA::cudart)
target_link_libraries(${PROJECT_NAME} general CUDA::cufft)

if (IS_LINUX)
    target_link_libraries(${PROJECT_NAME} general TBB::tbb TBB::tbbmalloc)

else (IS_LINUX)
    target_link_libraries(${PROJECT_NAME} general TBB::tbb TBB::tbbmalloc TBB::tbbmalloc_proxy)

    add_custom_command(TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${OpenCASCADE_DLL} ${TARGET_OUTPUT_PATH}
    )
endif (IS_LINUX)