cmake_minimum_required (VERSION 3.10)
PROJECT(prepare_evaluate_gt_feature)


FILE(GLOB SOURCE_FILE *.cpp *.h)

find_package(TBB CONFIG REQUIRED)
find_package(TinyNPY CONFIG REQUIRED)


find_package(CUDAToolkit)

add_executable(${PROJECT_NAME} ${SOURCE_FILE})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "GSP_Field")

target_include_directories(${PROJECT_NAME} PUBLIC ../)
target_include_directories(${PROJECT_NAME} PUBLIC ${ROOT_PATH}/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${BVH_DIR}/src)

target_link_libraries(${PROJECT_NAME} general bvh_distance_queries)
target_link_libraries(${PROJECT_NAME} general C)
link_general(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} general CUDA::cudart)
target_link_libraries(${PROJECT_NAME} general CUDA::cufft)

if (IS_LINUX)
    target_link_libraries(${PROJECT_NAME} general TBB::tbb TBB::tbbmalloc)
    target_link_libraries(${PROJECT_NAME} general TinyNPYstatic)
else (IS_LINUX)
    target_link_libraries(${PROJECT_NAME} general TBB::tbb TBB::tbbmalloc TBB::tbbmalloc_proxy)
    target_link_libraries(${PROJECT_NAME} general TinyNPY)
endif (IS_LINUX)