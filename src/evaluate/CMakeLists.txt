cmake_minimum_required (VERSION 3.10)
PROJECT(evaluate)


FILE(GLOB SOURCE_FILE *.cpp *.h)

find_package(TBB CONFIG REQUIRED)
find_package(hungarian CONFIG REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILE})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "GSP_Field")

target_include_directories(${PROJECT_NAME} PUBLIC ../)
target_include_directories(${PROJECT_NAME} PUBLIC ../../../include)

target_link_libraries(${PROJECT_NAME} general C)
link_general(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} general hungarian::hungarian)

if (IS_LINUX)
    target_link_libraries(${PROJECT_NAME} general TBB::tbb TBB::tbbmalloc)

else (IS_LINUX)
    target_link_libraries(${PROJECT_NAME} general TBB::tbb TBB::tbbmalloc TBB::tbbmalloc_proxy)
endif (IS_LINUX)