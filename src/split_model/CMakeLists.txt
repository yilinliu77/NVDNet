cmake_minimum_required (VERSION 3.10)
PROJECT(split_model)


FILE(GLOB SOURCE_FILE *.cpp *.h)

#find_package(yaml-cpp CONFIG REQUIRED)
find_package(ryml CONFIG REQUIRED)

find_package(TBB CONFIG REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILE})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "GSP_Field")

target_include_directories(${PROJECT_NAME} PUBLIC ../../../include)

target_link_libraries(${PROJECT_NAME} general C)
link_general(${PROJECT_NAME})
#target_link_libraries(${PROJECT_NAME} general yaml-cpp)
target_link_libraries(${PROJECT_NAME} general TBB::tbb TBB::tbbmalloc)
target_link_libraries(${PROJECT_NAME} general ryml::ryml)
