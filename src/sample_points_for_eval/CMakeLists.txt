cmake_minimum_required (VERSION 3.10)
PROJECT(sample_points_for_eval)

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

add_executable(${PROJECT_NAME} ${SOURCE_FILE})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "GSP_Field")

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCASCADE_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC "${ROOT_PATH}/external/GTE/GTE")

target_link_libraries(${PROJECT_NAME} general C)
target_link_libraries(${PROJECT_NAME} general ${OpenCASCADE_LIBRARIES})
target_link_libraries(${PROJECT_NAME} general TBB::tbb TBB::tbbmalloc)

if (IS_LINUX)
else (IS_LINUX)
    add_custom_command(TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${OpenCASCADE_DLL} ${TARGET_OUTPUT_PATH}
    )
endif (IS_LINUX)
