#!/bin/bash

read -p "Windows? (Y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[yY]$ ]]
then
    with_windows=true
else
    with_windows=false
fi

git submodule update --init external/elsed external/exiv2 external/colmap

mkdir -p external/exiv2/build && cd external/exiv2/build && cmake .. -DCMAKE_TOOLCHAIN_FILE=$vcpkg -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_INSTALL_PREFIX=./install/ -DEXIV2_ENABLE_BROTLI=off &&  cmake --build . --target install --config Release -j16 && cd ../../../

#mkdir -p external/elsed/build && cd external/elsed/build && cmake .. -DCMAKE_TOOLCHAIN_FILE=$vcpkg -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_INSTALL_PREFIX=./install/ && cmake --build . --config Release -j16 && cd ../../../

# #######################################################################################
# Third party (colmap)
# Add "-T v142" on windows when runing cmake. It is a bug in visual studio 2023
# #######################################################################################

if $with_windows
then
    mv external/colmap/CMakeLists.txt external/colmap/CMakeLists.txt.bak && sed 's/find_package(FreeImage REQUIRED)/find_package(freeimage CONFIG REQUIRED)/g; s/${FREEIMAGE_LIBRARIES}/freeimage::FreeImage/g' external/colmap/CMakeLists.txt.bak >> external/colmap/CMakeLists.txt

    mv external/colmap/cmake/CMakeConfig.cmake.in external/colmap/cmake/CMakeConfig.cmake.in.bak && sed 's/find_package(FreeImage QUIET)/find_package(freeimage CONFIG QUIET)/g; ;s/find_package(FreeImage REQUIRED)/find_package(freeimage CONFIG REQUIRED)/g; s/${FREEIMAGE_LIBRARIES}/freeimage::FreeImage/g' external/colmap/cmake/CMakeConfig.cmake.in.bak >> external/colmap/cmake/CMakeConfig.cmake.in

    mkdir -p external/colmap/build && cd external/colmap/build && cmake .. -DCMAKE_TOOLCHAIN_FILE=$vcpkg -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_INSTALL_PREFIX=./install/release -T v142 &&  cmake --build . --target install --config Debug -j16 && cmake --build . --target install --config Release -j16 && cd ../../../
else
    vcpkg install colmap[cuda]
fi