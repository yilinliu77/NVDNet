#!/bin/bash

read -p "Windows? (Y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[yY]$ ]]
then
    with_windows=true
else
    with_windows=false
fi

DEPENDENCIES="yaml-cpp hdf5[threadsafe] tinyexif eigen3 ceres[core,lapack,suitesparse] tinyobjloader proj pixman glib argparse pcl[surface-on-nurbs] opencv4[contrib,dnn,eigen,nonfree,opengl,openmp,sfm] cgal tinyxml2 glog cpr jsoncpp glm nanoflann embree3 nanoflann tinyply glew glfw3 imgui[opengl3-binding,glfw-binding] boost-program-options boost-test freeimage expat openblas inih[cpp] tk tinynpy ryml pybind11"

if $with_windows
then
  echo "Built on Windows"
else
  echo "Built on Linux"
  apt update && apt install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev libtool gfortran libdbus-1-dev libxi-dev libxtst-dev zip at libxt-dev gperf libxaw7-dev cifs-utils build-essential g++ gfortran libx11-dev libxkbcommon-x11-dev libxi-dev libgl1-mesa-dev libglu1-mesa-dev mesa-common-dev libxinerama-dev libxxf86vm-dev libxcursor-dev yasm libnuma1 libnuma-dev libtool-bin flex bison libbison-dev autoconf libudev-dev libncurses5-dev libtool libxrandr-dev xutils-dev dh-autoreconf autoconf-archive libgles2-mesa-dev ruby-full pkg-config meson nasm cmake ninja-build libxkbcommon-dev libxcb-keysyms1-dev libxcb-image0-dev libxcb-shm0-dev libxcb-icccm4-dev libxcb-sync-dev libxcb-xfixes0-dev libxcb-shape0-dev libxcb-randr0-dev libxcb-render-util0-dev libxcb-xinerama0-dev libxcb-xkb-dev libxcb-xinput-dev libxext-dev libxfixes-dev libxrender-dev libxcb1-dev libx11-xcb-dev libxcb-glx0-dev libxcb-util0-dev  libdbus-1-dev libxi-dev libxtst-dev libxkbcommon-x11-dev libxext-dev libxfixes-dev libxrender-dev libxcb1-dev libx11-xcb-dev libxcb-glx0-dev libxcb-util0-dev -y
fi

ROOTDIR=$PWD

vcpkg_bin=$ROOTDIR/external/vcpkg/vcpkg

cd $ROOTDIR  && $vcpkg_bin install --keep-going --only-downloads ${DEPENDENCIES} && $vcpkg_bin install --keep-going ${DEPENDENCIES} && $vcpkg_bin install --keep-going ${DEPENDENCIES}

# #######################################################################################
# tool function
# #######################################################################################
function with_backoff {
  local max_attempts=${ATTEMPTS-9}
  local timeout=10
  local attempt=0
  local exitCode=0

  while [[ $attempt < $max_attempts ]]
  do
    "$@"
    exitCode=$?

    if [[ $exitCode == 0 ]]
    then
      break
    fi

    echo "Failure! Retrying in $timeout.." 1>&2
    sleep $timeout
    attempt=$(( attempt + 1 ))
    timeout=$(( timeout))
  done

  if [[ $exitCode != 0 ]]
  then
    echo "You've failed me for the last time! ($@)" 1>&2
  fi

  return $exitCode
}

# #######################################################################################
# Conda
# #######################################################################################
function conda {
  curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe --output Miniconda3-latest-Windows-x86_64.exe
}

# #######################################################################################
# VCPKG
# #######################################################################################

function vcpkg {
  cd $ROOTDIR
  git submodule update external/vcpkg
  cd $ROOTDIR/external/vcpkg

  if $with_windows
  then
    ./bootstrap-vcpkg.bat
  else
    ./bootstrap-vcpkg.sh
  fi

  ./vcpkg install vcpkg-cmake

  # CMake
  # cmake_dir=$(ls $ROOTDIR/external/vcpkg/downloads/tools/ | grep cmake)
  # cmake_ver=$(ls $ROOTDIR/external/vcpkg/downloads/tools/$cmake_dir | grep cmake)

  # ninja_dir=$(ls $ROOTDIR/external/vcpkg/downloads/tools/ | grep ninja)
  # ninja_ver=$(ls $ROOTDIR/external/vcpkg/downloads/tools/$ninja_dir)

  # echo "export PATH=$ROOTDIR/external/vcpkg/:$ROOTDIR/external/vcpkg/downloads/tools/$cmake_dir/$cmake_ver/bin/:$ROOTDIR/external/vcpkg/downloads/tools/$ninja_dir/$ninja_ver:\$PATH" >> ~/.bashrc
  # echo "export PATH=$ROOTDIR/external/vcpkg/:$ROOTDIR/external/vcpkg/downloads/tools/$cmake_dir/$cmake_ver/bin/:$ROOTDIR/external/vcpkg/downloads/tools/$ninja_dir/$ninja_ver:\$PATH" >> ~/.zshrc

  # echo "export vcpkg=$ROOTDIR/external/vcpkg/scripts/buildsystems/vcpkg.cmake" >> ~/.bashrc
  # echo "export vcpkg=$ROOTDIR/external/vcpkg/scripts/buildsystems/vcpkg.cmake" >> ~/.zshrc
}

# #######################################################################################
# AIRSIM
# #######################################################################################
function airsim {
  cd $ROOTDIR/external/Airsim && ./build.cmd
}

# #######################################################################################
# Dependencies
# #######################################################################################
function build_dependencis
{
  
  # #######################################################################################
  # Third party (libnabo)
  # #######################################################################################
  cd $ROOTDIR && mkdir -p external/libnabo/build && cd external/libnabo/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./ -DCMAKE_TOOLCHAIN_FILE=${vcpkg} &&  cmake --build . --target install --config Release -j16 && cd ../../../

  # #######################################################################################
  # Third party (libpointmatcher)
  # #######################################################################################
  cd $ROOTDIR && mkdir -p external/libpointmatcher/build && cd external/libpointmatcher/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./ -DCMAKE_TOOLCHAIN_FILE=${vcpkg} && cmake --build . --target install --config Release -j16 && cd ../../../

  # #######################################################################################
  # Third party (PROJ)
  # #######################################################################################
  conda install sqlite curl libtiff mkl-dnn -y
  # conda install sqlite libtiff curl cmake -y
  cd $ROOTDIR/external/PROJ && mkdir -p build && cd build && cmake .. -DCMAKE_LIBRARY_PATH:FILEPATH="$CONDA_PREFIX/Library/lib" -DBUILD_SHARED_LIBS=false -DCMAKE_INCLUDE_PATH:FILEPATH="$CONDA_PREFIX/Library/include" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./installed -DBUILD_APPS=off -DBUILD_TESTING=off && cmake --build . --target install --config Release -j16 && cd ../../../

  # #######################################################################################
  # Third party (maxflow)
  # #######################################################################################
  cd $ROOTDIR && mkdir -p external/maxflow/build && cd external/maxflow/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./ &&  cmake --build . --target install --config Release -j16 && cd ../../../

  # #######################################################################################
  # Third party (colmap)
  # #######################################################################################
  cd $ROOTDIR && mkdir -p external/colmap/build && cd external/colmap/build && cmake .. -DCMAKE_TOOLCHAIN_FILE=$vcpkg -DCMAKE_INSTALL_PREFIX=./install/release &&  cmake --build . --target install --config Release -j16  && cmake .. -DCMAKE_TOOLCHAIN_FILE=$vcpkg -DCMAKE_INSTALL_PREFIX=./install/debug &&  cmake --build . --target install --config Debug -j16 && cd ../../../

  # #######################################################################################
  # Third party (osqp)
  # #######################################################################################
  cd $ROOTDIR && mkdir -p external/osqp/build && cd external/osqp/build && cmake .. -DCMAKE_TOOLCHAIN_FILE=$vcpkg -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_INSTALL_PREFIX=./install/ &&  cmake --build . --target install --config Debug -j16  && cmake --build . --target install --config Release -j16 && cd ../../../

  # cd $ROOTDIR && mkdir external/pangolin/build && cd external/pangolin/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./ -DCMAKE_TOOLCHAIN_FILE=${vcpkg} -DBUILD_PANGOLIN_FFMPEG=false -DMSVC_USE_STATIC_CRT=false  && cmake --build . --target install --config RelWithDebInfo -j16 &&  cmake --build . --target install --config Release -j16 && cd ../../../

  # #######################################################################################
  # Third party (exiv2)
  # #######################################################################################
  cd $ROOTDIR && mkdir -p external/exiv2/build && cd external/exiv2/build && cmake .. -DCMAKE_TOOLCHAIN_FILE=$vcpkg -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_INSTALL_PREFIX=./install/ -DEXIV2_ENABLE_BROTLI=off &&  cmake --build . --target install --config Debug -j16  && cmake --build . --target install --config Release -j16 && cd ../../../

  # #######################################################################################
  # Third party (libtorch)
  # #######################################################################################
  if [ ! -d "$ROOTDIR/external/libtorch" ]; then
      cd $ROOTDIR/external && curl -q https://download.pytorch.org/libtorch/cu113/libtorch-win-shared-with-deps-1.11.0%2Bcu113.zip --output libtorch-win-shared-with-deps-1.11.0+cu113.zip && unzip -q libtorch-win-shared-with-deps-1.11.0+cu113.zip
  fi

  # #######################################################################################
  # Third party (OpenCascade)
  # #######################################################################################
  cd $ROOTDIR/external && wget https://github.com/Open-Cascade-SAS/OCCT/archive/refs/tags/V7_7_2.zip && unzip -q V7_7_2.zip && mkdir -p OCCT-7_7_2/build && cd OCCT-7_7_2/build && cmake .. -DCMAKE_TOOLCHAIN_FILE=$vcpkg -DCMAKE_INSTALL_PREFIX=./install/ -DBUILD_MODULE_Draw=off -DUSE_TK=off -DUSE_FREETYPE=off &&  cmake --build . --target install --config RelWithDebInfo -j16 && cd ../../../
}

# mkdir $ROOTDIR/build && cd $ROOTDIR/build && cmake .. -DCMAKE_TOOLCHAIN_FILE=${vcpkg} -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
