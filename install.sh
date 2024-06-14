DEPENDENCIES="yaml-cpp hdf5[cpp] eigen3 tinyobjloader pixman glib argparse opencv4[core,eigen,tbb,jpeg,contrib,nonfree] cgal glog jsoncpp glm nanoflann tinyply boost-program-options boost-test expat openblas inih[cpp] tinynpy ryml pybind11 hungarian"

apt install -y libpython3.10-dev

cd external/vcpkg && ./bootstrap-vcpkg.sh && cd ../../

./external/vcpkg/vcpkg install --keep-going --only-downloads $DEPENDENCIES && ./external/vcpkg/vcpkg install --keep-going --only-downloads $DEPENDENCIES && ./external/vcpkg/vcpkg install --keep-going --only-downloads $DEPENDENCIES && ./external/vcpkg/vcpkg install $DEPENDENCIES

cd external && wget https://github.com/Open-Cascade-SAS/OCCT/archive/refs/tags/V7_7_2.zip && unzip -q V7_7_2.zip && mkdir -p OCCT-7_7_2/build && cd OCCT-7_7_2/build && cmake .. -DCMAKE_INSTALL_PREFIX=./install/ -DBUILD_MODULE_Draw=off -DUSE_TK=off -DUSE_FREETYPE=off &&  cmake --build . --target install --config RelWithDebInfo -j8 && cd ../../../

echo "export LD_LIBRARY_PATH=/root/NVDNet/external/OCCT-7_7_2/build/install/lib:$LD_LIBRARY_PATH" >> ~/.zshrc

mkdir build && cd build && cmake .. -DCMAKE_TOOLCHAIN_FILE=../external/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release && make -j8