DEPENDENCIES="yaml-cpp hdf5 eigen3 tinyobjloader pixman glib argparse opencv4[core,eigen,tbb,jpeg,contrib,nonfree] cgal glog jsoncpp glm nanoflann tinyply boost-program-options boost-test expat openblas inih[cpp] tinynpy ryml pybind11"

# sed -i -e "s/tukaani-project/bminor/g" externel/vcpkg/ports/liblzma/portfile.cmake

./externel/vcpkg/vcpkg install --keep-going --only-downloads $DEPENDENCIES && vcpkg install --keep-going --only-downloads $DEPENDENCIES && vcpkg install --keep-going --only-downloads $DEPENDENCIES && vcpkg install $DEPENDENCIES
