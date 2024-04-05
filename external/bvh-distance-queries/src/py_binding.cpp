#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

#include "bvh.h"

PYBIND11_MODULE(cuda_distance, m) {
    m.doc() = "Calculate distance through cuda";
    m.def("query", &bvh_distance_queries,
        "Calculate distance");
}