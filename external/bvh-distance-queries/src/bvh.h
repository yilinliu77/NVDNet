#ifndef BVH_H
#define BVH_H
#include <iostream>
#include <limits>
#include <vector>

// #include <torch/extension.h>

#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include "aabb.hpp"
#include "triangle.hpp"

std::tuple<std::vector<double>, std::vector<double>, std::vector<long long>, std::vector<double>> bvh_distance_queries(
    const std::vector<double>& triangles, const std::vector<double>& points,
    int queue_size = 128, long long num_points_batch = 16777216, const int id_gpu = -1);

template <typename T>
struct BVHNode {
public:
    AABB<T> bbox;

    // __host__ __device__
    // BVHNode(): left(nullptr), right(nullptr), tri_ptr(nullptr), idx(-1);

    int tri_id = -1;
    int left = -1;
    int right = -1;
    int parent = -1;
    __host__ __device__ inline bool isLeaf() const { return tri_id != -1; }

    int idx;
};

class MyBVH
{
public:
    thrust::device_vector<BVHNode<double>>* _d_bvh_nodes;
    thrust::device_vector<Triangle<double>>* _d_triangles;

    MyBVH(const std::vector<std::vector<double>>& vertices,
        const std::vector<std::vector<int>>& faces);
    ~MyBVH()
    {
    }

    std::tuple<std::vector<double>, std::vector<double>, std::vector<long long>, std::vector<double>> query(
        const std::vector<double>& points, int queue_size = 128, const long long num_points_batch = 16777216, const int id_gpu = 0
    );
};



#endif
