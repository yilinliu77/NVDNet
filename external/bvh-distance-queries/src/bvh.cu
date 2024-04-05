#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "device_launch_parameters.h"
#include "bvh.h"

#include <algorithm>
#include <iomanip>
#include <numeric>

#include "defs.hpp"
#include "double_vec_ops.h"
#include "helper_math.h"
#include "math_utils.hpp"
#include "priority_queue.hpp"


// Number of threads per block for CUDA kernel launch
#ifndef NUM_THREADS
#define NUM_THREADS 256
#endif

#ifndef FORCE_INLINE
#define FORCE_INLINE 1
#endif /* ifndef FORCE_INLINE */

#ifndef BVH_PROFILING
#define BVH_PROFILING 0
#endif /* ifndef BVH_PROFILING */

#ifndef ERROR_CHECKING
#define ERROR_CHECKING 1
#endif /* ifndef ERROR_CHECKING */

// Macro for checking cuda errors following a cuda launch or api call
#if ERROR_CHECKING == 1
#define cudaCheckError()                                             \
    {                                                                \
        cudaDeviceSynchronize();                                     \
        cudaError_t e = cudaGetLastError();                          \
        if (e != cudaSuccess) {                                      \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
                cudaGetErrorString(e));                              \
            throw;                                                 \
        }                                                            \
    }
#else
#define cudaCheckError()
#endif

template <typename T>
std::ostream& operator<<(std::ostream& os, const vec3<T>& x)
{
    os << x.x << ", " << x.y << ", " << x.z;
    return os;
}

std::ostream& operator<<(std::ostream& os, const vec3<float>& x)
{
    os << x.x << ", " << x.y << ", " << x.z;
    return os;
}

std::ostream& operator<<(std::ostream& os, const vec3<double>& x)
{
    os << x.x << ", " << x.y << ", " << x.z;
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, vec3<T> x)
{
    os << x.x << ", " << x.y << ", " << x.z;
    return os;
}

__host__ __device__ inline double3 fmin(const double3& a, const double3& b)
{
    return make_double3(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
}

__host__ __device__ inline double3 fmax(const double3& a, const double3& b)
{
    return make_double3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));
}

template <typename T>
__host__ __device__ T pointToTriangleDistance(vec3<T> p,
    TrianglePtr<T> tri_ptr)
{
    vec3<T> a = tri_ptr->v0;
    vec3<T> b = tri_ptr->v1;
    vec3<T> c = tri_ptr->v2;

    vec3<T> ba = b - a;
    vec3<T> pa = p - a;
    vec3<T> cb = c - b;
    vec3<T> pb = p - b;
    vec3<T> ac = a - c;
    vec3<T> pc = p - c;
    vec3<T> nor = cross(ba, ac);

    return (sign<T>(dot(cross(ba, nor), pa)) + sign<T>(dot(cross(cb, nor), pb)) + sign<T>(dot(cross(ac, nor), pc)) < 2.0)
        ? min(min(dot2<T>(ba * clamp(dot(ba, pa) / dot2<T>(ba), 0.0, 1.0) - pa),
                  dot2<T>(cb * clamp(dot(cb, pb) / dot2<T>(cb), 0.0, 1.0) - pb)),
            dot2<T>(ac * clamp(dot(ac, pc) / dot2<T>(ac), 0.0, 1.0) - pc))
        : dot(nor, pa) * dot(nor, pa) / dot2<T>(nor);
}

template <typename T>
__host__ __device__ T pointToTriangleDistance(vec3<T> p, const Triangle<T>& tri,
    vec3<T>* closest_bc,
    vec3<T>* closest_point)
{
    vec3<T> a = tri.v0;
    vec3<T> b = tri.v1;
    vec3<T> c = tri.v2;

    // Check if P in vertex region outside A
    vec3<T> ab = b - a;
    vec3<T> ac = c - a;
    vec3<T> ap = p - a;
    T d1 = dot(ab, ap);
    T d2 = dot(ac, ap);
    if (d1 <= static_cast<T>(0) && d2 <= static_cast<T>(0)) {
        *closest_point = a;
        *closest_bc = make_vec3<T>(static_cast<T>(1), static_cast<T>(0), static_cast<T>(0));
        return dot(ap, ap);
    }
    // Check if P in vertex region outside B
    vec3<T> bp = p - b;
    T d3 = dot(ab, bp);
    T d4 = dot(ac, bp);

    if (d3 >= static_cast<T>(0) && d4 <= d3) {
        *closest_point = b;
        *closest_bc = make_vec3<T>(static_cast<T>(0), static_cast<T>(1), static_cast<T>(0));
        return dot(bp, bp);
    }
    // Check if P in edge region of AB, if so return projection of P onto AB
    T vc = d1 * d4 - d3 * d2;
    if (vc <= static_cast<T>(0) && d1 >= static_cast<T>(0) && d3 <= static_cast<T>(0)) {
        T v = d1 / (d1 - d3);
        *closest_point = a + v * ab;
        *closest_bc = make_vec3<T>(static_cast<T>(1 - v), static_cast<T>(v), static_cast<T>(0));
        return dot(p - *closest_point, p - *closest_point);
    }
    // Check if P in vertex region outside C
    vec3<T> cp = p - c;
    T d5 = dot(ab, cp);
    T d6 = dot(ac, cp);
    if (d6 >= static_cast<T>(0) && d5 <= d6) {
        *closest_point = c;
        *closest_bc = make_vec3<T>(0, 0, 1);
        return dot(cp, cp);
    }
    // Check if P in edge region of AC, if so return projection of P onto AC
    T vb = d5 * d2 - d1 * d6;
    if (vb <= static_cast<T>(0) && d2 >= static_cast<T>(0) && d6 <= static_cast<T>(0)) {
        T w = d2 / (d2 - d6);
        *closest_point = a + w * ac;
        *closest_bc = make_vec3<T>(static_cast<T>(1 - w), static_cast<T>(0), static_cast<T>(w));
        return dot(p - *closest_point, p - *closest_point);
    }
    // Check if P in edge region of BC, if so return projection of P onto BC
    T va = d3 * d6 - d5 * d4;
    if (va <= static_cast<T>(0) && (d4 - d3) >= static_cast<T>(0) && (d5 - d6) >= static_cast<T>(0)) {
        T w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        *closest_point = b + w * (c - b);
        *closest_bc = make_vec3<T>(static_cast<T>(0), static_cast<T>(1 - w), static_cast<T>(w));
        return dot(p - *closest_point, p - *closest_point);
    }
    // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    T denom = static_cast<T>(1) / (va + vb + vc);
    T v = vb * denom;
    T w = vc * denom;
    *closest_point = a + v * ab + w * ac;
    *closest_bc = make_vec3<T>(static_cast<T>(1 - v - w), static_cast<T>(v), static_cast<T>(w));
    return dot(p - *closest_point, p - *closest_point);
}

template <typename T>
__global__ void ComputeTriBoundingBoxes(Triangle<T>* triangles,
    int num_triangles, AABB<T>* bboxes)
{
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_triangles;
         idx += blockDim.x * gridDim.x) {
        bboxes[idx] = triangles[idx].ComputeBBox();
    }
    return;
}



template <typename T>
__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
    bool
    checkOverlap(const AABB<T>& bbox1, const AABB<T>& bbox2)
{
    return (bbox1.min_t.x <= bbox2.max_t.x) && (bbox1.max_t.x >= bbox2.min_t.x) && (bbox1.min_t.y <= bbox2.max_t.y) && (bbox1.max_t.y >= bbox2.min_t.y) && (bbox1.min_t.z <= bbox2.max_t.z) && (bbox1.max_t.z >= bbox2.min_t.z);
}

template <typename T, int StackSize = 32>
__device__ __host__ T traverseBVHStack(
    const Triangle<double>* triangles,
    const vec3<T>& queryPoint, const BVHNode<T>* tree,
    long long* closest_face, vec3<T>* closest_bc,
    vec3<T>* closestPoint)
{
    int stack[StackSize];
    int* stackPtr = stack;
    *stackPtr++ = -1; // push

    int cur_node = 0;
    T closest_distance = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;

    do {
        // Check each child node for overlap.
        const BVHNode<T>& node = tree[cur_node];
        const BVHNode<T>& childL = tree[node.left];
        const BVHNode<T>& childR = tree[node.right];

        T distance_left = pointToAABBDistance<T>(queryPoint, childL.bbox);
        T distance_right = pointToAABBDistance<T>(queryPoint, childR.bbox);

        bool checkL = distance_left <= closest_distance;
        bool checkR = distance_right <= closest_distance;

        if (checkL && childL.isLeaf()) {
            // If  the child is a leaf then
            vec3<T> curr_clos_point;
            vec3<T> curr_closest_bc;

            T distance_left = pointToTriangleDistance<T>(
                queryPoint, triangles[childL.tri_id], &curr_closest_bc, &curr_clos_point);
            if (distance_left <= closest_distance) {
                closest_distance = distance_left;
                *closest_face = childL.tri_id;
                *closestPoint = curr_clos_point;
                *closest_bc = curr_closest_bc;
            }
        }

        if (checkR && childR.isLeaf()) {
            // If  the child is a leaf then
            vec3<T> curr_clos_point;
            vec3<T> curr_closest_bc;

            T distance_right = pointToTriangleDistance<T>(
                queryPoint, triangles[childR.tri_id], &curr_closest_bc, &curr_clos_point);
            if (distance_right <= closest_distance) {
                closest_distance = distance_right;
                *closest_face = childR.tri_id;
                *closestPoint = curr_clos_point;
                *closest_bc = curr_closest_bc;
            }
        }
        // Query overlaps an internal node => traverse.
        bool traverseL = (checkL && !childL.isLeaf());
        bool traverseR = (checkR && !childR.isLeaf());

        if (!traverseL && !traverseR) {
            cur_node = *--stackPtr; // pop
        }
        else {
            cur_node = (traverseL) ? node.left : node.right;
            if (traverseL && traverseR) {
                *stackPtr++ = node.right; // push
            }
        }
    } while (cur_node != -1);

    return closest_distance;
}

template <typename T, int QueueSize = 32>
__global__ void findNearestNeighbor(
    const Triangle<double>* triangles,
    vec3<T>* query_points, T* distances,
    vec3<T>* closest_points,
    long long* closest_faces,
    vec3<T>* closest_bcs,
    const BVHNode<T>* root, int num_points,
    bool use_stack = true)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_points;
         idx += blockDim.x * gridDim.x) {
        vec3<T> query_point = query_points[idx];

        long long closest_face;
        vec3<T> closest_bc;
        vec3<T> closest_point;

        T closest_distance;

        closest_distance = traverseBVHStack<T, QueueSize>(
            triangles,
            query_point, root, &closest_face, &closest_bc, &closest_point);

        distances[idx] = sqrt(closest_distance);
        closest_points[idx] = closest_point;
        closest_faces[idx] = closest_face;
        closest_bcs[idx] = closest_bc;
    }
    return;
}

template <typename T>
__global__ void copy_to_tensor(T* dest, T* source, int* ids, int num_elements)
{
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_elements;
         idx += blockDim.x * gridDim.x) {
        // dest[idx] = source[ids[idx]];
        dest[ids[idx]] = source[idx];
    }
    return;
}

__global__
void test(vec3<double>* queryPoint, int num_points, Triangle<double>* tri, int num_triangles, double* result)
{

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_points; idx += blockDim.x * gridDim.x)
    {
        double min_dis = DBL_MAX;
        int id_tri;
        vec3<double> curr_closest_bc;
        vec3<double> curr_clos_point;
        for (int i_tri = 0; i_tri < num_triangles; ++i_tri)
        {
            double distance = pointToTriangleDistance<double>(queryPoint[idx], tri[i_tri], &curr_closest_bc, &curr_clos_point);
            if (distance < min_dis)
            {
                min_dis = distance;
                id_tri = i_tri;
            }
        }
        result[idx] = min_dis;
    }
    return;
}

AABB<double> UpdateNodeBounds(const std::vector<int>& triangle_ids, const std::vector<AABB<double>>& bboxs)
{
    AABB<double> scene_box = std::accumulate(triangle_ids.begin(), triangle_ids.end(), AABB<double>(),
        [&bboxs](const AABB<double>& v_box, const int id_tri)
        {
            return v_box + bboxs[id_tri];
        });
    return scene_box;
}

void Subdivide(
    const int v_current_id, int& v_node_used,
    std::vector<BVHNode<double>>& v_node, 
    const std::vector<double3>& centroids,
    const std::vector<AABB<double>>& bboxs,
    std::vector<int>& v_ids)
{
    double3 extent = v_node[v_current_id].bbox.max_t - v_node[v_current_id].bbox.min_t;
    int axis = 0;
    if (extent.x > extent.y)
    {
	    if (extent.x > extent.z) axis = 0;
	    else axis = 2;
    }
    else
    {
	    if (extent.y > extent.z) axis = 1;
	    else axis = 2;
    }
    const int mid = v_ids.size() / 2;
    if (axis == 0)
        std::nth_element(v_ids.begin(), v_ids.begin() + mid, v_ids.end(),
            [&centroids](const int id1, const int id2) {return centroids[id1].x < centroids[id2].x; });
    else if (axis ==1)
        std::nth_element(v_ids.begin(), v_ids.begin() + mid, v_ids.end(),
            [&centroids](const int id1, const int id2) {return centroids[id1].y < centroids[id2].y; });
    else
        std::nth_element(v_ids.begin(), v_ids.begin() + mid, v_ids.end(),
            [&centroids](const int id1, const int id2) {return centroids[id1].z < centroids[id2].z; });

    const int left = v_node_used;
    const int right = v_node_used + 1;
    v_node[v_current_id].left = left;
    v_node[v_current_id].right = right;

    std::vector<int> left_ids(v_ids.begin(), v_ids.begin() + mid);
    std::vector<int> right_ids(v_ids.begin() + mid, v_ids.end());
    v_node[left].bbox = UpdateNodeBounds(left_ids, bboxs);
    v_node[right].bbox = UpdateNodeBounds(right_ids, bboxs);

    v_node[left].parent = v_current_id;
    v_node[right].parent = v_current_id;
    v_node_used += 2;
    if (left_ids.size() > 1)
		Subdivide(left, v_node_used, v_node, centroids, bboxs, left_ids);
    else
        v_node[left].tri_id = left_ids[0];
	
    if (right_ids.size() > 1)
		Subdivide(right, v_node_used, v_node, centroids, bboxs, right_ids);
    else
        v_node[right].tri_id = right_ids[0];
    return;
}

std::vector<BVHNode<double>> buildBVH_sequential(
    thrust::host_vector<Triangle<double>>& v_triangles)
{
    const int num_triangles = v_triangles.size();
    std::vector<double3> centroids(num_triangles);
    std::vector<AABB<double>> bboxs(num_triangles);
    for (int i = 0; i < num_triangles; i++)
    {
        centroids[i] = (v_triangles[i].v0 + v_triangles[i].v1 + v_triangles[i].v2) / 3.;
        bboxs[i] = v_triangles[i].ComputeBBox();
    }

    std::vector<BVHNode<double>> bvh_nodes(num_triangles * 2 - 1);
    int rootNodeIdx = 0, nodesUsed = 1;

    // assign all triangles to root node
    BVHNode<double>& root = bvh_nodes[0];
    std::vector<int> triangle_ids(num_triangles, 0);
    std::iota(triangle_ids.begin(), triangle_ids.end(), 0);
    
    root.bbox = UpdateNodeBounds(triangle_ids, bboxs);
    // subdivide recursively
    Subdivide(rootNodeIdx, nodesUsed, bvh_nodes, centroids,bboxs, triangle_ids);

    // Debug
    // double3 p = make_double3(-1., -1., -1.);
    // long long closest_face;
    // double3 closet_bcs, closet_points;
    // std::cout << traverseBVHStack_h(v_triangles.data(), p, &bvh_nodes[0], &closest_face, &closet_bcs, &closet_points);

    return bvh_nodes;
}

void bvh_distance_queries_kernel(
    const BVHNode<double>* bvh_tree,
    const thrust::device_vector<Triangle<double>>& triangles, const thrust::device_vector<vec3<double>>& points,
    thrust::device_vector<double>& distances, thrust::device_vector<double>& closest_points,
    thrust::device_vector<long long>& closest_faces, thrust::device_vector<double>& closest_bcs,
    int queue_size)
{
    const auto num_triangles = triangles.size();
    const auto num_points = points.size();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    thrust::device_vector<int> triangle_ids(num_triangles);

    // Build BVH
    thrust::device_vector<BVHNode<double>> leaf_nodes(num_triangles);
    thrust::device_vector<BVHNode<double>> internal_nodes(num_triangles - 1);

    const Triangle<double>* triangle_T_ptr = triangles.data().get();

    int blockSize = NUM_THREADS;

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int gridSize = std::min(
        32 * numSMs, static_cast<int>((num_points + blockSize - 1) / blockSize));

    // Calculate
    double* distances_ptr;
    cudaMalloc((void**)&distances_ptr, num_points * sizeof(double));
    cudaCheckError();

    vec3<double>* closest_points_ptr;
    cudaMalloc((void**)&closest_points_ptr, num_points * sizeof(vec3<double>));
    cudaCheckError();

    long long* closest_faces_ptr;
    cudaMalloc((void**)&closest_faces_ptr, num_points * sizeof(long long));
    cudaCheckError();

    vec3<double>* closest_bcs_ptr;
    cudaMalloc((void**)&closest_bcs_ptr, num_points * sizeof(vec3<double>));
    cudaCheckError();

    vec3<double>* points_ptr = (vec3<double>*)points.data().get();

    if (queue_size == 32) {
        findNearestNeighbor<double, 32> << <gridSize, NUM_THREADS, 0, stream >> > (
            triangle_T_ptr,
            points_ptr, distances_ptr, closest_points_ptr,
            closest_faces_ptr, closest_bcs_ptr,
            bvh_tree, num_points);
    }
    else if (queue_size == 64) {
        findNearestNeighbor<double, 64> << <gridSize, NUM_THREADS, 0, stream >> > (
            triangle_T_ptr,
            points_ptr, distances_ptr, closest_points_ptr,
            closest_faces_ptr, closest_bcs_ptr,
            bvh_tree, num_points);
    }
    else if (queue_size == 128) {
        findNearestNeighbor<double, 128> << <gridSize, NUM_THREADS, 0, stream >> > (
            triangle_T_ptr,
            points_ptr, distances_ptr, closest_points_ptr,
            closest_faces_ptr, closest_bcs_ptr,
            bvh_tree, num_points);
    }
    else if (queue_size == 256) {
        findNearestNeighbor<double, 256> << <gridSize, NUM_THREADS, 0, stream >> > (
            triangle_T_ptr,
            points_ptr, distances_ptr, closest_points_ptr,
            closest_faces_ptr, closest_bcs_ptr,
            bvh_tree, num_points);
    }
    else if (queue_size == 512) {
        findNearestNeighbor<double, 512> << <gridSize, NUM_THREADS, 0, stream >> > (
            triangle_T_ptr,
            points_ptr, distances_ptr, closest_points_ptr,
            closest_faces_ptr, closest_bcs_ptr,
            bvh_tree, num_points);
    }
    else if (queue_size == 1024) {
        findNearestNeighbor<double, 1024> << <gridSize, NUM_THREADS, 0, stream >> > (
            triangle_T_ptr,
            points_ptr, distances_ptr, closest_points_ptr,
            closest_faces_ptr, closest_bcs_ptr,
            bvh_tree, num_points);
    }
    cudaCheckError();

    // Export data
    double* distances_dest_ptr = distances.data().get();
    vec3<double>* closest_points_dest_ptr = (vec3<double>*)closest_points.data().get();
    vec3<double>* closest_bcs_dest_ptr = (vec3<double>*)closest_bcs.data().get();
    long long* closest_faces_dest_ptr = closest_faces.data().get();

    cudaMemcpy(distances_dest_ptr, distances_ptr,
        num_points * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(closest_points_dest_ptr, closest_points_ptr,
        num_points * sizeof(vec3<double>),
        cudaMemcpyDeviceToDevice);
    cudaMemcpy(closest_bcs_dest_ptr, closest_bcs_ptr,
        num_points * sizeof(vec3<double>), cudaMemcpyDeviceToDevice);
    cudaMemcpy(closest_faces_dest_ptr, closest_faces_ptr,
        num_points * sizeof(long long), cudaMemcpyDeviceToDevice);
    // test<<<gridSize, NUM_THREADS>>>(points_ptr, num_points, triangles_ptr, num_triangles, distances_dest_ptr);
    // cudaCheckError();

    cudaFree(distances_ptr);
    cudaFree(closest_points_ptr);
    cudaFree(closest_faces_ptr);
    cudaFree(closest_bcs_ptr);
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<long long>, std::vector<double>> bvh_distance_queries(
    const std::vector<double>& triangles, const std::vector<double>& points,
    int queue_size, const long long num_points_batch, const int id_gpu
    )
{
    if (id_gpu != -1)
        cudaSetDevice(id_gpu);

    const long long num_points = points.size() / 3;
    const long long num_triangles = triangles.size() / 9;

	//std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    int num_batch = std::ceil((double)num_points / num_points_batch);
    num_batch = std::max(num_batch, 1);

    thrust::host_vector<Triangle<double>> _triangles(num_triangles);

    auto triangle_ptr = _triangles.data();
    for (int i = 0; i < num_triangles; ++i)
    {
        triangle_ptr[i].v0 = make_double3(triangles[i * 9 + 0], triangles[i * 9 + 1], triangles[i * 9 + 2]);
        triangle_ptr[i].v1 = make_double3(triangles[i * 9 + 3], triangles[i * 9 + 4], triangles[i * 9 + 5]);
        triangle_ptr[i].v2 = make_double3(triangles[i * 9 + 6], triangles[i * 9 + 7], triangles[i * 9 + 8]);
    }

    std::vector<BVHNode<double>> bvh_nodes = buildBVH_sequential(_triangles);
    thrust::device_vector<BVHNode<double>> bvh_nodes_d(bvh_nodes);
    thrust::device_vector<Triangle<double>> _d_triangles = _triangles;

    std::vector<double> h_distances(num_points);
    std::vector<double> h_closest_points(num_points*3);
    std::vector<long long> h_closest_faces(num_points);
    std::vector<double> h_closest_bcs(num_points*3);

    for (int i_batch = 0; i_batch < num_batch; ++i_batch)
    {
        const int num_current_batch = std::min(num_points_batch, (int)num_points - i_batch * num_points_batch);
        const int id_start = i_batch * num_points_batch;

        thrust::device_vector<double> d_distances(num_current_batch, -1);
        thrust::device_vector<double> d_closest_points(num_current_batch * 3, 0);
        thrust::device_vector<long long> d_closest_faces(num_current_batch, 0);
        thrust::device_vector<double> d_closest_bcs(num_current_batch * 3, 0);

        thrust::host_vector<vec3<double>> _points(num_current_batch);
        auto points_ptr = _points.data();
        for (int i = 0; i < num_current_batch; ++i)
            points_ptr[i] = make_double3(points[(i + id_start) * 3 + 0], points[(i + id_start) * 3 + 1], points[(i + id_start) * 3 + 2]);
        thrust::device_vector<vec3<double>> _d_points = _points;

        bvh_distance_queries_kernel(
            bvh_nodes_d.data().get(),
            _d_triangles, _d_points,
            d_distances, d_closest_points, d_closest_faces, d_closest_bcs, queue_size);

        thrust::copy(d_distances.begin(), d_distances.end(), h_distances.begin() + id_start);
        thrust::copy(d_closest_points.begin(), d_closest_points.end(), h_closest_points.begin() + id_start * 3);
        thrust::copy(d_closest_faces.begin(), d_closest_faces.end(), h_closest_faces.begin() + id_start);
        thrust::copy(d_closest_bcs.begin(), d_closest_bcs.end(), h_closest_bcs.begin() + id_start * 3);

        _d_points.clear();
        d_distances.clear();
        d_closest_points.clear();
        d_closest_faces.clear();
        d_closest_bcs.clear();
    }

    _triangles.clear();
    _d_triangles.clear();
    bvh_nodes.clear();
    bvh_nodes_d.clear();
    return { h_distances , h_closest_points , h_closest_faces ,h_closest_bcs };
}

MyBVH::MyBVH(const std::vector<std::vector<double>>& vertices,
    const std::vector<std::vector<int>>& faces)
{
    std::vector<double> cuda_triangles(faces.size() * 9);
    for (int i_face = 0; i_face < faces.size(); ++i_face)
    {
        cuda_triangles[i_face * 9 + 0] = vertices[faces[i_face][0]][0];
        cuda_triangles[i_face * 9 + 1] = vertices[faces[i_face][0]][1];
        cuda_triangles[i_face * 9 + 2] = vertices[faces[i_face][0]][2];
        cuda_triangles[i_face * 9 + 3] = vertices[faces[i_face][1]][0];
        cuda_triangles[i_face * 9 + 4] = vertices[faces[i_face][1]][1];
        cuda_triangles[i_face * 9 + 5] = vertices[faces[i_face][1]][2];
        cuda_triangles[i_face * 9 + 6] = vertices[faces[i_face][2]][0];
        cuda_triangles[i_face * 9 + 7] = vertices[faces[i_face][2]][1];
        cuda_triangles[i_face * 9 + 8] = vertices[faces[i_face][2]][2];
    }

    const long long num_triangles = cuda_triangles.size() / 9;

    thrust::host_vector<Triangle<double>> _triangles(num_triangles);
    auto triangle_ptr = _triangles.data();
    for (int i = 0; i < num_triangles; ++i)
    {
        triangle_ptr[i].v0 = make_double3(cuda_triangles[i * 9 + 0], cuda_triangles[i * 9 + 1], cuda_triangles[i * 9 + 2]);
        triangle_ptr[i].v1 = make_double3(cuda_triangles[i * 9 + 3], cuda_triangles[i * 9 + 4], cuda_triangles[i * 9 + 5]);
        triangle_ptr[i].v2 = make_double3(cuda_triangles[i * 9 + 6], cuda_triangles[i * 9 + 7], cuda_triangles[i * 9 + 8]);
    }

    _d_bvh_nodes = new thrust::device_vector<BVHNode<double>>(buildBVH_sequential(_triangles));
    _d_triangles = new thrust::device_vector<Triangle<double>>(_triangles);

    _triangles.clear();
    return;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<long long>, std::vector<double>> MyBVH::query(
    const std::vector<double>& points, int queue_size, const long long num_points_batch, const int id_gpu
)
{
    const long long num_points = points.size() / 3;
    int num_batch = std::ceil((double)num_points / num_points_batch);
    num_batch = std::max(num_batch, 1);

    std::vector<double> h_distances(num_points);
    std::vector<double> h_closest_points(num_points * 3);
    std::vector<long long> h_closest_faces(num_points);
    std::vector<double> h_closest_bcs(num_points * 3);
    for (int i_batch = 0; i_batch < num_batch; ++i_batch)
    {
        const int num_current_batch = std::min(num_points_batch, (int)num_points - i_batch * num_points_batch);
        const int id_start = i_batch * num_points_batch;

        thrust::device_vector<double> d_distances(num_current_batch, -1);
        thrust::device_vector<double> d_closest_points(num_current_batch * 3, 0);
        thrust::device_vector<long long> d_closest_faces(num_current_batch, 0);
        thrust::device_vector<double> d_closest_bcs(num_current_batch * 3, 0);

        thrust::host_vector<vec3<double>> _points(num_current_batch);
        auto points_ptr = _points.data();
        for (int i = 0; i < num_current_batch; ++i)
            points_ptr[i] = make_double3(points[(i + id_start) * 3 + 0], points[(i + id_start) * 3 + 1], points[(i + id_start) * 3 + 2]);
        thrust::device_vector<vec3<double>> _d_points = _points;

        bvh_distance_queries_kernel(
            _d_bvh_nodes->data().get(),
            *_d_triangles, _d_points,
            d_distances, d_closest_points, d_closest_faces, d_closest_bcs, queue_size);

        thrust::copy(d_distances.begin(), d_distances.end(), h_distances.begin() + id_start);
        thrust::copy(d_closest_points.begin(), d_closest_points.end(), h_closest_points.begin() + id_start * 3);
        thrust::copy(d_closest_faces.begin(), d_closest_faces.end(), h_closest_faces.begin() + id_start);
        thrust::copy(d_closest_bcs.begin(), d_closest_bcs.end(), h_closest_bcs.begin() + id_start * 3);

        _d_points.clear();
        d_distances.clear();
        d_closest_points.clear();
        d_closest_faces.clear();
        d_closest_bcs.clear();
    }
    return { h_distances , h_closest_points , h_closest_faces ,h_closest_bcs };
}