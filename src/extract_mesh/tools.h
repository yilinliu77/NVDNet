#pragma once

#include "common_util.h"
#include "cgal_tools.h"

void add_neighbours(const int x, const int y, const int z, std::queue<std::tuple<int, int, int>>& v_queues);


void add_neighbours(const int x, const int y, const int z,
                    const Eigen::Tensor<double, 4>& v_surface_points,
                    std::queue<Eigen::Vector3i>& v_queues,
                    const double v_epsilon
);

bool check_range(const int x, const int y, const int z, const int resolution);

Eigen::Vector3d get_vector(const Eigen::Tensor<double, 4>& v_tensor, const long long x, const long long y,
                           const long long z);

void export_points(const std::string& v_path, const Eigen::Tensor<bool, 3>& consistent_flags, const int resolution);

void export_points(const std::string& v_path, const Eigen::Tensor<bool, 3>& consistent_flags, const int resolution,
                   const Eigen::Tensor<double, 4>& features, const double threshold);

inline Eigen::Vector3i to_voxel_coor(const Eigen::Vector3d& v_coor, const double res)
{
    return Eigen::round((v_coor.array() + 1) / 2 * (res - 1)).cast<int>();
}

inline bool within_bounds(const int x, const int y, const int z, const Eigen::VectorXi& v_bounds)
{
    return x >= v_bounds[0] && x <= v_bounds[3] && y >= v_bounds[1] && y <= v_bounds[4] && z >= v_bounds[2] && z <= v_bounds[5];
}
