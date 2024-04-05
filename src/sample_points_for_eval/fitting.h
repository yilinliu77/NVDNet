#pragma once

#include <common_util.h>

#include "shape2d.h"
#include "shape3d.h"

std::pair<Plane_3, double> fit_plane(const Cluster& v_cluster);

std::pair<std::shared_ptr<Shape>, double> fit_vertex(
	const std::vector<Eigen::Vector3d>& v_points,
	const Cluster& v_cluster,
	const double v_epsilon
);

std::pair<std::shared_ptr<Shape>, double> fit_curve(
	const std::vector<Eigen::Vector3d>& v_points,
	const Cluster& v_cluster,
	const gp_Pln& v_plane,
	const std::string& v_type
);

// std::pair<Shape3D*, double> fit_surface(
// 	const std::vector<gte::Vector3<double>>& gte_data,
// 	const Cluster& v_cluster,
// 	const std::string& v_type,
// 	const double epsilon
// );

std::pair<std::shared_ptr<Shape>, double> fit_surface(
	const std::vector<gte::Vector3<double>>& gte_data,
	const Cluster& v_cluster,
	const std::string& v_type
);

std::pair<std::shared_ptr<Shape>, double> fit_surface(
	const std::vector<Eigen::Vector3d>& v_points,
	const Cluster& v_cluster,
	const std::string& v_type
);

std::shared_ptr<Shape> fall_back_ransac(const Cluster& v_cluster, const double epsilon, const std::string& v_type="", const double radius=-1.);

std::vector<std::shared_ptr<Shape>> fitting(
	const std::vector<Cluster>& v_input, 
	const double epsilon,
	const int num_fitting_points
);


std::shared_ptr<Shape> check_valid_ellipse(std::shared_ptr<Shape>& v_shape, const std::vector<Eigen::Vector3d>& inliers,
	const double epsilon=0.001);

void prepare_gte_data(
	const std::vector<Eigen::Vector3d>& v_points,
	std::vector<gte::Vector3<double>>& data1,
	const int max_samples = 10000
);