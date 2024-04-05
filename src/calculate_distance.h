#pragma once
#include "common_util.h"
#include "cgal_tools.h"
#include "bvh.h"


Eigen::Vector2d normal_vector_to_angle(const Eigen::Vector3d& v_dir)
{
	double phi = std::atan2(v_dir[1], v_dir[0]);
	double theta = std::acos(v_dir[2]);

	phi = phi - M_PI * 2 * std::floor(phi / (M_PI * 2));
	theta = theta - M_PI * 2 * std::floor(theta / (M_PI * 2));
	return { phi, theta };
}

Eigen::Vector3d normal_angle_to_vector(const Eigen::Vector2d& v_angle)
{
	return {
		std::sin(v_angle[1]) * std::cos(v_angle[0]),
		std::sin(v_angle[1]) * std::sin(v_angle[0]),
		std::cos(v_angle[1])
	};
}

std::tuple<double, size_t, Eigen::Vector3d, Point_3> calculate_dis_cuda(
	const int i_point,
	const std::vector<double>& bvh_distances, const std::vector<double>& bvh_closest_points,
	const std::vector<double>& bvh_closest_bcs, const std::vector<long long>& bvh_closest_faces)
{
	Eigen::Vector3d uvs(bvh_closest_bcs[i_point * 3 + 0], bvh_closest_bcs[i_point * 3 + 1],
		bvh_closest_bcs[i_point * 3 + 2]);
	Point_3 closest_point(bvh_closest_points[i_point * 3 + 0], bvh_closest_points[i_point * 3 + 1],
		bvh_closest_points[i_point * 3 + 2]);
	return { bvh_distances[i_point], (size_t)bvh_closest_faces[i_point], uvs, closest_point };
}

void calculate_distance(
	const std::vector<Point_3>& query_points,
	const std::vector<Point_3>& vertices,
	const std::vector<std::array<int, 3>>& faces,
	const std::vector<Triangle_3>& v_triangles,
	const std::vector<long long>& surface_id_to_primitives,
	const std::vector<std::vector<long long>>& face_edge_indicator,
	const long long num_curves, const bool is_log,
	Eigen::Tensor<unsigned short, 2, Eigen::RowMajor>& feature,
	std::vector<int>& closest_primitives
)
{
	auto timer = recordTime();

	// Input
	std::vector<double> cuda_triangles(v_triangles.size() * 9);
	for (int i_face = 0; i_face < v_triangles.size(); ++i_face)
	{
		cuda_triangles[i_face * 9 + 0] = v_triangles[i_face][0][0];
		cuda_triangles[i_face * 9 + 1] = (v_triangles[i_face][0][1]);
		cuda_triangles[i_face * 9 + 2] = (v_triangles[i_face][0][2]);
		cuda_triangles[i_face * 9 + 3] = (v_triangles[i_face][1][0]);
		cuda_triangles[i_face * 9 + 4] = (v_triangles[i_face][1][1]);
		cuda_triangles[i_face * 9 + 5] = (v_triangles[i_face][1][2]);
		cuda_triangles[i_face * 9 + 6] = (v_triangles[i_face][2][0]);
		cuda_triangles[i_face * 9 + 7] = (v_triangles[i_face][2][1]);
		cuda_triangles[i_face * 9 + 8] = (v_triangles[i_face][2][2]);
	}
	std::vector<double> cuda_queries(query_points.size() * 3);
	for (int i_point = 0; i_point < query_points.size(); ++i_point)
	{
		cuda_queries[i_point * 3 + 0] = query_points[i_point][0];
		cuda_queries[i_point * 3 + 1] = query_points[i_point][1];
		cuda_queries[i_point * 3 + 2] = query_points[i_point][2];
	}
	profileTime(timer, "Distance: BVH Prepare", is_log);

	// Output
	std::vector<double> bvh_distances, bvh_closest_points, bvh_closest_bcs;
	std::vector<long long> bvh_closest_faces;

	std::tie(bvh_distances, bvh_closest_points, bvh_closest_faces, bvh_closest_bcs) = bvh_distance_queries(
		cuda_triangles, cuda_queries,
		512);
	profileTime(timer, "Distance: BVH Compute", is_log);

	const double epsilon = 1e-8;
	for (int i_point = 0; i_point < query_points.size(); ++i_point)
	{
		long long target_face;
		Eigen::Vector3d uvs;
		double distance;
		Point_3 closest_point;
		std::tie(distance, target_face, uvs, closest_point) = calculate_dis_cuda(
			i_point, bvh_distances, bvh_closest_points, bvh_closest_bcs, bvh_closest_faces
		);

		Eigen::Vector3d gradient_dir = Eigen::Vector3d(
			closest_point[0] - query_points[i_point][0],
			closest_point[1] - query_points[i_point][1],
			closest_point[2] - query_points[i_point][2]
		).normalized();
		feature(i_point, 0) = static_cast<unsigned short>(std::round((distance / 2.0) * 65535));

		auto phi_theta = normal_vector_to_angle(gradient_dir);
		feature(i_point, 1) = static_cast<unsigned short>(std::round(phi_theta[0] / (M_PI * 2) * 65535));
		feature(i_point, 2) = static_cast<unsigned short>(std::round(phi_theta[1] / (M_PI * 2) * 65535));

		const auto& id_vertices = faces[target_face];

		Eigen::Vector3d v1 = cgal_2_eigen_point<double>(vertices[id_vertices[1]]) - cgal_2_eigen_point<double>(vertices[id_vertices[0]]);
		Eigen::Vector3d v2 = cgal_2_eigen_point<double>(vertices[id_vertices[2]]) - cgal_2_eigen_point<double>(vertices[id_vertices[1]]);
		Eigen::Vector3d surface_normal = v1.cross(v2).normalized();
		auto normal_phi_theta = normal_vector_to_angle(surface_normal);
		feature(i_point, 3) = static_cast<unsigned short>(std::round(normal_phi_theta[0] / (M_PI * 2) * 65535));
		feature(i_point, 4) = static_cast<unsigned short>(std::round(normal_phi_theta[1] / (M_PI * 2) * 65535));

		closest_primitives[i_point] = surface_id_to_primitives[target_face];
		const auto& edge_indicator = face_edge_indicator[target_face];

		if (edge_indicator[0] < 0 && edge_indicator[1] < 0 && edge_indicator[2] < 0)
			continue;

		if (uvs[0] > epsilon && uvs[1] > epsilon && uvs[2] > epsilon)
			continue;

		bool is_corner = false;
		for (int i_v = 0; i_v < 3; i_v++)
		{
			if (uvs[i_v] > 1 - epsilon && edge_indicator[i_v] > num_curves)
			{
				closest_primitives[i_point] = edge_indicator[i_v];
				is_corner = true;
				break;
			}
		}

		if (!is_corner)
		{
			if (uvs[0] > 1 - epsilon && edge_indicator[0] >= 0)
				closest_primitives[i_point] = edge_indicator[0];
			else if (uvs[1] > 1 - epsilon && edge_indicator[1] >= 0)
				closest_primitives[i_point] = edge_indicator[1];
			else if (uvs[2] > 1 - epsilon && edge_indicator[2] >= 0)
				closest_primitives[i_point] = edge_indicator[2];
			else if (uvs[0] < epsilon && edge_indicator[1] >= 0 && edge_indicator[2] >= 0 && edge_indicator[1] ==
				edge_indicator[2])
				closest_primitives[i_point] = edge_indicator[1];
			else if (uvs[1] < epsilon && edge_indicator[0] >= 0 && edge_indicator[2] >= 0 && edge_indicator[0] ==
				edge_indicator[2])
				closest_primitives[i_point] = edge_indicator[0];
			else if (uvs[2] < epsilon && edge_indicator[0] >= 0 && edge_indicator[1] >= 0 && edge_indicator[0] ==
				edge_indicator[1])
				closest_primitives[i_point] = edge_indicator[0];
			else
				continue;
		}
		if (closest_primitives[i_point] == -1)
			throw;
	}
	profileTime(timer, "Distance: Extract", is_log);
	return;
}

// Only calculate the closest_primitives, do not store features
void calculate_distance(
	const std::vector<Point_3>& query_points,
	const std::vector<Triangle_3>& v_triangles,
	const std::vector<long long>& surface_id_to_primitives,
	const std::vector<std::vector<long long>>& face_edge_indicator,
	const long long num_curves, const bool is_log,
	std::vector<int>& closest_primitives
)
{
	auto timer = recordTime();

	// Input
	std::vector<double> cuda_triangles(v_triangles.size() * 9);
	for (int i_face = 0; i_face < v_triangles.size(); ++i_face)
	{
		cuda_triangles[i_face * 9 + 0] = v_triangles[i_face][0][0];
		cuda_triangles[i_face * 9 + 1] = v_triangles[i_face][0][1];
		cuda_triangles[i_face * 9 + 2] = v_triangles[i_face][0][2];
		cuda_triangles[i_face * 9 + 3] = v_triangles[i_face][1][0];
		cuda_triangles[i_face * 9 + 4] = v_triangles[i_face][1][1];
		cuda_triangles[i_face * 9 + 5] = v_triangles[i_face][1][2];
		cuda_triangles[i_face * 9 + 6] = v_triangles[i_face][2][0];
		cuda_triangles[i_face * 9 + 7] = v_triangles[i_face][2][1];
		cuda_triangles[i_face * 9 + 8] = v_triangles[i_face][2][2];
	}
	std::vector<double> cuda_queries(query_points.size() * 3);
	for (int i_point = 0; i_point < query_points.size(); ++i_point)
	{
		cuda_queries[i_point * 3 + 0] = query_points[i_point][0];
		cuda_queries[i_point * 3 + 1] = query_points[i_point][1];
		cuda_queries[i_point * 3 + 2] = query_points[i_point][2];
	}
	profileTime(timer, "Distance: BVH Prepare", is_log);

	auto [bvh_distances, bvh_closest_points, bvh_closest_faces, bvh_closest_bcs] = bvh_distance_queries(
		cuda_triangles, cuda_queries,
		512);
	profileTime(timer, "Distance: BVH Compute", is_log);

	const double epsilon = 1e-8;
	for (int i_point = 0; i_point < query_points.size(); ++i_point)
	{
		const long long target_face = bvh_closest_faces[i_point];
		const Eigen::Vector3d uvs(
			bvh_closest_bcs[i_point * 3 + 0], 
			bvh_closest_bcs[i_point * 3 + 1],
			bvh_closest_bcs[i_point * 3 + 2]);
		
		closest_primitives[i_point] = surface_id_to_primitives[target_face];
		const auto& edge_indicator = face_edge_indicator[target_face];

		if (edge_indicator[0] < 0 && edge_indicator[1] < 0 && edge_indicator[2] < 0)
			continue;

		if (uvs[0] > epsilon && uvs[1] > epsilon && uvs[2] > epsilon)
			continue;

		bool is_corner = false;
		for (int i_v = 0; i_v < 3; i_v++)
		{
			if (uvs[i_v] > 1 - epsilon && edge_indicator[i_v] > num_curves)
			{
				closest_primitives[i_point] = edge_indicator[i_v];
				is_corner = true;
				break;
			}
		}

		if (!is_corner)
		{
			if (uvs[0] > 1 - epsilon && edge_indicator[0] >= 0)
				closest_primitives[i_point] = edge_indicator[0];
			else if (uvs[1] > 1 - epsilon && edge_indicator[1] >= 0)
				closest_primitives[i_point] = edge_indicator[1];
			else if (uvs[2] > 1 - epsilon && edge_indicator[2] >= 0)
				closest_primitives[i_point] = edge_indicator[2];
			else if (uvs[0] < epsilon && edge_indicator[1] >= 0 && edge_indicator[2] >= 0 && edge_indicator[1] ==
				edge_indicator[2])
				closest_primitives[i_point] = edge_indicator[1];
			else if (uvs[1] < epsilon && edge_indicator[0] >= 0 && edge_indicator[2] >= 0 && edge_indicator[0] ==
				edge_indicator[2])
				closest_primitives[i_point] = edge_indicator[0];
			else if (uvs[2] < epsilon && edge_indicator[0] >= 0 && edge_indicator[1] >= 0 && edge_indicator[0] ==
				edge_indicator[1])
				closest_primitives[i_point] = edge_indicator[0];
			else
				continue;
		}
		assert(closest_primitives[i_point] != -1);
	}
	profileTime(timer, "Distance: Extract", is_log);
	return;
}

// Only store features
void calculate_distance(
	const std::vector<Point_3>& query_points,
	const std::vector<Point_3>& vertices,
	const std::vector<std::vector<int>>& faces,
	const bool is_log,
	Eigen::Tensor<unsigned short, 2, Eigen::RowMajor>& feature
)
{
	auto timer = recordTime();

	// Input
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
	std::vector<double> cuda_queries(query_points.size() * 3);
	for (int i_point = 0; i_point < query_points.size(); ++i_point)
	{
		cuda_queries[i_point * 3 + 0] = query_points[i_point][0];
		cuda_queries[i_point * 3 + 1] = query_points[i_point][1];
		cuda_queries[i_point * 3 + 2] = query_points[i_point][2];
	}
	profileTime(timer, "Distance: BVH Prepare", is_log);

	// Output
	std::vector<double> bvh_distances, bvh_closest_points, bvh_closest_bcs;
	std::vector<long long> bvh_closest_faces;

	std::tie(bvh_distances, bvh_closest_points, bvh_closest_faces, bvh_closest_bcs) = bvh_distance_queries(
		cuda_triangles, cuda_queries,
		512);
	profileTime(timer, "Distance: BVH Compute", is_log);

	for (int i_point = 0; i_point < query_points.size(); ++i_point)
	{
		long long target_face = bvh_closest_faces[i_point];
		double distance = bvh_distances[i_point];
		Point_3 closest_point(
			bvh_closest_points[i_point * 3 + 0],
			bvh_closest_points[i_point * 3 + 1],
			bvh_closest_points[i_point * 3 + 2]
		);

		Eigen::Vector3d gradient_dir = Eigen::Vector3d(
			closest_point[0] - query_points[i_point][0],
			closest_point[1] - query_points[i_point][1],
			closest_point[2] - query_points[i_point][2]
		).normalized();
		feature(i_point, 0) = static_cast<unsigned short>(std::round((distance / 2.0) * 65535));

		auto phi_theta = normal_vector_to_angle(gradient_dir);
		feature(i_point, 1) = static_cast<unsigned short>(std::round(phi_theta[0] / (M_PI * 2) * 65535));
		feature(i_point, 2) = static_cast<unsigned short>(std::round(phi_theta[1] / (M_PI * 2) * 65535));

		const auto& id_vertices = faces[target_face];

		Eigen::Vector3d v1 = cgal_2_eigen_point<double>(vertices[id_vertices[1]]) - cgal_2_eigen_point<double>(vertices[id_vertices[0]]);
		Eigen::Vector3d v2 = cgal_2_eigen_point<double>(vertices[id_vertices[2]]) - cgal_2_eigen_point<double>(vertices[id_vertices[1]]);
		Eigen::Vector3d surface_normal = v1.cross(v2).normalized();
		auto normal_phi_theta = normal_vector_to_angle(surface_normal);
		feature(i_point, 3) = static_cast<unsigned short>(std::round(normal_phi_theta[0] / (M_PI * 2) * 65535));
		feature(i_point, 4) = static_cast<unsigned short>(std::round(normal_phi_theta[1] / (M_PI * 2) * 65535));
	}
	profileTime(timer, "Distance: Extract", is_log);
	return;
}