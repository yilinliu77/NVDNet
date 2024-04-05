#pragma once
#include <cuda_runtime_api.h>

#include "common_util.h"
#include <yaml-cpp/yaml.h>
#include <argparse/argparse.hpp>

#include "tools.h"

#include <pcl/impl/point_types.hpp>

#include <tbb/tbb.h>

#include "bvh.h"
#include "model_tools.h"
#include "kd_tree_helper.h"

#include <mutex>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/polygon_mesh_to_polygon_soup.h>

#include "writer.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct Curve
{
	std::string type;

	Eigen::Vector3d location;
	double radius;

	std::vector<double> knots;
	int degree;
	std::vector<Eigen::Vector3d> poles;

	Eigen::Vector3d direction;

	Eigen::Vector3d focus1;
	Eigen::Vector3d focus2;
	double maj_radius;
	double min_radius;
	std::unordered_set<int> vert_indices;
};

struct Surface
{
	std::string type;

	std::vector<int> face_indices;
};

Eigen::Vector3d convert_vector(const YAML::Node& v_node)
{
	return Eigen::Vector3d(v_node[0].as<double>(), v_node[1].as<double>(), v_node[2].as<double>());
}

std::vector<double> convert_knot(const YAML::Node& v_node)
{
	std::vector<double> knots(v_node.size());
	for (int i = 0; i < v_node.size(); ++i)
		knots[i] = v_node[i].as<double>();
	return knots;
}

std::vector<Eigen::Vector3d> convert_poles(const YAML::Node& v_node)
{
	std::vector<Eigen::Vector3d> poles(v_node.size());
	for (int i = 0; i < v_node.size(); ++i)
	{
		poles[i] = Eigen::Vector3d(v_node[i][0].as<double>(), v_node[i][1].as<double>(), v_node[i][2].as<double>());
	}
	return poles;
}

std::pair<std::vector<Curve>, std::vector<Surface>> filter_primitives(
	const YAML::Node& v_yaml,
	const std::vector<std::vector<int>>& v_faces,
	const int i_component,
	const std::vector<int>& id_vertex_component_maps,
	const std::vector<int>& id_face_component_maps
)
{
	std::vector<Curve> curves;
	std::vector<Surface> surfaces;
	for (int i_curve = 0; i_curve < v_yaml["curves"].size(); ++i_curve)
	{
		const int example_vert_id = v_yaml["curves"][i_curve]["vert_indices"][0].as<int>();
		if (id_vertex_component_maps.at(example_vert_id) != i_component)
			continue;

		if (v_yaml["curves"][i_curve]["type"].as<std::string>() == "Circle")
		{
			bool is_new_primitive = true;
			auto type = v_yaml["curves"][i_curve]["type"].as<std::string>();
			auto location = convert_vector(v_yaml["curves"][i_curve]["location"]);
			auto radius = v_yaml["curves"][i_curve]["radius"].as<double>();
			for (int i_exist = 0; i_exist < curves.size(); ++i_exist)
			{
				if (curves[i_exist].type == type &&
					curves[i_exist].location == location &&
					std::abs(curves[i_exist].radius - radius) < 1e-8)
				{
					for (const auto& item : v_yaml["curves"][i_curve]["vert_indices"])
						curves[i_exist].vert_indices.insert(item.as<int>());
					is_new_primitive = false;
					break;
				}
			}
			if (is_new_primitive)
			{
				Curve curve;
				curve.type = type;
				curve.location = location;
				curve.radius = radius;
				for (const auto& item : v_yaml["curves"][i_curve]["vert_indices"])
					curve.vert_indices.insert(item.as<int>());
				curves.emplace_back(curve);
			}
		}
		else if (v_yaml["curves"][i_curve]["type"].as<std::string>() == "BSpline")
		{
			LOG(ERROR) << "Found BSpline";
			throw "";
			bool is_new_primitive = true;
			auto type = v_yaml["curves"][i_curve]["type"].as<std::string>();
			auto knots = convert_knot(v_yaml["curves"][i_curve]["knots"]);
			auto poles = convert_poles(v_yaml["curves"][i_curve]["poles"]);
			auto degree = v_yaml["curves"][i_curve]["degree"].as<int>();
			for (int i_exist = 0; i_exist < curves.size(); ++i_exist)
			{
				if (curves[i_exist].type == type &&
					curves[i_exist].knots == knots &&
					curves[i_exist].degree == degree &&
					curves[i_exist].poles == poles)
				{
					for (const auto& item : v_yaml["curves"][i_curve]["vert_indices"])
						curves[i_exist].vert_indices.insert(item.as<int>());
					is_new_primitive = false;
					break;
				}
			}
			if (is_new_primitive)
			{
				Curve curve;
				curve.type = type;
				curve.knots = knots;
				curve.degree = degree;
				curve.poles = poles;
				for (const auto& item : v_yaml["curves"][i_curve]["vert_indices"])
					curve.vert_indices.insert(item.as<int>());
				curves.emplace_back(curve);
			}
		}
		else if (v_yaml["curves"][i_curve]["type"].as<std::string>() == "Line")
		{
			bool is_new_primitive = true;
			auto type = v_yaml["curves"][i_curve]["type"].as<std::string>();
			auto location = convert_vector(v_yaml["curves"][i_curve]["location"]);
			auto direction = convert_vector(v_yaml["curves"][i_curve]["direction"]);
			for (int i_exist = 0; i_exist < curves.size(); ++i_exist)
			{
				if (curves[i_exist].type == type &&
					curves[i_exist].direction == direction &&
					curves[i_exist].location == location)
				{
					for (const auto& item : v_yaml["curves"][i_curve]["vert_indices"])
						curves[i_exist].vert_indices.insert(item.as<int>());
					is_new_primitive = false;
					break;
				}
			}
			if (is_new_primitive)
			{
				Curve curve;
				curve.type = type;
				curve.direction = direction;
				curve.location = location;
				for (const auto& item : v_yaml["curves"][i_curve]["vert_indices"])
					curve.vert_indices.insert(item.as<int>());
				curves.emplace_back(curve);
			}
		}
		else if (v_yaml["curves"][i_curve]["type"].as<std::string>() == "Ellipse")
		{
			bool is_new_primitive = true;
			auto type = v_yaml["curves"][i_curve]["type"].as<std::string>();
			auto focus1 = convert_vector(v_yaml["curves"][i_curve]["focus1"]);
			auto focus2 = convert_vector(v_yaml["curves"][i_curve]["focus2"]);
			auto maj_radius = (v_yaml["curves"][i_curve]["maj_radius"]).as<double>();
			auto min_radius = (v_yaml["curves"][i_curve]["min_radius"]).as<double>();
			for (int i_exist = 0; i_exist < curves.size(); ++i_exist)
			{
				if (curves[i_exist].type == v_yaml["curves"][i_curve]["type"].as<std::string>() &&
					curves[i_exist].focus1 == focus1 &&
					curves[i_exist].focus2 == focus2 &&
					curves[i_exist].maj_radius == maj_radius &&
					curves[i_exist].min_radius == min_radius)
				{
					for (const auto& item : v_yaml["curves"][i_curve]["vert_indices"])
						curves[i_exist].vert_indices.insert(item.as<int>());
					is_new_primitive = false;
					break;
				}
			}
			if (is_new_primitive)
			{
				Curve curve;
				curve.type = v_yaml["curves"][i_curve]["type"].as<std::string>();
				curve.focus1 = focus1;
				curve.focus2 = focus2;
				curve.maj_radius = maj_radius;
				curve.min_radius = min_radius;
				for (const auto& item : v_yaml["curves"][i_curve]["vert_indices"])
					curve.vert_indices.insert(item.as<int>());
				curves.emplace_back(curve);
			}
		}
		else
			return {std::vector<Curve>(), surfaces};
	}

	// Change vertex index to the split mesh
	for (int i_curve = 0; i_curve < curves.size(); ++i_curve)
	{
		std::unordered_set<int> local_set;
		for (const auto& id : curves[i_curve].vert_indices)
			local_set.insert(id);
		curves[i_curve].vert_indices = local_set;
	}

	std::vector<int> corner_points;
	std::unordered_map<int, std::unordered_set<int>> id_primitive_per_vertex;
	for (int i_curve = 0; i_curve < curves.size(); ++i_curve)
	{
		for (const auto& vert_id : curves[i_curve].vert_indices)
		{
			if (std::find(corner_points.begin(), corner_points.end(), vert_id) != corner_points.end())
				continue;

			if (id_primitive_per_vertex.find(vert_id) != id_primitive_per_vertex.end())
			{
				id_primitive_per_vertex.erase(vert_id);
				corner_points.push_back(vert_id);
			}
			else
				id_primitive_per_vertex[vert_id] = std::unordered_set<int>();
		}
	}

	// Check the minimum index
	for (const auto& curve : curves)
		for (const auto idx : curve.vert_indices)
			if (idx < 0)
				return {std::vector<Curve>(), surfaces};

	for (int i_surface = 0; i_surface < v_yaml["surfaces"].size(); ++i_surface)
	{
		if (id_face_component_maps.at(v_yaml["surfaces"][i_surface]["face_indices"][0].as<int>()) != i_component)
			continue;

		Surface surface;
		for (const auto id_face_yaml : v_yaml["surfaces"][i_surface]["face_indices"])
		{
			const int id_face = id_face_yaml.as<int>();
			for (auto vert_id : v_faces[id_face])
			{
				vert_id = vert_id;
				if (id_primitive_per_vertex.find(vert_id) != id_primitive_per_vertex.end())
					id_primitive_per_vertex[vert_id].insert(i_surface);
			}
			surface.type = v_yaml["surfaces"][i_surface]["type"].as<std::string>();
			surface.face_indices.emplace_back(id_face);
		}
		surfaces.emplace_back(surface);
	}

	std::vector<Curve> filtered_curves;
	for (int i_curve = 0; i_curve < curves.size(); ++i_curve)
	{
		std::unordered_set<int> neighbour_primitives;
		for (const auto& vert_id : curves[i_curve].vert_indices)
		{
			if (id_primitive_per_vertex.find(vert_id) == id_primitive_per_vertex.end())
				continue;
			for (const auto& item : id_primitive_per_vertex[vert_id])
				neighbour_primitives.insert(item);
		}
		if (neighbour_primitives.size() > 1)
			filtered_curves.emplace_back(curves[i_curve]);
	}

	return {filtered_curves, surfaces};
}

std::tuple<
	std::vector<long long>,
	std::vector<std::vector<long long>>,
	std::vector<std::pair<int, std::vector<int>>>
> calculate_indices(const std::vector<Curve>& curves,
                    const std::vector<Surface>& surfaces,
                    const std::vector<Point_3>& vertices,
                    const std::vector<std::vector<int>>& faces
)
{
	const long long num_faces = static_cast<long long>(faces.size());
	const long long num_vertices = static_cast<long long>(vertices.size());
	const long long num_curves = static_cast<long long>(curves.size());
	const long long num_surfaces = static_cast<long long>(surfaces.size());
	long long num_primitives = num_curves + num_surfaces;

	std::vector<long long> vert_id_to_primitives(num_vertices, -1);
	std::vector<long long> surface_id_to_primitives(num_faces, 0);

	std::vector<std::pair<int, std::vector<int>>> id_corner_points;
	for (int id_curve = 0; id_curve < num_curves; ++id_curve)
	{
		const auto& curve = curves[id_curve];
		for (const auto id_vert : curve.vert_indices)
		{
			if (vert_id_to_primitives[id_vert] != -1)
			{
				auto id_corner = std::find_if(id_corner_points.begin(), id_corner_points.end(),
				                              [&id_vert](const auto& item) { return item.first == id_vert; });

				if (id_corner == id_corner_points.end())
				{
					vert_id_to_primitives[id_vert] = num_primitives + id_corner_points.size();
					id_corner_points.emplace_back(id_vert, std::vector<int>());
					id_corner = id_corner_points.end() - 1;
				}
				else
				{
					vert_id_to_primitives[id_vert] = num_primitives +
						std::distance(id_corner_points.begin(), id_corner);
				}
				id_corner->second.push_back(id_curve);
				if (vert_id_to_primitives[id_vert] < num_primitives)
					id_corner->second.push_back(vert_id_to_primitives[id_vert]);
			}
			else
			{
				vert_id_to_primitives[id_vert] = id_curve;
			}
		}
	}

	int num_corner_points = id_corner_points.size();
	num_primitives += num_corner_points;

	std::vector<std::vector<long long>> face_edge_indicator(num_faces, std::vector<long long>(3, -1));
	for (int id_surface = 0; id_surface < num_surfaces; ++id_surface)
	{
		auto surface = surfaces[id_surface];
		for (auto id_face : surface.face_indices)
		{
			for (int idx = 0; idx < faces[id_face].size(); idx++)
			{
				int primitive_id = vert_id_to_primitives[faces[id_face][idx]];
				if (primitive_id >= num_curves + num_surfaces)
				{
					face_edge_indicator[id_face][idx] = primitive_id;
				}
				else if (primitive_id > -1)
				{
					face_edge_indicator[id_face][idx] = primitive_id;
				}
			}

			surface_id_to_primitives[id_face] = id_surface + num_curves;
		}
	}

	return {surface_id_to_primitives, face_edge_indicator, id_corner_points};
}

typedef CGAL::AABB_triangle_primitive<K, std::vector<Triangle_3>::iterator, CGAL::Tag_true> My_Primitive;
typedef CGAL::AABB_traits<K, My_Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> My_tree;

Eigen::Vector2d normal_vector_to_angle(const Eigen::Vector3d& v_dir)
{
	double phi = std::atan2(v_dir[1], v_dir[0]);
	double theta = std::acos(v_dir[2]);

	phi = phi - M_PI * 2 * std::floor(phi / (M_PI * 2));
	theta = theta - M_PI * 2 * std::floor(theta / (M_PI * 2));
	return {phi, theta};
}

Eigen::Vector3d normal_angle_to_vector(const Eigen::Vector2d& v_angle)
{
	return {
		std::sin(v_angle[1]) * std::cos(v_angle[0]),
		std::sin(v_angle[1]) * std::sin(v_angle[0]),
		std::cos(v_angle[1])
	};
}

std::tuple<double, size_t, Eigen::Vector3d, Point_3> calculate_dis_cgal(
	My_tree& tree, std::vector<Triangle_3>& triangles, const Point_3& v_point)
{
	auto cgal_result = tree.closest_point_and_primitive(v_point);
	const size_t cgal_target_face = std::distance(triangles.begin(), cgal_result.second);
	const double area = std::sqrt(CGAL::squared_area(
		triangles[cgal_target_face][0],
		triangles[cgal_target_face][1],
		triangles[cgal_target_face][2]));
	const double area1 = std::sqrt(CGAL::squared_area(
		cgal_result.first,
		triangles[cgal_target_face][1],
		triangles[cgal_target_face][2]));
	const double area2 = std::sqrt(CGAL::squared_area(
		triangles[cgal_target_face][0],
		cgal_result.first,
		triangles[cgal_target_face][2]));
	const double area3 = std::sqrt(CGAL::squared_area(
		triangles[cgal_target_face][0],
		triangles[cgal_target_face][1],
		cgal_result.first));
	Eigen::Vector3d cgal_uvs(area1 / area, area2 / area, area3 / area);

	double cgal_distance = std::sqrt((cgal_result.first - v_point).squared_length());

	return {cgal_distance, cgal_target_face, cgal_uvs, cgal_result.first};
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
	return {bvh_distances[i_point], (size_t)bvh_closest_faces[i_point], uvs, closest_point};
}

void calculate_distance(
	const std::vector<Point_3>& query_points,
	const std::vector<Point_3>& vertices,
	const std::vector<std::vector<int>>& faces,
	const std::vector<Triangle_3>& v_triangles,
	const std::unordered_map<int, int> to_original_face_id,
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

	tbb::parallel_for(tbb::blocked_range<size_t>(0, query_points.size()),
		[&](const tbb::blocked_range<size_t>& r)
		{
			for (int i_point = r.begin(); i_point < r.end(); ++i_point)
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

				const auto& id_vertices = faces[to_original_face_id.at(target_face)];

				Eigen::Vector3d v1 = cgal_2_eigen_point<double>(vertices[id_vertices[1]]) - cgal_2_eigen_point<double>(vertices[id_vertices[0]]);
				Eigen::Vector3d v2 = cgal_2_eigen_point<double>(vertices[id_vertices[2]]) - cgal_2_eigen_point<double>(vertices[id_vertices[1]]);
				Eigen::Vector3d surface_normal = v1.cross(v2).normalized();
				auto normal_phi_theta = normal_vector_to_angle(surface_normal);
				feature(i_point, 3) = static_cast<unsigned short>(std::round(normal_phi_theta[0] / (M_PI * 2) * 65535));
				feature(i_point, 4) = static_cast<unsigned short>(std::round(normal_phi_theta[1] / (M_PI * 2) * 65535));

				closest_primitives[i_point] = surface_id_to_primitives[to_original_face_id.at(target_face)];
				const auto& edge_indicator = face_edge_indicator[to_original_face_id.at(target_face)];

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
		});
	profileTime(timer, "Distance: Extract", is_log);
	return;
}
#pragma optimize ("", off)
#pragma optimize ("", on)


std::vector<std::vector<int>> generateAdjacencyList(const std::vector<std::vector<int>>& B, int numVertices) {
	std::vector<std::vector<int>> adjacencyList(numVertices);

	for (const auto& face : B) {
		for (size_t i = 0; i < face.size(); ++i) {
			for (size_t j = i + 1; j < face.size(); ++j) {
				adjacencyList[face[i]].push_back(face[j]);
				adjacencyList[face[j]].push_back(face[i]);
			}
		}
	}

	return adjacencyList;
}

void DFS(
	int v, 
	const std::vector<std::vector<int>>& adjacencyList, 
	std::vector<bool>& visited, 
	int componentId, 
	std::vector<int>& vertexToComponent)
{
	std::vector<int> queue;
	queue.push_back(v);
	while(!queue.empty())
	{
		int v = queue.back();
		queue.pop_back();
		visited[v] = true;
		vertexToComponent[v] = componentId;
		for (const int& neighbor : adjacencyList[v]) {
			if (!visited[neighbor]) {
				queue.push_back(neighbor);
			}
		}
	}
}



class Executer
{
private:
    std::vector<fs::path> tasks;
	bool is_log;
	bool is_udf_feature;
	bool is_poisson;
	bool is_point_feature;

	int resolution;
	fs::path output_root;

    const std::vector<Eigen::Vector3i>* m_source_coords;
	const std::vector<std::vector<int>>* m_target_coords;
	const std::vector<std::vector<bool>>* m_valid_flag;

	std::mutex* gpu_mutex;
	int max_task_per_gpu;
	int num_gpus;
	std::vector<double>* m_time_statics;
	std::vector<Writter*>* m_writers;

	std::atomic<size_t>* counter;

public:
	Executer(
		const std::vector<fs::path>& v_tasks,
		const std::vector<Eigen::Vector3i>* v_source_coords,
		const std::vector<std::vector<int>>* v_target_coords,
		const std::vector<std::vector<bool>>* v_valid_flag,
		const bool is_log, const int resolution,
		const fs::path& output_root,
		bool is_udf_feature,
		bool is_poisson,
		bool is_point_feature,
		std::mutex* gpu_mutex,
		int max_task_per_gpu,
		int num_gpus,
		std::vector<double>& time_statics,
		std::vector<Writter*>& writers,
		std::atomic<size_t>& counter

	): tasks(v_tasks), m_source_coords(v_source_coords), m_target_coords(v_target_coords), m_valid_flag(v_valid_flag), is_log(is_log),
		resolution(resolution), output_root(output_root), is_udf_feature(is_udf_feature), is_poisson(is_poisson), is_point_feature(is_point_feature),
		m_time_statics(&time_statics), m_writers(&writers), gpu_mutex(gpu_mutex), max_task_per_gpu(max_task_per_gpu), counter(&counter), num_gpus(num_gpus)

	{

	}
	void operator()( const tbb::blocked_range<size_t>& r ) const
	{
		const std::vector<Eigen::Vector3i>& source_coords = *m_source_coords;
		const std::vector<std::vector<int>>& target_coords = *m_target_coords;
		const std::vector<std::vector<bool>>& valid_flag = *m_valid_flag;

		std::vector<Point_3> query_points(source_coords.size());
		for (int i = 0; i < source_coords.size(); ++i)
		{
			query_points[i] = Point_3(
				((double)(source_coords)[i][0] / (resolution - 1) * 2 - 1),
				((double)(source_coords)[i][1] / (resolution - 1) * 2 - 1),
				((double)(source_coords)[i][2] / (resolution - 1) * 2 - 1)
			);
		}

		std::vector<double>& time_statics = *m_time_statics;
		std::vector<Writter*>& writers = *m_writers;
		for(size_t i_task=r.begin(); i_task !=r.end(); ++i_task)
		{
			const int frac_length = std::max((int)tasks.size() / 100, 1);
			const int num_finished = counter->fetch_add(1);
			if (num_finished % frac_length == 0 && num_finished !=0 )
			{
				LOG(INFO) << "Processing " << num_finished << " / " << tasks.size() << " tasks...";
				for (int i = 0; i < writers.size(); ++i)
					std::cout << i << ": " << writers[i]->get_size();
				std::cout << "\n";

				for (int i = 0; i < time_statics.size(); ++i)
					std::cout  << time_statics[i] << "; ";
				std::cout << "\n";
			}

			std::string prefix = tasks[i_task].filename().string();
			LOG_IF(INFO, is_log) << "=================== " << prefix << " ===================";
			auto timer = recordTime();
			// 1. Read input file and normalize coordinates and split the mesh
			std::vector<int> id_vertex_component_maps; // which component does this vertex belong to
			std::vector<int> id_face_component_maps; // which component does this face belong to

			std::vector<std::vector<Triangle_3>> triangles;
			std::vector<std::unordered_map<int, int>> to_original_Face_id;
			std::vector<Point_3> total_vertices;
			std::vector<std::vector<int>> total_faces;

			YAML::Node config;
			int num_mesh_components;
			// Read and preprocess the mesh and primitives
			{
				auto timer_io = recordTime();
				fs::path obj_file, feature_file;
				for (fs::directory_iterator it_file(tasks[i_task]); it_file != fs::directory_iterator(); ++it_file)
				{
					const std::string name = it_file->path().string();
					if (name.find("features") != name.npos)
						feature_file = it_file->path();
					else if (name.find("trimesh") != name.npos)
						obj_file = it_file->path();
				}

				// Read obj
				std::tie(total_vertices, total_faces) = read_obj(obj_file);
				time_statics[0] += profileTime(timer_io, "IO 1: ", is_log);

				// Read feature curves and surfaces
				config = YAML::LoadFile(feature_file.string());
				time_statics[1] += profileTime(timer_io, "IO 2: ", is_log);

				// Perform DFS on the input mesh to split it into several components
				{
					int numVertices = total_vertices.size();
					auto adjacencyList = generateAdjacencyList(total_faces, numVertices);
					std::vector<bool> visited(numVertices, false);
					id_vertex_component_maps.resize(numVertices, -1);
					id_face_component_maps.resize(total_faces.size(), -1);

					int componentId = 0;
					for (int v = 0; v < numVertices; ++v) {
						if (!visited[v]) {
							DFS(v, adjacencyList, visited, componentId, id_vertex_component_maps);
							++componentId;
						}
					}
					num_mesh_components = componentId;
					for (int i = 0; i < total_faces.size(); ++i) {
						id_face_component_maps[i] = id_vertex_component_maps[total_faces[i][0]]; // Assuming a face will always have at least one vertex
					}
				}

				// Compute id_face_component_maps and id_vertex_component_maps
				// Store triangles for each component
				{
					triangles.resize(num_mesh_components);
					to_original_Face_id.resize(num_mesh_components);
					for (int i = 0; i < total_vertices.size(); ++i)
						id_vertex_component_maps[i] = id_vertex_component_maps[i];
					for (int i = 0; i < total_faces.size(); ++i)
					{
						id_face_component_maps[i] = id_face_component_maps[i];

						to_original_Face_id[id_face_component_maps[i]][triangles[id_face_component_maps[i]].size()] = i;
						triangles[id_face_component_maps[i]].emplace_back(
							total_vertices[total_faces[i][0]],
							total_vertices[total_faces[i][1]],
							total_vertices[total_faces[i][2]]
						);
					}
				}

				// Normalize each component
				{
					std::vector<std::vector<Point_3>> aabb_calculation(num_mesh_components);
					for (int i = 0; i < total_vertices.size(); ++i)
						aabb_calculation[id_vertex_component_maps[i]].push_back(total_vertices[i]);

					for (int i_component = 0; i_component < num_mesh_components; ++i_component)
					{
						// min_x,min_y,min_z,max_x,max_y,max_z,center_x,center_y,center_z,diag
						const auto bounds = calculate_aabb(aabb_calculation[i_component]);
						// Normalize obj
						for (int i_triangle = 0; i_triangle < triangles[i_component].size(); ++i_triangle)
						{
							triangles[i_component][i_triangle] = Triangle_3(
								Point_3(
									(triangles[i_component][i_triangle][0].x() - bounds[6]) / bounds[9] * 2,
									(triangles[i_component][i_triangle][0].y() - bounds[7]) / bounds[9] * 2,
									(triangles[i_component][i_triangle][0].z() - bounds[8]) / bounds[9] * 2
								),
								Point_3(
									(triangles[i_component][i_triangle][1].x() - bounds[6]) / bounds[9] * 2,
									(triangles[i_component][i_triangle][1].y() - bounds[7]) / bounds[9] * 2,
									(triangles[i_component][i_triangle][1].z() - bounds[8]) / bounds[9] * 2
								),
								Point_3(
									(triangles[i_component][i_triangle][2].x() - bounds[6]) / bounds[9] * 2,
									(triangles[i_component][i_triangle][2].y() - bounds[7]) / bounds[9] * 2,
									(triangles[i_component][i_triangle][2].z() - bounds[8]) / bounds[9] * 2
								)
							);
						}

						if (false)
						{
							std::vector<double> poisson_set(10000 * 6);
							{
								Point_set p = sample_poisson_points(triangles[i_component], 10000);
								CGAL::IO::write_point_set(
									(output_root / "pointcloud" / (ffmt("%d_%d.ply") % prefix % i_component).str()).string(),
									p);
							}
						}
					}
				}
				time_statics[2] += profileTime(timer_io, "IO 3: ", is_log);
			}
			profileTime(timer, "IO: ", is_log);
			// continue;

			// #pragma omp parallel for
			for (int i_component = 0; i_component < num_mesh_components; ++i_component)
			{
				if (triangles[i_component].size() <= 10)
					continue;

				// 2. Filter the input curves and surfaces
				std::vector<Curve> curves;
				std::vector<Surface> surfaces;
				std::tie(curves, surfaces) = filter_primitives(
					config, total_faces,
					i_component,
					id_vertex_component_maps, id_face_component_maps
				);

				if (curves.empty() && surfaces.empty())
					continue;

				time_statics[3] += profileTime(timer, "Filter: ", is_log);

				// 3. Calculate the per face index
				std::vector<long long> surface_id_to_primitives;
				std::vector<std::vector<long long>> face_edge_indicator;
				std::vector<std::pair<int, std::vector<int>>> id_corner_points;
				std::tie(surface_id_to_primitives, face_edge_indicator, id_corner_points) = calculate_indices(
					curves, surfaces, total_vertices, total_faces);

				const long long num_curves = static_cast<long long>(curves.size());
				const long long num_surfaces = static_cast<long long>(surfaces.size());
				const long long num_corner_points = static_cast<long long>(id_corner_points.size());
				const long long num_primitives = num_curves + num_surfaces + num_corner_points;
				time_statics[4] += profileTime(timer, "Index: ", is_log);

				// 4. Calculate the closest primitive and distance for query points
				std::vector<int> closest_primitives(query_points.size());
				Eigen::Tensor<unsigned short, 2, Eigen::RowMajor> udf_feature(query_points.size(), 5);
				{
					try
					{
						bool succesful = false;
						while (!succesful)
						{
							for (int i = 0; i < max_task_per_gpu * num_gpus; ++i)
							{
								if (gpu_mutex[i].try_lock())
								{
									cudaSetDevice(i % num_gpus);
									calculate_distance(
										query_points, total_vertices, total_faces, triangles[i_component],
										to_original_Face_id[i_component],
										surface_id_to_primitives, face_edge_indicator,
										num_curves, is_log, udf_feature, closest_primitives);
									succesful = true;
									gpu_mutex[i].unlock();
									break;
								}
							}
							if (!succesful)
								override_sleep(3);
						}
					}
					catch (...)
					{
						LOG(ERROR) << prefix << " failed";
						exit(0);
					}
				}
				time_statics[5] += profileTime(timer, "Distance: ", is_log);

				// Flags
				Eigen::Tensor<unsigned int, 1, Eigen::RowMajor> flags;
				{
					std::vector<unsigned int> voronoi_edge(query_points.size(), 0);
					tbb::parallel_for(tbb::blocked_range<int>(0, query_points.size()),
						[&](const tbb::blocked_range<int>& r)
						{
							for (int i_point = r.begin(); i_point != r.end(); ++i_point)
							{
								unsigned int data = 0;
								for (int i_neighbour = 0; i_neighbour < target_coords.size(); ++i_neighbour)
								{
									if (valid_flag[i_neighbour][i_point] && closest_primitives[i_point] != closest_primitives[
										target_coords[i_neighbour][i_point]])
										data |= (1 << i_neighbour);
									else
										data &= ~(1 << i_neighbour);
									// if (valid_flag[i_neighbour][i_point] && closest_primitives[i_point] != closest_primitives[
										// target_coords[i_neighbour][i_point]])
										// data = 1;
								}
								voronoi_edge[i_point] = data;
							}
						});
					flags = Eigen::TensorMap<Eigen::Tensor<
						unsigned int, 1, Eigen::RowMajor>>(
							voronoi_edge.data(), resolution * resolution * resolution);
				}
				
				// Sparse features
				Eigen::Tensor<unsigned short, 2, Eigen::RowMajor> point_feature;
				std::vector<unsigned short> poisson_set_data;
				Point_set poisson_set_cgal(true);
				if (is_poisson)
				{
					// Sample points
					std::vector<double> poisson_set(10000 * 6);
					{
						poisson_set_cgal = sample_poisson_points(triangles[i_component], 10000);
						if (poisson_set_cgal.number_of_points() != 10000)
						{
							LOG(INFO) << "Wrong sampling points " << prefix;
							exit(-1);
						}
						for (int i = 0; i < poisson_set_cgal.size(); ++i)
						{
							poisson_set[i * 6 + 0] = poisson_set_cgal.point(i).x();
							poisson_set[i * 6 + 1] = poisson_set_cgal.point(i).y();
							poisson_set[i * 6 + 2] = poisson_set_cgal.point(i).z();
							poisson_set[i * 6 + 3] = poisson_set_cgal.normal(i).x();
							poisson_set[i * 6 + 4] = poisson_set_cgal.normal(i).y();
							poisson_set[i * 6 + 5] = poisson_set_cgal.normal(i).z();
						}
					}

					// Normalization
					poisson_set_data.resize(poisson_set.size());
					for (int i = 0; i < poisson_set_data.size(); ++i)
						poisson_set_data[i] = static_cast<unsigned short>((poisson_set[i] - (-1)) / 2 * 65535);
					
				}
				if (is_poisson && is_point_feature)
				{
					point_feature = Eigen::Tensor<unsigned short, 2, Eigen::RowMajor>(
					resolution * resolution * resolution, 5);
					my_kd_tree_t* kdtree = initialize_kd_tree(poisson_set_cgal);

					tbb::parallel_for(tbb::blocked_range<int>(0, source_coords.size()),
						[&](const tbb::blocked_range<int>& r)
						{
							for (int i_point = r.begin(); i_point != r.end(); ++i_point)
							{
								Eigen::Vector3f p = source_coords[i_point].cast<float>().array() / (resolution - 1) * 2 - 1;
								auto result = search_k_neighbour(*kdtree, p, 1);
								const double distance = std::sqrt(result.second[0]);
								const Eigen::Vector3d direction = (
									cgal_2_eigen_point<double>(poisson_set_cgal.point(result.first[0])) - p.cast<double>()).
									normalized();
								auto phi_theta = normal_vector_to_angle(direction);

								point_feature(i_point, 0) = std::round(distance / 2 * 65535);
								point_feature(i_point, 1) = std::round(phi_theta[0] / (M_PI * 2) * 65535);
								point_feature(i_point, 2) = std::round(phi_theta[1] / (M_PI * 2) * 65535);
								Eigen::Vector3d normal = cgal_vector_2_eigend(poisson_set_cgal.normal(result.first[0])).
									normalized();
								phi_theta = normal_vector_to_angle(normal);
								point_feature(i_point, 3) = std::round(phi_theta[0] / (M_PI * 2) * 65535);
								point_feature(i_point, 4) = std::round(phi_theta[1] / (M_PI * 2) * 65535);
							}
						});

					delete kdtree;
					profileTime(timer, "Point based udf: ", is_log);
				}

				time_statics[6] += profileTime(timer, "Extraction: ", is_log);

				// 6. Write
				{
					std::shared_ptr<unsigned short[]> feature_ptr, point_feature_ptr, poisson_set_ptr;
					std::shared_ptr<unsigned int[]> flags_ptr;
					if (is_udf_feature)
					{
						feature_ptr=std::shared_ptr<unsigned short[]>(new unsigned short[resolution * resolution * resolution * 5]);
						std::copy_n(udf_feature.data(), udf_feature.size(), feature_ptr.get());
					}
					if (is_poisson)
					{
						poisson_set_ptr= std::shared_ptr<unsigned short[]>(new unsigned short[10000 * 6]);
						std::copy_n(poisson_set_data.data(), poisson_set_data.size(), poisson_set_ptr.get());
					}
					if (is_point_feature)
					{
						point_feature_ptr = std::shared_ptr<unsigned short[]>(new unsigned short[resolution * resolution * resolution * 5]);
						std::copy_n(point_feature.data(), point_feature.size(), point_feature_ptr.get());
					}
					flags_ptr = std::shared_ptr<unsigned int[]>(new unsigned int[resolution * resolution * resolution]);
					std::copy_n(flags.data(), flags.size(), flags_ptr.get());
					const int int_prefix = std::atoi(prefix.c_str());
					const int id_writer = int_prefix % writers.size();
					while (writers[id_writer]->get_size() > 50)
						override_sleep(1);
					writers[id_writer]->m_mutex.lock();
					writers[id_writer]->m_queues.emplace(
						i_component, prefix, flags_ptr, feature_ptr, point_feature_ptr, poisson_set_ptr);
					writers[id_writer]->m_mutex.unlock();
				}
				

				time_statics[7] += profileTime(timer, "IO out 1: ", is_log);

				if (is_log)
				{
					fs::path local_root(output_root / (prefix + "_" + std::to_string(i_component)));
					checkFolder(local_root);

					CGAL::IO::write_point_set((local_root / "0sampled_points.ply").string(), poisson_set_cgal);

					if (is_point_feature)
					{
						Point_set surface_points(true);
						surface_points.resize(resolution* resolution* resolution);
						for (int x = 0; x < resolution; ++x)
							for (int y = 0; y < resolution; ++y)
								for (int z = 0; z < resolution; ++z)
								{
									Eigen::Vector3d p(x, y, z);
									p = p.array() / (resolution - 1) * 2 - 1;
									const int i_point = x * resolution * resolution + y * resolution + z;
									const double distance = point_feature(i_point, 0) / 65535. * 2;
									const double phi = point_feature(i_point, 1) / 65535. * 2 * M_PI;
									const double theta = point_feature(i_point, 2) / 65535. * 2 * M_PI;
									const double normal_phi = point_feature(i_point, 3) / 65535. * 2 * M_PI;
									const double normal_theta = point_feature(i_point, 4) / 65535. * 2 * M_PI;
									Eigen::Vector3d direction = normal_angle_to_vector({ phi, theta });
									Eigen::Vector3d normal = normal_angle_to_vector({ normal_phi, normal_theta });
									Eigen::Vector3d s = p + direction * distance;
									surface_points.point(x * resolution * resolution + y * resolution + z) = eigen_2_cgal_point(
										s);
									surface_points.normal(x * resolution * resolution + y * resolution + z) =
										eigen_2_cgal_vector(normal);
								}
						CGAL::IO::write_point_set((local_root / "0recover_surface_points.ply").string(), surface_points);
					}
					
					Surface_mesh total_mesh;
					for (int i_face = 0; i_face < triangles[i_component].size(); ++i_face)
					{
						total_mesh.add_face(
							std::vector{
								total_mesh.add_vertex(triangles[i_component][i_face][0]),
								total_mesh.add_vertex(triangles[i_component][i_face][1]),
								total_mesh.add_vertex(triangles[i_component][i_face][2])
							});
					}

					Point_set boundary_points;
					for (int i_point = 0; i_point < query_points.size(); ++i_point)
					{
						if (flags(i_point) != 0)
							boundary_points.insert(query_points[i_point]);
					}

					// Write boundary edges
					if (false)
					{
						throw;
						std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> lines;
						for (int i_point = 0; i_point < query_points.size(); ++i_point)
						{
							for (int i_neighbour = 0; i_neighbour < target_coords.size(); ++i_neighbour)
							{
								if (flags(i_point) & (1 << i_neighbour))
								{
									lines.emplace_back(
										cgal_2_eigen_point<double>(query_points[i_point]),
										cgal_2_eigen_point<double>(query_points[target_coords[i_neighbour][i_point]]));
								}
							}
						}
						std::ofstream ofs((local_root / "0boundary_lines.ply").string(),
										std::ios::binary | std::ios::out
						);
						ofs << "ply\n";
						ofs << "format binary_little_endian 1.0\n";
						ofs << "element vertex " << lines.size() * 2 << "\n";
						ofs << "property double x\n";
						ofs << "property double y\n";
						ofs << "property double z\n";
						ofs << "element edge " << lines.size() << "\n";
						ofs << "property int vertex1\n";
						ofs << "property int vertex2\n";
						ofs << "end_header\n";
						for (const auto& line : lines)
						{
							ofs.write(reinterpret_cast<const char*>(line.first.data()), sizeof(double) * 3);
							ofs.write(reinterpret_cast<const char*>(line.second.data()), sizeof(double) * 3);
						}
						for (int i_line = 0; i_line < lines.size(); ++i_line)
						{
							int d[2] = {i_line * 2, i_line * 2 + 1};
							ofs.write(reinterpret_cast<const char*>(&d), sizeof(int) * 2);
						}
						ofs.close();
					}

					if(is_udf_feature)
					{
						Point_set projected_points;
						for (int i_point = 0; i_point < query_points.size(); ++i_point)
						{
							double phi = udf_feature(i_point, 1) / 65535. * 2 * M_PI;
							double theta = udf_feature(i_point, 2) / 65535. * 2 * M_PI;
							double distance = udf_feature(i_point, 0) / 65535. * 2;

							double dx = std::sin(theta) * std::cos(phi);
							double dy = std::sin(theta) * std::sin(phi);
							double dz = std::cos(theta);
							projected_points.insert(query_points[i_point] + Vector_3(dx, dy, dz) * distance);
						}
						CGAL::IO::write_point_set((local_root / "0projected_point.ply").string(), projected_points);

					}
					
					std::vector<Surface_mesh> viz_meshes(num_primitives);
					// Add mesh
					for (int i_face = 0; i_face < triangles[i_component].size(); ++i_face)
					{
						const long long id_max = *std::max_element(face_edge_indicator[i_face].begin(),
																face_edge_indicator[i_face].end());
						long long target_mesh;
						if (id_max >= num_curves + num_surfaces)
							target_mesh = id_max;
						else if (id_max < 0)
							target_mesh = surface_id_to_primitives[i_face];
						else
							target_mesh = id_max;

						viz_meshes[target_mesh].add_face(
							std::vector{
								viz_meshes[target_mesh].add_vertex(triangles[i_component][i_face][0]),
								viz_meshes[target_mesh].add_vertex(triangles[i_component][i_face][1]),
								viz_meshes[target_mesh].add_vertex(triangles[i_component][i_face][2])
							}
						);
					}

					// Add query points
					for (int i_point = 0; i_point < query_points.size(); ++i_point)
					{
						viz_meshes[closest_primitives[i_point]].add_vertex(query_points[i_point]);
					}

					CGAL::IO::write_polygon_mesh(
						(local_root / "0total_mesh.ply").string(),
						total_mesh);
					for (int i_primitive = 0; i_primitive < viz_meshes.size(); ++i_primitive)
					{
						std::string primitive_type = "curve";
						if (i_primitive > curves.size() && i_primitive < curves.size() + surfaces.size())
							primitive_type = "surface";
						else if (i_primitive >= curves.size() + surfaces.size())
							primitive_type = "corner";

						CGAL::IO::write_polygon_mesh(
							(local_root / (ffmt("%s_%d.ply") % primitive_type % i_primitive).str()).string(),
							viz_meshes[i_primitive]);
						//CGAL::IO::write_point_set((fmt("point_%d.ply") % i_primitive).str(), points[i_primitive]);
						//write_ply((fmt("mesh_%d.ply") % i_primitive).str(), primitive_meshes[i_primitive]);
					}
					CGAL::IO::write_point_set((local_root / "0boundary.ply").string(), boundary_points);
				}

				time_statics[8] += profileTime(timer, "IO out 2: ", is_log);
			}
		}
	}
	~Executer(){}
};

