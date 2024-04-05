#include <cuda_runtime_api.h>
#include <tbb/tbb.h>

#include "cgal_tools.h"
#include "common_util.h"

#include <argparse/argparse.hpp>


#include "calculate_distance.h"
#include "TinyNPY.h"

int main(int argc, char* argv[])
{
	argparse::ArgumentParser program("prepare_data_3d");
	{
		LOG(INFO) << "enter the arguments: data_root";
		program.add_description("data_root");
		program.add_argument("input_model").required();
		program.add_argument("output_folder").required();
		program.parse_args(argc, argv);
	}

	fs::path data_root(program.get<std::string>("input_model"));
	fs::path output_root(program.get<std::string>("output_folder"));

	safeCheckFolder(output_root);

	// 1. Read and preprocess the mesh and primitives
	std::vector<Point_3> total_vertices;
	std::vector<std::vector<int>> total_faces;
	std::vector<std::vector<double>> _vertices;
	{
		CGAL::IO::read_polygon_soup(
			data_root.string(),
			total_vertices,
			total_faces
		);

		Eigen::Vector3d bbox_min = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
		Eigen::Vector3d bbox_max = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
		for (const auto& v : total_vertices)
		{
			bbox_min[0] = std::min(bbox_min[0], v.x());
			bbox_min[1] = std::min(bbox_min[1], v.y());
			bbox_min[2] = std::min(bbox_min[2], v.z());

			bbox_max[0] = std::max(bbox_max[0], v.x());
			bbox_max[1] = std::max(bbox_max[1], v.y());
			bbox_max[2] = std::max(bbox_max[2], v.z());
		}
		Eigen::Vector3d bbox_center = (bbox_min + bbox_max) / 2.;
		double diagonal = (bbox_max - bbox_min).norm();
		for(auto& v: total_vertices)
		{
			v = Point_3(
				(v.x() - bbox_center[0]) / diagonal * 2,
				(v.y() - bbox_center[1]) / diagonal * 2,
				(v.z() - bbox_center[2]) / diagonal * 2
			);
		}

		CGAL::IO::write_PLY((output_root / "normalized_scene.ply").string(), total_vertices, total_faces);

		_vertices.resize(total_vertices.size());
		for (int i = 0; i < total_vertices.size(); ++i)
		{
			_vertices[i].resize(3);
			_vertices[i][0] = total_vertices[i].x();
			_vertices[i][1] = total_vertices[i].y();
			_vertices[i][2] = total_vertices[i].z();
		}
	}

	// Generate query points
	const int resolution = 256;
	const int patch_size = 32;
	const int num_grids = 10;
	const int total_resolution = resolution * num_grids;

	MyBVH bvh(_vertices, total_faces);
	LOG(INFO) << "Read and build BVH done";

	for(int i=0;i< num_grids * num_grids * num_grids;++i)
	{
		LOG(INFO) << "Processing " << i << " / " << num_grids * num_grids * num_grids;
		const int x_start = (i / num_grids / num_grids) * resolution;
		const int y_start = (i / num_grids % num_grids) * resolution;
		const int z_start = (i % num_grids) * resolution;

		// Prepare query points
		std::vector<Eigen::Vector3d> viz_points;
		std::vector<double> query_points(resolution * resolution * resolution * 3);
		for (int j = 0; j < query_points.size() / 3; ++j)
		{
			const int x = j / resolution / resolution;
			const int y = j / resolution % resolution;
			const int z = j % resolution;

			query_points[j * 3 + 0] = (double)(x_start + x) / (total_resolution - 1) * 2 - 1;
			query_points[j * 3 + 1] = (double)(y_start + y) / (total_resolution - 1) * 2 - 1;
			query_points[j * 3 + 2] = (double)(z_start + z) / (total_resolution - 1) * 2 - 1;
			viz_points.emplace_back(
				query_points[j * 3 + 0],
				query_points[j * 3 + 1],
				query_points[j * 3 + 2]
			);
		}
		export_points(output_root / "test.ply", viz_points);
		// continue;
		// Calculate distance
		Eigen::Tensor<unsigned short, 2, Eigen::RowMajor> feature(resolution * resolution * resolution, 5);
		bool is_valid = false;
		{
			std::vector<double> bvh_distances, bvh_closest_points, bvh_closest_bcs;
			std::vector<long long> bvh_closest_faces;

			std::tie(bvh_distances, bvh_closest_points, bvh_closest_faces, bvh_closest_bcs) = bvh.query(query_points);

			tbb::parallel_for(tbb::blocked_range<int>(0, query_points.size() / 3), [&](const auto& r0)
				{
					for (int i_point = r0.begin(); i_point < r0.end(); ++i_point)
					{
						long long target_face = bvh_closest_faces[i_point];
						double distance = bvh_distances[i_point];
						if (distance < 1e-2)
							is_valid = true;
						Point_3 closest_point(
							bvh_closest_points[i_point * 3 + 0],
							bvh_closest_points[i_point * 3 + 1],
							bvh_closest_points[i_point * 3 + 2]
						);

						Eigen::Vector3d gradient_dir = Eigen::Vector3d(
							closest_point[0] - query_points[i_point * 3 + 0],
							closest_point[1] - query_points[i_point * 3 + 1],
							closest_point[2] - query_points[i_point * 3 + 2]
						).normalized();
						feature(i_point, 0) = static_cast<unsigned short>(std::round((distance / 2.0) * 65535));

						auto phi_theta = normal_vector_to_angle(gradient_dir);
						feature(i_point, 1) = static_cast<unsigned short>(std::round(phi_theta[0] / (M_PI * 2) * 65535));
						feature(i_point, 2) = static_cast<unsigned short>(std::round(phi_theta[1] / (M_PI * 2) * 65535));

						const auto& id_vertices = total_faces[target_face];

						Eigen::Vector3d v1 = cgal_2_eigen_point<double>(total_vertices[id_vertices[1]]) - cgal_2_eigen_point<double>(total_vertices[id_vertices[0]]);
						Eigen::Vector3d v2 = cgal_2_eigen_point<double>(total_vertices[id_vertices[2]]) - cgal_2_eigen_point<double>(total_vertices[id_vertices[1]]);
						Eigen::Vector3d surface_normal = v1.cross(v2).normalized();
						auto normal_phi_theta = normal_vector_to_angle(surface_normal);
						feature(i_point, 3) = static_cast<unsigned short>(std::round(normal_phi_theta[0] / (M_PI * 2) * 65535));
						feature(i_point, 4) = static_cast<unsigned short>(std::round(normal_phi_theta[1] / (M_PI * 2) * 65535));
					}
				});
		}

		if (!is_valid)
			continue;

		std::vector<unsigned short> features_vec(resolution * resolution* resolution * 5);
		std::copy_n(feature.data(), features_vec.size(), features_vec.begin());

		NpyArray::SaveNPY(
			(output_root / (std::to_string(i) + ".npy")).string(),
			features_vec,
			NpyArray::shape_t{ (size_t)resolution, (size_t)resolution, (size_t)resolution, 5 }
		);
	}

	LOG(INFO) << "Done";
	return 0;
}
