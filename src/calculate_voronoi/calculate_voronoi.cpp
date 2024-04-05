#include <cuda_runtime_api.h>
#include <tbb/tbb.h>

#include "cgal_tools.h"
#include "common_util.h"
#include "model_tools.h"

#include <argparse/argparse.hpp>

#include "calculate_distance.h"
#include "calculate_indices.h"

std::vector<fs::path> generate_task_list(const fs::path& v_data_input)
{
	std::vector<fs::path> task_files;

	std::vector<std::string> target_prefix;
	
	for (fs::directory_iterator it_file(v_data_input/"mesh"); it_file != fs::directory_iterator(); ++it_file)
		target_prefix.push_back(it_file->path().filename().stem().string());
	LOG(INFO) << "We have " << target_prefix.size() << " targets in total.";

	//Check existing
	std::vector<std::string> existing_ids;
	fs::directory_iterator it_file(v_data_input / "progress");
	fs::directory_iterator it_end;
	for(;it_file!=it_end;++it_file)
	{
		const std::string full_name = it_file->path().filename().string();
		if (full_name.size() < 14)
			continue;

		if (full_name.substr(full_name.size() - 14, 14) != ".whole_voronoi")
			continue;
		existing_ids.push_back(it_file->path().filename().stem().string());
	}

	#pragma omp parallel for
	for (int i = 0; i < target_prefix.size(); ++i)
	{
		std::string prefix = target_prefix[i];

		if (std::find(existing_ids.begin(), existing_ids.end(), prefix) != existing_ids.end())
			continue;

#pragma omp critical
		{
			task_files.push_back(v_data_input / prefix);
		}
	}

	std::sort(task_files.begin(), task_files.end(), [](const fs::path& item1, const fs::path& item2)
	{
		return item1.filename().string() < item2.filename().string();
	});
	return task_files;
}

int main(int argc, char* argv[])
{
	argparse::ArgumentParser program("prepare_data_3d");
	{
		LOG(INFO) << "enter the arguments: data_root";
		program.add_description("data_root");
		program.add_argument("data_root").required();
		program.add_argument("resolution").required().scan<'i',int>();
		program.add_argument("is_pure_quadric").implicit_value(true).default_value(false);
		program.parse_args(argc, argv);
	}

	fs::path output_root(program.get<std::string>("data_root"));
	const int res = program.get<int>("resolution");
	const bool is_pure_quadric = program.get<bool>("is_pure_quadric");
	if(res*res*res%8!=0)
	{
		LOG(ERROR) << "The resolution should be a multiple of 8";
		return 0;
	}

	safeCheckFolder(output_root);
	safeCheckFolder(output_root/"voronoi");

	auto task_files = generate_task_list(
		output_root
	);

	LOG(INFO) << ffmt("We have %d valid task") % task_files.size();
	if (task_files.empty())
		return 0;

	const int max_task_per_gpu = 8;
	int num_gpus;
	cudaGetDeviceCount(&num_gpus);
	std::mutex gpu_mutex[100];
	std::vector<Point_3> query_points(res*res*res);
	for(int i=0;i<res*res*res;++i)
	{
		const int x = i / res / res;
		const int y = (i / res) % res;
		const int z = i % res;
		query_points[i] = Point_3(
			(double)x / (res - 1) * 2 - 1,
			(double)y / (res - 1) * 2 - 1,
			(double)z / (res - 1) * 2 - 1
		);
	}

	// Test
	if(false)
	{
		std::ifstream ifs("G:/Dataset/GSP/GSP_debug/voronoi/00000017_0.bin");
		std::vector<char> buffer(std::istreambuf_iterator<char>(ifs), {});
		std::vector<bool> flags(buffer.size() * 8, false);
		for(int i=0;i< buffer.size();++i)
		{
			for(int j=0;j<8;++j)
			{
				flags[i * 8 + j] = (buffer[i] >> j) & 1;
			}
		}

		std::vector<Point_3> points(query_points);
		points.erase(std::remove_if(points.begin(), points.end(),
			[&flags, &points](const Point_3& item)
			{
				return !flags[&item - &points[0]];
			}), points.end());
		export_points((output_root / "debug" / (ffmt("%d.ply") % 1).str()).string(), points);
	}

	std::atomic<int> progress(0);

	tbb::parallel_for(tbb::blocked_range<int>(0, task_files.size()), [&](const tbb::blocked_range<int>& range)
		{
			for (int i_task = range.begin(); i_task != range.end(); ++i_task)
			{
				progress.fetch_add(1);
				if (progress.load() % 100 == 0)
					LOG(INFO) << ffmt("Progress: %d/%d") % progress.load() % task_files.size();

				std::string prefix = task_files[i_task].filename().string();

				std::vector<Point_3> points;
				std::vector<Triangle_3> triangles;
				std::vector<Vector_3> normals;
				std::vector<std::array<int, 3>> face_indices;
				read_model((output_root / "mesh" / (prefix + ".ply")).string(), points, normals, face_indices, triangles);

				if (triangles.size() <= 10)
					continue;

				ryml::Tree config;
				{
					std::ifstream ifs((output_root / "yml" / (prefix + ".yml")).string());
					std::stringstream buffer;
					buffer << ifs.rdbuf();
					ifs.close();
					config = ryml::parse_in_arena(ryml::to_csubstr(buffer.str()));
					// config = YAML::LoadFile((output_root / "yml" / (prefix + ".yml")).string());

				}

				// 2. Filter the input curves and surfaces
				std::vector<Curve> curves;
				std::vector<Surface> surfaces;
				Eigen::Vector4d bounds_vector;
				bounds_vector.setConstant(0.); // Bounds are not used in the function below
				std::tie(curves, surfaces) = filter_primitives(
					config, face_indices, is_pure_quadric, bounds_vector
				);

				if (curves.empty() && surfaces.empty())
					continue;

				// 3. Calculate the per face index
				std::vector<long long> surface_id_to_primitives;
				std::vector<std::vector<long long>> face_edge_indicator;
				std::vector<std::pair<int, std::vector<int>>> id_corner_points;
				Eigen::MatrixXi adj_matrix;
				std::tie(surface_id_to_primitives, face_edge_indicator, id_corner_points, adj_matrix) = calculate_indices(
					curves, surfaces, points, face_indices);

				const long long num_curves = static_cast<long long>(curves.size());
				const long long num_surfaces = static_cast<long long>(surfaces.size());
				const long long num_corner_points = static_cast<long long>(id_corner_points.size());
				const long long num_primitives = num_curves + num_surfaces + num_corner_points;

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
										query_points, points, face_indices, triangles,
										surface_id_to_primitives, face_edge_indicator,
										num_curves, false, udf_feature, closest_primitives);
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

				// Debug
				if (true)
				{
					std::vector<Point_3> points_(query_points);
					points_.erase(std::remove_if(points_.begin(), points_.end(),
						[&closest_primitives, &points_, &num_curves](const Point_3& item)
						{
							return closest_primitives[&item - &points_[0]] >= num_curves;
						}), points_.end());
					// export_points((output_root / "debug" / (ffmt("%d_curve_points.ply") % prefix).str()).string(), points_);

					points_.clear();
					for(int i=0; i < face_edge_indicator.size(); ++i)
					{
						for(int j=0;j<3;++j)
						{
							if (face_edge_indicator[i][j] != -1)
							{
								points_.push_back(points[face_indices[i][j]]);
							}
						}
					}
					export_points((output_root / "debug" / (ffmt("%d_curve_points.ply") % prefix).str()).string(), points_);

					points_.clear();
					for(int i=0; i < id_corner_points.size(); ++i)
					{
						points_.push_back(points[id_corner_points[i].first]);
					}
					export_points((output_root / "debug" / (ffmt("%d_corner_points.ply") % prefix).str()).string(), points_);
				}

				// Flags
				std::vector<bool> voronoi_edge(query_points.size(), 0);
				{
					tbb::parallel_for(tbb::blocked_range<int>(0, query_points.size()), 
						[&](const tbb::blocked_range<int>& r2)
					{
						for (int i_point = r2.begin(); i_point != r2.end(); ++i_point)
						{
							const int x = i_point / res / res;
							const int y = (i_point / res) % res;
							const int z = i_point % res;

							bool is_voronoi_edge = false;
							for (int dx = -1; dx <= 1; dx++)
								for (int dy = -1; dy <= 1; dy++)
									for (int dz = -1; dz <= 1; dz++)
									{
										const int nx = x + dx; const int ny = y + dy; const int nz = z + dz;
										if (nx < 0 || nx >= res || ny < 0 || ny >= res || nz < 0 || nz >= res)
											continue;
										if (closest_primitives[i_point] != closest_primitives[nx * res * res + ny * res + nz])
											is_voronoi_edge = true;
									}

							voronoi_edge[i_point] = is_voronoi_edge;
							}
						});
				}

				// Write the voronoi flag to a binary file
				{
					const int new_size = voronoi_edge.size() / 8;
					std::vector<char> new_array(new_size, 0);
					for (int i = 0; i < new_array.size(); ++i)
						for (int j = 0; j < 8; ++j)
							new_array[i] |= voronoi_edge[i * 8 + j] << j;

					std::ofstream fout((output_root / "voronoi" / (prefix + ".bin")).string(), std::ios::binary);
					fout.write(new_array.data(), new_array.size() * sizeof(char));
					fout.close();
				}

				std::ofstream out((output_root / "progress" / (ffmt("%d.whole_voronoi") % prefix).str()).string());
				out << "";
				out.close();

				// Log
				if (true)
				{
					safeCheckFolder(output_root / "debug");
					std::vector<Point_3> points(query_points);
					points.erase(std::remove_if(points.begin(), points.end(),
						[&voronoi_edge, &points](const Point_3& item)
						{
							return !voronoi_edge[&item - &points[0]];
						}), points.end());
					export_points((output_root / "debug" / (ffmt("%d.ply") % prefix).str()).string(), points);
				}
			}

		});
	
	LOG(INFO) << "Done";
	return 0;
}
