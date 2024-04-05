#include <cuda_runtime_api.h>

#include "cgal_tools.h"
#include "common_util.h"
#include "model_tools.h"

#include <argparse/argparse.hpp>
#include "calculate_distance.h"
#include "kd_tree_helper.h"
#include "read_primitives_from_yml.h"
#include "calculate_indices.h"

#include "writer.h"

#include <tbb/tbb.h>


std::vector<fs::path> generate_task_list(const fs::path& v_data_input)
{
	std::vector<fs::path> task_files;

	std::vector<std::string> target_prefix;

	for (fs::directory_iterator it_file(v_data_input / "mesh"); it_file != fs::directory_iterator(); ++it_file)
		target_prefix.push_back(it_file->path().filename().stem().string());
	LOG(INFO) << "We have " << target_prefix.size() << " targets in total.";

	//Check existing
	std::vector<std::string> existing_ids;
	fs::directory_iterator it_file(v_data_input / "progress");
	fs::directory_iterator it_end;
	for (; it_file != it_end; ++it_file)
	{
		const std::string full_name = it_file->path().filename().string();
		if (full_name.substr(full_name.size() - 6, 6) != ".patch")
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

//#pragma optimize("", off)
int main(int argc, char* argv[])
{
	// tbb::global_control global_limit1(tbb::global_control::thread_stack_size, 100 * 1024 * 1024);
	argparse::ArgumentParser program("prepare udf data for training; will read the mesh model you specified");
	{
		LOG(INFO) << "enter the arguments: data_root mesh_folder flag_folder output_folder resolution";
		program.add_description("data_root");
		program.add_argument("data_root").required();
		program.add_argument("mesh_folder").required();
		program.add_argument("flag_folder").required();
		program.add_argument("num_cpus").scan<'i', int>().required();
		program.parse_args(argc, argv);
	}

	const int patch_resolution = 32;
	const int num_max_points = 1024;

	fs::path output_root(program.get<std::string>("data_root"));
	fs::path poisson_folder(output_root / "poisson");
	fs::path mesh_folder(output_root / program.get<std::string>("mesh_folder"));
	fs::path flag_folder(output_root / program.get<std::string>("flag_folder"));
	int num_cpus = program.get<int>("num_cpus");
	tbb::global_control global_limit2(tbb::global_control::max_allowed_parallelism, num_cpus);

	safeCheckFolder(output_root);
	safeCheckFolder(output_root / "seeds"); // Store the seed points
	safeCheckFolder(output_root / "patch"); // Store the patch point clouds
	safeCheckFolder(flag_folder); // Store the training.h5

	auto task_files = generate_task_list(
		output_root
	);

	LOG(INFO) << ffmt("We have %d valid task") % task_files.size();
	if (task_files.empty())
		return 0;

	const std::vector<double> grid_radius{ 0.05, 0.1, 0.2 };
	const int num_sample_points = 400;

	H5Writter writer(flag_folder, patch_resolution, grid_radius.size() * num_sample_points);
	std::thread writer_thread(&H5Writter::store_data, &writer);

	const int max_task_per_gpu = 8;
	int num_gpus;
	cudaGetDeviceCount(&num_gpus);
	std::mutex gpu_mutex[100];
	std::atomic<int> progress(0);
	std::atomic<int> num_regenerate_patch(0);

	std::vector<double> time_statics(10, 0.);
	bool is_log = false;

	// tbb::parallel_for(tbb::blocked_range<int>(0, 10),
	tbb::parallel_for(tbb::blocked_range<int>(0, task_files.size()),
		[&](const tbb::blocked_range<int>& r)
		{
			for (int i_task = r.begin(); i_task != r.end(); ++i_task)
			{
				progress.fetch_add(1);
				if (progress.load() % 100 == 0)
					LOG(INFO) << ffmt("Progress: %d/%d") % progress.load() % task_files.size();

				std::string prefix = task_files[i_task].filename().string();
				int prefix1 = std::atoi(prefix.substr(0, prefix.find_first_of('_')).c_str());
				int prefix2 = std::atoi(prefix.substr(prefix.find_first_of('_') + 1).c_str());

				auto timer = recordTime();
				// 1. Read mesh
				std::vector<Point_3> points;
				std::vector<Triangle_3> triangles;
				// std::vector<Vector_3> normals;
				std::vector<std::array<int, 3>> face_indices;
				{
					try
					{
						CGAL::IO::read_polygon_soup((output_root / "mesh" / (prefix + ".ply")).string(),
							points, face_indices);
						triangles.resize(face_indices.size());
						for(int i=0;i<face_indices.size();++i)
							triangles[i] = Triangle_3(points[face_indices[i][0]], points[face_indices[i][1]], points[face_indices[i][2]]);
						// read_model(, points, normals, face_indices, triangles);
					}
					catch (const std::exception& e)
					{
						LOG(INFO) << prefix << " " << e.what();
						continue;
					}
				}
				if (face_indices.empty())
					continue;
				time_statics[0] += profileTime(timer, "Read mesh", is_log);

				// 2. Read yml
				ryml::Tree config;
				{
					std::ifstream ifs((output_root / "yml" / (prefix + ".yml")).string());
					std::stringstream buffer;
					buffer << ifs.rdbuf();
					ifs.close();
					config = ryml::parse_in_arena(ryml::to_csubstr(buffer.str()));
					// config = YAML::LoadFile((output_root / "yml" / (prefix + ".yml")).string());

				}
				if (config["curves"].empty() || config["surfaces"].empty())
					continue;
				time_statics[1] += profileTime(timer, "Read yml", is_log);

				// 3. Sample seed points
				Point_set seed_points;
				Point_set poisson_points(true);
				{
					// Read points
					// CGAL::IO::read_point_set((poisson_folder / (prefix + ".ply")).string(), poisson_points);
					poisson_points = sample_poisson_points(triangles, 10000);

					// Sample points on curves
					std::ifstream ifs((output_root / "bbox" / (prefix + ".txt")).string());
					Eigen::Vector4d bbox;
					ifs >> bbox(0) >> bbox(1) >> bbox(2) >> bbox(3);
					ifs.close();
					Point_set points_on_curve = sample_points_on_curve(config, bbox, 1000).second;

					// fps sampling
					const std::vector<int> fps_indices = fps_sampling(points_on_curve, 200);
					for (const auto& item : fps_indices)
						seed_points.insert(points_on_curve.point(item));

					std::default_random_engine mt;
					std::uniform_int_distribution<int> dist(0, poisson_points.size() - 1);
					for (int i = 0; i < 200; ++i)
						seed_points.insert(poisson_points.point(dist(mt)));

					CGAL::IO::write_point_set((output_root / "seeds" / (prefix + ".ply")).string(), seed_points);
				}
				time_statics[2] += profileTime(timer, "Seed points", is_log);

				// 4. Calculate indices
				std::vector<long long> surface_id_to_primitives;
				std::vector<std::vector<long long>> face_edge_indicator;
				int num_curves;
				{
					// Filter the input curves and surfaces
					std::vector<Curve> curves;
					std::vector<Surface> surfaces;
					std::tie(curves, surfaces) = filter_primitives(
						config, face_indices, true
					);
					num_curves = curves.size();
					// Calculate the per face index
					std::vector<std::pair<int, std::vector<int>>> id_corner_points;
					std::tie(surface_id_to_primitives, face_edge_indicator, id_corner_points) = calculate_indices(
						curves, surfaces, points, face_indices);
				}
				time_statics[3] += profileTime(timer, "Indices", is_log);

				// 5. Generate the query points and the input point patch
				std::vector<Point_3> total_query_points;
				std::vector<std::vector<Eigen::Vector3d>> input_point_patches;
				std::vector<std::unordered_set<int>> input_point_patch_ids;
				{
					auto index_map = poisson_points.property_map<int>("face_index").first;
					input_point_patches.resize(seed_points.size() * grid_radius.size());
					input_point_patch_ids.resize(seed_points.size() * grid_radius.size());

					std::uniform_int_distribution<int> und(0, seed_points.size() - 1);
					std::default_random_engine mt;
					for (int i_seed = 0; i_seed < seed_points.size(); ++i_seed)
					{
						Eigen::Vector3d seed_point = cgal_2_eigen_point<double>(seed_points.point(i_seed));
						for (int i_iter = 0; i_iter < grid_radius.size(); ++i_iter)
						{
							const double local_grid_radius = grid_radius[i_iter];
							std::vector<Eigen::Vector3d>& centralized_points_in_range = input_point_patches[
								i_seed * grid_radius.size() + i_iter];
							std::unordered_set<int>& id_primitives = input_point_patch_ids[i_seed * grid_radius.size() + i_iter];
							std::vector<Eigen::Vector3d> points_in_range;
							std::vector<Eigen::Vector3d> query_points;

							// 5.1 get range points
							int num_attempt = 0;
							do
							{
								Eigen::AlignedBox3d box;
								box.min() = seed_point - Eigen::Vector3d::Constant(local_grid_radius);
								box.max() = seed_point + Eigen::Vector3d::Constant(local_grid_radius);

								id_primitives.clear();
								points_in_range.clear();
								centralized_points_in_range.clear();

								for (int i_point = 0; i_point < poisson_points.size(); ++i_point)
								{
									const auto& point = cgal_2_eigen_point<double>(poisson_points.point(i_point));
									const auto& bbox_min = box.min();
									const auto& bbox_max = box.max();

									if (point.x() < bbox_min.x() || point.y() < bbox_min.y() || point.z() < bbox_min.z())
										continue;
									if (point.x() > bbox_max.x() || point.y() > bbox_max.y() || point.z() > bbox_max.z())
										continue;

									points_in_range.push_back(point);
									centralized_points_in_range.emplace_back(
										(point - seed_point) / local_grid_radius
									);
									const int id_face = index_map[i_point];

									id_primitives.insert(surface_id_to_primitives[id_face]);
									if (face_edge_indicator[id_face][0] != -1)
										id_primitives.insert(face_edge_indicator[id_face][0]);
									if (face_edge_indicator[id_face][1] != -1)
										id_primitives.insert(face_edge_indicator[id_face][1]);
									if (face_edge_indicator[id_face][2] != -1)
										id_primitives.insert(face_edge_indicator[id_face][2]);
								}

								if (points_in_range.size() >= 5)
									break;

								int new_i_seed = und(mt);
								seed_point = cgal_2_eigen_point<double>(seed_points.point(new_i_seed));
								num_regenerate_patch.fetch_add(1);
							} while (num_attempt < 100);
							if (num_attempt == 100)
							{
								LOG(INFO) << "Failed to generate patch for " << prefix << " " << i_seed << " " << i_iter;
								exit(0);
							}

							// 5.2 generate query points
							const double length_per_voxel = local_grid_radius * 2 / patch_resolution;
							for (int ix = -patch_resolution / 2; ix < patch_resolution / 2; ++ix)
								for (int iy = -patch_resolution / 2; iy < patch_resolution / 2; ++iy)
									for (int iz = -patch_resolution / 2; iz < patch_resolution / 2; ++iz)
									{
										const Eigen::Vector3d query_point = seed_point + Eigen::Vector3d(ix, iy, iz) *
											length_per_voxel;
										query_points.push_back(query_point);
										total_query_points.emplace_back(query_point.x(), query_point.y(), query_point.z());
									}

							if (false)
							{
								export_points("query.ply", query_points);
								export_points("source.ply", points_in_range);
							}
							continue;
						}
					}
				}
				time_statics[4] += profileTime(timer, "Query points and input patches", is_log);

				// 6. Calculate the closest primitive and distance for query points
				std::vector<int> closest_primitives(total_query_points.size());
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
									total_query_points,
									triangles,
									surface_id_to_primitives, face_edge_indicator,
									num_curves, false,
									closest_primitives
								);
								succesful = true;
								gpu_mutex[i].unlock();
								break;
							}
						}
						if (!succesful)
							override_sleep(3);
					}
				}
				time_statics[5] += profileTime(timer, "Cuda distance", is_log);

				// 7. Extract the voronoi from closest_primitives
				std::shared_ptr<char[]> voronoi_edge(new char[closest_primitives.size() / 8]);
				{
					std::fill_n(voronoi_edge.get(), closest_primitives.size() / 8, 0);

					// tbb::parallel_for(tbb::blocked_range<int>(0, input_point_patches.size()),
						// [&](const tbb::blocked_range<int>& r1)
						// {
							// for (int i_patch = r1.begin(); i_patch != r1.end(); ++i_patch)
							for (int i_patch = 0; i_patch < input_point_patches.size(); ++i_patch)
							{
								const auto& id_primitives = input_point_patch_ids[i_patch];
								const int total_points_per_patch = patch_resolution * patch_resolution * patch_resolution;
								for (int i_point = 0; i_point < total_points_per_patch; ++i_point)
								{
									const int id_start = total_points_per_patch * i_patch;
									const int x = i_point / patch_resolution / patch_resolution;
									const int y = i_point / patch_resolution % patch_resolution;
									const int z = i_point % patch_resolution;

									bool is_voronoi_edge = false;
									for (int dx = -1; dx <= 1; dx++)
										for (int dy = -1; dy <= 1; dy++)
											for (int dz = -1; dz <= 1; dz++)
											{
												const int nx = x + dx; const int ny = y + dy; const int nz = z + dz;
												if (nx < 0 || nx >= patch_resolution || ny < 0 || ny >= patch_resolution || nz < 0 || nz >= patch_resolution)
													continue;
												const int id1 = closest_primitives[id_start + i_point];
												const int id2 = closest_primitives[id_start + nx * patch_resolution * patch_resolution + ny * patch_resolution + nz];

												if (id1 != id2 &&
													id_primitives.find(id1) != id_primitives.end() &&
													id_primitives.find(id2) != id_primitives.end()
													)
													is_voronoi_edge = true;
											}
									int id = (i_patch * total_points_per_patch + i_point) / 8;

									voronoi_edge[id] |= is_voronoi_edge << (i_point % 8);
								}
							}
						// });
				}
				time_statics[6] += profileTime(timer, "voronoi_edge", is_log);

				// 8. Write
				{
					const int num_patches = input_point_patches.size();
					std::shared_ptr<short[]> points_ptr(new short[num_patches * num_max_points * 3]);
					std::shared_ptr<char[]> point_flags_ptr(new char[num_patches * num_max_points / 8]);
					std::fill_n(point_flags_ptr.get(), num_patches* num_max_points / 8, 0);
					for (int i_patch = 0; i_patch < input_point_patches.size(); ++i_patch)
					{
						for(int i_point = 0;i_point < num_max_points;++i_point)
						{
							const int id_point = i_patch * num_max_points * 3 + i_point * 3;
							const int id_flag = i_patch * num_max_points / 8 + i_point / 8;

							int id_flag_local = i_point / 8;
							if (i_point < input_point_patches[i_patch].size())
							{
								points_ptr[id_point + 0] = (short)(input_point_patches[i_patch][i_point](0) * 32767);
								points_ptr[id_point + 1] = (short)(input_point_patches[i_patch][i_point](1) * 32767);
								points_ptr[id_point + 2] = (short)(input_point_patches[i_patch][i_point](2) * 32767);
								point_flags_ptr[id_flag] |= (1 << (i_point % 8));
							}
							else
							{
								points_ptr[i_patch * num_max_points * 3 + i_point * 3 + 0] = 0;
								points_ptr[i_patch * num_max_points * 3 + i_point * 3 + 1] = 0;
								points_ptr[i_patch * num_max_points * 3 + i_point * 3 + 2] = 0;
								point_flags_ptr[id_flag_local] |= (0 << (i_point % 8));
							}
						}
						// export_points(
							// (output_root / "patch" / (ffmt("%d_%d.ply") % prefix % i_patch).str()).string(),
							// input_point_patches[i_patch]
						// );
					}


					while (writer.m_queues.size() > 100)
						override_sleep(4);
					writer.m_mutex.lock();
					writer.m_queues.emplace(prefix1, prefix2, voronoi_edge, points_ptr, point_flags_ptr);
					writer.m_mutex.unlock();
				}
				time_statics[7] += profileTime(timer, "Write", is_log);

				// 9. Debug
				if (false)
				{
					Point_set p;
					for (int i = 0; i < total_query_points.size(); ++i)
					{
						if (voronoi_edge[i] == 1)
							p.insert(total_query_points[i]);
					}
					CGAL::IO::write_point_set((output_root / "debug" / (prefix + ".ply")).string(), p);
				}

				// 10. Calculate the closest primitive and distance for query points
				std::ofstream out((output_root / "progress" / (ffmt("%d.patch") % prefix).str()).string());
				out << "";
				out.close();
			}
		});

	writer.need_terminal = true;
	writer_thread.join();

	for (const auto& item : time_statics)
		std::cout << item << "; ";
	std::cout << std::endl;

	LOG(INFO) << "Done";
	return 0;
}
