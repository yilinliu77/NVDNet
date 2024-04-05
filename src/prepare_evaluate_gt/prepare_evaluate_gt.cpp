#include <cuda_runtime_api.h>
#include <tbb/tbb.h>

#include "cgal_tools.h"
#include "common_util.h"
#include "model_tools.h"

#include <unordered_set>
#include <argparse/argparse.hpp>
#include <CGAL/grid_simplify_point_set.h>



#include <ryml/ryml.hpp>
#include <ryml/ryml_std.hpp>

#include "read_primitives_from_yml.h"
#include "calculate_indices.h"
#include "calculate_distance.h"
#include "prepare_data_3d/tools.h"

std::vector<fs::path> generate_task_list(const fs::path& v_data_input, const fs::path& v_data_output,
                                         const std::string& v_target_list,
                                         const int id_start, const int id_end)
{
	std::vector<fs::path> task_files;

	std::vector<std::string> target_prefix;
	if (v_target_list != "")
	{
		std::ifstream ifs(v_target_list);
		std::string line;
		while (!ifs.eof())
		{
			std::getline(ifs, line);
			if (line.size() < 3)
				break;
			// line = split_string(line, std::string("_"))[0];
			if (!line.empty() && line[line.size() - 1] == '\r')
				line.erase(line.size() - 1);
			target_prefix.push_back(line);
		}
		ifs.close();
		LOG(INFO) << "Use " << v_target_list << ", we have " << target_prefix.size() << " targets.";
	}
	else
	{
		for (fs::directory_iterator it_file(v_data_input); it_file != fs::directory_iterator(); ++it_file)
		{
			target_prefix.push_back(it_file->path().filename().string());
		}
		LOG(INFO) << "We have " << target_prefix.size() << " targets in total.";
	}

	std::unordered_set<std::string> target_set(target_prefix.begin(), target_prefix.end());
	target_prefix.clear();
	target_prefix.insert(target_prefix.end(), target_set.begin(), target_set.end());

	//Check existing
	std::vector<unsigned int> existing_ids;
	fs::directory_iterator it_file(v_data_output / "progress");
	fs::directory_iterator it_end;
	for (; it_file != it_end; ++it_file)
	{
		const std::string full_name = it_file->path().filename().string();
		const std::string name = full_name.substr(0, 8);
		const std::string stage = full_name.substr(8, 1);
		if (std::atoi(stage.c_str()) != 0)
			continue;
		existing_ids.push_back(std::atoi(name.c_str()));
	}

	#pragma omp parallel for
	for (int i = 0; i < target_prefix.size(); ++i)
	{
		std::string prefix = target_prefix[i];
		const int file_idx = std::atoi(prefix.c_str());
		if (file_idx >= id_start && file_idx < id_end)
		{
			const int file_count = std::count_if(
				boost::filesystem::directory_iterator(v_data_input / prefix),
				boost::filesystem::directory_iterator(),
				static_cast<bool(*)(const boost::filesystem::path&)>(boost::filesystem::is_regular_file)
			);

			if (std::find(existing_ids.begin(), existing_ids.end(), std::atoi(prefix.c_str())) != existing_ids.end())
				continue;

			if (file_count >= 2)
			{
				#pragma omp critical
				{
					task_files.push_back(v_data_input / prefix);
				}
			}
		}
	}

	std::sort(task_files.begin(), task_files.end(), [](const fs::path& item1, const fs::path& item2)
		{
			return std::atoi(item1.filename().string().c_str()) < std::atoi(item2.filename().string().c_str());
		});
	return task_files;
}

int main(int argc, char* argv[])
{
	// tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);
	argparse::ArgumentParser program("prepare_data_3d");
	{
		LOG(INFO) << "enter the arguments: data_root";
		program.add_description("data_root");
		program.add_argument("data_root").required();
		program.add_argument("output_root").required();
		program.add_argument("id_start").required().scan<'i', int>();
		program.add_argument("id_end").required().scan<'i', int>();
		program.add_argument("--target_list").default_value(std::string(""));
		program.parse_args(argc, argv);
	}

	fs::path data_root(program.get<std::string>("data_root"));
	fs::path output_root(program.get<std::string>("output_root"));
	int id_start = program.get<int>("id_start");
	int id_end = program.get<int>("id_end");

	std::string target_list = program.get<std::string>("--target_list");

	safeCheckFolder(output_root);
	safeCheckFolder(output_root/"progress");
	safeCheckFolder(output_root/"gt"/"vertices");
	safeCheckFolder(output_root/"gt"/"curves");
	safeCheckFolder(output_root/"gt"/"surfaces");
	safeCheckFolder(output_root/"gt"/"voronoi");
	safeCheckFolder(output_root/"gt"/"topology");
	safeCheckFolder(output_root/"mesh");
	safeCheckFolder(output_root/"poisson");

	auto task_files = generate_task_list(
		data_root,
		output_root,
		target_list,
		id_start,
		id_end
	);

	LOG(INFO) << ffmt("We have %d valid task") % task_files.size();
	if (task_files.empty())
		return 0;

	const int max_task_per_gpu = 8;
	int num_gpus;
	cudaGetDeviceCount(&num_gpus);
	std::mutex gpu_mutex[100];

	std::atomic<int> progress(0);

	// Generate query points
	const int resolution = 256;
	std::vector<Point_3> total_query_points(resolution * resolution * resolution);
	for (int i = 0; i < total_query_points.size(); ++i)
	{
		const int x = i / resolution / resolution;
		const int y = i / resolution % resolution;
		const int z = i % resolution;
		total_query_points[i] = Point_3(
			(double)x / (resolution - 1) * 2 - 1,
			(double)y / (resolution - 1) * 2 - 1,
			(double)z / (resolution - 1) * 2 - 1
		);
	}

	tbb::parallel_for(tbb::blocked_range<int>(0, task_files.size()),
		[&](const tbb::blocked_range<int>& r)
		{
			for (int i_task = r.begin(); i_task != r.end(); ++i_task)
			{
				progress.fetch_add(1);
				if (progress.load() % 100 == 0)
					LOG(INFO) << ffmt("Progress: %d/%d") % progress.load() % task_files.size();

				std::string prefix = task_files[i_task].filename().string();
				auto timer_io = recordTime();

				// 1. Read and preprocess the mesh and primitives
				std::vector<Point_3> total_vertices;
				std::vector<std::vector<int>> total_faces;
				ryml::Tree config;
				std::vector<double> bounds;
				{
					fs::path obj_file, feature_file;
					for (fs::directory_iterator it_file(task_files[i_task]); it_file != fs::directory_iterator(); ++it_file)
					{
						const std::string name = it_file->path().string();
						if (name.find("features") != name.npos)
							feature_file = it_file->path();
						else if (name.find("trimesh") != name.npos)
							obj_file = it_file->path();
					}
					std::tie(total_vertices, total_faces) = read_obj(obj_file);
					bounds = calculate_aabb(total_vertices);
					bounds[9] *= 1.25; // Scale the shape to [-0.8,+0.8]
					std::ifstream ifs(feature_file.string());
					std::stringstream buffer;
					buffer << ifs.rdbuf();
					ifs.close();
					config = ryml::parse_in_arena(ryml::to_csubstr(buffer.str()));
				}

				// 2. Calculate indices
				std::vector<long long> surface_id_to_primitives;
				std::vector<std::vector<long long>> face_edge_indicator;
				std::vector<Triangle_3> triangles;
				int num_curves, num_surfaces;
				{
					std::vector<std::array<int, 3>> face_indices(total_faces.size());
					for (int i = 0; i < face_indices.size(); ++i)
					{
						face_indices[i][0] = total_faces[i][0];
						face_indices[i][1] = total_faces[i][1];
						face_indices[i][2] = total_faces[i][2];
					}
					Eigen::Vector4d bounds_vector(bounds[6], bounds[7], bounds[8], bounds[9]);

					// Filter the input curves and surfaces
					std::vector<Curve> curves;
					std::vector<Surface> surfaces;
					std::tie(curves, surfaces) = filter_primitives(
						config, face_indices, true, bounds_vector
					);
					num_curves = curves.size();
					num_surfaces = surfaces.size();
					// Calculate the per face index
					std::vector<std::pair<int, std::vector<int>>> id_corner_points;
					Eigen::MatrixXi adj_matrix;
					std::tie(surface_id_to_primitives, face_edge_indicator, id_corner_points, adj_matrix) = calculate_indices(
						curves, surfaces, total_vertices, face_indices);

					// 3. Sample points on vertices, curves and faces
					for (auto& item : total_vertices)
						item = Point_3(
							(item.x() - bounds[6]) / bounds[9] * 2,
							(item.y() - bounds[7]) / bounds[9] * 2,
							(item.z() - bounds[8]) / bounds[9] * 2
						);

					for (auto& item : total_faces)
						triangles.emplace_back(
							total_vertices[item[0]],
							total_vertices[item[1]],
							total_vertices[item[2]]
						);

					Point_set corner_points;
					corner_points.resize(id_corner_points.size());
					for(int i=0;i<id_corner_points.size();++i)
						corner_points.point(i) = total_vertices[id_corner_points[i].first];

					const double num_per_m = 1000;
					const double num_per_m2 = 10000;
					auto sample_points_curves = sample_points_on_curve(
						curves, total_vertices, num_per_m);

					// Remove duplicate points
					// auto iterator_to_first_to_remove = CGAL::grid_simplify_point_set(corner_points, 0.001); // optional
					// corner_points.remove(iterator_to_first_to_remove, corner_points.end());
					// corner_points.collect_garbage();

					// Write points and curves
					CGAL::IO::write_point_set(
						(output_root / "gt" / "vertices" / (ffmt("%d.ply") % prefix).str()).string(),
						corner_points);
					CGAL::IO::write_point_set((output_root / "gt" / "curves" / (ffmt("%d.ply") % prefix).str()).string(),
						sample_points_curves
					);

					// Write sample points on faces
					Point_set face_points = sample_points_according_density(triangles, num_per_m2);
					auto primitive_index = face_points.add_property_map("primitive_index", 0).first;
					{
						const auto face_index = face_points.property_map<int>("face_index").first;

						for (int i = 0; i < face_points.size(); ++i)
							primitive_index[i] = surface_id_to_primitives[face_index[i]] - num_curves;

						CGAL::IO::write_point_set(
							(output_root / "gt" / "surfaces" / (ffmt("%d.ply") % prefix).str()).string(),
							face_points);
					}

					// Write mesh and poisson points
					{
						Surface_mesh mesh;
						const auto color_table = get_color_table_bgr2();
						auto index_map = mesh.add_property_map<Surface_mesh::Face_index, int>("primitive_index").first;
						auto red_map = mesh.add_property_map<Surface_mesh::Face_index, uchar>("red").first;
						auto green_map = mesh.add_property_map<Surface_mesh::Face_index, uchar>("green").first;
						auto blue_map = mesh.add_property_map<Surface_mesh::Face_index, uchar>("blue").first;

						std::vector<CGAL::Color> fcolors(triangles.size());
						for(int i=0;i<triangles.size();++i)
						{
							const auto color = color_table[i % color_table.size()];
							fcolors[i] = CGAL::Color(color[2], color[1], color[0]);
						}

						std::vector<Surface_mesh::Vertex_index> vertices;
						for (auto& item : total_vertices)
							vertices.emplace_back(mesh.add_vertex(item));

						for (int i_face = 0; i_face < total_faces.size(); ++i_face)
						{
							const auto pointer = mesh.add_face(
								vertices[total_faces[i_face][0]],
								vertices[total_faces[i_face][1]],
								vertices[total_faces[i_face][2]]
							);
							index_map[pointer] = surface_id_to_primitives[i_face] - num_curves;
							red_map[pointer] = color_table[surface_id_to_primitives[i_face] % color_table.size()][2];
							green_map[pointer] = color_table[surface_id_to_primitives[i_face] % color_table.size()][1];
							blue_map[pointer] = color_table[surface_id_to_primitives[i_face] % color_table.size()][0];
						}

						CGAL::IO::write_PLY(
							(output_root / "mesh" / (ffmt("%d.ply") % prefix).str()).string(),
							mesh);

						// Write poisson points
						std::vector<double> poisson_set(10000 * 6);
						Point_set p = sample_poisson_points(triangles, 10000);
						CGAL::IO::write_point_set(
							(output_root / "poisson" / (ffmt("%d.ply") % prefix).str()).string(),
							p);
					}
					// continue;
					// Write topology
					{
						std::ofstream ofs((output_root / "gt" / "topology" / (ffmt("%d.txt") % prefix).str()).string());
						ofs << "FE" << std::endl;
						for (int i_surface = 0; i_surface < surfaces.size(); ++i_surface)
						{
							ofs << i_surface;
							for (int i_curve = 0; i_curve < num_curves; ++i_curve)
								if (adj_matrix(i_curve, i_surface + num_curves))
								{
									ofs << ffmt(" %d") % i_curve;
									// Debug; Check connectivity
									if (false)
									{
										Point_set surface;
										for(int i=0;i< face_points.size();++i)
											if (primitive_index[i] == i_surface)
												surface.insert(face_points.point(i));
										CGAL::IO::write_point_set(
											(output_root / "gt" / "temps.ply").string(),
											surface);
										Point_set curve;
										const auto curve_index = sample_points_curves.property_map<int>("primitive_index").first;
										for (int i = 0; i < sample_points_curves.size(); ++i)
											if (curve_index[i] == i_curve)
												curve.insert(sample_points_curves.point(i));
										CGAL::IO::write_point_set(
											(output_root / "gt" / "tempc.ply").string(),
											curve);
										std::cout << 1;
									}
								}
							ofs << std::endl;
						}
						ofs << "EV" << std::endl;

						for(int i_curve=0;i_curve<num_curves;++i_curve)
						{
							ofs << i_curve;
							for (int i_vertex = 0; i_vertex < id_corner_points.size(); ++i_vertex)
								if (adj_matrix(i_curve, i_vertex + num_curves + surfaces.size()))
								{
									ofs << ffmt(" %d") % i_vertex;
									if (false)
									{
										Point_set curve;
										const auto curve_index = sample_points_curves.property_map<int>("primitive_index").first;
										for (int i = 0; i < sample_points_curves.size(); ++i)
											if (curve_index[i] == i_curve)
												curve.insert(sample_points_curves.point(i));
										CGAL::IO::write_point_set(
											(output_root / "gt" / "tempc.ply").string(),
											curve);
										Point_set vertex;
										for (int i = 0; i < id_corner_points.size(); ++i)
											if (i == i_vertex)
												vertex.insert(total_vertices[id_corner_points[i].first]);
										CGAL::IO::write_point_set(
											(output_root / "gt" / "tempv.ply").string(),
											vertex);
										std::cout << 1;
									}
								}
							ofs << std::endl;
						}
						ofs.close();
					}

					// Check topology
					{
						for (int i_curve = 0; i_curve < num_curves; ++i_curve)
						{
							int count = 0;
							for (int i_vertex = 0; i_vertex < id_corner_points.size(); ++i_vertex)
								if (adj_matrix(i_curve, num_surfaces + num_curves + i_vertex))
									count++;
							if (count != 2)
							{
								// Point_set curve;
								// const auto curve_index = sample_points_curves.property_map<int>("primitive_index").first;
								// for (int i = 0; i < sample_points_curves.size(); ++i)
								// 	if (curve_index[i] == i_curve)
								// 		curve.insert(sample_points_curves.point(i));
								// CGAL::IO::write_point_set(
								// 	(output_root / "gt" / "tempc.ply").string(),
								// 	curve);
								// LOG(INFO) << ffmt("%d: Curve %d has %d vertices") % prefix % i_curve % count;
							}
						}
					}
				}
				// continue;
				// 4. Compute closest_primitives
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

				// 5. Extract the voronoi
				std::vector<char> voronoi_edge(closest_primitives.size() / 8);
				{
					std::fill_n(voronoi_edge.begin(), closest_primitives.size() / 8, 0);

					Point_set voronoi_boundaries;
					for (int i_point = 0; i_point < closest_primitives.size(); ++i_point)
					{
						const int x = i_point / resolution / resolution;
						const int y = i_point / resolution % resolution;
						const int z = i_point % resolution;

						bool is_voronoi_edge = false;
						for (int dx = -1; dx <= 1; dx++)
							for (int dy = -1; dy <= 1; dy++)
								for (int dz = -1; dz <= 1; dz++)
								{
									const int nx = x + dx; const int ny = y + dy; const int nz = z + dz;
									if (nx < 0 || nx >= resolution || ny < 0 || ny >= resolution || nz < 0 || nz >= resolution)
										continue;
									const int id1 = closest_primitives[i_point];
									const int id2 = closest_primitives[nx * resolution * resolution + ny * resolution + nz];

									if (id1 != id2)
										is_voronoi_edge = true;
								}
						int id = i_point / 8;
						voronoi_edge[id] |= is_voronoi_edge << (i_point % 8);
						// if (is_voronoi_edge)
							// voronoi_boundaries.insert(total_query_points[i_point]);
					}

					// Write the vector of char to files
					std::ofstream ofs((output_root / "gt" / "voronoi" / (ffmt("%d") % prefix).str()).string());
					ofs.write(voronoi_edge.data(), voronoi_edge.size());
					ofs.close();

					// CGAL::IO::write_point_set((output_root / "gt" / "voronoi" / (ffmt("%d.ply") % prefix).str()).string(), voronoi_boundaries);
				}

				std::ofstream out((output_root / "progress" / (ffmt("%d.0") % prefix).str()).string());
				out << "";
				out.close();
			}
		}
	);
	

	LOG(INFO) << "Done";
	return 0;
}
