#include <tbb/tbb.h>

#include "cgal_tools.h"
#include "common_util.h"
#include "model_tools.h"

#include <unordered_set>
#include <argparse/argparse.hpp>
// #include <yaml-cpp/emitter.h>
// #include <yaml-cpp/node/node.h>
// #include <yaml-cpp/node/parse.h>
// #include <yaml-cpp/yaml.h>
#include <ryml/ryml.hpp>
#include <ryml/ryml_std.hpp>

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
	for(;it_file!=it_end;++it_file)
	{
		const std::string full_name = it_file->path().filename().string();
		const std::string name = full_name.substr(0,8);
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

			if (std::find(existing_ids.begin(), existing_ids.end(), std::atoi(prefix.c_str()))!=existing_ids.end())
				continue;

			if (file_count > 2)
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

std::vector<std::string> split_string(const std::string& v_str, const std::string& v_splitter)
{
	std::vector<std::string> tokens;
	boost::split(tokens, v_str, boost::is_any_of(v_splitter));
	return tokens;
}

std::pair<std::vector<Point_3>, std::vector<std::vector<int>>> read_obj(const fs::path& obj_file)
{
	std::vector<Point_3> vertices;
	std::vector<std::vector<int>> faces;

	std::ifstream ifs(obj_file.string());
	std::string line;
	std::getline(ifs, line);
	while (line.size() > 3)
	{
		const auto tokens = split_string(line, " ");
		if (tokens[0] == "v")
		{
			vertices.emplace_back(
				std::stod(tokens[1]),
				std::stod(tokens[2]),
				std::stod(tokens[3])
			);
		}
		else if (tokens[0] == "f")
		{
			faces.emplace_back(std::vector<int>{
				std::atoi(tokens[1].substr(0, tokens[1].find_first_of("//")).c_str()) - 1,
					std::atoi(tokens[2].substr(0, tokens[2].find_first_of("//")).c_str()) - 1,
					std::atoi(tokens[3].substr(0, tokens[3].find_first_of("//")).c_str()) - 1
			});
		}
		std::getline(ifs, line);
	}

	ifs.close();
	return { vertices, faces };
}

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
	while (!queue.empty())
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

std::vector<double> calculate_aabb(const std::vector<Point_3>& v_points)
{
	double min_x = 99999, min_y = 99999, min_z = 99999;
	double max_x = -99999, max_y = -99999, max_z = -99999;
	for (const auto& item : v_points)
	{
		min_x = item.x() < min_x ? item.x() : min_x;
		min_y = item.y() < min_y ? item.y() : min_y;
		min_z = item.z() < min_z ? item.z() : min_z;

		max_x = item.x() > max_x ? item.x() : max_x;
		max_y = item.y() > max_y ? item.y() : max_y;
		max_z = item.z() > max_z ? item.z() : max_z;
	}

	double center_x = (min_x + max_x) / 2;
	double center_y = (min_y + max_y) / 2;
	double center_z = (min_z + max_z) / 2;
	double diag = std::sqrt(std::pow(max_x - min_x, 2) + std::pow(max_y - min_y, 2) + std::pow(max_z - min_z, 2));
	return { min_x,min_y,min_z,max_x,max_y,max_z,center_x,center_y,center_z,diag };
}


int main(int argc, char* argv[])
{
	argparse::ArgumentParser program("prepare_data_3d");
	{
		LOG(INFO) << "enter the arguments: data_root output_root num_cpus resolution id_start id_end is_log";
		program.add_description("data_root output_root resolution id_start id_end is_log");
		program.add_argument("data_root").required();
		program.add_argument("output_root").required();
		program.add_argument("id_start").required().scan<'i', int>();
		program.add_argument("id_end").required().scan<'i', int>();
		program.add_argument("--target_list").default_value(std::string(""));
		program.parse_args(argc, argv);
	}

	// tbb::global_control limit(tbb::global_control::max_allowed_parallelism, 1);

	fs::path data_root(program.get<std::string>("data_root"));
	fs::path output_root(program.get<std::string>("output_root"));
	const int id_start = program.get<int>("id_start");
	const int id_end = program.get<int>("id_end");
	std::string target_list = program.get<std::string>("--target_list");

	assert(id_end >= id_start);

	safeCheckFolder(output_root);
	safeCheckFolder(output_root/"mesh");
	safeCheckFolder(output_root/"yml");
	safeCheckFolder(output_root/"poisson");
	safeCheckFolder(output_root/"progress");
	safeCheckFolder(output_root/"bbox");

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

	std::atomic<int> progress(0);

	tbb::parallel_for(tbb::blocked_range<int>(0, task_files.size()),
		[&](const tbb::blocked_range<int>& r1)
		{
			for (int i_task = r1.begin(); i_task != r1.end(); ++i_task)
			{
				progress.fetch_add(1);
				if (progress.load() % 100 == 0)
					LOG(INFO) << ffmt("Progress: %d/%d") % progress.load() % task_files.size();

				std::string prefix = task_files[i_task].filename().string();
				int num_mesh_components;
				// Read and preprocess the mesh and primitives
				{
					auto timer_io = recordTime();
					fs::path obj_file, feature_file;
					for (fs::directory_iterator it_file(task_files[i_task]); it_file != fs::directory_iterator(); ++it_file)
					{
						const std::string name = it_file->path().string();
						if (name.find("features") != name.npos)
							feature_file = it_file->path();
						else if (name.find("trimesh") != name.npos)
							obj_file = it_file->path();
					}

					// Read obj
					std::vector<Point_3> total_vertices;
					std::vector<std::vector<int>> total_faces;

					std::tie(total_vertices, total_faces) = read_obj(obj_file);

					int num_vertices = total_vertices.size();
					int num_faces = total_faces.size();

					std::vector<int> id_vertex_component_maps(total_vertices.size(), -1); // which component does this vertex belong to
					std::vector<int> id_face_component_maps(total_faces.size(), -1); // which component does this face belong to

					// Perform DFS on the input mesh to split it into several components
					{
						auto adjacencyList = generateAdjacencyList(total_faces, num_vertices);
						std::vector<bool> visited(num_vertices, false);

						int componentId = 0;
						for (int v = 0; v < num_vertices; ++v) {
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

					std::vector<std::unordered_map<int, int>> to_original_Face_id(num_mesh_components);
					std::vector<int> to_new_vertex_id(num_vertices);
					std::vector<int> to_new_Face_id(num_faces);

					std::vector<std::vector<Point_3>> split_vertices(num_mesh_components);
					std::vector<std::vector<std::vector<int>>> split_triangles(num_mesh_components);
					// Compute id_face_component_maps and id_vertex_component_maps
					// Store triangles for each component
					{
						for (int i = 0; i < total_vertices.size(); ++i)
						{
							int component_id = id_vertex_component_maps[i];
							int current_vertex_num = split_vertices[component_id].size();
							split_vertices[component_id].push_back(total_vertices[i]);
							to_new_vertex_id[i] = current_vertex_num;
						}

						for (int i = 0; i < total_faces.size(); ++i)
						{
							int component_id = id_face_component_maps[i];
							int current_face_num = split_triangles[component_id].size();
							to_original_Face_id[component_id][current_face_num] = i;
							to_new_Face_id[i] = current_face_num;
							split_triangles[component_id].emplace_back(std::vector<int>{
								to_new_vertex_id[total_faces[i][0]],
									to_new_vertex_id[total_faces[i][1]],
									to_new_vertex_id[total_faces[i][2]]
							}
							);
						}
					}

					// Normalize each component
					tbb::parallel_for(tbb::blocked_range<int>(0, num_mesh_components),
						[&](const tbb::blocked_range<int>& r2)
						{
							for (int i_component = r2.begin(); i_component != r2.end(); ++i_component)
							{
								// min_x,min_y,min_z,max_x,max_y,max_z,center_x,center_y,center_z,diag
								const auto bounds = calculate_aabb(split_vertices[i_component]);
								for (auto& item : split_vertices[i_component])
									item = Point_3(
										(item.x() - bounds[6]) / bounds[9] * 2,
										(item.y() - bounds[7]) / bounds[9] * 2,
										(item.z() - bounds[8]) / bounds[9] * 2
									);

								std::vector<Triangle_3> triangles;
								for (auto& item : split_triangles[i_component])
									triangles.emplace_back(
										split_vertices[i_component][item[0]],
										split_vertices[i_component][item[1]],
										split_vertices[i_component][item[2]]
									);

								CGAL::IO::write_PLY(
									(output_root / "mesh" / (ffmt("%d_%d.ply") % prefix % i_component).str()).string(),
									split_vertices[i_component], split_triangles[i_component]);

								std::ofstream ofs((output_root / "bbox" / (ffmt("%d_%d.txt") % prefix % i_component).str()).string());
								Eigen::Vector4d bbox(bounds[6], bounds[7], bounds[8], bounds[9]);
								ofs << bbox;
								ofs.close();

								std::vector<double> poisson_set(10000 * 6);
								Point_set p = sample_poisson_points(triangles, 10000);

								CGAL::IO::write_point_set(
									(output_root / "poisson" / (ffmt("%d_%d.ply") % prefix % i_component).str()).string(),
									p);
							}

						}
					);

					// Read feature curves and surfaces
					std::ifstream ifs(feature_file.string());
					std::stringstream buffer;
					buffer << ifs.rdbuf();
					ifs.close();
					ryml::Tree config = ryml::parse_in_arena(ryml::to_csubstr(buffer.str()));
					// YAML::Node config = YAML::LoadFile(feature_file.string());
					std::vector<bool> curve_flags(config["curves"].num_children(), false);
					std::vector<bool> surface_flags(config["surfaces"].num_children(), false);

					const auto original_curves = config["curves"];
					const auto original_surfaces = config["surfaces"];

					tbb::parallel_for(tbb::blocked_range<int>(0, num_mesh_components),
						[&](const tbb::blocked_range<int>& r3)
						{
							for (int i_component = r3.begin(); i_component != r3.end(); ++i_component)
							{
								ryml::Tree new_tree = config;
								auto new_curves = new_tree["curves"];
								auto new_surfaces = new_tree["surfaces"];

								for (int i_curve = original_curves.num_children() - 1; i_curve >=0 ; --i_curve)
								{
									if (original_curves[i_curve]["vert_indices"].num_children() == 0)
										throw;
									int original_vert;
									original_curves[i_curve]["vert_indices"][0] >> original_vert;
									const int id_component = id_vertex_component_maps[original_vert];
									if (id_component != i_component)
									{
										new_curves.remove_child(i_curve);
									}
									else
									{
										if (curve_flags[i_curve])
											throw;
										curve_flags[i_curve] = true;
										for (int i_vert = 0; i_vert < original_curves[i_curve]["vert_indices"].num_children(); ++i_vert)
										{
											int original_vert;
											original_curves[i_curve]["vert_indices"][i_vert] >> original_vert;
											new_curves[i_curve]["vert_indices"][i_vert] << to_new_vertex_id[original_vert];
										}
									}
								}

								for (int i_surface = original_surfaces.num_children() - 1; i_surface >=0; --i_surface)
								{
									if (original_surfaces[i_surface]["face_indices"].num_children() == 0)
										throw;
									int original_face;
									original_surfaces[i_surface]["face_indices"][0] >> original_face;
									const int id_component = id_face_component_maps[original_face];
									if (id_component != i_component)
									{
										new_tree["surfaces"].remove_child(i_surface);
									}
									else
									{
										if (surface_flags[i_surface])
											throw;
										surface_flags[i_surface] = true;
										for (int i_face = 0; i_face < original_surfaces[i_surface]["face_indices"].num_children(); ++i_face)
										{
											int original_face;
											original_surfaces[i_surface]["face_indices"][i_face] >> original_face;
											new_surfaces[i_surface]["face_indices"][i_face] << to_new_Face_id[original_face];
										}
									}
								}
								// Write to file
								std::ofstream out((output_root / "yml" / (ffmt("%d_%d.yml") % prefix % i_component).str()).string());
								out << new_tree;
								out.close();
							}
						}
						);
					
					std::ofstream out((output_root / "progress" / (ffmt("%d.0") % prefix).str()).string());
					out << "";
					out.close();
				}
			}

		}
	);
	
	LOG(INFO) << "Done";
	return 0;
}
