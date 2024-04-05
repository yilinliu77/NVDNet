#include <tbb/tbb.h>

#include "cgal_tools.h"
#include "common_util.h"
#include "model_tools.h"

#include <unordered_set>
#include <argparse/argparse.hpp>
#include <CGAL/grid_simplify_point_set.h>

#include "kd_tree_helper.h"
#include <hungarian.h>

void debug_matched_shapes(const std::vector<Point_set>& pred_curves_split, const std::vector<Point_set>& gt_curves_split, 
	const Eigen::MatrixXd& distances)
{
	for (int i_pred = 0; i_pred < pred_curves_split.size(); ++i_pred)
	{
		Point_set p;
		auto red_map = p.add_property_map<uchar>("red", 0).first;
		auto green_map = p.add_property_map<uchar>("green", 0).first;
		auto blue_map = p.add_property_map<uchar>("blue", 0).first;
		for (int i = 0; i < pred_curves_split[i_pred].size(); ++i)
		{
			auto pp = p.insert(pred_curves_split[i_pred].point(i));
			red_map[*pp] = 255;
		}
		Eigen::RowVector2<Eigen::Index> idx;
		distances.row(i_pred).minCoeff(&idx.x(), &idx.y());
		for (int i = 0; i < gt_curves_split[idx[1]].size(); ++i)
			p.insert(gt_curves_split[idx[1]].point(i));

		CGAL::IO::write_point_set("temp.ply", p);
		LOG(INFO) << distances.row(i_pred).minCoeff();
	}
}

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

std::vector<Eigen::Vector2i> hungarian_match(const Eigen::MatrixXi& v_cost)
{
	const int num_pred = v_cost.rows();
	const int num_gt = v_cost.cols();
	// const int side_length = std::max(num_pred, num_gt);

	hungarian_problem_t problem;
	int** cost = new int* [num_pred];
	for (int i = 0; i < num_pred; ++i)
	{
		cost[i] = new int[num_gt];
		for (int j = 0; j < num_gt; ++j)
			if (i >= num_pred || j >= num_gt)
				cost[i][j] = 0;
			else
				cost[i][j] = v_cost(i, j);
	}

	int matrix_size = hungarian_init(&problem, cost, num_pred, num_gt, HUNGARIAN_MODE_MINIMIZE_COST);
	//hungarian_print_costmatrix(&problem);
	hungarian_solve(&problem);
	//hungarian_print_assignment(&problem);

	std::vector<Eigen::Vector2i> matched_index;
	for (int i_pred = 0; i_pred < num_pred; ++i_pred)
	{
		for (int j = 0; j < num_gt; ++j)	
			if (problem.assignment[i_pred][j] == 1)
				matched_index.emplace_back(i_pred, j);

	}
	hungarian_free(&problem);
	for (int i = 0; i < num_pred; ++i)
		free(cost[i]);
	free(cost);

	return matched_index;
}

#pragma optimize("", off)
Eigen::Vector3d pr_compute(const std::vector<Eigen::Vector2i>& matched_index, const Eigen::MatrixXd& v_cost, const double distance_threshold)
{
	int num_true_positive = 0;
	for (int i_match = 0; i_match < matched_index.size(); ++i_match)
	{
		const auto& index = matched_index[i_match];
		// LOG(INFO) << v_cost(index[0], index[1]);
		if (v_cost(index[0], index[1]) < distance_threshold)
		{
			num_true_positive++;
		}
	}
	const double precision = num_true_positive / (v_cost.rows() + 1e-6);
	const double recall = num_true_positive / (v_cost.cols() + 1e-6);
	const double f1 = 2 * precision * recall / (precision + recall + 1e-6);

	return { precision, recall, f1 };
}

inline
bool read_json_points(const fs::path& v_path, Point_set& corners, std::vector<Point_set>& curves, std::vector<Point_set>& surfaces)
{
	if (!fs::exists(v_path))
	{
		LOG(ERROR) << "File not exist: " << v_path;
		return false;
	}

	// load json file
	std::ifstream ifs(v_path.string());
	ifs.open(v_path.string());

	assert(ifs.is_open());

	Json::Reader reader;
	Json::Value root;
	if (reader.parse(ifs, root))
	{
		for (auto& corner : root["corners"])
		{
			int point_number = static_cast<int>(corner["pts"].size()) / 3;
			for (int i = 0; i < point_number; i++)
			{
				Point_3 p = Point_3(corner["pts"][i * 3].asDouble(), corner["pts"][i * 3 + 1].asDouble(), corner["pts"][i * 3 + 2].asDouble());
				corners.insert(p);
			}
		}

		for (auto& curve : root["curves"])
		{
			Point_set curve_points;
			int point_number = static_cast<int>(curve["pts"].size()) / 3;
			for (int i = 0; i < point_number; i++)
			{
				Point_3 p = Point_3(curve["pts"][i * 3].asDouble(), curve["pts"][i * 3 + 1].asDouble(), curve["pts"][i * 3 + 2].asDouble());
				curve_points.insert(p);
			}
			curves.emplace_back(curve_points);
		}
		for (auto& patch : root["patches"])
		{
			Point_set surface_points;
			int point_number = static_cast<int>(patch["grid"].size()) / 3;
			for (int i = 0; i < point_number; i++)
			{
				Point_3 p = Point_3(patch["grid"][i * 3].asDouble(), patch["grid"][i * 3 + 1].asDouble(), patch["grid"][i * 3 + 2].asDouble());
				surface_points.insert(p);
			}
			surfaces.emplace_back(surface_points);
		}
	}
	// LOG(INFO) << ffmt("Vertices: %d, Curves: %d, Surfaces: %d") % corners.size() % curves.size() % surfaces.size();

	return true;
}

struct MyColor
{
  unsigned int r, g, b;

  MyColor(unsigned int r, unsigned int g, unsigned int b) : r(r), g(g), b(b) {}

  bool operator==(const MyColor &other) const
  { 
	return (r == other.r && g == other.g && b == other.b);
  }
};

template <>
struct std::hash<MyColor>
{
  std::size_t operator()(const MyColor& color) const
  {
    return color.r ^ color.g ^ color.b;
  }
};

std::vector<double> calculate_aabb(const Point_set& v_points)
{
	double min_x = 99999, min_y = 99999, min_z = 99999;
	double max_x = -99999, max_y = -99999, max_z = -99999;
	for (auto it = v_points.begin(); it != v_points.end(); ++it)
	{
		auto& item = v_points.point(*it);
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


bool read_xyzc_points(const fs::path& v_path, Point_set& pred, std::vector<Point_set>& pred_split)
{
	std::fstream fin(v_path.string());

	std::string line;
	std::unordered_map<int, int> pred_primitive_index_map;
	while (std::getline(fin, line))
	{
		std::vector<std::string> data;
		boost::split(data, line, boost::is_any_of(" "));
		Point_3 p(
			std::stod(data[0]), 
			std::stod(data[1]), 
			std::stod(data[2])
			);
		int id = (int)std::stod(data[3]);
		if (pred_primitive_index_map.find(id) == pred_primitive_index_map.end())
		{
			pred_primitive_index_map[id] = pred_split.size();
			pred_split.push_back(Point_set());
		}
		pred_split[pred_primitive_index_map[id]].insert(p);
		pred.insert(p);
	}
	pred_primitive_index_map.clear();
	fin.close();

	return true;
}

void read_topology(const std::string& v_str,
	std::vector<std::vector<int>>& FE,
	std::vector<std::vector<int>>& EV
	)
{
	std::ifstream ifs(v_str);
	if (!ifs.is_open())
		return;
	while (!ifs.eof())
	{
		std::string line;
		std::getline(ifs, line);
		std::getline(ifs, line);
		while (line.substr(0, 2) != "EV")
		{
			std::vector<std::string> tokens;
			boost::split(tokens, line, boost::is_any_of(" "));
			const int cur_surface = std::stoi(tokens[0]);
			for (int i = 1; i < tokens.size(); ++i)
				FE[cur_surface].push_back(std::stoi(tokens[i]));
			std::getline(ifs, line);
		}
		std::getline(ifs, line);
		while (!line.empty())
		{
			std::vector<std::string> tokens;
			boost::split(tokens, line, boost::is_any_of(" "));
			const int cur_curve = std::stoi(tokens[0]);
			for (int i = 1; i < tokens.size(); ++i)
				EV[cur_curve].push_back(std::stoi(tokens[i]));
			std::getline(ifs, line);
		}
	}
	ifs.close();
}

int main(int argc, char* argv[])
{
	// tbb::global_control limit(tbb::global_control::max_allowed_parallelism, 1);
	argparse::ArgumentParser program("evaluate");
	{
		LOG(INFO) << "enter the arguments: data_root";
		program.add_argument("prediction_folder").required();
		program.add_argument("gt_folder").required();
		program.add_argument("output_folder").required();
		program.add_argument("--complex_gen_input").default_value(false).implicit_value(true);
		program.add_argument("--point2CAD_input").default_value(false).implicit_value(true);
		program.add_argument("--xyzc_input").default_value(false).implicit_value(true);
		program.add_argument("--chamfer").default_value(false).implicit_value(true);
		program.add_argument("--matched").default_value(false).implicit_value(true);
		program.add_argument("--topology").default_value(false).implicit_value(true);
		program.add_argument("--prefix").default_value(std::string(""));
		program.add_argument("--dist").default_value(0.1).scan<'g',double>();
		program.parse_args(argc, argv);
	}

	std::string one_prefix(program.get<std::string>("--prefix"));
	fs::path prediction_folder(program.get<std::string>("prediction_folder"));
	fs::path gt_folder(program.get<std::string>("gt_folder"));
	fs::path output_folder(program.get<std::string>("output_folder"));

	safeCheckFolder(output_folder);

	const bool is_complex_gen = program.get<bool>("--complex_gen_input");
	const bool is_point2CAD = program.get<bool>("--point2CAD_input");
	const bool is_xyzc = program.get<bool>("--xyzc_input");
	const bool is_chamfer = program.get<bool>("--chamfer");
	const bool is_matched = program.get<bool>("--matched");
	const bool is_topology = program.get<bool>("--topology");

	const double distance_threshold = program.get<double>("--dist");
	std::vector<double> distance_thresholds;
	if (distance_threshold < 0)
		distance_thresholds = std::vector<double>{ 0.005, 0.01, 0.02, 0.05, 0.1 };
	else
		distance_thresholds.push_back(distance_threshold);

	if (!fs::exists(prediction_folder) || !fs::exists(gt_folder))
	{
		LOG(ERROR) << "Input folders do not exist";
		LOG(ERROR) << prediction_folder;
		LOG(ERROR) << gt_folder;
		return 0;
	}

	std::vector<std::string> tasks;

	if (one_prefix.empty())
	{
		for (fs::directory_iterator it_file(gt_folder / "surfaces"); it_file != fs::directory_iterator(); ++it_file)
		{
			const std::string prefix = it_file->path().stem().string();	
			tasks.push_back(prefix);
		}
	}
	else
		tasks.push_back(one_prefix);

	std::vector<double> metrics(20, 0.); // vertex_cd, curve_cd, surface_cd, vertex_p, vertex_r, vertex_f, curve_p, curve_r, curve_f, surface_p, surface_r, surface_f, num_v, num_c, num_s, num_gt_v, num_gt_c, num_gt_s
	std::mutex mutex;
	int id_count = 0;
	tbb::parallel_for(tbb::blocked_range<int>(0, tasks.size()), [&](const tbb::blocked_range<int>& r0)
	{
		for(int i_task=r0.begin();i_task!=r0.end();++i_task)
		{
			const std::string& prefix = tasks[i_task];

			// TODO : if match.txt and chamfer.txt exist , then load data
			
			if (false && fs::exists((output_folder / (prefix + "_matched.txt"))) && fs::exists((output_folder / (prefix + "_cd.txt"))))
			{
				Eigen::Vector3d prf_vertices, prf_curves, prf_surfaces; // Precision and recall and f1-score
				prf_vertices.setConstant(0.);
				prf_curves.setConstant(0.);
				prf_surfaces.setConstant(0.);

				std::ifstream ifs((output_folder / (prefix + "_matched.txt")).string());

				if (ifs.is_open()) 
				{
					ifs >> prf_vertices.x() >> prf_vertices.y() >> prf_vertices.z();
					ifs >> prf_curves.x() >> prf_curves.y() >> prf_curves.z();
					ifs >> prf_surfaces.x() >> prf_surfaces.y() >> prf_surfaces.z();
					ifs.close();
				} 

				double vertices_chamfer, curves_chamfer, surfaces_chamfer;

				ifs = std::ifstream((output_folder / (prefix + "_cd.txt")).string());
				if (ifs.is_open())
				{
					ifs >> vertices_chamfer >> curves_chamfer >> surfaces_chamfer;
					ifs.close();
				}


				mutex.lock();
				metrics[0] += vertices_chamfer;
				metrics[1] += curves_chamfer;
				metrics[2] += surfaces_chamfer;
				LOG(INFO) << ffmt("%d: Vertices: %d; Curves: %d; Surfaces: %d") % prefix % vertices_chamfer % curves_chamfer % surfaces_chamfer;

				metrics[3] += prf_vertices.x();
				metrics[4] += prf_vertices.y();
				metrics[5] += prf_vertices.z();
				metrics[6] += prf_curves.x();
				metrics[7] += prf_curves.y();
				metrics[8] += prf_curves.z();
				metrics[9] += prf_surfaces.x();
				metrics[10] += prf_surfaces.y();
				metrics[11] += prf_surfaces.z();
				LOG(INFO) << ffmt("%d: precision: %d; recall: %d; f1: %d") % prefix %
					prf_vertices.x() % prf_vertices.y() % prf_vertices.z();
				LOG(INFO) << ffmt("%d: precision: %d; recall: %d; f1: %d") % prefix %
					prf_curves.x() % prf_curves.y() % prf_curves.z();
				LOG(INFO) << ffmt("%d: precision: %d; recall: %d; f1: %d") % prefix %
					prf_surfaces.x() % prf_surfaces.y() % prf_surfaces.z();
				mutex.unlock();
			}
			else
			{
				Point_set pred_vertices, pred_curves, pred_surfaces;
				Point_set gt_vertices, gt_curves, gt_surfaces;
				std::vector<Point_set> pred_curves_split, pred_surfaces_split;
				std::vector<Point_set> gt_curves_split, gt_surfaces_split;
				int num_pred_vertices = 0, num_gt_vertices = 0;
				int num_pred_curves = 0, num_gt_curves = 0;
				int num_pred_surfaces = 0, num_gt_surfaces = 0;

				// Load gt connectivity
				std::vector<std::vector<int>> gt_FE, gt_EV;
				// Load pred connectivity
				std::vector<std::vector<int>> pred_FE, pred_EV;

				// Load gt
				{
					read_points(gt_folder / "vertices" / (prefix + ".ply"), gt_vertices);
					read_points(gt_folder / "curves" / (prefix + ".ply"), gt_curves);
					read_points(gt_folder / "surfaces" / (prefix + ".ply"), gt_surfaces);
					num_gt_vertices = gt_vertices.size();

					// Curves
					auto gt_primitive_index = gt_curves.property_map<int>("primitive_index").first;
					for (int i_p = 0; i_p < gt_curves.size(); ++i_p)
						num_gt_curves = std::max(num_gt_curves,gt_primitive_index[i_p] + 1);

					gt_curves_split.resize(num_gt_curves);
					for (int i_p = 0; i_p < gt_curves.size(); ++i_p)
						gt_curves_split[gt_primitive_index[i_p]].insert(gt_curves.point(i_p));

					// Surfaces
					gt_primitive_index = gt_surfaces.property_map<int>("primitive_index").first;

					for (int i_p = 0; i_p < gt_surfaces.size(); ++i_p)
						num_gt_surfaces = std::max(num_gt_surfaces, gt_primitive_index[i_p] + 1);

					gt_surfaces_split.resize(num_gt_surfaces);
					for (int i_p = 0; i_p < gt_surfaces.size(); ++i_p)
						gt_surfaces_split[gt_primitive_index[i_p]].insert(gt_surfaces.point(i_p));

					// Check
					{
						if (num_gt_vertices == 0)
							std::cout << ffmt("%d num_gt_vertices=0") % prefix << std::endl;
						if (num_gt_curves == 0)
							std::cout << ffmt("%d num_gt_curves=0") % prefix << std::endl;
						if (num_gt_surfaces == 0)
							std::cout << ffmt("%d num_gt_surfaces=0") % prefix << std::endl;

						for (int ic=0;ic<num_gt_curves;++ic)
							if (gt_curves_split[ic].size() == 0)
								std::cout << ffmt("%d gt_curves_split[%d].size()=0") % prefix % ic << std::endl;

						for (int is=0;is<num_gt_surfaces;++is)
							if (gt_surfaces_split[is].size() == 0)
								std::cout << ffmt("%d gt_surfaces_split[%d].size()=0") % prefix % is << std::endl;
					}

					gt_FE.resize(num_gt_surfaces);
					gt_EV.resize(num_gt_curves);
					read_topology((gt_folder / "topology" / (prefix + ".txt")).string(), gt_FE, gt_EV);
				}
				
				// Old ComplexGen
				if (is_complex_gen)
				{
					// Before trim
					if (false)
					{
						CGAL::IO::read_point_set(
							(prediction_folder / "sample_on_geom_refine" / "vertices" / (ffmt("%d.ply") % prefix).str()).string(), pred_vertices);
						CGAL::IO::read_point_set(
							(prediction_folder / "sample_on_geom_refine" / "curves" / (ffmt("%d.ply") % prefix).str()).string(), pred_curves);
						CGAL::IO::read_point_set(
							(prediction_folder / "sample_on_geom_refine" / "surfaces" / (ffmt("%d.ply") % prefix).str()).string(), pred_surfaces);

						num_pred_vertices = pred_vertices.size();

						// Curves
						auto pred_primitive_index1 = pred_curves.normal_map();
						for (int i_p = 0; i_p < pred_curves.size(); ++i_p)
							if ((int)pred_primitive_index1[i_p][0] > num_pred_curves)
								num_pred_curves = (int)pred_primitive_index1[i_p][0];
						num_pred_curves += 1;

						pred_curves_split.resize(num_pred_curves);
						for (int i_p = 0; i_p < pred_curves.size(); ++i_p)
							pred_curves_split[(int)pred_primitive_index1[i_p][0]].insert(pred_curves.point(i_p));

						// Surfaces
						auto pred_primitive_index2 = pred_surfaces.normal_map();
						for (int i_p = 0; i_p < pred_surfaces.size(); ++i_p)
							if ((int)pred_primitive_index2[i_p][0] > num_pred_surfaces)
								num_pred_surfaces = (int)pred_primitive_index2[i_p][0];
						num_pred_surfaces += 1;

						pred_surfaces_split.resize(num_pred_surfaces);
						for (int i_p = 0; i_p < pred_surfaces.size(); ++i_p)
							pred_surfaces_split[(int)pred_primitive_index2[i_p][0]].insert(pred_surfaces.point(i_p));
					}
					// After trim
					else
					{
						// sample_on_geom_refine: normals is primitive index
						// sample_on_cut_grouped: has property map primitive index
						CGAL::IO::read_point_set(
							(prediction_folder / "sample_on_geom_refine" / "vertices" / (ffmt("%d.ply") % prefix).str()).string(), pred_vertices);
						CGAL::IO::read_point_set(
							(prediction_folder / "sample_on_geom_refine" / "curves" / (ffmt("%d.ply") % prefix).str()).string(), pred_curves);
						CGAL::IO::read_point_set(
							(prediction_folder / "sample_on_cut_grouped" / (ffmt("%d.ply") % prefix).str()).string(), pred_surfaces);

						num_pred_vertices = pred_vertices.size();

						// Curves
						auto pred_primitive_index1 = pred_curves.normal_map();

						for (int i_p = 0; i_p < pred_curves.size(); ++i_p)
							if ((int)pred_primitive_index1[i_p][0] > num_pred_curves)
								num_pred_curves = (int)pred_primitive_index1[i_p][0];
						num_pred_curves += 1;

						pred_curves_split.resize(num_pred_curves);
						for (int i_p = 0; i_p < pred_curves.size(); ++i_p)
							pred_curves_split[(int)pred_primitive_index1[i_p][0]].insert(pred_curves.point(i_p));

						// Surfaces
						auto pred_primitive_index = pred_surfaces.property_map<int>("primitive_index").first;
						for (int i_p = 0; i_p < pred_surfaces.size(); ++i_p)
							if (pred_primitive_index[i_p] > num_pred_surfaces)
								num_pred_surfaces = pred_primitive_index[i_p];
						num_pred_surfaces += 1;

						pred_surfaces_split.resize(num_pred_surfaces);
						for (int i_p = 0; i_p < pred_surfaces.size(); ++i_p)
							pred_surfaces_split[pred_primitive_index[i_p]].insert(pred_surfaces.point(i_p));

						pred_FE.resize(num_pred_surfaces);
						pred_EV.resize(num_pred_curves);
						read_topology((prediction_folder / "topo" / (ffmt("%d.txt") % prefix).str()).string(), pred_FE, pred_EV);
					}
				}
				else if (is_point2CAD)
				{
					// Load prediction
					read_points(prediction_folder / prefix / "clipped" / "remove_duplicates_corners.ply", pred_vertices);
					read_xyzc_points(prediction_folder / prefix / "clipped" / "curve_points.xyzc", pred_curves, pred_curves_split);
					read_points(prediction_folder / prefix / "clipped" / "mesh_transformed_sampled.ply", pred_surfaces);
					num_pred_vertices = pred_vertices.size();
					num_pred_curves = pred_curves_split.size();

					// Surfaces
					auto pred_primitive_index = pred_surfaces.property_map<int>("primitive_index").first;
					for (int i_p = 0; i_p < pred_surfaces.size(); ++i_p)
						if (pred_primitive_index[i_p] > num_pred_surfaces)
							num_pred_surfaces = pred_primitive_index[i_p];
					num_pred_surfaces += 1;

					pred_surfaces_split.resize(num_pred_surfaces);
					for (int i_p = 0; i_p < pred_surfaces.size(); ++i_p)
						pred_surfaces_split[pred_primitive_index[i_p]].insert(pred_surfaces.point(i_p));

					pred_FE.resize(num_pred_surfaces);
					pred_EV.resize(num_pred_curves);
					read_topology((prediction_folder / prefix / "topo" / "topo_fix.txt").string(), pred_FE, pred_EV);
				}
				else if (is_xyzc)
				{
					read_xyzc_points(prediction_folder / (prefix + ".xyzc"), pred_surfaces, pred_surfaces_split);
					num_pred_surfaces = pred_surfaces_split.size();
				}
				else
				{
					// Load prediction
					read_points(prediction_folder / prefix / "eval" / "vertices.ply", pred_vertices);
					read_points(prediction_folder / prefix / "eval" / "curves.ply", pred_curves);
					read_points(prediction_folder / prefix / "eval" / "surfaces.ply", pred_surfaces);

					num_pred_vertices = pred_vertices.size();

					// Curves
					auto pred_primitive_index = pred_curves.property_map<int>("primitive_index").first;
					for (int i_p = 0; i_p < pred_curves.size(); ++i_p)
						num_pred_curves = std::max(num_pred_curves, pred_primitive_index[i_p] + 1);

					pred_curves_split.resize(num_pred_curves);
					for (int i_p = 0; i_p < pred_curves.size(); ++i_p)
						pred_curves_split[pred_primitive_index[i_p]].insert(pred_curves.point(i_p));
			
					// Surfaces
					pred_primitive_index = pred_surfaces.property_map<int>("primitive_index").first;
					for (int i_p = 0; i_p < pred_surfaces.size(); ++i_p)
						num_pred_surfaces = std::max(num_pred_surfaces, pred_primitive_index[i_p] + 1);

					pred_surfaces_split.resize(num_pred_surfaces);
					for (int i_p = 0; i_p < pred_surfaces.size(); ++i_p)
						pred_surfaces_split[pred_primitive_index[i_p]].insert(pred_surfaces.point(i_p));


				// Filter out bank

					// Check
					{
						if (num_pred_vertices == 0)
							std::cout << ffmt("%d num_pred_vertices=0") % prefix << std::endl;
						if (num_pred_curves == 0)
							std::cout << ffmt("%d num_pred_curves=0") % prefix << std::endl;
						if (num_pred_surfaces == 0)
							std::cout << ffmt("%d num_pred_surfaces=0") % prefix << std::endl;

						for (int ic = 0; ic < num_pred_curves; ++ic)
							if (pred_curves_split[ic].empty())
								std::cout << ffmt("%d pred_curves_split[%d].size()=0") % prefix % ic << std::endl;

						for (int is = 0; is < num_pred_surfaces; ++is)
							if (pred_surfaces_split[is].empty())
								std::cout << ffmt("%d pred_surfaces_split[%d].size()=0") % prefix % is << std::endl;
					}

					if (is_topology)
					{
						pred_FE.resize(num_pred_surfaces);
						pred_EV.resize(num_pred_curves);
						read_topology((prediction_folder /prefix/ "eval" / "adj_matrix.txt").string(), pred_FE, pred_EV);
					}
				}
				
				// Filter out blank
				// Modify topological relationship correspondingly

				{
					for (int i_p = pred_surfaces_split.size() - 1; i_p >= 0; --i_p)
					{
						if (!pred_surfaces_split[i_p].empty())
							continue;
						LOG(ERROR) << "Pred Blank surface: " << i_p << " " << prefix;
						pred_surfaces_split.erase(pred_surfaces_split.begin() + i_p);
						if (is_topology)
							pred_FE.erase(pred_FE.begin() + i_p);
						num_pred_surfaces -= 1;
					}

					for (int i_c = pred_curves_split.size() - 1; i_c >= 0; --i_c)
					{
						if (!pred_curves_split[i_c].empty())
							continue;
						LOG(ERROR) << "Pred Blank curve: " << i_c << " " << prefix;
						pred_curves_split.erase(pred_curves_split.begin() + i_c);
						pred_EV.erase(pred_EV.begin() + i_c);
						for(auto& surface: pred_FE)
							for (int i = surface.size() - 1; i >= 0; --i)
							{
								if (surface[i] > i_c)
									surface[i] -= 1;
								else if (surface[i] == i_c)
									surface.erase(surface.begin() + i);
							}
						num_pred_curves -= 1;
					}

					// GT primitives will not have zero samples
				}

				// Build kdtree
				const auto kd_pred_vertices_data = initialize_kd_data(pred_vertices);
				std::shared_ptr<my_kd_tree_t> kd_pred_vertices(initialize_kd_tree(kd_pred_vertices_data));
				const auto kd_gt_vertices_data = initialize_kd_data(gt_vertices);
				std::shared_ptr<my_kd_tree_t> kd_gt_vertices(initialize_kd_tree(kd_gt_vertices_data));
				const auto kd_pred_curves_data = initialize_kd_data(pred_curves);
				std::shared_ptr<my_kd_tree_t> kd_pred_curves(initialize_kd_tree(kd_pred_curves_data));
				const auto kd_gt_curves_data = initialize_kd_data(gt_curves);
				std::shared_ptr<my_kd_tree_t> kd_gt_curves(initialize_kd_tree(kd_gt_curves_data));
				const auto kd_pred_surfaces_data = initialize_kd_data(pred_surfaces);
				std::shared_ptr<my_kd_tree_t> kd_pred_surfaces(initialize_kd_tree(kd_pred_surfaces_data));
				const auto kd_gt_surfaces_data = initialize_kd_data(gt_surfaces);
				std::shared_ptr<my_kd_tree_t> kd_gt_surfaces(initialize_kd_tree(kd_gt_surfaces_data));

				// Calculate chamfer
				if (is_chamfer)
				{
					double vertices_chamfer, curves_chamfer, surfaces_chamfer;

					// Vertices
					if (gt_vertices.empty())
						vertices_chamfer = 0.;
					else if (!pred_vertices.empty())
					{
						Eigen::VectorXd pred_distances(pred_vertices.size());
						pred_distances.setConstant(0.);

						Eigen::VectorXd gt_distances(gt_vertices.size());
						gt_distances.setConstant(0.);

						for (int i = 0; i < pred_vertices.size(); ++i)
							pred_distances[i] = std::sqrt(search_k_neighbour(*kd_gt_vertices, cgal_2_eigen_point<float>(pred_vertices.point(i)), 1).second[0]);
						for (int i = 0; i < gt_vertices.size(); ++i)
							gt_distances[i] = std::sqrt(search_k_neighbour(*kd_pred_vertices, cgal_2_eigen_point<float>(gt_vertices.point(i)), 1).second[0]);

						vertices_chamfer = pred_distances.mean() + gt_distances.mean();
					}
					else
					{
						vertices_chamfer = 1;
					}
					// Curves
					if (gt_curves.empty())
						curves_chamfer = 0.;
					else if (!pred_curves.empty())
					{
						Eigen::VectorXd pred_distances(pred_curves.size());
						pred_distances.setConstant(0.);

						Eigen::VectorXd gt_distances(gt_curves.size());
						gt_distances.setConstant(0.);

						for (int i = 0; i < pred_curves.size(); ++i)
							pred_distances[i] = std::sqrt(search_k_neighbour(*kd_gt_curves, cgal_2_eigen_point<float>(pred_curves.point(i)), 1).second[0]);
						for (int i = 0; i < gt_curves.size(); ++i)
							gt_distances[i] = std::sqrt(search_k_neighbour(*kd_pred_curves, cgal_2_eigen_point<float>(gt_curves.point(i)), 1).second[0]);

						curves_chamfer = pred_distances.mean() + gt_distances.mean();
					}
					else 
					{
						curves_chamfer = 1;
					}
					// Surfaces
					if (gt_surfaces.empty())
						surfaces_chamfer = 0.;
					else if (!pred_surfaces.empty())
					{
						Eigen::VectorXd pred_distances(pred_surfaces.size());
						pred_distances.setConstant(0.);

						Eigen::VectorXd gt_distances(gt_surfaces.size());
						gt_distances.setConstant(0.);

						for (int i = 0; i < pred_surfaces.size(); ++i)
							pred_distances[i] = std::sqrt(search_k_neighbour(*kd_gt_surfaces, cgal_2_eigen_point<float>(pred_surfaces.point(i)), 1).second[0]);
						for (int i = 0; i < gt_surfaces.size(); ++i)
							gt_distances[i] = std::sqrt(search_k_neighbour(*kd_pred_surfaces, cgal_2_eigen_point<float>(gt_surfaces.point(i)), 1).second[0]);

						surfaces_chamfer = pred_distances.mean() + gt_distances.mean();
					}
					else
					{
						surfaces_chamfer = 1;
					}

					std::ofstream ofs;
					ofs = std::ofstream((output_folder / (prefix + "_cd.txt")).string());

					// if (is_complex_gen)
					// 	ofs = std::ofstream((prediction_folder / (prefix+"_cd.txt")).string());
					// else
					// 	ofs = std::ofstream((prediction_folder / prefix / "eval" / "eval_cd.txt").string());
					ofs << vertices_chamfer << " " << curves_chamfer << " " << surfaces_chamfer << std::endl;
					ofs.close();

					mutex.lock();
					if (!one_prefix.empty())
						LOG(INFO) << ffmt("%d: Vertices: %d; Curves: %d; Surfaces: %d") % prefix % vertices_chamfer % curves_chamfer % surfaces_chamfer;
					metrics[0] += vertices_chamfer;
					metrics[1] += curves_chamfer;
					metrics[2] += surfaces_chamfer;
					mutex.unlock();
				}

				// Calculate matched loss
				if (is_matched)
				{
					// Compute match
					std::vector<Eigen::Vector2i> vertex_matches, curve_matches, surface_matches;
					Eigen::MatrixXd vertex_distances, curve_distances, surface_distances;
					// Vertices
					if (num_pred_vertices != 0 && num_gt_vertices !=0)
					{
						Eigen::MatrixXd distance_pred_2_gt(num_pred_vertices, num_gt_vertices);
						for (int i_pred_vertex = 0; i_pred_vertex < num_pred_vertices; ++i_pred_vertex)
						{
							for (int i_gt_vertex = 0; i_gt_vertex < num_gt_vertices; ++i_gt_vertex)
								distance_pred_2_gt(i_pred_vertex, i_gt_vertex) = std::sqrt(
									(gt_vertices.point(i_gt_vertex) - pred_vertices.point(i_pred_vertex)).squared_length());
						}

						vertex_distances = distance_pred_2_gt;
						Eigen::MatrixXi distances_int = (vertex_distances * 10000).cast<int>();
						vertex_matches = hungarian_match(distances_int);
					}

					// Curves
					if (num_pred_curves != 0 && num_gt_curves != 0)
					{
						std::vector<matrix_t> kd_pred_curves_split_data(num_pred_curves), kd_gt_curves_split_data(num_gt_curves);
						std::vector<std::shared_ptr<my_kd_tree_t>> kd_pred_curves_split(num_pred_curves), kd_gt_curves_split(num_gt_curves);
						for (int i_c = 0; i_c < pred_curves_split.size(); ++i_c)
						{
							kd_pred_curves_split_data[i_c] = initialize_kd_data(pred_curves_split[i_c]);
							kd_pred_curves_split[i_c] = initialize_kd_tree(kd_pred_curves_split_data[i_c]);
						}
						for (int i_c = 0; i_c < gt_curves_split.size(); ++i_c)
						{
							kd_gt_curves_split_data[i_c] = initialize_kd_data(gt_curves_split[i_c]);
							kd_gt_curves_split[i_c] = initialize_kd_tree(kd_gt_curves_split_data[i_c]);
						}

						Eigen::MatrixXd distance_pred_2_gt(num_pred_curves, num_gt_curves);
						distance_pred_2_gt.fill(0.);
						tbb::parallel_for(tbb::blocked_range<int>(0, num_pred_curves),
							[&](const tbb::blocked_range<int>& r)
							{
								for (int i_pred_curve = r.begin(); i_pred_curve != r.end(); ++i_pred_curve)
								{
									std::vector<double> dis(num_gt_curves, 0.);
									for (int i_point = 0; i_point < pred_curves_split[i_pred_curve].size(); ++i_point)
									{
										const auto& point = pred_curves_split[i_pred_curve].point(i_point);
										for (int i_gt_curve = 0; i_gt_curve < num_gt_curves; ++i_gt_curve)
											dis[i_gt_curve] += std::sqrt(search_k_neighbour(*kd_gt_curves_split[i_gt_curve], cgal_2_eigen_point<float>(point), 1).second[0]);
									}
									for (int i_gt_curve = 0; i_gt_curve < num_gt_curves; ++i_gt_curve)
									{
										distance_pred_2_gt(i_pred_curve, i_gt_curve) = dis[i_gt_curve] / (pred_curves_split[i_pred_curve].size() + 1e-6);
									}
								}
							});

						Eigen::MatrixXd distance_gt_2_pred(num_pred_curves, num_gt_curves);
						distance_gt_2_pred.fill(0.);
						tbb::parallel_for(tbb::blocked_range<int>(0, num_gt_curves),
							[&](const tbb::blocked_range<int>& r)
							{
								for (int i_gt_curve = r.begin(); i_gt_curve < r.end(); ++i_gt_curve)
								{
									std::vector<double> dis(num_pred_curves, 0.);
									for (int i_point = 0; i_point < gt_curves_split[i_gt_curve].size(); ++i_point)
									{
										const auto& point = gt_curves_split[i_gt_curve].point(i_point);
										for (int i_pred_curve = 0; i_pred_curve < num_pred_curves; ++i_pred_curve)
											dis[i_pred_curve] += std::sqrt(search_k_neighbour(*kd_pred_curves_split[i_pred_curve], cgal_2_eigen_point<float>(point), 1).second[0]);
									}
									for (int i_pred_curve = 0; i_pred_curve < num_pred_curves; ++i_pred_curve)
									{
										distance_gt_2_pred(i_pred_curve, i_gt_curve) = dis[i_pred_curve] / (gt_curves_split[i_gt_curve].size() + 1e-6);
									}
								}
							});

						curve_distances = (distance_pred_2_gt + distance_gt_2_pred) / 2;
						Eigen::MatrixXi distances_int = (curve_distances * 10000).cast<int>();
						curve_matches = hungarian_match(distances_int);
					}

					// Surfaces
					if (num_pred_surfaces != 0 && num_gt_surfaces != 0)
					{
						std::vector<matrix_t> kd_pred_surfaces_split_data(num_pred_surfaces), kd_gt_surfaces_split_data(num_gt_surfaces);
						std::vector<std::shared_ptr<my_kd_tree_t>> kd_pred_surfaces_split(num_pred_surfaces), kd_gt_surfaces_split(num_gt_surfaces);
						for (int i_c = 0; i_c < pred_surfaces_split.size(); ++i_c)
						{
							kd_pred_surfaces_split_data[i_c] = initialize_kd_data(pred_surfaces_split[i_c]);
							kd_pred_surfaces_split[i_c] = (initialize_kd_tree(kd_pred_surfaces_split_data[i_c]));
						}
						for (int i_c = 0; i_c < gt_surfaces_split.size(); ++i_c)
						{
							kd_gt_surfaces_split_data[i_c] = initialize_kd_data(gt_surfaces_split[i_c]);
							kd_gt_surfaces_split[i_c] = initialize_kd_tree(kd_gt_surfaces_split_data[i_c]);
						}

						Eigen::MatrixXd distance_pred_2_gt(num_pred_surfaces, num_gt_surfaces);
						distance_pred_2_gt.fill(0.);
						tbb::parallel_for(tbb::blocked_range<int>(0, num_pred_surfaces),
							[&](const tbb::blocked_range<int>& r)
							{
								for (int i_pred_surface = r.begin(); i_pred_surface != r.end(); ++i_pred_surface)
								{
									std::vector<double> dis(num_gt_surfaces, 0.);
									for (int i_point = 0; i_point < pred_surfaces_split[i_pred_surface].size(); ++i_point)
									{
										const auto& point = pred_surfaces_split[i_pred_surface].point(i_point);
										for (int i_gt_surface = 0; i_gt_surface < num_gt_surfaces; ++i_gt_surface)
											dis[i_gt_surface] += std::sqrt(search_k_neighbour(*kd_gt_surfaces_split[i_gt_surface], cgal_2_eigen_point<float>(point), 1).second[0]);
									}
									for (int i_gt_curve = 0; i_gt_curve < num_gt_surfaces; ++i_gt_curve)
									{
										distance_pred_2_gt(i_pred_surface, i_gt_curve) = dis[i_gt_curve] / (pred_surfaces_split[i_pred_surface].size() + 1e-6);
									}
								}
							});


						Eigen::MatrixXd distance_gt_2_pred(num_pred_surfaces, num_gt_surfaces);
						distance_gt_2_pred.fill(0.);
						tbb::parallel_for(tbb::blocked_range<int>(0, num_gt_surfaces),
							[&](const tbb::blocked_range<int>& r)
							{
								for (int i_gt_surface = r.begin(); i_gt_surface != r.end(); ++i_gt_surface)
								{
									std::vector<double> dis(num_pred_surfaces, 0.);
									for (int i_point = 0; i_point < gt_surfaces_split[i_gt_surface].size(); ++i_point)
									{
										const auto& point = gt_surfaces_split[i_gt_surface].point(i_point);
										for (int i_pred_surface = 0; i_pred_surface < num_pred_surfaces; ++i_pred_surface)
											dis[i_pred_surface] += std::sqrt(search_k_neighbour(*kd_pred_surfaces_split[i_pred_surface], cgal_2_eigen_point<float>(point), 1).second[0]);
									}
									for (int i_pred_surface = 0; i_pred_surface < num_pred_surfaces; ++i_pred_surface)
									{
										distance_gt_2_pred(i_pred_surface, i_gt_surface) = dis[i_pred_surface] / (gt_surfaces_split[i_gt_surface].size() + 1e-6);
									}
								}
							});
						surface_distances = (distance_gt_2_pred + distance_pred_2_gt) / 2;
						Eigen::MatrixXi distances_int = (surface_distances * 10000).cast<int>();
						surface_matches = hungarian_match(distances_int);
					}

					Eigen::Vector3d prf_vertices, prf_curves, prf_surfaces; // Precision and recall and f1-score
					prf_vertices.setConstant(0.);
					prf_curves.setConstant(0.);
					prf_surfaces.setConstant(0.);
					double f1_FE = 0, f1_EV = 0;
					for(int i_thresh=0;i_thresh<distance_thresholds.size();++i_thresh)
					{
						// Vertices
						{
							if (num_gt_vertices == 0)
								prf_vertices = Eigen::Vector3d(1., 1., 1.);
							else if (num_pred_vertices == 0)
								prf_vertices = Eigen::Vector3d(0., 0., 0.);
							else
								prf_vertices += pr_compute(vertex_matches, vertex_distances, distance_thresholds[i_thresh]);
						}
						// Curves
						{
							if (num_gt_curves == 0)
								prf_curves = Eigen::Vector3d(1., 1., 1.);
							else if (num_pred_curves == 0)
								prf_curves = Eigen::Vector3d(0., 0., 0.);
							else
							{
								Eigen::MatrixXd distances;
								if (is_complex_gen && false)
								{
									std::vector<std::shared_ptr<my_kd_tree_t>> kd_pred_curves_split(num_pred_curves), kd_gt_curves_split(num_gt_curves);
									for (int i_c = 0; i_c < pred_curves_split.size(); ++i_c)
										kd_pred_curves_split[i_c].reset(initialize_kd_tree(pred_curves_split[i_c]));
									for (int i_c = 0; i_c < gt_curves_split.size(); ++i_c)
										kd_gt_curves_split[i_c].reset(initialize_kd_tree(gt_curves_split[i_c]));
									Eigen::MatrixXd distance_gt_2_pred(num_pred_curves, num_gt_curves);
									distance_gt_2_pred.fill(0.);
									tbb::parallel_for(tbb::blocked_range<int>(0, num_gt_curves),
										[&](const tbb::blocked_range<int>& r)
										{
											for (int i_gt_curve = r.begin(); i_gt_curve < r.end(); ++i_gt_curve)
											{
												std::vector<double> dis(num_pred_curves, 0.);
												for (int i_point = 0; i_point < gt_curves_split[i_gt_curve].size(); ++i_point)
												{
													const auto& point = gt_curves_split[i_gt_curve].point(i_point);
													for (int i_pred_curve = 0; i_pred_curve < num_pred_curves; ++i_pred_curve)
														dis[i_pred_curve] += std::sqrt(search_k_neighbour(*kd_pred_curves_split[i_pred_curve], cgal_2_eigen_point<float>(point), 1).second[0]);
												}
												for (int i_pred_curve = 0; i_pred_curve < num_pred_curves; ++i_pred_curve)
												{
													distance_gt_2_pred(i_pred_curve, i_gt_curve) = dis[i_pred_curve] / (gt_curves_split[i_gt_curve].size() + 1e-6);
												}
											}
										});
									distances = distance_gt_2_pred;
								}
								prf_curves += pr_compute(curve_matches, curve_distances, distance_thresholds[i_thresh]);
							}
						}
						// Surfaces
						{
							if (num_gt_surfaces == 0)
								prf_surfaces = Eigen::Vector3d(1., 1., 1.);
							else if (num_pred_surfaces == 0)
								prf_surfaces = Eigen::Vector3d(0., 0., 0.);
							else
							{
								if (is_complex_gen && false)
								{
									std::vector<std::shared_ptr<my_kd_tree_t>> kd_pred_surfaces_split(num_pred_surfaces), kd_gt_surfaces_split(num_gt_surfaces);
									for (int i_c = 0; i_c < pred_surfaces_split.size(); ++i_c)
										kd_pred_surfaces_split[i_c].reset(initialize_kd_tree(pred_surfaces_split[i_c]));
									for (int i_c = 0; i_c < gt_surfaces_split.size(); ++i_c)
										kd_gt_surfaces_split[i_c].reset(initialize_kd_tree(gt_surfaces_split[i_c]));

									Eigen::MatrixXd distance_gt_2_pred(num_pred_surfaces, num_gt_surfaces);
									tbb::parallel_for(tbb::blocked_range<int>(0, num_gt_surfaces),
										[&](const tbb::blocked_range<int>& r)
										{
											for (int i_gt_surface = r.begin(); i_gt_surface != r.end(); ++i_gt_surface)
											{
												std::vector<double> dis(num_pred_surfaces, 0.);
												for (int i_point = 0; i_point < gt_surfaces_split[i_gt_surface].size(); ++i_point)
												{
													const auto& point = gt_surfaces_split[i_gt_surface].point(i_point);
													for (int i_pred_surface = 0; i_pred_surface < num_pred_surfaces; ++i_pred_surface)
														dis[i_pred_surface] += std::sqrt(search_k_neighbour(*kd_pred_surfaces_split[i_pred_surface], cgal_2_eigen_point<float>(point), 1).second[0]);
												}
												for (int i_pred_surface = 0; i_pred_surface < num_pred_surfaces; ++i_pred_surface)
												{
													distance_gt_2_pred(i_pred_surface, i_gt_surface) = dis[i_pred_surface] / (gt_surfaces_split[i_gt_surface].size() + 1e-6);
												}
											}
										});
									// distances = distance_gt_2_pred;
								}
								prf_surfaces += pr_compute(surface_matches, surface_distances, distance_thresholds[i_thresh]);
							}
						}
					}

					prf_vertices /= distance_thresholds.size();
					prf_curves /= distance_thresholds.size();
					prf_surfaces /= distance_thresholds.size();

					if (is_topology)
					{
						std::vector<int> pv_to_gv(num_pred_vertices, -1), pc_to_gc(num_pred_curves, -1), ps_to_gs(num_pred_surfaces, -1);
						std::vector<int> gv_to_pv(num_gt_vertices, -1), gc_to_pc(num_gt_curves, -1), gs_to_ps(num_gt_surfaces, -1);

						for(const auto& item: vertex_matches)
						{
							pv_to_gv[item.x()] = item.y();
							gv_to_pv[item.y()] = item.x();
						}
						for (const auto& item : curve_matches)
						{
							pc_to_gc[item.x()] = item.y();
							gc_to_pc[item.y()] = item.x();
						}
						for (const auto& item: surface_matches)
						{
							ps_to_gs[item.x()] = item.y();
							gs_to_ps[item.y()] = item.x();
						}

						// FE
						{
							int true_count = 0;
							for (int i_surface = 0; i_surface < pred_FE.size(); ++i_surface)
							{
								for (int i_curve = 0; i_curve < pred_FE[i_surface].size(); ++i_curve)
								{
									if (ps_to_gs[i_surface] == -1 || pc_to_gc[pred_FE[i_surface][i_curve]] == -1)
										continue;
									int i_gt_surface = ps_to_gs[i_surface];
									int i_gt_curve = pc_to_gc[pred_FE[i_surface][i_curve]];
									if (std::find(gt_FE[i_gt_surface].begin(), gt_FE[i_gt_surface].end(), i_gt_curve) == gt_FE[i_gt_surface].end())
										continue;
									true_count++;
								}
							}

							int total_pred_relation = 0, total_gt_relation = 0;
							for (const auto& item : pred_FE)
								total_pred_relation += item.size();
							for (const auto& item : gt_FE)
								total_gt_relation += item.size();

							const double precision = true_count / (total_pred_relation+1e-6);
							const double recall = true_count / (total_gt_relation+1e-6);
							f1_FE = 2 * precision * recall / (precision + recall + 1e-6);
						}

						// EV
						{
							int true_count = 0;
							for (int i_curve = 0; i_curve < pred_EV.size(); ++i_curve)
							{
								for (int i_vertex = 0; i_vertex < pred_EV[i_curve].size(); ++i_vertex)
								{
									if (pc_to_gc[i_curve] == -1 || pv_to_gv[pred_EV[i_curve][i_vertex]] == -1)
										continue;
									int i_gt_curve = pc_to_gc[i_curve];
									int i_gt_vertex = pv_to_gv[pred_EV[i_curve][i_vertex]];
									if (std::find(gt_EV[i_gt_curve].begin(), gt_EV[i_gt_curve].end(), i_gt_vertex) == gt_EV[i_gt_curve].end())
										continue;
									true_count++;
								}
							}

							int total_pred_relation = 0, total_gt_relation = 0;
							for (const auto& item : pred_EV)
								total_pred_relation += item.size();
							for (const auto& item : gt_EV)
								total_gt_relation += item.size();

							const double precision = true_count / (total_pred_relation+1e-6);
							const double recall = true_count / (total_gt_relation+1e-6);
							f1_EV = 2 * precision * recall / (precision + recall + 1e-6);
						}

					}

					mutex.lock();
					metrics[3] += prf_vertices.x();
					metrics[4] += prf_vertices.y();
					metrics[5] += prf_vertices.z();
					metrics[6] += prf_curves.x();
					metrics[7] += prf_curves.y();
					metrics[8] += prf_curves.z();
					metrics[9] += prf_surfaces.x();
					metrics[10] += prf_surfaces.y();
					metrics[11] += prf_surfaces.z();
					metrics[18] += f1_FE;
					metrics[19] += f1_EV;

					if (!one_prefix.empty())
					{
						LOG(INFO) << ffmt("%d: Vertices: num_pred: %d; num_gt: %d; precision: %d; recall: %d; f1: %d") % prefix %
							num_pred_vertices % num_gt_vertices % prf_vertices.x() % prf_vertices.y() % prf_vertices.z();
						LOG(INFO) << ffmt("%d: Curves: num_pred: %d; num_gt: %d; precision: %d; recall: %d; f1: %d") % prefix %
							num_pred_curves % num_gt_curves % prf_curves.x() % prf_curves.y() % prf_curves.z();
						LOG(INFO) << ffmt("%d: Surfaces: num_pred: %d; num_gt: %d; precision: %d; recall: %d; f1: %d") % prefix %
							num_pred_surfaces % num_gt_surfaces % prf_surfaces.x() % prf_surfaces.y() % prf_surfaces.z();
						LOG(INFO) << ffmt("%d: FE: %d; EV: %d") % prefix %
							f1_FE % f1_EV;
					}
					mutex.unlock();

					std::ofstream ofs;
					ofs = std::ofstream((output_folder / (prefix + "_matched.txt")).string());

					ofs << prf_vertices.x() << " " << prf_vertices.y() << " " << prf_vertices.z() << std::endl;
					ofs << prf_curves.x() << " " << prf_curves.y() << " " << prf_curves.z() << std::endl;
					ofs << prf_surfaces.x() << " " << prf_surfaces.y() << " " << prf_surfaces.z() << std::endl;
					ofs.close();
				}

				mutex.lock();
				metrics[12] += num_pred_vertices;
				metrics[13] += num_pred_curves;
				metrics[14] += num_pred_surfaces;
				metrics[15] += num_gt_vertices;
				metrics[16] += num_gt_curves;
				metrics[17] += num_gt_surfaces;
				id_count++;
				if (id_count % 100 == 0)
					LOG(INFO) << ffmt("%d/%d") % id_count % tasks.size();
				mutex.unlock();
			}
		}
	});

	for(auto& item: metrics)
		item /= tasks.size();

	LOG(INFO) << ffmt("%d tasks; Vcd=%.3f; Ccd=%.3f; Scd=%.3f; Vp=%.3f; Vr=%.3f; Vf=%.3f; Cp=%.3f; Cr=%.3f; Cf=%.3f; Sp=%.3f; Sr=%.3f; Sf=%.3f") %
		tasks.size() % metrics[0] % metrics[1] % metrics[2] % metrics[3] % metrics[4] % metrics[5] % metrics[6] % metrics[7] % metrics[8] % metrics[9] % metrics[10] % metrics[11];
	LOG(INFO) << ffmt("%.5f %.5f %.5f %.1f %.1f %.1f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f") %
		metrics[0] % metrics[1] % metrics[2] % metrics[12] % metrics[13] % metrics[14] % metrics[5] % metrics[8] % metrics[11] % metrics[3] % metrics[6] % metrics[9] % metrics[4] % metrics[7] % metrics[10];
	LOG(INFO) << ffmt("GT numbers: %.3f %.3f %.3f") %
		metrics[15] % metrics[16] % metrics[17];
	LOG(INFO) << ffmt("FE: %.3f; EV: %.3f") %
		metrics[18] % metrics[19];

	LOG(INFO) << "Done";
	return 0;
}
