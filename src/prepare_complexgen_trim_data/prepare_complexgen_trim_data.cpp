#include <cuda_runtime_api.h>
#include <tbb/tbb.h>

#include "cgal_tools.h"
#include "common_util.h"
#include <argparse/argparse.hpp>

#include <unordered_set>
#include <CGAL/Polygon_mesh_processing/bbox.h>

#include "model_tools.h"

bool ends_with(std::string const& value, std::string const& ending)
{
	if (ending.size() > value.size()) return false;
	return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

std::vector<fs::path> generate_task_list(const fs::path& v_data_input, const int id_start, const int id_end)
{
	std::vector<fs::path> task_files;

	std::vector<std::string> target_prefix;

	for (fs::directory_iterator it_file(v_data_input); it_file != fs::directory_iterator(); ++it_file)
	{
		if (!ends_with(it_file->path().filename().string(), "_cut_grouped.obj"))
			continue;
		const int file_idx = std::atoi(it_file->path().filename().string().substr(0, 8).c_str());
		if (file_idx >= id_start && file_idx < id_end)
		{
			target_prefix.push_back(it_file->path().filename().string());
		}
	}
		
	std::unordered_set<std::string> target_set(target_prefix.begin(), target_prefix.end());
	target_prefix.clear();
	target_prefix.insert(target_prefix.end(), target_set.begin(), target_set.end());

	for (auto& prefix : target_prefix)
	{
		task_files.push_back(v_data_input / prefix);
	}

	return task_files;
}

inline
bool my_obj_loader(const std::string& file, std::vector<Point_3>& points, 
	std::vector<std::vector<std::vector<std::size_t>>>& faces)
{
	std::ifstream ifs(file, std::ios::binary);
	if (!ifs.is_open()) {
		LOG(ERROR) << ffmt("Unable to open the file: %s") % file;
		return false;
	}

	std::string line, s;
	Point_3 p;
	while (std::getline(ifs, line))
	{
		if (line.empty())
			continue;
		std::istringstream iss(line);
		if (!(iss >> s))
			continue;
		if (s == "v")
		{
			if (!(iss >> p))
			{
				LOG(ERROR) << "Error while reading OBJ vertex";
				return false;
			}

			points.push_back(p);
		}
		else if (s == "f")
		{
			std::vector<std::size_t> face;
			std::size_t idx;
			while (iss >> idx)
			{
				face.push_back(idx - 1);
			}
			faces.back().push_back(face);
		}
		else if (s == "g")
		{
			faces.push_back({});
		}
	}

	ifs.close();

	return true;
}

void scale_point_set(const std::string& file_idx, Point_set& point_set, const fs::path& v_gt_root)
{
	Point_set gt_point_set;
	CGAL::IO::read_point_set((v_gt_root / "surfaces" / (file_idx + ".ply")).string(), gt_point_set);


	std::vector<double> max_coord{ 0, 0, 0 }, min_coord{ 9999, 9999, 9999 };
	for (int i_p = 0; i_p < gt_point_set.size(); ++i_p)
	{
		auto& p = gt_point_set.point(i_p);

		max_coord[0] = std::max(max_coord[0], p.x());
		max_coord[1] = std::max(max_coord[1], p.y());
		max_coord[2] = std::max(max_coord[2], p.z());

		min_coord[0] = std::min(min_coord[0], p.x());
		min_coord[1] = std::min(min_coord[1], p.y());
		min_coord[2] = std::min(min_coord[2], p.z());
	}

	double scale = std::max(max_coord[0] - min_coord[0], std::max(max_coord[1] - min_coord[1], max_coord[2] - min_coord[2]));
	auto translate = Vector_3(
		(max_coord[0] + min_coord[0]) / 2,
		(max_coord[1] + min_coord[1]) / 2,
		(max_coord[2] + min_coord[2]) / 2
	);

	for (auto it = point_set.begin(); it != point_set.end(); ++it)
	{
		auto& p = point_set.point(*it);
		p = Point_3(
			p.x() * scale + translate.x(),
			p.y() * scale + translate.y(), 
			p.z() * scale + translate.z()
		);
	}
}

int main(int argc, char* argv[])
{
	// tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);
	argparse::ArgumentParser program("prepare_complexgen_trim_data");
	{
		LOG(INFO) << "enter the arguments: data_root";
		program.add_description("data_root");
		program.add_argument("data_root").required();
		program.add_argument("output_root").required();
		program.add_argument("gt_root").required();
		program.add_argument("test_ids_root").required();
		program.add_argument("--prefix").default_value(std::string(""));
		program.parse_args(argc, argv);
	}

	fs::path data_root(program.get<std::string>("data_root"));
	fs::path output_root(program.get<std::string>("output_root"));
	fs::path gt_root(program.get<std::string>("gt_root"));
	fs::path test_ids_root(program.get<std::string>("test_ids_root"));
	std::string prefix = program.get<std::string>("--prefix");



	safeCheckFolder(output_root);

	std::vector<fs::path> task_files;
	if (prefix.empty())
	{
		std::ifstream ifs(test_ids_root.string());

		std::string line;
		while (std::getline(ifs, line))
		{
			if (line.empty())
				continue;
			task_files.push_back(data_root / (line + "_extraction_cut_grouped.obj"));
		}
	}
	else
		task_files.push_back(data_root / (prefix + "_extraction_cut_grouped.obj"));
	
	LOG(INFO) << ffmt("We have %d valid task") % task_files.size();
	if (task_files.empty())
		return 0;


	//#pragma omp parallel for num_threads(omp_get_num_procs() / 2)
	for (int i = 0; i < task_files.size(); ++i)
	{
		auto& file = task_files[i];
		std::string file_idx = file.filename().string().substr(0, 8);
		std::ifstream ifs(file.string());

		// load group of obj
		std::vector<Point_3> points;
		std::vector<std::vector<std::vector<std::size_t> >> faces_ref;
		my_obj_loader(file.string(), points, faces_ref);
		LOG(INFO) << ffmt("Loaded %d groups of obj from % s") % faces_ref.size() % file.string();

		Point_set output_points;
		auto primitive_index = output_points.add_property_map<int>("primitive_index", 0).first;
		for (int i_group = 0; i_group < faces_ref.size(); ++i_group)
		{
			auto& faces = faces_ref[i_group];
			// sample points
			std::vector<Triangle_3> triangles(faces.size());
			for (int i_face = 0; i_face < faces.size(); ++i_face)
			{
				auto& face = faces[i_face];
				triangles[i_face] = Triangle_3(
					points[face[0]],
					points[face[1]],
					points[face[2]]
				);
			}

			Point_set surface_points = sample_points_according_density(triangles, 10000);

			for (auto it = surface_points.begin(); it != surface_points.end(); ++it)
			{
				auto p = output_points.insert(surface_points.point(*it));
				primitive_index[*p] = i_group;
			}
		}
		scale_point_set(file_idx, output_points, gt_root);
		auto output_path = output_root / (file_idx + ".ply");
		CGAL::IO::write_point_set(output_path.string(), output_points);

		//Point_set test;
		//CGAL::IO::read_point_set(output_path.string(), test);
		//LOG(INFO) << test.property_map<int>("primitive_index").second;
	}

	return 0;	
}
