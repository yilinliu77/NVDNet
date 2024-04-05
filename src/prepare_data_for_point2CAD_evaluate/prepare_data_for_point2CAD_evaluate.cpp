#include <tbb/tbb.h>

#include "cgal_tools.h"
#include "common_util.h"
#include <argparse/argparse.hpp>

#include <unordered_set>

#include "model_tools.h"
#include "kd_tree_helper.h"

void colorize_point_set(Point_set& v_points, const std::string& v_name)
{
	const auto index_map = v_points.property_map<int>(v_name).first;
	if (index_map == nullptr)
	{
		LOG(ERROR) << "Cannot find index map";
		return;
	}
	const auto color_map = get_color_table_bgr2();
	auto rmap = v_points.add_property_map<uchar>("red").first;
	auto gmap = v_points.add_property_map<uchar>("green").first;
	auto bmap = v_points.add_property_map<uchar>("blue").first;
	for (int i = 0; i < v_points.size(); ++i)
	{
		rmap[i] = color_map[index_map[i] % color_map.size()][2];
		gmap[i] = color_map[index_map[i] % color_map.size()][1];
		bmap[i] = color_map[index_map[i] % color_map.size()][0];
	}
}

Point_set sample_points(const std::vector<Triangle_3>& v_mesh, const int v_num_points, const std::vector<int>& primitive_index)
{
	std::mt19937 gen; std::uniform_real_distribution<double> dist(0.0f, 1.0f);
	Point_set o_point_set(true);
	auto index_map = o_point_set.add_property_map<int>("face_index", 0).first;
	auto primitive_map = o_point_set.add_property_map<int>("primitive_index", -1).first;

	double total_area = 0.;
	// #pragma omp parallel for reduction(+:total_area)
	for (int i_face = 0; i_face < v_mesh.size(); ++i_face)
		total_area += std::sqrt(v_mesh[i_face].squared_area());

	double point_per_area = (double)v_num_points / total_area;

	// #pragma omp parallel for
	for (int i_face = 0; i_face < v_mesh.size(); ++i_face)
	{
		Point_3 vertexes[3];
		vertexes[0] = v_mesh[i_face].vertex(0);
		vertexes[1] = v_mesh[i_face].vertex(1);
		vertexes[2] = v_mesh[i_face].vertex(2);


		Vector_3 normal = CGAL::cross_product(vertexes[1] - vertexes[0], vertexes[2] - vertexes[0]);
		normal /= std::sqrt(normal.squared_length());

		double area = std::sqrt(v_mesh[i_face].squared_area());

		double face_samples = area * point_per_area;
		uint num_face_samples = face_samples;

		if (dist(gen) < (face_samples - static_cast<double>(num_face_samples))) {
			num_face_samples += 1;
		}

		for (uint j = 0; j < num_face_samples; ++j) {
			double r1 = dist(gen);
			double r2 = dist(gen);

			double tmp = std::sqrt(r1);
			double u = 1.0f - tmp;
			double v = r2 * tmp;

			double w = 1.0f - v - u;
			auto point = Point_3(
				u * vertexes[0].x() + v * vertexes[1].x() + w * vertexes[2].x(),
				u * vertexes[0].y() + v * vertexes[1].y() + w * vertexes[2].y(),
				u * vertexes[0].z() + v * vertexes[1].z() + w * vertexes[2].z()
			);
			// #pragma omp critical
			{
				auto it = *o_point_set.insert(point, normal);
				index_map[it] = i_face;
				primitive_map[it] = primitive_index[i_face];
			}
		}
	}
	return o_point_set;
}


Point_set sample_points_according_density(const std::vector<Triangle_3>& v_mesh, const float v_num_points_per_m2, const std::vector<int>& primitive_index)
{
	double total_area = 0.;
#pragma omp parallel for reduction(+:total_area)
	for (int i_face = 0; i_face < v_mesh.size(); ++i_face)
		total_area += std::sqrt(v_mesh[i_face].squared_area());
	return sample_points(v_mesh, std::ceil(v_num_points_per_m2 * total_area), primitive_index);
}

struct Hash {
	std::size_t operator()(const CGAL::Color& color) const {
		std::size_t hash_value = 0;
		std::vector<unsigned char> data;
		data.push_back(color.red());
		data.push_back(color.green());
		data.push_back(color.blue());
		for (const auto& component : data) {
			hash_value ^= std::hash<unsigned char>()(component) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
		}
		return hash_value;
	}
};


int main(int argc, char* argv[])
{
	// tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);
	argparse::ArgumentParser program("prepare_data_for_point2CAD_evaluate");
	{
		LOG(INFO) << "enter the arguments: data_root";
		program.add_description("data_root");
		program.add_argument("data_root").required();
		program.add_argument("output_root").required();
		program.add_argument("color_root").required();
		program.add_argument("--prefix").default_value(std::string(""));
		program.parse_args(argc, argv);
	}

	std::string prefix = program.get<std::string>("--prefix");

	fs::path data_root(program.get<std::string>("data_root"));
	fs::path output_root(program.get<std::string>("output_root"));
	fs::path color_root(program.get<std::string>("color_root"));

	safeCheckFolder(output_root);

	std::vector<std::pair<fs::path, std::string>> task_files;
	if (prefix.empty())
	{
		for (fs::directory_iterator it_file(data_root); it_file != fs::directory_iterator(); ++it_file)
			task_files.push_back(std::make_pair(it_file->path() / "clipped" / "mesh_transformed.ply", it_file->path().filename().string()));
		
	}
	else
		task_files.push_back(std::make_pair(data_root / prefix / "clipped" / "mesh_transformed.ply", prefix));

	std::ifstream fin((color_root / "color.txt").string());
	std::unordered_map<CGAL::Color, int, Hash> color_map;

	std::string line;
	while (std::getline(fin, line))
	{
		std::stringstream ssin(line);
		int r, g, b;
		ssin >> r >> g >> b;
		CGAL::Color color(static_cast<unsigned char>(r), static_cast<unsigned char>(g), static_cast<unsigned char>(b));
		color_map[color] = color_map.size();
		//LOG(INFO) << static_cast<int>(color.red()) << " " << static_cast<int>(color.green()) << " " << static_cast<int>(color.blue()) << " " << color_map[color];
	}

	fin.close();


	LOG(INFO) << ffmt("We have %d task files in total") % task_files.size();
	std::string wrong_files;
#pragma omp parallel for
	for (int i_task = 0; i_task < task_files.size(); ++i_task)
	{
		auto& task = task_files[i_task].first;
		auto& file_idx = task_files[i_task].second;

		std::vector<Point_3> points;
		std::vector<std::vector<std::size_t>> faces;
		std::vector<Triangle_3> triangles;
		std::vector<CGAL::Color> f_colors;
		// Although there is no v_color, read it before f_color.
		std::vector<CGAL::Color> v_colors;
		std::vector<int> primitive_index;

		if (!CGAL::IO::read_PLY(task.string(), points, faces,
			CGAL::parameters::vertex_color_output_iterator(std::back_inserter(v_colors))
			.face_color_output_iterator(std::back_inserter(f_colors))
			.use_binary_mode(true)
		))
		{
			wrong_files += file_idx + " ";
			continue;
		}

		LOG(INFO) << static_cast<int>(f_colors.back().red()) << " " << static_cast<int>(f_colors.back().green()) << " " << static_cast<int>(f_colors.back().blue());

		primitive_index.resize(faces.size());
		for (int i_face = 0; i_face < faces.size(); ++i_face)
		{
			const auto& face = faces[i_face];
			triangles.emplace_back(
				points[face[0]],
				points[face[1]],
				points[face[2]]
			);
			auto color = f_colors[i_face];
			primitive_index[i_face] = color_map[color];
		}

		const int num_points_per_m2 = 10000;
		Point_set mesh_transformed_sampled = sample_points_according_density(triangles, num_points_per_m2, primitive_index);

		colorize_point_set(mesh_transformed_sampled, "primitive_index");
		CGAL::IO::write_point_set((output_root / file_idx / "clipped" / "mesh_transformed_sampled.ply").string(), mesh_transformed_sampled);
	}

	if (!wrong_files.empty())
		LOG(ERROR) << ffmt("%s files are wrong") % wrong_files;

	return 0;
}
