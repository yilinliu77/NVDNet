#pragma once

#include "common_util.h"
#include "cgal_tools.h"
#include <boost/algorithm/string.hpp>

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
	return {vertices, faces};
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
	return {min_x,min_y,min_z,max_x,max_y,max_z,center_x,center_y,center_z,diag};
}
