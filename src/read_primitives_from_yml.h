#pragma once

#include "common_util.h"

#include <gp_Lin.hxx>
#include <gp_Pnt.hxx>
#include <gp_Circ.hxx>
#include <gp_Elips.hxx>
#include <ElCLib.hxx>

#include "calculate_indices.h"

// #pragma optimize("", off)
// bbox: center_x, center_y, center_z, diagonal_length
Point_set sample_points_on_curve(
	const std::vector<Curve>& config, 
	const std::vector<Point_3>& vertices, 
	const double num_per_m)
{
	Point_set sample_points_curves;
	auto curve_index_map = sample_points_curves.add_property_map("primitive_index", 0).first;

	int num_curves = 0;
	for (int i_curve = 0; i_curve < config.size(); ++i_curve)
	{
		// Calculate the length between mint and maxt along the line
		for (const auto vertex_indices : config[i_curve].vert_indices)
		{
			for (int iv = 0; iv < vertex_indices.size() - 1; ++iv)
			{
				const auto& v0 = cgal_2_eigen_point<double>(vertices[vertex_indices[iv]]);
				const auto& v1 = cgal_2_eigen_point<double>(vertices[vertex_indices[iv + 1]]);
				const double length = (v0 - v1).norm();
				const int num_samples = std::ceil(num_per_m * length);
				for (int i_sample = 0; i_sample < num_samples; ++i_sample)
				{
					const double t = i_sample / num_samples;
					const auto p = v0 + t * (v1 - v0);
					curve_index_map[*sample_points_curves.insert(Point_3(p.x(), p.y(), p.z()))] = i_curve;
				}
				curve_index_map[*sample_points_curves.insert(Point_3(v1.x(), v1.y(), v1.z()))] = i_curve;
			}
		}
	}
	return sample_points_curves;
}