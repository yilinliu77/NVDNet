#pragma once

#include "common_util.h"
#include "cgal_tools.h"

// #include <yaml-cpp/emitter.h>
// #include <yaml-cpp/node/node.h>
// #include <yaml-cpp/node/parse.h>
// #include <yaml-cpp/yaml.h>

#include "gp_Lin.hxx"
#include "gp_Pnt.hxx"
#include "gp_Circ.hxx"
#include "gp_Elips.hxx"
#include "ElCLib.hxx"

#include <ryml/ryml.hpp>
#include <ryml/ryml_std.hpp>

struct Curve
{
	std::string type;

	std::string attribute;

	std::vector<std::vector<int>> vert_indices;

	gp_Lin line;
	gp_Circ circle;
	gp_Elips ellipse;
};

struct Surface
{
	std::string type;

	std::vector<int> face_indices;
};

gp_Pnt normalize(const gp_Pnt& pos, const Eigen::Vector4d& bbox)
{
	return gp_Pnt(
		(pos.X() - bbox[0]) / bbox[3] * 2,
		(pos.Y() - bbox[1]) / bbox[3] * 2,
		(pos.Z() - bbox[2]) / bbox[3] * 2
	);
}

gp_Lin read_line(const ryml::ConstNodeRef& config, const Eigen::Vector4d& bbox)
{
	double x, y, z, dx, dy, dz;
	config["location"][0] >> x;
	config["location"][1] >> y;
	config["location"][2] >> z;
	config["direction"][0] >> dx;
	config["direction"][1] >> dy;
	config["direction"][2] >> dz;
	gp_Pnt pos(
		x, y, z
	);
	pos = normalize(pos, bbox);
	gp_Dir dir(
		dx, dy, dz
	);

	gp_Lin line(pos, dir);
	return line;
}

gp_Circ read_circle(const ryml::ConstNodeRef& config, const Eigen::Vector4d& bbox)
{
	double x, y, z, xx, xy, xz, zx, zy, zz;
	config["location"][0] >> x;
	config["location"][1] >> y;
	config["location"][2] >> z;
	config["x_axis"][0] >> xx;
	config["x_axis"][1] >> xy;
	config["x_axis"][2] >> xz;
	config["z_axis"][0] >> zx;
	config["z_axis"][1] >> zy;
	config["z_axis"][2] >> zz;

	gp_Pnt pos(
		x, y, z
	);
	pos = normalize(pos, bbox);

	gp_Dir xdir(
		xx, xy, xz
	);
	gp_Dir zdir(
		zx, zy, zz
	);
	double radius;
	config["radius"] >> radius;
	radius = radius / bbox[3] * 2;
	gp_Circ circle(
		gp_Ax2(pos, zdir, xdir),
		radius
	);
	return circle;
}

gp_Elips read_ellipse(const ryml::ConstNodeRef& config, const Eigen::Vector4d& bbox)
{
	double fx1, fy1, fz1, fx2, fy2, fz2, z, xx, xy, xz, zx, zy, zz;
	config["focus1"][0] >> fx1;
	config["focus1"][1] >> fy1;
	config["focus1"][2] >> fz1;
	config["focus2"][0] >> fx2;
	config["focus2"][1] >> fy2;
	config["focus2"][2] >> fz2;
	config["x_axis"][0] >> xx;
	config["x_axis"][1] >> xy;
	config["x_axis"][2] >> xz;
	config["z_axis"][0] >> zx;
	config["z_axis"][1] >> zy;
	config["z_axis"][2] >> zz;

	gp_Pnt focus1(
		fx1, fy1, fz1
	);
	gp_Pnt focus2(
		fx2, fy2, fz2
	);
	gp_Pnt center = (focus1.XYZ() + focus2.XYZ()) / 2;
	center = normalize(center, bbox);

	gp_Dir xdir(
		xx, xy, xz
	);
	gp_Dir zdir(
		zx, zy, zz
	);
	double r1, r2;
	config["maj_radius"] >> r1;
	config["min_radius"] >> r2;
	const double maj_radius = r1 / bbox[3] * 2;
	const double min_radius = r2 / bbox[3] * 2;
	gp_Elips elips(
		gp_Ax2(center, zdir, xdir),
		maj_radius,
		min_radius
	);
	return elips;
}

/*
std::pair<std::vector<Curve>, std::vector<Surface>> filter_primitives(
	const ryml::Tree& v_yaml,
	const std::vector<std::array<int, 3>>& v_faces,
	const bool is_pure_quadric
)
{
	std::vector<Curve> curves;
	std::vector<Surface> surfaces;
	for (int i_curve = 0; i_curve < v_yaml["curves"].num_children(); ++i_curve)
	{
		std::string type, sharp;
		v_yaml["curves"][i_curve]["type"] >> type;
		v_yaml["curves"][i_curve]["sharp"] >> sharp;
		// if (sharp!="true")
			// continue;

		Curve curve;
		for (int i = 0; i < v_yaml["curves"][i_curve]["vert_indices"].num_children(); ++i)
		{
			int a;
			v_yaml["curves"][i_curve]["vert_indices"][i] >> a;
			curve.vert_indices.push_back(a);
		}

		if (type == "Circle")
		{
			bool is_new_primitive = true;
			std::string location, radius;
			v_yaml["curves"][i_curve]["location"] >> location;
			v_yaml["curves"][i_curve]["radius"] >> radius;
			for (int i_exist = 0; i_exist < curves.size(); ++i_exist)
			{
				if (curves[i_exist].type == type &&
					curves[i_exist].location == location &&
					curves[i_exist].radius == radius)
				{
					for (int i = 0; i < v_yaml["curves"][i_curve]["vert_indices"].num_children(); ++i)
					{
						int a;
						v_yaml["curves"][i_curve]["vert_indices"][i] >> a;
						curves[i_exist].vert_indices.push_back(a);
					}
					is_new_primitive = false;
					break;
				}
			}
			if (is_new_primitive)
			{
				Curve curve;
				curve.type = type;
				curve.location = location;
				curve.radius = radius;

				for (int i = 0; i < v_yaml["curves"][i_curve]["vert_indices"].num_children(); ++i)
				{
					int a;
					v_yaml["curves"][i_curve]["vert_indices"][i] >> a;
					curve.vert_indices.push_back(a);
				}
				curves.emplace_back(curve);
			}
		}
		else if (type == "BSpline")
		{
			if (is_pure_quadric)
			{
				LOG(ERROR) << "Found BSpline";
				throw "";
			}
			bool is_new_primitive = true;
			std::stringstream knots, poles, degree;
			knots << v_yaml["curves"][i_curve]["knots"];
			poles << v_yaml["curves"][i_curve]["poles"];
			degree << v_yaml["curves"][i_curve]["degree"];
			for (int i_exist = 0; i_exist < curves.size(); ++i_exist)
			{
				if (curves[i_exist].type == type &&
					curves[i_exist].knots == knots.str() &&
					curves[i_exist].degree == degree.str() &&
					curves[i_exist].poles == poles.str())
				{
					for (int i = 0; i < v_yaml["curves"][i_curve]["vert_indices"].num_children(); ++i)
					{
						int a;
						v_yaml["curves"][i_curve]["vert_indices"][i] >> a;
						curves[i_exist].vert_indices.push_back(a);
					}
					is_new_primitive = false;
					break;
				}
			}
			if (is_new_primitive)
			{
				Curve curve;
				curve.type = type;
				curve.knots = knots.str();
				curve.degree = degree.str();
				curve.poles = poles.str();
				for (int i = 0; i < v_yaml["curves"][i_curve]["vert_indices"].num_children(); ++i)
				{
					int a;
					v_yaml["curves"][i_curve]["vert_indices"][i] >> a;
					curve.vert_indices.push_back(a);
				}
				curves.emplace_back(curve);
			}
		}
		else if (type == "Line")
		{
			bool is_new_primitive = true;
			curve.line = read_line(v_yaml["curves"][i_curve], v_bounds);

			std::string location, direction;
			v_yaml["curves"][i_curve]["location"] >> location;
			v_yaml["curves"][i_curve]["direction"] >> direction;
			for (int i_exist = 0; i_exist < curves.size(); ++i_exist)
			{
				if (curves[i_exist].type == type &&
					curves[i_exist].direction == direction &&
					curves[i_exist].location == location)
				{
					for (int i = 0; i < v_yaml["curves"][i_curve]["vert_indices"].num_children(); ++i)
					{
						int a;
						v_yaml["curves"][i_curve]["vert_indices"][i] >> a;
						curves[i_exist].vert_indices.push_back(a);
					}
					is_new_primitive = false;
					break;
				}
			}
			if (is_new_primitive)
			{
				curve.type = type;
				curve.direction = direction;
				curve.location = location;
				curves.emplace_back(curve);
			}
		}
		else if (v_yaml["curves"][i_curve]["type"].val() == "Ellipse")
		{
			bool is_new_primitive = true;
			std::string focus1, focus2, maj_radius, min_radius;
			v_yaml["curves"][i_curve]["focus1"] >> focus1;
			v_yaml["curves"][i_curve]["focus2"] >> focus2;
			v_yaml["curves"][i_curve]["maj_radius"] >> maj_radius;
			v_yaml["curves"][i_curve]["min_radius"] >> min_radius;
			for (int i_exist = 0; i_exist < curves.size(); ++i_exist)
			{
				if (curves[i_exist].type == type &&
					curves[i_exist].focus1 == focus1 &&
					curves[i_exist].focus2 == focus2 &&
					curves[i_exist].maj_radius == maj_radius &&
					curves[i_exist].min_radius == min_radius)
				{
					for (int i = 0; i < v_yaml["curves"][i_curve]["vert_indices"].num_children(); ++i)
					{
						int a;
						v_yaml["curves"][i_curve]["vert_indices"][i] >> a;
						curves[i_exist].vert_indices.push_back(a);
					}
					is_new_primitive = false;
					break;
				}
			}
			if (is_new_primitive)
			{
				Curve curve;
				curve.type = type;
				curve.focus1 = focus1;
				curve.focus2 = focus2;
				curve.maj_radius = maj_radius;
				curve.min_radius = min_radius;
				for (int i = 0; i < v_yaml["curves"][i_curve]["vert_indices"].num_children(); ++i)
				{
					int a;
					v_yaml["curves"][i_curve]["vert_indices"][i] >> a;
					curve.vert_indices.push_back(a);
				}
				curves.emplace_back(curve);
			}
		}
		else
			return { std::vector<Curve>(), surfaces };
	}

	// Convert the vert indices to set
	for (int i_curve = 0; i_curve < curves.size(); ++i_curve)
	{
		std::unordered_set<int> local_set;
		for (const auto& id : curves[i_curve].vert_indices)
			local_set.insert(id);
		curves[i_curve].vert_indices = local_set;
	}

	// Initialize the nearby surface id in for each vertex
	// For vertex that is assigned by two different curves, it is a corner point
	std::vector<int> corner_points;
	std::unordered_map<int, std::unordered_set<int>> id_primitive_per_vertex;
	for (int i_curve = 0; i_curve < curves.size(); ++i_curve)
	{
		for (const auto& vert_id : curves[i_curve].vert_indices)
		{
			if (std::find(corner_points.begin(), corner_points.end(), vert_id) != corner_points.end())
				continue;

			if (id_primitive_per_vertex.find(vert_id) != id_primitive_per_vertex.end())
			{
				id_primitive_per_vertex.erase(vert_id);
				corner_points.push_back(vert_id);
			}
			else
				id_primitive_per_vertex[vert_id] = std::unordered_set<int>();
		}
	}

	// Check the minimum index
	for (const auto& curve : curves)
		for (const auto idx : curve.vert_indices)
			if (idx < 0)
				return { std::vector<Curve>(), surfaces };

	// For each surface, iterate all the faces and the corresponding vertices
	// Assign the surface id to the vertex
	for (int i_surface = 0; i_surface < v_yaml["surfaces"].num_children(); ++i_surface)
	{
		Surface surface;
		for (const auto& id_face_yaml : v_yaml["surfaces"][i_surface]["face_indices"])
		{
			int id_face;
			id_face_yaml >> id_face;
			for (auto vert_id : v_faces[id_face])
			{
				vert_id = vert_id;
				if (id_primitive_per_vertex.find(vert_id) != id_primitive_per_vertex.end())
					id_primitive_per_vertex[vert_id].insert(i_surface);
			}
			v_yaml["surfaces"][i_surface]["type"] >> surface.type;
			surface.face_indices.emplace_back(id_face);
		}
		surfaces.emplace_back(surface);
	}

	// For each vertex, the set should store the nearby surface id
	// If the neighbour surface ids contains only one surface, it is a curve inside a surface
	// We then simply ignore it
	std::vector<Curve> filtered_curves;
	for (int i_curve = 0; i_curve < curves.size(); ++i_curve)
	{
		std::unordered_set<int> neighbour_primitives;
		for (const auto& vert_id : curves[i_curve].vert_indices)
		{
			if (id_primitive_per_vertex.find(vert_id) == id_primitive_per_vertex.end())
				continue;
			for (const auto& item : id_primitive_per_vertex[vert_id])
				neighbour_primitives.insert(item);
		}
		if (neighbour_primitives.size() > 1)
			filtered_curves.emplace_back(curves[i_curve]);
	}

	return { filtered_curves, surfaces };
}
*/

std::pair<std::vector<Curve>, std::vector<Surface>> filter_primitives(
	const ryml::Tree& v_yaml,
	const std::vector<std::array<int, 3>>& v_faces,
	const bool is_pure_quadric,
	const Eigen::Vector4d& v_bounds
)
{
	std::vector<Curve> curves;
	std::vector<Surface> surfaces;
	for (int i_curve = 0; i_curve < v_yaml["curves"].num_children(); ++i_curve)
	{
		std::string type, sharp;
		v_yaml["curves"][i_curve]["type"] >> type;
		v_yaml["curves"][i_curve]["sharp"] >> sharp;
		// if (sharp!="true")
			// continue;

		std::vector<int> vert_indices;
		for (int i = 0; i < v_yaml["curves"][i_curve]["vert_indices"].num_children(); ++i)
		{
			int a;
			v_yaml["curves"][i_curve]["vert_indices"][i] >> a;
			vert_indices.push_back(a);
		}

		if (type == "Circle")
		{
			bool is_new_primitive = true;

			std::stringstream ss_attribute;
			std::string attribute;
			ss_attribute << v_yaml["curves"][i_curve]["location"];
			ss_attribute << v_yaml["curves"][i_curve]["radius"];
			attribute = ss_attribute.str();
			for (int i_exist = 0; i_exist < curves.size(); ++i_exist)
			{
				if (curves[i_exist].type == type &&
					curves[i_exist].attribute == attribute)
				{
					curves[i_exist].vert_indices.push_back(vert_indices);
					is_new_primitive = false;
					break;
				}
			}
			if (is_new_primitive)
			{
				Curve curve;
				curve.circle = read_circle(v_yaml["curves"][i_curve], v_bounds);
				curve.type = type;
				curve.attribute = attribute;
				curve.vert_indices.push_back(vert_indices);
				curves.emplace_back(curve);
			}
			
		}
		else if (type == "BSpline")
		{
			bool is_new_primitive = true;
			if (is_pure_quadric)
			{
				LOG(ERROR) << "Found BSpline";
				throw "";
			}

			std::stringstream ss_attribute;
			std::string attribute;
			ss_attribute << v_yaml["curves"][i_curve]["knots"];
			ss_attribute << v_yaml["curves"][i_curve]["poles"];
			ss_attribute << v_yaml["curves"][i_curve]["degree"];
			attribute = ss_attribute.str();

			for (int i_exist = 0; i_exist < curves.size(); ++i_exist)
			{
				if (curves[i_exist].type == type &&
					curves[i_exist].attribute == attribute)
				{
					curves[i_exist].vert_indices.push_back(vert_indices);
					is_new_primitive = false;
					break;
				}
			}
			if (is_new_primitive)
			{
				Curve curve;
				curve.type = type;
				curve.attribute = attribute;
				curve.vert_indices.push_back(vert_indices);
				curves.emplace_back(curve);
			}
		}
		else if (type == "Line")
		{
			bool is_new_primitive = true;

			std::stringstream ss_attribute;
			std::string attibute;
			ss_attribute << v_yaml["curves"][i_curve]["location"];
			ss_attribute << v_yaml["curves"][i_curve]["direction"];
			attibute = ss_attribute.str();
			for (int i_exist = 0; i_exist < curves.size(); ++i_exist)
			{
				if (curves[i_exist].type == type &&
					curves[i_exist].attribute == attibute)
				{
					curves[i_exist].vert_indices.push_back(vert_indices);
					is_new_primitive = false;
					break;
				}
			}
			if (is_new_primitive)
			{
				Curve curve;
				curve.line = read_line(v_yaml["curves"][i_curve], v_bounds);
				curve.type = type;
				curve.attribute = attibute;
				curve.vert_indices.push_back(vert_indices);
				curves.emplace_back(curve);
			}
			
		}
		else if (type == "Ellipse")
		{
			bool is_new_primitive = true;

			std::stringstream ss_attribute;
			std::string attribute;
			ss_attribute << v_yaml["curves"][i_curve]["focus1"];
			ss_attribute << v_yaml["curves"][i_curve]["focus2"];
			ss_attribute << v_yaml["curves"][i_curve]["maj_radius"];
			ss_attribute << v_yaml["curves"][i_curve]["min_radius"];
			attribute = ss_attribute.str();

			for (int i_exist = 0; i_exist < curves.size(); ++i_exist)
			{
				if (curves[i_exist].type == type &&
					curves[i_exist].attribute == attribute)
				{
					curves[i_exist].vert_indices.push_back(vert_indices);
					is_new_primitive = false;
					break;
				}
			}
			if (is_new_primitive)
			{
				Curve curve;
				curve.ellipse = read_ellipse(v_yaml["curves"][i_curve], v_bounds);
				curve.type = type;
				curve.attribute = attribute;
				curve.vert_indices.push_back(vert_indices);
				curves.emplace_back(curve);
			}
		}
		else
			return { std::vector<Curve>(), surfaces };
	}

	// Initialize the nearby surface id in for each vertex
	// For vertex that is assigned by two different curves, it is a corner point
	std::vector<int> corner_points;
	std::unordered_map<int, std::unordered_set<int>> id_primitive_per_vertex;
	for (int i_curve = 0; i_curve < curves.size(); ++i_curve)
	{
		for (const auto& vert_indices : curves[i_curve].vert_indices)
			for (const auto& vert_id : vert_indices)
			{
				if (std::find(corner_points.begin(), corner_points.end(), vert_id) != corner_points.end())
					continue;

				if (id_primitive_per_vertex.find(vert_id) != id_primitive_per_vertex.end())
				{
					id_primitive_per_vertex.erase(vert_id);
					corner_points.push_back(vert_id);
				}
				else
					id_primitive_per_vertex[vert_id] = std::unordered_set<int>();
			}
	}

	// Check the minimum index
	for (const auto& curve : curves)
		for (const auto& vert_indices : curve.vert_indices)
			for (const auto idx : vert_indices)
				if (idx < 0)
					return { std::vector<Curve>(), surfaces };

	// For each surface, iterate all the faces and the corresponding vertices
	// Assign the surface id to the vertex
	for (int i_surface = 0; i_surface < v_yaml["surfaces"].num_children(); ++i_surface)
	{
		Surface surface;
		if (v_yaml["surfaces"][i_surface]["face_indices"].num_children() == 0)
			continue;
		for (const auto& id_face_yaml : v_yaml["surfaces"][i_surface]["face_indices"])
		{
			int id_face;
			id_face_yaml >> id_face;
			for (auto vert_id : v_faces[id_face])
			{
				vert_id = vert_id;
				if (id_primitive_per_vertex.find(vert_id) != id_primitive_per_vertex.end())
					id_primitive_per_vertex[vert_id].insert(i_surface);
			}
			v_yaml["surfaces"][i_surface]["type"] >> surface.type;
			surface.face_indices.emplace_back(id_face);
		}
		surfaces.emplace_back(surface);
	}

	// For each vertex, the set should store the nearby surface id
	// If the neighbour surface ids contains only one surface, it is a curve inside a surface
	// We then simply ignore it
	std::vector<Curve> filtered_curves;
	for (int i_curve = 0; i_curve < curves.size(); ++i_curve)
	{
		std::unordered_set<int> neighbour_primitives;
		for (const auto& vert_indices : curves[i_curve].vert_indices)
			for (const auto& vert_id : vert_indices)
			{
				if (id_primitive_per_vertex.find(vert_id) == id_primitive_per_vertex.end())
					continue;
				for (const auto& item : id_primitive_per_vertex[vert_id])
					neighbour_primitives.insert(item);
			}
		if (neighbour_primitives.size() > 1)
			filtered_curves.emplace_back(curves[i_curve]);
	}

	return { filtered_curves, surfaces };
}

std::tuple<
	std::vector<long long>,
	std::vector<std::vector<long long>>,
	std::vector<std::pair<int, std::vector<int>>>,
	Eigen::MatrixXi
> calculate_indices(const std::vector<Curve>& curves,
	const std::vector<Surface>& surfaces,
	const std::vector<Point_3>& vertices,
	const std::vector<std::array<int, 3>>& faces
)
{
	const long long num_faces = static_cast<long long>(faces.size());
	const long long num_vertices = static_cast<long long>(vertices.size());
	const long long num_curves = static_cast<long long>(curves.size());
	const long long num_surfaces = static_cast<long long>(surfaces.size());
	long long num_primitives = num_curves + num_surfaces;

	std::vector<long long> vert_id_to_primitives(num_vertices, -1);
	std::vector<long long> surface_id_to_primitives(num_faces, 0);

	std::vector<std::pair<int, std::vector<int>>> id_corner_points;
	for (int id_curve = 0; id_curve < num_curves; ++id_curve)
	{
		const auto& curve = curves[id_curve];
		for (const auto vert_indices : curve.vert_indices)
			for (const auto id_vert : vert_indices)
			{
				if (vert_id_to_primitives[id_vert] == id_curve)
					continue;
				if (vert_id_to_primitives[id_vert] != -1)
				{
					auto id_corner = std::find_if(id_corner_points.begin(), id_corner_points.end(),
						[&id_vert](const auto& item) { return item.first == id_vert; });

					if (id_corner == id_corner_points.end())
					{
						std::vector<int> temp{ id_curve, (int)vert_id_to_primitives[id_vert]};
						vert_id_to_primitives[id_vert] = num_primitives + id_corner_points.size();
						id_corner_points.emplace_back(id_vert, temp);
						id_corner = id_corner_points.end() - 1;
					}
					else
					{
						vert_id_to_primitives[id_vert] = num_primitives +
							std::distance(id_corner_points.begin(), id_corner);
						id_corner->second.emplace_back(id_curve);
					}
				}
				else
				{
					vert_id_to_primitives[id_vert] = id_curve;
				}
			}
	}

	int num_corner_points = id_corner_points.size();
	num_primitives += num_corner_points;
	Eigen::MatrixXi adj_matrix(num_primitives, num_primitives);
	adj_matrix.setConstant(0);

	for(int i_corner = 0;i_corner<id_corner_points.size();++i_corner)
	{
		for (const auto id_curve : id_corner_points[i_corner].second)
		{
			adj_matrix(num_curves + num_surfaces + i_corner, id_curve) = 1;
			adj_matrix(id_curve, num_curves + num_surfaces + i_corner) = 1;
		}
	}
	
	std::vector<std::vector<long long>> face_edge_indicator(num_faces, std::vector<long long>(3, -1));
	for (int id_surface = 0; id_surface < num_surfaces; ++id_surface)
	{
		auto surface = surfaces[id_surface];
		for (auto id_face : surface.face_indices)
		{
			for (int idx = 0; idx < faces[id_face].size(); idx++)
			{
				int primitive_id = vert_id_to_primitives[faces[id_face][idx]];
				if (primitive_id >= num_curves + num_surfaces)
				{
					face_edge_indicator[id_face][idx] = primitive_id;
				}
				else if (primitive_id > -1)
				{
					adj_matrix(primitive_id, num_curves + id_surface) = 1;
					adj_matrix(num_curves + id_surface, primitive_id) = 1;
					face_edge_indicator[id_face][idx] = primitive_id;
				}
			}

			surface_id_to_primitives[id_face] = id_surface + num_curves;
		}
	}

	return { surface_id_to_primitives, face_edge_indicator, id_corner_points, adj_matrix };
}