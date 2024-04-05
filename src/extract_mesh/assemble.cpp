#include "assemble.h"

#include "fitting.h"
#include "kd_tree_helper.h"
#include "IntAna_QuadQuadGeo.hxx"
#include "IntAna_IntQuadQuad.hxx"
#include "IntAna_Quadric.hxx"
#include "GeomAPI_ExtremaCurveCurve.hxx"
#include "Geom_Line.hxx"
#include "Geom_Circle.hxx"
#include "Geom_Ellipse.hxx"
#include "merge_shape.h"

#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <StlAPI_Writer.hxx>
#include <RWObj_CafWriter.hxx>
#include <TopoDS_Wire.hxx>
#include <TopoDS_Face.hxx>
#include <BRepLib.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepBuilderAPI_MakeShell.hxx>
#include <BRep_Builder.hxx>

#include <Extrema_ExtPS.hxx>

#include "tools.h"

// #pragma optimize ("", off)
// #pragma optimize ("", on)

void export_point_cloud(const std::string& v_path, const std::vector<Eigen::Vector3d>& points)
{
	Point_set p;
	p.resize(points.size());
	for (int i = 0; i < points.size(); ++i)
		p.point(i) = eigen_2_cgal_point(points[i]);
	CGAL::IO::write_point_set(v_path, p);
}

Point_set to_point_set(const std::vector<Eigen::Vector3d>& v_points)
{
	Point_set p;
	p.resize(v_points.size());
	for (int i = 0; i < v_points.size(); ++i)
		p.point(i) = eigen_2_cgal_point(v_points[i]);
	return p;
}

std::vector<std::shared_ptr<Shape2D>> find_intersection(const std::shared_ptr<Shape3D> v_shape1, const std::shared_ptr<Shape3D> v_shape2)
{
	IntAna_QuadQuadGeo inter;
	if (v_shape1->detail_type == "plane" && v_shape2->detail_type == "plane")
	{
		gp_Pln plane1, plane2;
		plane1 = dynamic_pointer_cast<MyPlane>(v_shape1)->plane;
		plane2 = dynamic_pointer_cast<MyPlane>(v_shape2)->plane;
		inter.Perform(plane1, plane2, 1e-3, 1e-3);
	}
	else if (v_shape1->detail_type == "plane" && v_shape2->detail_type == "cylinder" ||
		v_shape1->detail_type == "cylinder" && v_shape2->detail_type == "plane")
	{
		gp_Pln plane;
		gp_Cylinder cylinder;
		if (v_shape1->detail_type == "plane")
		{
			plane = dynamic_pointer_cast<MyPlane>(v_shape1)->plane;
			cylinder = dynamic_pointer_cast<MyCylinder>(v_shape2)->cylinder;
		}
		else
		{
			plane = dynamic_pointer_cast<MyPlane>(v_shape2)->plane;
			cylinder = dynamic_pointer_cast<MyCylinder>(v_shape1)->cylinder;
		}
		inter.Perform(plane, cylinder, 1e-3, 1e-3, 1e-3);
	}
	else if (v_shape1->detail_type == "plane" && v_shape2->detail_type == "cone" ||
		v_shape1->detail_type == "cone" && v_shape2->detail_type == "plane")
	{
		gp_Pln plane;
		gp_Cone cone;
		if (v_shape1->detail_type == "plane")
		{
			plane = dynamic_pointer_cast<MyPlane>(v_shape1)->plane;
			cone = dynamic_pointer_cast<MyCone>(v_shape2)->cone;
		}
		else
		{
			plane = dynamic_pointer_cast<MyPlane>(v_shape2)->plane;
			cone = dynamic_pointer_cast<MyCone>(v_shape1)->cone;
		}
		inter.Perform(plane, cone, 1e-3, 1e-3);
	}
	else if (v_shape1->detail_type == "plane" && v_shape2->detail_type == "sphere" ||
		v_shape1->detail_type == "sphere" && v_shape2->detail_type == "plane")
	{
		gp_Pln plane;
		gp_Sphere sphere;
		if (v_shape1->detail_type == "plane")
		{
			plane = dynamic_pointer_cast<MyPlane>(v_shape1)->plane;
			sphere = dynamic_pointer_cast<MySphere>(v_shape2)->sphere;
		}
		else
		{
			plane = dynamic_pointer_cast<MyPlane>(v_shape2)->plane;
			sphere = dynamic_pointer_cast<MySphere>(v_shape1)->sphere;
		}
		inter.Perform(plane, sphere);
	}
	else if (v_shape1->detail_type == "plane" && v_shape2->detail_type == "torus" ||
		v_shape1->detail_type == "torus" && v_shape2->detail_type == "plane")
	{
		gp_Pln plane;
		gp_Torus torus;
		if (v_shape1->detail_type == "plane")
		{
			plane = dynamic_pointer_cast<MyPlane>(v_shape1)->plane;
			torus = dynamic_pointer_cast<MyTorus>(v_shape2)->torus;
		}
		else
		{
			plane = dynamic_pointer_cast<MyPlane>(v_shape2)->plane;
			torus = dynamic_pointer_cast<MyTorus>(v_shape1)->torus;
		}
		inter.Perform(plane, torus, 1e-3);
	}
	else
		return {};

	if (!inter.IsDone())
		return {};
	std::vector<std::shared_ptr<Shape2D>> result_shapes;
	if (inter.TypeInter() == IntAna_Line)
	{
		for(int i=0;i<inter.NbSolutions();++i)
		{
			gp_Lin line = inter.Line(i+1);
			MyLine* shape = new MyLine();
			shape->line = line;
			shape->type = "curve";
			shape->detail_type = "line";
			gp_Dir normal;
			if (line.Direction().Dot(gp_Dir(0, 0, 1)) > 0.9999)
				normal = gp_Dir(0, 1, 0).Crossed(line.Direction());
			else
				normal = gp_Dir(0, 0, 1).Crossed(line.Direction());
			shape->m_plane = gp_Pln(line.Location(), normal);
			std::shared_ptr<Shape2D> result_shape;
			result_shape.reset(shape);
			result_shapes.push_back(result_shape);
		}
	}
	else if (inter.TypeInter() == IntAna_Circle)
	{
		gp_Circ circle = inter.Circle(1);
		MyCircle* shape = new MyCircle();
		shape->circle = circle;
		shape->type = "curve";
		shape->detail_type = "circle";
		shape->m_plane=gp_Pln(circle.Location(), circle.Axis().Direction());
		std::shared_ptr<Shape2D> result_shape;
		result_shape.reset(shape);
		result_shapes.push_back(result_shape);
	}
	else if (inter.TypeInter() == IntAna_PointAndCircle)
	{
		gp_Circ circle = inter.Circle(2);
		MyCircle* shape = new MyCircle();
		shape->circle = circle;
		shape->type = "curve";
		shape->detail_type = "circle";
		shape->m_plane = gp_Pln(circle.Location(), circle.Axis().Direction());
		std::shared_ptr<Shape2D> result_shape;
		result_shape.reset(shape);
		result_shapes.push_back(result_shape);
	}
	else if (inter.TypeInter() == IntAna_Ellipse)
	{
		gp_Elips ellipse = inter.Ellipse(1);
		MyEllipse* shape = new MyEllipse();
		shape->ellipse = ellipse;
		shape->type = "curve";
		shape->detail_type = "ellipse";
		shape->m_plane = gp_Pln(ellipse.Location(), ellipse.Axis().Direction());
		std::shared_ptr<Shape2D> result_shape;
		result_shape.reset(shape);
		result_shapes.push_back(result_shape);
	}
	else if (inter.TypeInter() == IntAna_NoGeometricSolution)
	{
		return {};
	}
	else
		return {};
	return result_shapes;
}

Shape* find_intersection(const Shape2D* v_shape1, const Shape2D* v_shape2)
{
	GeomAPI_ExtremaCurveCurve inter;
	if (v_shape1->detail_type == "line" && v_shape2->detail_type == "line")
	{
		gp_Lin s1, s2;
		s1 = ((MyLine*)v_shape1)->line;
		s2 = ((MyLine*)v_shape2)->line;
		Handle(Geom_Line) l1 = new Geom_Line(s1), l2 = new Geom_Line(s2);
		inter.Init(l1, l2);
	}
	else if (v_shape1->detail_type == "line" && v_shape2->detail_type == "circle" ||
		v_shape1->detail_type == "circle" && v_shape2->detail_type == "line")
	{
		gp_Lin s1;
		gp_Circ s2;
		Handle(Geom_Line) l1;
		Handle(Geom_Circle) l2;
		if (v_shape1->detail_type == "line")
		{
			s1 = ((MyLine*)v_shape1)->line;
			s2 = ((MyCircle*)v_shape2)->circle;
		}
		else
		{
			s1 = ((MyLine*)v_shape2)->line;
			s2 = ((MyCircle*)v_shape1)->circle;
		}
		l1 = new Geom_Line(s1);
		l2 = new Geom_Circle(s2);
		inter.Init(l1, l2);
	}
	else if (v_shape1->detail_type == "line" && v_shape2->detail_type == "ellipse" ||
		v_shape1->detail_type == "ellipse" && v_shape2->detail_type == "line")
	{
		gp_Lin s1;
		gp_Elips s2;
		Handle(Geom_Line) l1;
		Handle(Geom_Ellipse) l2;
		if (v_shape1->detail_type == "line")
		{
			s1 = ((MyLine*)v_shape1)->line;
			s2 = ((MyEllipse*)v_shape2)->ellipse;
		}
		else
		{
			s1 = ((MyLine*)v_shape2)->line;
			s2 = ((MyEllipse*)v_shape1)->ellipse;
		}
		l1 = new Geom_Line(s1);
		l2 = new Geom_Ellipse(s2);
		inter.Init(l1, l2);
	}
	else if (v_shape1->detail_type == "circle" && v_shape2->detail_type == "ellipse" ||
		v_shape1->detail_type == "ellipse" && v_shape2->detail_type == "circle")
	{
		gp_Circ s1;
		gp_Elips s2;
		Handle(Geom_Circle) l1;
		Handle(Geom_Ellipse) l2;
		if (v_shape1->detail_type == "circle")
		{
			s1 = ((MyCircle*)v_shape1)->circle;
			s2 = ((MyEllipse*)v_shape2)->ellipse;
		}
		else
		{
			s1 = ((MyCircle*)v_shape2)->circle;
			s2 = ((MyEllipse*)v_shape1)->ellipse;
		}
		l1 = new Geom_Circle(s1);
		l2 = new Geom_Ellipse(s2);
		inter.Init(l1, l2);
	}
	else
		return nullptr;

	if (inter.NbExtrema() == 0)
	{
		return nullptr;
	}
	gp_Pnt p1, p2;
	inter.NearestPoints(p1, p2);
	Shape1D* shape = new Shape1D;
	shape->vertex = Eigen::Vector3d(p1.X()+p2.X(), p1.Y()+p2.Y(), p1.Z()+p2.Z()) / 2;
	shape->type = "vertex";
	shape->detail_type = "vertex";
	return shape;
}

void check_adj(const Eigen::MatrixXi& adjacency_matrix, const int num_shapes)
{
	if (num_shapes!= adjacency_matrix.rows())
		LOG(INFO) << "num_shapes != adjacency_matrix.rows()";
	if (num_shapes != adjacency_matrix.cols())
		LOG(INFO) << "num_shapes != adjacency_matrix.cols()";

	for (int i = 0; i < num_shapes; ++i)
		for (int j = 0; j < num_shapes; ++j)
		{
			if (i == j)
				if (adjacency_matrix(i, j) > 0)
					LOG(INFO) << ffmt("(%d, %d) should be zero") % i % j;

			if ((adjacency_matrix(i, j) > 0 && adjacency_matrix(j, i) > 0) ||
				(adjacency_matrix(i, j) == 0 && adjacency_matrix(j, i) == 0))
				continue;
			else
			{
				LOG(INFO) << ffmt("(%d,%d): %d; (%d,%d): %d") % 
					i % j % adjacency_matrix(i, j) % j % i % adjacency_matrix(j, i);
			}
		}
}

void fix_adjacency_matrix(const std::vector<std::shared_ptr<Shape>>& shapes, Eigen::MatrixXi& adjacency_matrix, const double epsilon = 2e-2)
{
	// #pragma omp parallel for
	for (int i_shape1 = 0; i_shape1 < shapes.size(); ++i_shape1)
	{
		if (shapes[i_shape1]->type == "curve")
			shapes[i_shape1]->find_boundary();
		for (int i_shape2 = 0; i_shape2 < shapes.size(); ++i_shape2)
		{
			if (adjacency_matrix(i_shape1, i_shape2) == 0)
				continue;
			if (shapes[i_shape1]->type == "surface" && shapes[i_shape2]->type == "curve")
			{
				std::vector<double> errors(shapes[i_shape2]->inliers.size());
				for (int i = 0; i < shapes[i_shape2]->inliers.size(); ++i)
					errors[i] = shapes[i_shape1]->inlier_distance(shapes[i_shape2]->inliers[i]);
				const double error = std::accumulate(errors.begin(), errors.end(), 0.) / shapes[i_shape2]->inliers.size();
				if (error > epsilon)
				{
					adjacency_matrix(i_shape1, i_shape2) = 0;
					adjacency_matrix(i_shape2, i_shape1) = 0;
				}
			}
			else if (shapes[i_shape2]->type == "vertex")
			{
				
				const double error = shapes[i_shape1]->distance(dynamic_pointer_cast<Shape1D>(shapes[i_shape2])->vertex);
				if (error > epsilon)
				{
					adjacency_matrix(i_shape1, i_shape2) = 0;
					adjacency_matrix(i_shape2, i_shape1) = 0;
				}
			}
		}
	}
}

void debug_shapes(const std::vector<std::shared_ptr<Shape>>& v_shapes, const fs::path& v_output)
{
	Point_set p_inliers, p_samples;
	auto index_inliers = p_inliers.add_property_map("index", 0).first;
	auto index_samples = p_samples.add_property_map("index", 0).first;
	int num_curves = 0;
	for (const auto& curve : v_shapes)
	{
		if (curve->type != "curve")
			continue;
		for (const auto& coord : curve->inliers)
			index_inliers[*p_inliers.insert(eigen_2_cgal_point(coord))] = num_curves;

		Point_set pp = curve->sample_parametric(10000);
		for (const auto& coord : pp.points())
			index_samples[*p_samples.insert(coord)] = num_curves;
		num_curves++;

		colorize_point_set(p_inliers);
		colorize_point_set(p_samples);
	}
	CGAL::IO::write_point_set((v_output / "adding_curves/curves_after_adding_i.ply").string(), p_inliers);
	CGAL::IO::write_point_set((v_output / "adding_curves/curves_after_adding_s.ply").string(), p_samples);
}

std::pair<std::vector<std::shared_ptr<Shape>>, double> expand_segments(
	std::shared_ptr<Shape2D>& v_shape, 
	const Eigen::MatrixXi& v_connectivity, 
	const std::vector<Eigen::Vector3d>& v_boundary_point,
	const double epsilon
)
{
	const int num_points = v_boundary_point.size();
	std::vector<bool> local_visited(num_points, false);

	std::vector<std::shared_ptr<Shape>> curves;
	double avg_error = 0.;
	for (int i_p = 0; i_p < num_points; ++i_p)
	{
		if(local_visited[i_p]) // Skip if it has been already visited
			continue;

		local_visited[i_p] = true;
		if (v_shape->distance(v_boundary_point[i_p]) >= epsilon) // Skip if it is far away to the curve
			continue;

		v_shape->inliers.clear();

		std::vector<Eigen::Vector3d> inliers;

		std::queue<int> q;
		q.push(i_p);
		while (!q.empty())
		{
			const int id_cur = q.front();
			q.pop();

			local_visited[id_cur] = true;
			inliers.emplace_back(v_boundary_point[id_cur]);

			for (int j_p = 0; j_p < num_points; ++j_p)
			{
				if (id_cur == j_p || v_connectivity(id_cur, j_p) == 0 || local_visited[j_p] || 
					v_shape->distance(v_boundary_point[j_p]) >= epsilon)
					continue;
				local_visited[j_p] = true;
				q.push(j_p);
			}
		}

		if (inliers.size() < 5)
			continue;

		std::shared_ptr<Shape> new_curve;
		Cluster cluster;
		cluster.surface_points = inliers;
		if (v_shape->detail_type == "line")
		{
			const auto shape = dynamic_pointer_cast<MyLine>(v_shape);
			new_curve.reset(new MyLine());
			const auto new_shape = dynamic_pointer_cast<MyLine>(new_curve);
			new_shape->type = "curve";
			new_shape->detail_type = "line";
			new_shape->line = shape->line;
			new_shape->cluster = cluster;
			new_shape->m_plane = shape->m_plane;
			new_shape->inliers = inliers;
		}
		else if (v_shape->detail_type == "circle")
		{
			auto shape = dynamic_pointer_cast<MyCircle>(v_shape);
			new_curve.reset(new MyCircle());
			const auto new_shape = dynamic_pointer_cast<MyCircle>(new_curve);
			new_shape->type = "curve";
			new_shape->detail_type = "circle";
			new_shape->circle = shape->circle;
			new_shape->cluster = cluster;
			new_shape->inliers = inliers;
			new_shape->m_plane = shape->m_plane;
		}
		else if (v_shape->detail_type == "ellipse")
		{
			auto ellipse_shape = dynamic_pointer_cast<MyEllipse>(v_shape);
			auto shape = dynamic_pointer_cast<MyEllipse>(v_shape);

			new_curve.reset(new MyEllipse());
			const auto new_shape = dynamic_pointer_cast<MyEllipse>(new_curve);
			new_shape->type = "curve";
			new_shape->detail_type = "ellipse";
			new_shape->ellipse = shape->ellipse;
			new_shape->cluster = cluster;
			new_shape->inliers = inliers;
			new_shape->m_plane = shape->m_plane;

			// Test if we can use line to approximate the ellipse
			new_curve = check_valid_ellipse(new_curve, inliers);
			if (new_curve == nullptr)
				continue;
		}
		else
			throw;

		new_curve->find_boundary();
		curves.push_back(new_curve);

		double error = 0.;
		for(const auto& item: inliers)
			error += v_shape->distance(item);
		avg_error += error / inliers.size();
	}
	if (curves.empty())
		return { {}, 99999. };
	avg_error /= curves.size();
	return { curves, avg_error };
}

std::vector<std::shared_ptr<Shape>> expand_segments_bak_1227(
	std::shared_ptr<Shape2D>& v_shape,
	const Eigen::MatrixXi& v_connectivity,
	const std::vector<Eigen::Vector3d>& v_boundary_point,
	const double epsilon
)
{
	const int num_points = v_boundary_point.size();
	std::vector<bool> local_visited(num_points, false);

	std::vector<std::shared_ptr<Shape>> curves;
	for (int i_p = 0; i_p < num_points; ++i_p)
	{
		if (local_visited[i_p]) // Skip if it has been already visited
			continue;

		local_visited[i_p] = true;
		if (v_shape->distance(v_boundary_point[i_p]) >= epsilon) // Skip if it is far away to the curve
			continue;

		v_shape->inliers.clear();

		std::vector<Eigen::Vector3d> inliers;

		std::queue<int> q;
		q.push(i_p);
		while (!q.empty())
		{
			const int id_cur = q.front();
			q.pop();

			local_visited[id_cur] = true;
			inliers.emplace_back(v_boundary_point[id_cur]);

			for (int j_p = 0; j_p < num_points; ++j_p)
			{
				if (id_cur == j_p || v_connectivity(id_cur, j_p) == 0 || local_visited[j_p] ||
					v_shape->distance(v_boundary_point[j_p]) >= epsilon)
					continue;
				local_visited[j_p] = true;
				q.push(j_p);
			}
		}

		std::shared_ptr<Shape> new_curve;
		Cluster cluster;
		cluster.surface_points = inliers;
		if (v_shape->detail_type == "line")
		{
			auto shape = dynamic_pointer_cast<MyLine>(v_shape);
			const auto result = fit_curve(inliers, cluster, v_shape->m_plane, "line");
			if (result.second < epsilon)
				new_curve = result.first;
		}
		else if (v_shape->detail_type == "circle")
		{
			auto shape = dynamic_pointer_cast<MyCircle>(v_shape);
			const auto result = fit_curve(inliers, cluster, v_shape->m_plane, "circle");
			if (result.second < epsilon)
				new_curve = result.first;
		}
		else if (v_shape->detail_type == "ellipse")
		{
			auto shape = dynamic_pointer_cast<MyEllipse>(v_shape);
			const auto result = fit_curve(inliers, cluster, v_shape->m_plane, "ellipse");
			if (result.second < epsilon)
				new_curve = result.first;
		}
		else
			throw;

		if (new_curve == nullptr)
			continue;

		new_curve->find_boundary();
		curves.push_back(new_curve);
	}

	return curves;
}

Eigen::MatrixXi assemble(
	std::vector<std::shared_ptr<Shape>>& shapes, 
	const int resolution, 
	const Eigen::Tensor<double, 4>& v_surface_points,
	Point_set& v_boundary,
	const double common_points_threshold,
	const double shape_epsilon,
	const fs::path& v_output,
	const bool debug_viz)
{
	LOG(INFO) << "Start to compute connectivity";

	Eigen::MatrixXi adjacency_matrix(shapes.size(), shapes.size());
	adjacency_matrix.fill(0);
	{
		std::vector<std::vector<int>> id_map(resolution * resolution * resolution);

		// Compute the id map
		for (int i_shape1 = 0; i_shape1 < shapes.size(); ++i_shape1)
			for (const auto& coord : shapes[i_shape1]->cluster.coords)
				id_map[coord[0] * resolution * resolution + coord[1] * resolution + coord[2]].push_back(i_shape1);

		// Compute the adjacency_matrix
		for (int i_shape1 = 0; i_shape1 < shapes.size(); ++i_shape1)
		{
			for (const auto& coord : shapes[i_shape1]->cluster.coords)
			{
				for(int dx = -1;dx<=1;++dx)
					for(int dy = -1;dy<=1;++dy)
						for(int dz = -1;dz<=1;++dz)
						{
							if (!check_range(coord[0] + dx, coord[1] + dy, coord[2] + dz, resolution))
								continue;
							for(const auto& i_shape2 : id_map[(coord[0] + dx) * resolution * resolution + (coord[1] + dy) * resolution + (coord[2] + dz)])
								adjacency_matrix(i_shape1, i_shape2) += 1;
						}
			}
		}

		// Fix adj
		for (int i_shape1 = 0; i_shape1 < shapes.size(); ++i_shape1)
		{
			adjacency_matrix(i_shape1, i_shape1) = 0;
			for (int i_shape2 = i_shape1 + 1; i_shape2 < shapes.size(); ++i_shape2)
				adjacency_matrix(i_shape2, i_shape1) = adjacency_matrix(i_shape1, i_shape2);
		}
	}

	check_adj(adjacency_matrix, shapes.size());

	// Initialize data
	std::vector<int> id_vertices;
	std::vector<int> id_curves;
	std::vector<int> id_surfaces;
	for (int i_shape1 = 0; i_shape1 < shapes.size(); ++i_shape1)
	{
		if (shapes[i_shape1]->type == "surface")
			id_surfaces.push_back(i_shape1);
		else if (shapes[i_shape1]->type == "curve")
			id_curves.push_back(i_shape1);
		else if (shapes[i_shape1]->type == "vertex")
			id_vertices.push_back(i_shape1);
		else throw;
	}

	// Visualize before adding missing curves
	if (debug_viz)
	{
		checkFolder((v_output / "adding_curves").string());
		CGAL::IO::write_point_set((v_output/"adding_curves/boundaries.ply").string(), v_boundary);
		{
			Point_set p;
			auto index_map = p.add_property_map("index", 0).first;
			int i_c = 0;
			for (const int i_curve : id_curves)
			{
				Point_set pp = shapes[i_curve]->sample_parametric(10000);
				for (const auto& coord : pp.points())
					index_map[*p.insert(coord)] = i_c;
				i_c += 1;
			}
			colorize_point_set(p);
			CGAL::IO::write_point_set((v_output/"adding_curves/curves_before_adding.ply").string(), p);
		}
	}

	// Compute boundaries for each surface
	std::vector<std::shared_ptr<my_kd_tree_t>> boundary_kd_trees(shapes.size());
	for (int i_shape: id_surfaces)
	{
		const auto shape1 = shapes[i_shape];

		Point_set boundary_points;
		boundary_points.resize(shape1->boundary_points.size());
		for (int i = 0; i < shape1->boundary_points.size(); ++i)
			boundary_points.point(i) = eigen_2_cgal_point(shape1->boundary_points[i]);
		boundary_kd_trees[i_shape].reset(initialize_kd_tree(boundary_points));
	}

	const double nearby_threshold = 5e-2; // To decide whether two points on the boundary of the same shape are close
	const double curve_merge_epsilon = 2e-2;

	// Fix adjacency using the boundaries
	// If a surface and a curve is adjacent, the distance between the curve and the surface should be no more than shape_epsilon
	for (int i_surface : id_surfaces)
	{
		std::shared_ptr<Shape> surface = shapes[i_surface];
		for (int i_curve : id_curves)
		{
			if (adjacency_matrix(i_surface, i_curve) == 0)
				continue;
			std::shared_ptr<Shape> curve = shapes[i_curve];
			// Check validity
			std::vector<double> errors(curve->inliers.size());
			for (int i = 0; i < curve->inliers.size(); ++i)
				errors[i] = surface->distance(curve->inliers[i]);

			const double error = std::accumulate(errors.begin(), errors.end(), 0.) / curve->inliers.size();
			if (error > shape_epsilon)
			{
				adjacency_matrix(i_surface, i_curve) = 0;
				adjacency_matrix(i_curve, i_surface) = 0;
			}
		}
	}

	check_adj(adjacency_matrix, shapes.size());

	LOG(INFO) << "Start to add missing curves";
	Eigen::MatrixXi is_visited(shapes.size(), shapes.size());
	is_visited.setConstant(0);
	for (int ii_surface1 =0; ii_surface1 < id_surfaces.size();++ii_surface1)
	{
		const int i_surface = id_surfaces[ii_surface1];
		std::shared_ptr<Shape3D> shape1 = dynamic_pointer_cast<Shape3D>(shapes[i_surface]);
		if (shape1->type != "surface")
			throw;

		// Find adjacent surfaces and curves
		std::vector<int> adjacent_surface;
		std::vector<int> adjacent_curves;
		{
			for (const int i_shape2: id_surfaces)
			{
				if (i_surface == i_shape2 || adjacency_matrix(i_surface, i_shape2) == 0)
					continue;
				adjacent_surface.push_back(i_shape2);
			}

			for (int i_shape2 : id_curves)
			{
				if (adjacency_matrix(i_surface, i_shape2) == 0)
					continue;
				adjacent_curves.push_back(i_shape2);
			}
		}

		std::shared_ptr<my_kd_tree_t> boundary_kdtree = boundary_kd_trees[i_surface];
		const int num_boundary_points = shape1->boundary_points.size();
		Eigen::MatrixXi boundary_adjacency_matrix(num_boundary_points, num_boundary_points);
		boundary_adjacency_matrix.setZero();
		// Initial and visualize the boundary
		{
			Point_set boundary_points;
			boundary_points.resize(num_boundary_points);

			for (int i = 0; i < num_boundary_points; ++i)
				boundary_points.point(i) = eigen_2_cgal_point(shape1->boundary_points[i]);
			if (debug_viz)
				CGAL::IO::write_point_set((v_output/"adding_curves/single_boundary.ply").string(), boundary_points);

			for(int i=0;i<num_boundary_points;++i)
			{
				const Eigen::Vector3d& p = shape1->boundary_points[i];
				auto results = search_range(*boundary_kdtree, p.cast<float>(), nearby_threshold);
				for (const auto& result : results)
				{
					if (result.first == i)
						continue;
					boundary_adjacency_matrix(i, result.first) = 1;
				}
			}
		}

		// Compute new curves
		{
			std::vector<std::shared_ptr<Shape>> new_curves;
			std::vector<int> id_adj_surface;

			std::vector<bool> v_visited(num_boundary_points, false);
			for (int iadj=0;iadj< adjacent_surface.size();++iadj)
			{
				if (is_visited(i_surface, adjacent_surface[iadj]) == 1)
					continue;
				is_visited(i_surface, adjacent_surface[iadj]) = 1;
				is_visited(adjacent_surface[iadj], i_surface) = 1;
				const auto shape2 = dynamic_pointer_cast<Shape3D>(shapes[adjacent_surface[iadj]]);

				// Check the validity of the intersection
				// Find common points of two surfaces
				std::vector<Eigen::Vector3d> common_points;
				std::vector<int> id_adj; // The id of boundary points in surface 1
				{
					for (int ip1 = 0; ip1 < shape1->boundary_points.size(); ++ip1)
					{
						const auto result = search_k_neighbour(*boundary_kd_trees[adjacent_surface[iadj]],
							shape1->boundary_points[ip1].cast<float>(), 1);
						if (result.second[0] < common_points_threshold * common_points_threshold)
						{
							common_points.push_back(shape1->boundary_points[ip1]);
							id_adj.push_back(ip1);
						}
					}
				}
				// Debug the common points between two surfaces
				if (debug_viz)
				{
					export_point_cloud((v_output / "adding_curves/another_boundary.ply").string(), shape2->boundary_points);
					export_point_cloud((v_output / "adding_curves/common_points.ply").string(), common_points);
				}

				// Perform intersection
				std::vector<std::shared_ptr<Shape>> added_curves;
				if (shape1->detail_type == "cylinder" && shape2->detail_type == "cylinder"
					|| shape1->detail_type == "cone" && shape2->detail_type == "plane" || 
					shape1->detail_type == "plane" && shape2->detail_type == "cone" ||
					shape1->detail_type == "cylinder" && shape2->detail_type == "cone" ||
					shape1->detail_type == "cone" && shape2->detail_type == "cylinder")
				{
					IntAna_IntQuadQuad inter1;
					if (shape1->detail_type == "cylinder" && shape2->detail_type == "cylinder")
					{
						gp_Cylinder cyn1 = dynamic_pointer_cast<MyCylinder>(shape1)->cylinder;
						gp_Cylinder cyn2 = dynamic_pointer_cast<MyCylinder>(shape2)->cylinder;
						inter1.Perform(cyn1, cyn2, 1e-3);
					}
					else if (shape1->detail_type == "cone" && shape2->detail_type == "plane")
					{
						gp_Cone cyn1 = dynamic_pointer_cast<MyCone>(shape1)->cone;
						gp_Pln cyn2 = dynamic_pointer_cast<MyPlane>(shape2)->plane;
						inter1.Perform(cyn1, cyn2, 1e-3);
					}
					else if (shape2->detail_type == "cone" && shape1->detail_type == "plane")
					{
						gp_Cone cyn1 = dynamic_pointer_cast<MyCone>(shape2)->cone;
						gp_Pln cyn2 = dynamic_pointer_cast<MyPlane>(shape1)->plane;
						inter1.Perform(cyn1, cyn2, 1e-3);
					}
					else if (shape1->detail_type == "cylinder" && shape2->detail_type == "cone")
					{
						gp_Cylinder cyn1 = dynamic_pointer_cast<MyCylinder>(shape1)->cylinder;
						gp_Cone cyn2 = dynamic_pointer_cast<MyCone>(shape2)->cone;
						inter1.Perform(cyn1, cyn2, 1e-3);
					}
					else if (shape1->detail_type == "cone" && shape2->detail_type == "cylinder")
					{
						gp_Cylinder cyn1 = dynamic_pointer_cast<MyCylinder>(shape2)->cylinder;
						gp_Cone cyn2 = dynamic_pointer_cast<MyCone>(shape1)->cone;
						inter1.Perform(cyn1, cyn2, 1e-3);
					}
					if (!inter1.IsDone())
						continue;

					LOG(INFO) << "Final RANSAC on remaining points";
					{
						Cluster cluster;
						cluster.surface_points = common_points;
						int num_iter = 0;
						while (cluster.surface_points.size() > 20 && num_iter < 5)
						{
							std::shared_ptr<Shape> plane = fall_back_ransac(cluster, 0.005, "plane", 0.2);
							if (plane == nullptr)
								break;
							else
							{
								Cluster remain;
								plane->cluster.surface_points.clear();
								for (int i = 0; i < cluster.surface_points.size(); ++i)
									if (plane->distance(cluster.surface_points[i]) < 0.005)
										plane->cluster.surface_points.emplace_back(cluster.surface_points[i]);
									else
										remain.surface_points.push_back(cluster.surface_points[i]);
								export_point_cloud("1.ply", plane->cluster.surface_points);
								if (plane->cluster.surface_points.size() < 10)
									break;
								auto result_shape = fit_curve(plane->cluster.surface_points, plane->cluster, dynamic_pointer_cast<MyPlane>(plane)->plane, "ellipse");
								if (result_shape.first == nullptr || result_shape.second > shape_epsilon) // Threshold that marks invalid curve
									break;
								auto& ellipse = result_shape.first;
								if (ellipse->cluster.surface_points.size() < 10)
									break;

								ellipse->get_inliers(plane->cluster.surface_points, shape_epsilon);

								ellipse = check_valid_ellipse(ellipse, ellipse->inliers);
								if(ellipse == nullptr)
									break;
								ellipse->find_boundary();

								cluster = remain;

								LOG(INFO) << ffmt("Fallback ransac found %s of %d points in cluster") % ellipse->detail_type %
									ellipse->cluster.surface_points.size();
								// shapes.push_back(shape);
								added_curves.emplace_back(ellipse);
							}
							num_iter++;
						}
					}
				}
				else
				{
					std::vector<std::shared_ptr<Shape2D>> fixed_shapes;
					fixed_shapes = find_intersection(shape1, shape2);
					if (fixed_shapes.empty())
						continue;

					if (fixed_shapes.size() > 2)
					{
						LOG(ERROR) << "intersection.size()>2";
						// throw;
					}
					for (int i_intersection = 0; i_intersection < fixed_shapes.size(); ++i_intersection)
					{
						auto [local_curves, error] = expand_segments(
							fixed_shapes[i_intersection],
							boundary_adjacency_matrix(id_adj, id_adj),
							common_points, shape_epsilon);

						for(const auto& item: local_curves)
							added_curves.emplace_back(item);
					}

					if (added_curves.empty()) //Uninitialized
						continue;
				}

				// Debug the added curve between two surfaces
				if (debug_viz)
				{
					Point_set p;
					for(const auto& item: added_curves)
					{
						Point_set pp = item->sample_parametric(10000);
						for (const auto& coord : pp.points())
							p.insert(coord);
					}
					CGAL::IO::write_point_set((v_output / "adding_curves/added_points.ply").string(), p);
				}

				for (const auto& item : added_curves)
				{
					new_curves.push_back(item);
					id_adj_surface.push_back(adjacent_surface[iadj]);
				}
			}

			// Debug added curve in this surface
			if (debug_viz)
			{
				Point_set p_inliers, p_samples;
				auto index_inliers = p_inliers.add_property_map("index", 0).first;
				auto index_samples = p_samples.add_property_map("index", 0).first;
				int num_curves = 0;
				for (const auto& i_curve : new_curves)
				{
					if (i_curve->inliers.empty())
						continue;
					for (const auto& coord : i_curve->inliers)
						index_inliers[*p_inliers.insert(eigen_2_cgal_point(coord))] = num_curves;

					Point_set pp = i_curve->sample_parametric(10000);
					for (const auto& coord : pp.points())
						index_samples[*p_samples.insert(coord)] = num_curves;
					num_curves++;

					colorize_point_set(p_inliers);
					colorize_point_set(p_samples);
				}
				CGAL::IO::write_point_set((v_output / "adding_curves/curves_after_adding_i.ply").string(), p_inliers);
				CGAL::IO::write_point_set((v_output / "adding_curves/curves_after_adding_s.ply").string(), p_samples);
			}

			for (int i_new=0;i_new<new_curves.size();++i_new)
			{
				const int id_new_shape = shapes.size();
				shapes.push_back(new_curves[i_new]);

				adjacency_matrix.conservativeResize(shapes.size(), shapes.size());
				adjacency_matrix.row(id_new_shape).setZero();
				adjacency_matrix.col(id_new_shape).setZero();

				adjacency_matrix(i_surface, id_new_shape) = 1;
				adjacency_matrix(id_new_shape, i_surface) = 1;
				adjacency_matrix(id_adj_surface[i_new], id_new_shape) = 1;
				adjacency_matrix(id_new_shape, id_adj_surface[i_new]) = 1;
			}
		}
	}

	check_adj(adjacency_matrix, shapes.size());

	// debug the added curves
	if (debug_viz)
		debug_shapes(shapes, v_output);

	LOG(INFO) << "Start to merge curves";
	shapes = merge_shape(shapes, curve_merge_epsilon, resolution, adjacency_matrix, "curve");
	LOG(INFO) << ffmt("%d shapes remain") % shapes.size();
	check_adj(adjacency_matrix, shapes.size());
	if (debug_viz)
		debug_shapes(shapes, v_output);
	{
		id_surfaces.clear();
		id_curves.clear();
		id_vertices.clear();
		for (int i_shape1 = 0; i_shape1 < shapes.size(); ++i_shape1)
		{
			if (shapes[i_shape1]->type == "surface")
				id_surfaces.push_back(i_shape1);
			else if (shapes[i_shape1]->type == "curve")
				id_curves.push_back(i_shape1);
			else if (shapes[i_shape1]->type == "vertex")
				id_vertices.push_back(i_shape1);
			else throw;
		}
		for (const int i_curve : id_curves)
			shapes[i_curve]->find_boundary();
	}

	std::vector<int> remain_rows;
	for(int i= shapes.size()-1;i>=0;--i)
	{
		if (shapes[i]->inliers.size() < 10)
			shapes.erase(shapes.begin() + i);
		else
			remain_rows.push_back(i);
	}
	std::sort(remain_rows.begin(), remain_rows.end());
	const Eigen::MatrixXi a = adjacency_matrix(remain_rows, remain_rows);
	adjacency_matrix = a;
	check_adj(adjacency_matrix, shapes.size());

	id_vertices.clear();
	id_curves.clear();
	id_surfaces.clear();
	for (int i_shape1 = 0; i_shape1 < shapes.size(); ++i_shape1)
	{
		if (shapes[i_shape1]->type == "surface")
			id_surfaces.push_back(i_shape1);
		else if (shapes[i_shape1]->type == "curve")
			id_curves.push_back(i_shape1);
		else if (shapes[i_shape1]->type == "vertex")
			id_vertices.push_back(i_shape1);
		else throw;
	}

	// debug the added curves
	if (debug_viz)
		debug_shapes(shapes, v_output);

	fix_adjacency_matrix(shapes,adjacency_matrix, shape_epsilon);
	check_adj(adjacency_matrix, shapes.size());

	return adjacency_matrix;
}

void bak_build_occt()
{
	/*
	LOG(INFO) << "Start to add missing vertex";
	{
		std::vector<std::shared_ptr<my_kd_tree_t>> kdtrees(shapes.size(), nullptr);
		#pragma omp parallel for
		for (int i_shape = 0; i_shape < shapes.size(); ++i_shape)
		{
			std::vector<Eigen::Vector3d> p;
			for (int i = 0; i < shapes[i_shape]->inliers.size(); ++i)
				p.push_back(shapes[i_shape]->inliers[i]);
			kdtrees[i_shape].reset(initialize_kd_tree(to_point_set(p)));
		}

		std::vector<std::pair<int, int>> curve_topology;
		for (int i_surface : id_surfaces)
		{
			std::vector<int> curves;
			for (int i_curve : id_curves)
				if (adjacency_matrix(i_surface, i_curve) > 0)
					curves.push_back(i_curve);

			std::vector<Eigen::Vector3d> end_points;
			for (const auto& i_curve : curves)
			{
				Shape2D* curve = dynamic_pointer_cast<Shape2D>(shapes[i_curve]).get();
				Eigen::Vector3d p1 = curve->get_cartesian(curve->min_t);
				Eigen::Vector3d p2 = curve->get_cartesian(curve->max_t);
				end_points.push_back(p1);
				end_points.push_back(p2);
			}

			std::vector<bool> visited_flags(end_points.size(), false);
			for (int i_p = 0; i_p < end_points.size(); ++i_p)
			{
				if (visited_flags[i_p])
					continue;
				const int i_curve1 = i_p / 2;
				int best_end_point = -1;
				double best_distance = 100;
				for (int j_p = 0; j_p < end_points.size(); ++j_p)
				{
					const int i_curve2 = j_p / 2;
					if (i_curve1 == i_curve2 || visited_flags[j_p])
						continue;
					double distance = (end_points[i_p] - end_points[j_p]).norm();
					if (distance < best_distance)
					{
						best_distance = distance;
						best_end_point = j_p;
					}
				}
				if (best_distance < 5e-2)
				{
					const int i_curve2 = best_end_point / 2;
					Shape* new_shape = find_intersection(
						dynamic_pointer_cast<Shape2D>(shapes[curves[i_curve1]]).get(),
						dynamic_pointer_cast<Shape2D>(shapes[curves[i_curve2]]).get());
					if (new_shape != nullptr)
					{
						if ((((Shape1D*)new_shape)->vertex - end_points[best_end_point]).norm() > 5e-2)
						{
							delete new_shape;
							continue;
						}
						// visited_flags[i_p] = true;
						// visited_flags[best_end_point] = true;
						int existing_id = -1;
						for (const auto& id_vertex : id_vertices)
						{
							if (shapes[id_vertex]->distance(((Shape1D*)new_shape)->vertex) < 2e-2)
							{
								existing_id = id_vertex;
								break;
							}
						}
						if (existing_id == -1)
						{
							adjacency_matrix.conservativeResize(shapes.size() + 1, shapes.size() + 1);
							existing_id = shapes.size();
							adjacency_matrix.row(existing_id).fill(0);
							adjacency_matrix.col(existing_id).fill(0);
							id_vertices.push_back(existing_id);
							shapes.emplace_back(new_shape);
						}
						// Surface and vertex
						adjacency_matrix(i_surface, existing_id) = 1;
						adjacency_matrix(existing_id, i_surface) = 1;
						// Curve and curve
						adjacency_matrix(curves[i_curve1], curves[i_curve2]) = 1;
						adjacency_matrix(curves[i_curve2], curves[i_curve1]) = 1;
						// Curve and vertex
						adjacency_matrix(curves[i_curve1], existing_id) = 1;
						adjacency_matrix(existing_id, curves[i_curve1]) = 1;
						adjacency_matrix(curves[i_curve2], existing_id) = 1;
						adjacency_matrix(existing_id, curves[i_curve2]) = 1;
					}
				}
			}
		}
	}

	fix_adjacency_matrix(shapes, adjacency_matrix, 2e-2);
	check_adj(adjacency_matrix, shapes.size());

	// Remove isolate curves
	{
		for (int i_curve : id_curves)
		{
			std::vector<int> id_endpoints;
			for (int i_vertex : id_vertices)
				if (adjacency_matrix(i_curve, i_vertex) > 0)
					id_endpoints.push_back(i_vertex);

			if (id_endpoints.size() != 2)
			{
				LOG(INFO) << ffmt("Found iosolated curves: %d, which has %d endpoints") % i_curve % id_endpoints.size();
				adjacency_matrix.row(i_curve).fill(0);
				adjacency_matrix.col(i_curve).fill(0);
			}
		}
	}

	// debug the vertices and curves
	{
		{
			Point_set p;
			auto index_map = p.add_property_map("index", 0).first;
			int num_vertices = 0;
			for (int i = 0; i < shapes.size(); ++i)
			{
				if (shapes[i]->type == "vertex")
				{

					index_map[*p.insert(eigen_2_cgal_point(dynamic_pointer_cast<Shape1D>(shapes[i])->vertex))] = num_vertices;
					num_vertices++;
				}
			}
			colorize_point_set(p);
			CGAL::IO::write_point_set("temp/vertices_after_adding.ply", p);
		}

		for (int i_vertex : id_vertices)
		{
			Point_set p;
			auto index_map = p.add_property_map("index", 0).first;
			int num_curves = 1;

			index_map[*p.insert(eigen_2_cgal_point(dynamic_pointer_cast<Shape1D>(shapes[i_vertex])->vertex))] = 0;

			for (int i_curve : id_curves)
			{
				if (adjacency_matrix(i_curve, i_vertex) == 0)
					continue;
				for (const auto& item : shapes[i_curve]->inliers)
					index_map[*p.insert(eigen_2_cgal_point(item))] = num_curves;
				num_curves++;
			}
			colorize_point_set(p);
			CGAL::IO::write_point_set("temp/vertex_and_curve.ply", p);
		}
	}

	{
		adjacency_matrix;
		BRepLib::Precision(5e-2);
		BRep_Builder shell_maker;

		fs::path output_root = "temp/out";
		checkFolder(output_root);

		for (int i_surface : id_surfaces)
		{
			Point_set p;
			auto index_map = p.add_property_map("index", 0).first;
			BRepBuilderAPI_MakeWire wireMaker;
			int cur_iter = 0;

			std::queue<int> target_curves;
			std::vector<bool> visited_flag(shapes.size(), false);
			for (int i_curve : id_curves)
			{
				if (adjacency_matrix(i_surface, i_curve) == 0)
					continue;
				target_curves.push(i_curve);
				break;
			}

			if (target_curves.empty())
			{
				LOG(INFO) << ffmt("Surface %d has no curves") % i_surface;
				continue;
			}

			int past_id_vertex = -1;
			while (!target_curves.empty())
			{
				const int i_curve = target_curves.front();
				visited_flag[i_curve] = true;
				target_curves.pop();

				// Calculate the endpoints
				std::vector<Eigen::Vector3d> end_points;
				std::vector<int> id_vertices_local;
				for (const auto& i_vertex : id_vertices)
					if (adjacency_matrix(i_curve, i_vertex) != 0)
					{

						end_points.push_back(dynamic_pointer_cast<Shape1D>(shapes[i_vertex])->vertex);
						id_vertices_local.push_back(i_vertex);
					}
				if (end_points.size() != 2)
					// LOG(INFO) << i_curve;
					continue;

				if (past_id_vertex == -1)
					past_id_vertex = id_vertices_local[0];


				Shape2D* curve = dynamic_pointer_cast<Shape2D>(shapes[i_curve]).get();
				if (curve->detail_type == "line")
				{
					TopoDS_Edge edge = BRepBuilderAPI_MakeEdge(
						((MyLine*)curve)->line,
						gp_Pnt(end_points[0].x(), end_points[0].y(), end_points[0].z()),
						gp_Pnt(end_points[1].x(), end_points[1].y(), end_points[1].z())
					);
					wireMaker.Add(edge);
				}
				else if (curve->detail_type == "circle")
				{
					TopoDS_Edge edge = BRepBuilderAPI_MakeEdge(
						((MyCircle*)curve)->circle,
						gp_Pnt(end_points[0].x(), end_points[0].y(), end_points[0].z()),
						gp_Pnt(end_points[1].x(), end_points[1].y(), end_points[1].z())
					);
					wireMaker.Add(edge);
				}
				else if (curve->detail_type == "ellipse")
				{
					TopoDS_Edge edge = BRepBuilderAPI_MakeEdge(
						((MyEllipse*)curve)->ellipse,
						gp_Pnt(end_points[0].x(), end_points[0].y(), end_points[0].z()),
						gp_Pnt(end_points[1].x(), end_points[1].y(), end_points[1].z())
					);
					wireMaker.Add(edge);
				}
				else
					throw;

				for (const auto& item : shapes[i_curve]->inliers)
					index_map[*p.insert(eigen_2_cgal_point(item))] = cur_iter;
				cur_iter++;

				// Add the next
				for (const int i_target_curve : id_curves)
				{
					if (visited_flag[i_target_curve] ||
						adjacency_matrix(i_surface, i_target_curve) == 0)
						continue;

					int target_vertex = 0;
					if (past_id_vertex == id_vertices_local[0])
						target_vertex = 1;

					if (adjacency_matrix(i_target_curve, id_vertices_local[target_vertex]) == 0)
						continue;
					target_curves.push(i_target_curve);
					past_id_vertex = id_vertices_local[target_vertex];
					break;
				}
			}
			colorize_point_set(p);
			CGAL::IO::write_point_set("temp/temp.ply", p);

			TopoDS_Wire wire = wireMaker.Wire();
			bool is_closed = wire.Closed();
			LOG(INFO) << ffmt("Wire %d: Closed: %d") % i_surface % is_closed;

			if (is_closed)
			{
				std::shared_ptr<BRepBuilderAPI_MakeFace> faceMaker;
				if (shapes[i_surface]->detail_type == "plane")
					faceMaker.reset(new BRepBuilderAPI_MakeFace(dynamic_pointer_cast<MyPlane>(shapes[i_surface])->plane, wire, true));
				else if (shapes[i_surface]->detail_type == "cylinder")
					faceMaker.reset(new BRepBuilderAPI_MakeFace((dynamic_pointer_cast<MyCylinder>(shapes[i_surface]))->cylinder, wire, true));
				else if (shapes[i_surface]->detail_type == "sphere")
					faceMaker.reset(new BRepBuilderAPI_MakeFace(dynamic_pointer_cast<MySphere>(shapes[i_surface])->sphere, wire, true));
				else if (shapes[i_surface]->detail_type == "cone")
					faceMaker.reset(new BRepBuilderAPI_MakeFace(dynamic_pointer_cast<MyCone>(shapes[i_surface])->cone, wire, true));
				else if (shapes[i_surface]->detail_type == "torus")
					faceMaker.reset(new BRepBuilderAPI_MakeFace(dynamic_pointer_cast<MyTorus>(shapes[i_surface])->torus, wire, true));
				else
					throw;

				if (!faceMaker->IsDone())
				{
					throw;
				}
				TopoDS_Face triangleFace = faceMaker->Face();
				BRepMesh_IncrementalMesh mesh(triangleFace, 5e-2, false, 0.5);
				StlAPI_Writer objWriter;
				bool write_flag = objWriter.Write(triangleFace,
					(output_root / (ffmt("%d.stl") % i_surface).str()).string().c_str());
				LOG(INFO) << ffmt("Face flag %d: Write flag: %d") % mesh.GetStatusFlags() % write_flag;
			}
			else
			{
				auto surface = dynamic_pointer_cast<Shape3D>(shapes[i_surface]);

				std::vector<Point_2> parametrics(surface->inliers.size());
				#pragma omp parallel for
				for (int i = 0; i < surface->inliers.size(); ++i)
				{
					parametrics[i] = eigen_2_cgal_point(surface->get_parametric(surface->inliers[i]));
				}

				Alpha_shape_2 as(parametrics.begin(), parametrics.end(), 0.0001, Alpha_shape_2::GENERAL);
				std::vector<Triangle_3> mesh;

				for (Alpha_shape_2::Finite_faces_iterator it = as.finite_faces_begin();
					it != as.finite_faces_end(); ++it)
				{
					if (as.classify(it) != Alpha_shape_2::EXTERIOR)
					{
						Alpha_shape_2::Triangle triangle = as.triangle(it);
						Point_2 p1 = triangle.vertex(0);
						Point_2 p2 = triangle.vertex(1);
						Point_2 p3 = triangle.vertex(2);

						mesh.emplace_back(
							eigen_2_cgal_point(surface->get_cartesian(Eigen::Vector2d(p1.x(), p1.y()))),
							eigen_2_cgal_point(surface->get_cartesian(Eigen::Vector2d(p2.x(), p2.y()))),
							eigen_2_cgal_point(surface->get_cartesian(Eigen::Vector2d(p3.x(), p3.y())))
						);
					}
				}

				write_ply((output_root / (ffmt("%d.ply") % i_surface).str()).string().c_str(),
					mesh);
				LOG(INFO) << ffmt("Write ply with unclosed wires");
			}
		}

	}
	return adjacency_matrix;*/
}