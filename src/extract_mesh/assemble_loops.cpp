#include "assemble_loops.h"

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
#include <GeomAPI_ExtremaCurveCurve.hxx>
#include <Geom_Circle.hxx>
#include <Geom_Line.hxx>
#include <GeomAPI_IntCS.hxx>
#include <Geom2dAPI_InterCurveCurve.hxx>

#include "tools.h"
#include <tbb/tbb.h>
#include <IntTools_EdgeEdge.hxx>

#include "kd_tree_helper.h"
#include "merge_shape.h"

// #pragma optimize("", off)

std::vector<Eigen::Vector3d> find_curve_curve_intersection(const std::shared_ptr<Shape2D>& v_shape1, const std::shared_ptr<Shape2D>& v_shape2, const double v_tolerance)
{
	// GeomAPI_IntCS inter;
	GeomAPI_ExtremaCurveCurve inter;
	if (v_shape1->detail_type == "line" && v_shape2->detail_type == "line")
	{
		gp_Lin s1, s2;
		s1 = dynamic_pointer_cast<MyLine>(v_shape1)->line;
		s2 = dynamic_pointer_cast<MyLine>(v_shape2)->line;
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
			s1 = dynamic_pointer_cast<MyLine>(v_shape1)->line;
			s2 = dynamic_pointer_cast<MyCircle>(v_shape2)->circle;
		}
		else
		{
			s1 = dynamic_pointer_cast<MyLine>(v_shape2)->line;
			s2 = dynamic_pointer_cast<MyCircle>(v_shape1)->circle;
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
			s1 = dynamic_pointer_cast<MyLine>(v_shape1)->line;
			s2 = dynamic_pointer_cast<MyEllipse>(v_shape2)->ellipse;
		}
		else
		{
			s1 = dynamic_pointer_cast<MyLine>(v_shape2)->line;
			s2 = dynamic_pointer_cast<MyEllipse>(v_shape1)->ellipse;
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
			s1 = dynamic_pointer_cast<MyCircle>(v_shape1)->circle;
			s2 = dynamic_pointer_cast<MyEllipse>(v_shape2)->ellipse;
		}
		else
		{
			s1 = dynamic_pointer_cast<MyCircle>(v_shape2)->circle;
			s2 = dynamic_pointer_cast<MyEllipse>(v_shape1)->ellipse;
		}
		l1 = new Geom_Circle(s1);
		l2 = new Geom_Ellipse(s2);
		inter.Init(l1, l2);
	}
	else
		return {};

	if (inter.NbExtrema() == 0 || !inter.Extrema().IsDone() || inter.Extrema().IsParallel())
		return {};

	std::vector<Eigen::Vector3d> points;
	try
	{
		for (int i = 1; i <= inter.NbExtrema(); ++i)
		{
			gp_Pnt p1, p2;
			inter.Points(i, p1, p2);
			if (p1.Distance(p2) > v_tolerance)
				continue;
			points.emplace_back(Eigen::Vector3d(p1.X() + p2.X(), p1.Y() + p2.Y(), p1.Z() + p2.Z()) / 2);
		}
	}
	catch (...)
	{
		return {};
	}
	
	return points;
}

void debug_curve(const std::shared_ptr<Shape2D>& v_curves, const fs::path& v_output_path)
{
	std::vector<Eigen::Vector3d> points;
	for (const auto& item : v_curves->inliers)
		points.push_back(item);
	export_points(v_output_path, points);
}

void solve_vertex(std::vector<std::shared_ptr<Shape>>& v_shapes, const double vertex_threshold,
	const double resolution,
	Eigen::MatrixXi& v_adj_matrix, const fs::path& v_output_dir, bool debug_viz)
{
	const double intersection_tolerance = vertex_threshold;
	if (debug_viz)
		checkFolder(v_output_dir / "adding_vertex");
	Eigen::MatrixXi curve_adjacency(v_shapes.size(), v_shapes.size());
	curve_adjacency.setConstant(0);

	// Curve adjacency
	tbb::parallel_for(tbb::blocked_range<int>(0, v_shapes.size()), [&](const tbb::blocked_range<int>& r0)
		{
			for (int i_shape1 = r0.begin(); i_shape1 < r0.end(); ++i_shape1)
				for (int i_shape2 = i_shape1 + 1; i_shape2 < v_shapes.size(); ++i_shape2)
					if (v_shapes[i_shape1]->type == "curve" && v_shapes[i_shape2]->type == "curve")
					{
						const auto& shape1 = v_shapes[i_shape1];
						const auto& shape2 = v_shapes[i_shape2];
						double minimum_distance = 999.;
						for (int i = 0; i < shape2->inliers.size(); ++i)
							for (int j = 0; j < shape1->inliers.size(); ++j)
								minimum_distance = std::min(minimum_distance, (shape2->inliers[i] - shape1->inliers[j]).norm());

						if (minimum_distance < 5e-2) // threshold 1
						{
							curve_adjacency(i_shape1, i_shape2) = 1;
							curve_adjacency(i_shape2, i_shape1) = 1;
						}
					}
		});

	std::vector<std::shared_ptr<my_kd_tree_t>> kd_trees(v_shapes.size());
	std::vector<matrix_t> kd_trees_data(v_shapes.size());
	tbb::parallel_for(tbb::blocked_range<int>(0, v_shapes.size()), [&](const tbb::blocked_range<int>& r0)
		{
			for (int i_shape1 = r0.begin(); i_shape1 < r0.end(); ++i_shape1)
				if (v_shapes[i_shape1]->type == "curve")
				{
					kd_trees_data[i_shape1] = initialize_kd_data(dynamic_pointer_cast<Shape2D>(v_shapes[i_shape1])->boundary_points);
					kd_trees[i_shape1] = initialize_kd_tree(kd_trees_data[i_shape1]);
				}
		});

	std::vector<int> id_vertices;
	std::vector<int> id_curves;
	std::vector<int> id_surfaces;
	for (int i_shape1 = 0; i_shape1 < v_shapes.size(); ++i_shape1)
	{
		if (v_shapes[i_shape1]->type == "surface")
			id_surfaces.push_back(i_shape1);
		else if (v_shapes[i_shape1]->type == "curve")
			id_curves.push_back(i_shape1);
		else if (v_shapes[i_shape1]->type == "vertex")
			id_vertices.push_back(i_shape1);
		else throw;
	}

	// Compute vertex in each face individually
	std::vector<Eigen::Vector3d> all_added_vertices;
	std::vector<Eigen::Vector2i> id_vertex; // (id_curve1, id_curve2) -> id_vertex in `all_added_vertices`
	std::mutex adding_mutex;
	// tbb::parallel_for(tbb::blocked_range<int>(0, id_surfaces.size()), [&](const tbb::blocked_range<int>& r0)
		{
			// for (int ii_surface = r0.begin(); ii_surface < r0.end(); ++ii_surface)
			for (int ii_surface = 0; ii_surface < id_surfaces.size(); ++ii_surface)
			{
				const int i_surface = id_surfaces[ii_surface];
				const auto& surface = dynamic_pointer_cast<Shape3D>(v_shapes[i_surface]);

				std::vector<int> adjacent_curves;
				for (const int id_curve : id_curves)
				{
					if (v_adj_matrix(i_surface, id_curve) == 0)
						continue;
					adjacent_curves.push_back(id_curve);
				}

				if (adjacent_curves.empty())
				{
					LOG(INFO) << "No adjacent curves for surface " << i_surface;
					continue;
				}

				if (debug_viz)
				{
					std::vector<Eigen::Vector3d> viz_points;
					for (const auto& point : surface->inliers)
						viz_points.push_back(point);
					export_points(v_output_dir / "adding_vertex" / "surface.ply", viz_points);

					viz_points.clear();
					for(const auto& item: adjacent_curves)
						for(const auto& point: v_shapes[item]->inliers)
							viz_points.push_back(point);
					export_points(v_output_dir / "adding_vertex" / "all_curves_this_surface.ply", viz_points);
				}

				for (int ii_curve=0;ii_curve<adjacent_curves.size();++ii_curve)
				{
					const int i_curve1 = adjacent_curves[ii_curve];
					const auto& curve1 = dynamic_pointer_cast<Shape2D>(v_shapes[i_curve1]);

					for (int ii_curve2 = ii_curve+1; ii_curve2 < adjacent_curves.size(); ++ii_curve2)
					{
						const int i_curve2 = adjacent_curves[ii_curve2];
						if (curve_adjacency(i_curve1, i_curve2) == 0)
							continue;
						if (std::find(id_vertex.begin(), id_vertex.end(),Eigen::Vector2i(i_curve1, i_curve2)) != id_vertex.end())
							continue;

						const auto& curve2 = dynamic_pointer_cast<Shape2D>(v_shapes[i_curve2]);

						if (debug_viz)
						{
							debug_curve(curve1, v_output_dir / "adding_vertex" / "curve1.ply");
							debug_curve(curve2, v_output_dir / "adding_vertex" / "curve2.ply");
						}

						std::vector<Eigen::Vector3d> intersection_points = find_curve_curve_intersection(
							dynamic_pointer_cast<Shape2D>(curve1),
							dynamic_pointer_cast<Shape2D>(curve2), intersection_tolerance);

						double error = 9999;
						int id_best = -1;
						for(int i=0;i<intersection_points.size();++i)
						{
							if (surface->inlier_distance(intersection_points[i]) > intersection_tolerance)
								continue;

							double local_error1 = std::sqrt(search_k_neighbour(*kd_trees[i_curve1], intersection_points[i].cast<float>()).second[0]);
							double local_error2 = std::sqrt(search_k_neighbour(*kd_trees[i_curve2], intersection_points[i].cast<float>()).second[0]);
							double local_error = (local_error1 + local_error2) / 2;
							if (local_error < error)
							{
								error = local_error;
								id_best = i;
							}
						}

						if (intersection_points.empty() || id_best == -1)
							continue;

						adding_mutex.lock();
						all_added_vertices.push_back(intersection_points[id_best]);
						id_vertex.emplace_back(
							i_curve1,i_curve2);
						id_vertex.emplace_back(
							i_curve2, i_curve1);
						adding_mutex.unlock();

						if (debug_viz)
						{
							export_points(v_output_dir / "adding_vertex" / "intersection_points_all.ply", intersection_points);
							auto a = intersection_points[id_best];
							intersection_points.clear();
							intersection_points.push_back(a);
							export_points(v_output_dir / "adding_vertex" / "intersection_points_single.ply", intersection_points);
						}

						continue;
					}
				}
			}
		}
	// );
	if (debug_viz)
		export_points(v_output_dir / "adding_vertex" / "vertices_after_adding.ply", all_added_vertices);

	// Add them to the array
	{
		if (all_added_vertices.size()!=id_vertex.size()/2)
			throw;
		for(int i=0;i<all_added_vertices.size();++i)
		{
			std::shared_ptr<Shape1D> vertex(new Shape1D(all_added_vertices[i], Cluster(), { all_added_vertices[i] }));
			vertex->cluster.surface_points.push_back(all_added_vertices[i]);
			v_shapes.push_back(vertex);
			v_adj_matrix.conservativeResize(v_shapes.size(), v_shapes.size());
			v_adj_matrix.row(v_shapes.size() - 1).setZero();
			v_adj_matrix.col(v_shapes.size() - 1).setZero();
			v_adj_matrix(v_shapes.size() - 1, id_vertex[i * 2][0]) = 1;
			v_adj_matrix(v_shapes.size() - 1, id_vertex[i * 2][1]) = 1;
			v_adj_matrix(id_vertex[i * 2][0], v_shapes.size() - 1) = 1;
			v_adj_matrix(id_vertex[i * 2][1], v_shapes.size() - 1) = 1;
		}
	}

	v_shapes = merge_shape(v_shapes, 1e-2, resolution, v_adj_matrix, "vertex");
	if (debug_viz)
	{
		all_added_vertices.clear();
		for(const auto& item: v_shapes)
			if (item->type == "vertex")
				all_added_vertices.push_back(dynamic_pointer_cast<Shape1D>(item)->vertex);
		export_points(v_output_dir / "adding_vertex" / "vertices_after_adding.ply", all_added_vertices);
	}
	return;
}

void assemble_loops(
	std::vector<std::shared_ptr<Shape>>& v_shapes,
	const Eigen::Tensor<double, 4>& v_surface_points,
	Point_set& v_boundary)
{
	
}
