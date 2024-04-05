#include "shape3d.h"

BOOST_CLASS_EXPORT(Shape3D)
BOOST_CLASS_EXPORT(MyCylinder)
BOOST_CLASS_EXPORT(MyCone)
BOOST_CLASS_EXPORT(MySphere)
BOOST_CLASS_EXPORT(MyTorus)
BOOST_CLASS_EXPORT(MyEllipsoid)
BOOST_CLASS_EXPORT(MyPlane)

void Shape3D::find_boundary()
{
	std::vector<Point_2> parametrics(inliers.size());
	for (int i = 0; i < inliers.size(); ++i)
		parametrics[i] = eigen_2_cgal_point(get_parametric(inliers[i]));
	Alpha_shape_2 as(
		parametrics.begin(), parametrics.end(), 0.0009, Alpha_shape_2::GENERAL);

	const double sample_density = 0.001; // For sample boundary points
	boundary_points.clear();
	for (auto avit = as.finite_edges_begin(),
		avit_end = as.finite_edges_end();
		avit != avit_end; ++avit)
	{
		if (as.classify(*avit)!= Alpha_shape_2::REGULAR)
			continue;
		Point_2 p1 = as.segment(*avit).source();
		Point_2 p2 = as.segment(*avit).target();
		const int num_samples = std::ceil(std::sqrt(CGAL::squared_distance(p1, p2)) / sample_density);
		const Eigen::Vector2d step_dir((p2.x() - p1.x()) / num_samples, (p2.y() - p1.y()) / num_samples);
		for(int i = 0; i < num_samples; ++i)
			boundary_points.emplace_back(get_cartesian(
				Eigen::Vector2d(p1.x() , p1.y()) + step_dir * i));
		// boundary_points.emplace_back(get_cartesian(Eigen::Vector2d(p1.x(), p1.y())));
		boundary_points.emplace_back(get_cartesian(Eigen::Vector2d(p2.x(), p2.y())));
	}

	// Construct triangles
	m_boundary.clear();
	area_sum = 0;
	for (Alpha_shape_2::Finite_faces_iterator it = as.finite_faces_begin();
		it != as.finite_faces_end(); ++it)
	{
		if (as.classify(it) != Alpha_shape_2::EXTERIOR)
		{
			Alpha_shape_2::Triangle triangle = as.triangle(it);
			m_boundary.push_back(triangle);
			area_sum += triangle.area();
		}
	}
}

Eigen::Vector3d MyCylinder::get_cartesian(const Eigen::Vector2d& uv) const
{
	gp_Pnt p;
	ElSLib::CylinderD0(uv[0] * 2 * M_PI, uv[1], cylinder.Position(), cylinder.Radius(), p);
	return Eigen::Vector3d(p.X(), p.Y(), p.Z());
}

Eigen::Vector2d MyCylinder::get_parametric(const Eigen::Vector3d& v_point) const
{
	double u, v;
	ElSLib::CylinderParameters(cylinder.Position(), cylinder.Radius(), gp_Pnt(v_point[0], v_point[1], v_point[2]),
		u, v);
	return Eigen::Vector2d(u / 2 / M_PI, v);
}

Eigen::Vector3d MyCone::get_cartesian(const Eigen::Vector2d& v_point) const
{
	gp_Pnt p;
	ElSLib::ConeD0(v_point[0] * 2 * M_PI, v_point[1], cone.Position(), cone.RefRadius(), cone.SemiAngle(), p);
	return { p.X(), p.Y(), p.Z() };
}

Eigen::Vector2d MyCone::get_parametric(const Eigen::Vector3d& v_point) const
{
	double u, v;
	ElSLib::ConeParameters(cone.Position(), cone.RefRadius(), cone.SemiAngle(),
		gp_Pnt(v_point[0], v_point[1], v_point[2]), u, v);
	return { u / 2 / M_PI, v };
}

// #pragma optimize("", off)
// Surface_mesh MyCylinder::extract_alpha_mesh(const double epsilon) const
// {
// 	std::vector<Point_2> parametrics;
// 	std::vector<double> vs;
// 	for (int i = 0; i < cluster.surface_points.size(); ++i)
// 		if (distance(cluster.surface_points[i]) < epsilon)
// 		{
// 			parametrics.emplace_back(eigen_2_cgal_point(get_parametric(cluster.surface_points[i])));
// 			vs.push_back(parametrics.back().y());
// 			std::cout << parametrics.back().x() << ", " << parametrics.back().y() << std::endl;
// 		}
//
// 	P2DT2 pdt(PDT::Iso_rectangle_2(0, *std::min_element(vs.begin(), vs.end()), 2 * M_PI, *std::max_element(vs.begin(), vs.end())));
// 	pdt.insert(parametrics.begin(), parametrics.end(), true);
// 	if (pdt.is_triangulation_in_1_sheet())
// 		pdt.convert_to_1_sheeted_covering();
// 	// compute alpha shape
// 	PAlpha_shape_2 as(pdt);
// 	as.set_alpha(0.0009);
//
// 	// Construct triangles
// 	Surface_mesh triangles;
// 	for (PAlpha_shape_2::Finite_faces_iterator avit = as.finite_faces_begin();
// 		avit != as.finite_faces_end(); ++avit)
// 	{
// 		if (as.classify(avit) == Alpha_shape_2::EXTERIOR)
// 			continue;
// 		PAlpha_shape_2::Triangle triangle = as.triangle(avit);
// 		Point_2 p1 = as.triangle(avit).vertex(0);
// 		Point_2 p2 = as.triangle(avit).vertex(1);
// 		Point_2 p3 = as.triangle(avit).vertex(2);
// 		Point_3 p1_3d = eigen_2_cgal_point(get_cartesian(Eigen::Vector2d(p1.x(), p1.y())));
// 		Point_3 p2_3d = eigen_2_cgal_point(get_cartesian(Eigen::Vector2d(p2.x(), p2.y())));
// 		Point_3 p3_3d = eigen_2_cgal_point(get_cartesian(Eigen::Vector2d(p3.x(), p3.y())));
// 		triangles.add_face(
// 			triangles.add_vertex(p1_3d),
// 			triangles.add_vertex(p2_3d),
// 			triangles.add_vertex(p3_3d));
// 	}
// 	return triangles;
// }
