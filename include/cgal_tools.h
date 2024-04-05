#ifndef CGAL_TOOLS_H
#define CGAL_TOOLS_H

#include "common_util.h"

#include <tiny_obj_loader.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangle_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Polygon_2.h>
// #include <CGAL/polygon_mesh_processing.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/IO/read_ply_points.h>
#include <CGAL/IO/write_ply_points.h>
#include <CGAL/property_map.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/point_generators_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point_2;
typedef CGAL::Polygon_2<K> Polygon_2;
typedef CGAL::Line_2<K> Line_2;
typedef CGAL::Segment_2<K> Segment_2;
typedef CGAL::Vector_2<K> Vector_2;
typedef CGAL::Ray_2<K> Ray_2;

typedef K::Point_3 Point_3;
typedef CGAL::Plane_3<K> Plane_3;
typedef CGAL::Triangle_3<K> Triangle_3;
typedef CGAL::Line_3<K> Line_3;
typedef CGAL::Segment_3<K> Segment_3;
typedef CGAL::Direction_3<K> Direction_3;
typedef CGAL::Vector_3<K> Vector_3;
typedef CGAL::Surface_mesh<K::Point_3> Surface_mesh;
typedef CGAL::Polygon_2<K> Polygon_2;
typedef CGAL::Point_set_3<K::Point_3> Point_cloud;
typedef CGAL::Point_set_3<Point_3> Point_set;
typedef Surface_mesh::Face_index face_descriptor;
typedef Surface_mesh::Vertex_index vertex_descriptor;
typedef Surface_mesh::Halfedge_index halfedge_descriptor;

typedef CGAL::Polyhedron_3<K> Polyhedron_3;
typedef CGAL::AABB_face_graph_triangle_primitive<Polyhedron_3> Polyhedron_Primitive;
typedef CGAL::AABB_traits<K, Polyhedron_Primitive> Polyhedron_Traits;
typedef CGAL::AABB_tree<Polyhedron_Traits> Polyhedron_Tree;

typedef std::vector<Triangle_3>::iterator Triangle_Iterator;
typedef CGAL::AABB_triangle_primitive<K, Triangle_Iterator> Triangle_Primitive;
typedef CGAL::AABB_traits<K, Triangle_Primitive> Triangle_AABB_triangle_traits;
typedef CGAL::AABB_tree<Triangle_AABB_triangle_traits> Triangle_Tree;

template<typename T>
inline T cgal_normalized(const T& item)
{
	return item / std::sqrt(item.squared_length());
}

struct Rotated_box
{
	cv::RotatedRect cv_box;
	Eigen::AlignedBox3f box;
	float angle;
	Rotated_box() {};
	Rotated_box(const Eigen::AlignedBox3f& v_box) :box(v_box), angle(0.f)
	{
		cv_box = cv::RotatedRect(cv::Point2f(v_box.center().x(), v_box.center().y()), cv::Size2f(v_box.sizes().x(), v_box.sizes().y()), 0.f);
	}
	Rotated_box(const Eigen::AlignedBox3f& v_box, float v_angle_in_degree) :box(v_box), angle(v_angle_in_degree / 180.f * 3.1415926f)
	{
		cv_box = cv::RotatedRect(cv::Point2f(v_box.center().x(), v_box.center().y()), cv::Size2f(v_box.sizes().x(), v_box.sizes().y()), v_angle_in_degree);
	}

	bool inside_2d(const Eigen::Vector3f& v_point) const
	{
		Eigen::Vector2f point(v_point.x(), v_point.y());
		float s = std::sin(-angle);
		float c = std::cos(-angle);

		// set origin to rect center
		Eigen::Vector2f newPoint = point - Eigen::Vector2f(box.center().x(), box.center().y());
		// rotate
		newPoint = Eigen::Vector2f(newPoint.x() * c - newPoint.y() * s, newPoint.x() * s + newPoint.y() * c);
		// put origin back
		newPoint = newPoint + Eigen::Vector2f(box.center().x(), box.center().y());
		if (newPoint.x() >= box.min().x() && newPoint.x() <= box.max().x() && newPoint.y() >= box.min().y() && newPoint.y() <= box.max().y())
			return true;
		else
			return false;
	}
};


// @brief:
// @notice: Currently only transfer vertices to the cgal Surface mesh
// @param: `attrib_t, shape_t, material_t`
// @ret: Surface_mesh
Surface_mesh convert_obj_from_tinyobjloader_to_surface_mesh(
	const std::tuple<tinyobj::attrib_t, std::vector<tinyobj::shape_t>, std::vector<tinyobj::material_t>> v_obj_in);

Eigen::AlignedBox3f get_bounding_box(const Point_set& v_point_set);
Eigen::AlignedBox3f get_bounding_box(const std::vector<Point_3>& v_point_set);
Rotated_box get_bounding_box_rotated(const Point_set& v_point_set);

Eigen::Vector3d cgal_point_2_eigend(const Point_3& p);
Eigen::Vector3f cgal_point_2_eigen(const Point_3& p);
Eigen::Vector3f cgal_vector_2_eigen(const Vector_3& p);
Eigen::Vector3d cgal_vector_2_eigend(const Vector_3& p);
Point_3 eigen_2_cgal_point(const Eigen::Vector3f& p);
Vector_3 eigen_2_cgal_vector(const Eigen::Vector3f& p);

Point_3 eigen_2_cgal_point(const Eigen::Vector3d& p);
Vector_3 eigen_2_cgal_vector(const Eigen::Vector3d& p);

template <typename T>
Point_2 eigen_2_cgal_point(const Eigen::Vector2<T>& v)
{
	return Point_2(v[0], v[1]);
}

template <typename T>
Point_3 eigen_2_cgal_point(const Eigen::Vector3<T>& v)
{
	return Point_3(v[0], v[1], v[2]);
}

template <typename T>
Eigen::Vector2<T> cgal_2_eigen_point(const Point_2& v)
{
	return Eigen::Vector2<T>(v[0], v[1]);
}

template <typename T>
Eigen::Vector3<T> cgal_2_eigen_point(const Point_3& v)
{
	return Eigen::Vector3<T>(v[0], v[1], v[2]);
}

void write_ply(const fs::path& v_path, const std::vector<Triangle_3>& v_mesh);

inline
void export_points(const fs::path& v_path, const std::vector<Point_3>& points)
{
	Point_set boundary_points;
	boundary_points.resize(points.size());
	#pragma omp parallel for
	for (int i = 0; i < points.size(); ++i)
	{
		boundary_points.point(i) = points[i];
	}
	CGAL::IO::write_point_set(v_path.string(), boundary_points);
}

inline
void export_points(const fs::path& v_path, const std::vector<Eigen::Vector3d>& points)
{
	Point_set boundary_points;
	boundary_points.resize(points.size());
	#pragma omp parallel for
	for (int i = 0; i < points.size(); ++i)
	{
		boundary_points.point(i) = eigen_2_cgal_point(points[i]);
	}
	CGAL::IO::write_point_set(v_path.string(), boundary_points);
}

inline
bool read_points(const fs::path& v_path, Point_set& points)
{
	if (!fs::exists(v_path))
	{
		LOG(ERROR) << "File not exist: " << v_path;
		return false;
	}
	CGAL::IO::read_point_set(v_path.string(), points);
	return true;
}

inline
bool read_points(const fs::path& v_path, std::vector<Eigen::Vector3d>& points)
{
	Point_set boundary_points;
	read_points(v_path, boundary_points);

	points.resize(boundary_points.size());
	for (int i = 0; i < boundary_points.size(); ++i)
		points[i] = cgal_2_eigen_point<double>(boundary_points.point(i));
	return true;
}


std::vector<int> fps_sampling(const Point_set& v_point_set, const int num_points);


#endif // CGAL_TOOLS_H
