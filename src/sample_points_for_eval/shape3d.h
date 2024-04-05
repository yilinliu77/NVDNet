#pragma once

#include "cgal_tools.h"
#include "common_util.h"

#include "shape2d.h"

#include <Mathematics/Cylinder3.h>

#include <Mathematics/QuadricSurface.h>
#include <Mathematics/DistPointHyperellipsoid.h>

#include <gp_Cylinder.hxx>
#include <gp_Cone.hxx>
#include <gp_Sphere.hxx>
#include <gp_Torus.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <gp_Pln.hxx>
#include <ElSLib.hxx>

#include <boost/serialization/export.hpp>

#include <CGAL/grid_simplify_point_set.h>

#include <CGAL/Periodic_2_Delaunay_triangulation_traits_2.h>
#include <CGAL/Periodic_2_Delaunay_triangulation_2.h>

#include "kd_tree_helper.h"

typedef CGAL::Periodic_2_Delaunay_triangulation_traits_2<K>     PDT;
typedef CGAL::Periodic_2_triangulation_vertex_base_2<PDT>        PTV;
typedef CGAL::Alpha_shape_vertex_base_2<PDT, PTV>                 AsVb;
// Cell type
typedef CGAL::Periodic_2_triangulation_face_base_2<PDT>          PTF;
typedef CGAL::Alpha_shape_face_base_2<PDT, PTF>                   AsCb;
typedef CGAL::Triangulation_data_structure_2<AsVb, AsCb>        PTds;
typedef CGAL::Periodic_2_Delaunay_triangulation_2<PDT, PTds>      P2DT2;
typedef CGAL::Alpha_shape_2<P2DT2>                              PAlpha_shape_2;


class Shape3D : public Shape
{
public:
	std::vector<CGAL::Triangle_2<K>> m_boundary;
	double area_sum = 0;
	std::shared_ptr<my_kd_tree_t> m_kd_tree;
	matrix_t m_kdtree_data;

	Shape3D(){}
	Shape3D(
		const Cluster& v_cluster, 
		const std::vector<Eigen::Vector3d>& v_inliers,
		const std::string& v_type,
		const std::string& v_detail_type
	): Shape(v_cluster, v_inliers, v_type, v_detail_type)
	{
	}

	virtual Eigen::Vector3d get_cartesian(const Eigen::Vector2d& uv) const = 0;

	virtual Eigen::Vector2d get_parametric(const Eigen::Vector3d& v_point) const = 0;

	double distance(const Eigen::Vector3d& v_point) const override
	{
		return (get_cartesian(get_parametric(v_point)) - v_point).norm();
	}

	double inlier_distance(const Eigen::Vector3d& v_point) const override
	{
		return std::sqrt(search_k_neighbour(*m_kd_tree, v_point.cast<float>(), 1).second[0]);
	}


	void find_boundary() override;

	Point_set sample_parametric(const int num_samples = 10000) const override
	{
		// Calculate area and sample points
		std::vector<Point_2> sample_points;
		double num_per_area = num_samples / area_sum;
		std::mt19937_64 generator;
		std::uniform_real_distribution<double> distribution(0, 1);
		for (int i = 0; i < m_boundary.size(); ++i)
		{
			CGAL::Random_points_in_triangle_2<Point_2> g(m_boundary[i]);
			const double area = m_boundary[i].area();
			if (area * num_per_area < 1 && i != 0)
			{
				if (distribution(generator) < area * num_per_area)
					std::copy_n(g, 1, std::back_inserter(sample_points));
			}
			else
			{
				int num_samples_local = std::max(1, (int)(area * num_per_area));
				std::copy_n(g, num_samples_local, std::back_inserter(sample_points));
			}
		}

		Point_set point_set;
		for (int i = 0; i < sample_points.size(); ++i)
		{
			double theta = sample_points[i][0];
			double s = sample_points[i][1];
			point_set.insert(eigen_2_cgal_point(get_cartesian({theta, s})));
		}
		return point_set;
	}

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<Shape>(*this);
		ar& m_boundary;
		ar& area_sum;
	}
};

class MyPlane : public Shape3D
{
public:
	gp_Pln plane;

	Eigen::Vector3d get_cartesian(const Eigen::Vector2d& uv) const override
	{
		gp_Pnt p_3d;
		ElSLib::D0(uv.x(), uv.y(), plane, p_3d);
		return Eigen::Vector3d(p_3d.X(), p_3d.Y(), p_3d.Z());
	}

	Eigen::Vector2d get_parametric(const Eigen::Vector3d& v_point) const override
	{
		gp_Pnt p_3d(v_point.x(), v_point.y(), v_point.z());
		double u, v;
		ElSLib::Parameters(plane, p_3d, u, v);
		return Eigen::Vector2d(u, v);
	}

	double distance(const Eigen::Vector3d& v_point) const override
	{
		return plane.Distance((gp_Pnt(v_point.x(), v_point.y(), v_point.z())));
	}

	MyPlane(){}
	MyPlane(
		const Cluster& v_cluster,
		const gp_Pln& v_plane, const std::vector<Eigen::Vector3d>& v_inliers) :
		Shape3D(v_cluster, v_inliers, "surface", "plane"), plane(v_plane)
	{
		find_boundary();
	}

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<Shape3D>(*this);
		ar& plane;
	}
};

class MyCylinder : public Shape3D
{
public:
	std::vector<CGAL::Triangle_2<K>> m_boundary2;

	gp_Cylinder cylinder;
	MyCylinder(){}
	MyCylinder(
		const Cluster& v_cluster,
		const gp_Cylinder& cylinder, const std::vector<Eigen::Vector3d>& v_inliers)
		: Shape3D(v_cluster, v_inliers, "surface", "cylinder"), cylinder(cylinder)
	{
		find_boundary();
	}

	double distance(const Eigen::Vector3d& v_point) const override
	{
		// Calculate the distance
		Eigen::Vector3d center(cylinder.Location().X(), cylinder.Location().Y(), cylinder.Location().Z());
		Eigen::Vector3d center_to_point = v_point-center;
		Eigen::Vector3d cylinder_axis(cylinder.Axis().Direction().X(), cylinder.Axis().Direction().Y(), cylinder.Axis().Direction().Z());
		cylinder_axis.normalize();
		double a = std::abs(center_to_point.dot(cylinder_axis));
		double b = center_to_point.norm();
		double distance = std::sqrt(b*b - a*a);
		return std::abs(distance-cylinder.Radius());
		// return (get_cartesian(get_parametric(v_point)) - v_point).norm();
	}

	Eigen::Vector3d get_cartesian(const Eigen::Vector2d& uv) const override;

	Eigen::Vector2d get_parametric(const Eigen::Vector3d& v_point) const override;

	// Surface_mesh extract_alpha_mesh(const double epsilon) const;

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<Shape3D>(*this);
		ar& cylinder;
	}
};

class MyCone : public Shape3D
{
public:
	gp_Cone cone;
	MyCone(){}
	MyCone(
		const Cluster& v_cluster,
		const gp_Cone& cone, const std::vector<Eigen::Vector3d>& v_inliers)
		: Shape3D(v_cluster, v_inliers, "surface", "cone"), cone(cone)
	{
		find_boundary();
	}

	Eigen::Vector3d get_cartesian(const Eigen::Vector2d& v_point) const override;

	Eigen::Vector2d get_parametric(const Eigen::Vector3d& v_point) const override;

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<Shape3D>(*this);
		ar& cone;
	}
};

class MySphere : public Shape3D
{
public:
	gp_Sphere sphere;

	MySphere(){}
	MySphere(
		const Cluster& v_cluster,
		const gp_Sphere& sphere, const std::vector<Eigen::Vector3d>& v_inliers)
		: Shape3D(v_cluster, v_inliers, "surface", "sphere"), sphere(sphere)
	{
		find_boundary();
	}

	Eigen::Vector3d get_cartesian(const Eigen::Vector2d& v_point) const override
	{
		gp_Pnt p;
		ElSLib::SphereD0(v_point[0], v_point[1], sphere.Position(), sphere.Radius(), p);
		return {p.X(), p.Y(), p.Z()};
	}

	Eigen::Vector2d get_parametric(const Eigen::Vector3d& v_point) const override
	{
		double u, v;
		ElSLib::SphereParameters(sphere.Position(), sphere.Radius(),
		                         gp_Pnt(v_point[0], v_point[1], v_point[2]), u, v);
		return {u, v};
	}

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<Shape3D>(*this);
		ar& sphere;
	}
};

class MyTorus : public Shape3D
{
public:
	gp_Torus torus;

	MyTorus(){}
	MyTorus(
		const Cluster& v_cluster,
		const gp_Torus& torus, const std::vector<Eigen::Vector3d>& v_inliers)
		: Shape3D(v_cluster, v_inliers, "surface", "torus"), torus(torus)
	{
		find_boundary();
	}

	Eigen::Vector3d get_cartesian(const Eigen::Vector2d& v_point) const override
	{
		gp_Pnt p;
		ElSLib::TorusD0(v_point[0] * (2 * M_PI), v_point[1] * (2 * M_PI), torus.Position(), torus.MajorRadius(), torus.MinorRadius(), p);
		return {p.X(), p.Y(), p.Z()};
	}

	Eigen::Vector2d get_parametric(const Eigen::Vector3d& v_point) const override
	{
		double u, v;
		ElSLib::TorusParameters(torus.Position(), torus.MajorRadius(), torus.MinorRadius(),
		                        gp_Pnt(v_point[0], v_point[1], v_point[2]), u, v);
		u = u / (2 * M_PI);
		v = v / (2 * M_PI);
		return {u, v};
	}

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<Shape3D>(*this);
		ar& torus;
	}
};

class MyEllipsoid : public Shape3D
{
public:
	gte::Ellipsoid3<double> elips;
	Eigen::Matrix3d rotation;

	MyEllipsoid(){}
	MyEllipsoid(
		const Cluster& v_cluster,
		const gte::Ellipsoid3<double>& v_elips, const std::vector<Eigen::Vector3d>& v_inliers)
		: Shape3D(v_cluster, v_inliers, "surface", "ellipsoid"), elips(v_elips)
	{
		rotation << elips.axis[0][0], elips.axis[1][0], elips.axis[2][0],
			elips.axis[0][1], elips.axis[1][1], elips.axis[2][1],
			elips.axis[0][2], elips.axis[1][2], elips.axis[2][2];
		find_boundary();
	}

	Eigen::Vector3d get_cartesian(const Eigen::Vector2d& v_point) const override
	{
		double x = elips.extent[0] * std::sin(v_point[0]) * std::cos(v_point[1]);
		double y = elips.extent[1] * std::sin(v_point[0]) * std::sin(v_point[1]);
		double z = elips.extent[2] * std::cos(v_point[0]);

		const Eigen::Vector3d p = rotation * Eigen::Vector3d(x, y, z);

		return {p.x() + elips.center[0], p.y() + elips.center[1], p.z() + elips.center[2]};
	}

	Eigen::Vector2d get_parametric(const Eigen::Vector3d& v_point) const override
	{
		gte::Vector3<double> p({v_point[0], v_point[1], v_point[2]});
		gte::DCPQuery<double, gte::Vector3<double>, gte::Ellipsoid3<double>> query;

		gte::Vector3<double> n = query(p, elips).closest[1];
		n = n - elips.center;

		Eigen::Vector3d pp = rotation.inverse() * Eigen::Vector3d(n[0], n[1], n[2]);

		double theta = std::acos(pp[2] / elips.extent[2]);
		double phi = std::atan2(pp[1] / elips.extent[1] / std::sin(theta), pp[0] / elips.extent[0] / std::sin(theta));
		return {theta, phi};
	}

	double distance(const Eigen::Vector3d& v_point) const override
	{
		gte::Vector3<double> p({v_point[0], v_point[1], v_point[2]});
		gte::DCPQuery<double, gte::Vector3<double>, gte::Ellipsoid3<double>> query;
		return query(p, elips).distance;
	}

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& boost::serialization::base_object<Shape3D>(*this);
		ar& elips;
		ar& rotation;
	}
};

BOOST_SERIALIZATION_SPLIT_FREE(CGAL::Triangle_2<K>)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Cylinder)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Cone)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Sphere)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Torus)
BOOST_SERIALIZATION_SPLIT_FREE(gte::Ellipsoid3<double>)

namespace boost {
	namespace serialization {
		template<class Archive>
		void load(Archive& ar, CGAL::Triangle_2<K>& g, const unsigned int version)
		{
			double x1, y1, x2, y2, x3, y3;
			ar & x1 & y1 & x2 & y2 & x3 & y3;
			g = CGAL::Triangle_2<K>(
				Point_2(x1, y1),
				Point_2(x2, y2),
				Point_2(x3, y3)
			);
		}
		template<class Archive>
		void save(Archive& ar, const CGAL::Triangle_2<K>& g, const unsigned int version)
		{
			ar & g.vertex(0).x() & g.vertex(0).y();
			ar & g.vertex(1).x() & g.vertex(1).y();
			ar & g.vertex(2).x() & g.vertex(2).y();
		}

		template<class Archive>
		void load(Archive& ar, gp_Cylinder& g, const unsigned int version)
		{
			gp_Ax3 pos;
			double radius;
			ar& pos;
			ar& radius;
			g = gp_Cylinder(pos,radius);
		}
		template<class Archive>
		void save(Archive& ar, const gp_Cylinder& g, const unsigned int version)
		{
			ar& g.Position() & g.Radius();
		}

		template<class Archive>
		void load(Archive& ar, gp_Cone& g, const unsigned int version)
		{
			gp_Ax3 pos;
			double radius, semiangle;
			ar& pos;
			ar& radius & semiangle;
			g = gp_Cone(pos, semiangle, radius);
		}
		template<class Archive>
		void save(Archive& ar, const gp_Cone& g, const unsigned int version)
		{
			ar& g.Position() & g.RefRadius() & g.SemiAngle();
		}

		template<class Archive>
		void load(Archive& ar, gp_Sphere& g, const unsigned int version)
		{
			gp_Ax3 pos;
			double radius;
			ar& pos;
			ar& radius;
			g = gp_Sphere(pos, radius);
		}
		template<class Archive>
		void save(Archive& ar, const gp_Sphere& g, const unsigned int version)
		{
			ar& g.Position()& g.Radius();
		}

		template<class Archive>
		void load(Archive& ar, gp_Torus& g, const unsigned int version)
		{
			gp_Ax3 pos;
			double radius1, radius2;
			ar& pos;
			ar& radius1 & radius2;
			g = gp_Torus(pos, radius1, radius2);
		}
		template<class Archive>
		void save(Archive& ar, const gp_Torus& g, const unsigned int version)
		{
			ar& g.Position()& g.MajorRadius() & g.MinorRadius();
		}


		template<class Archive>
		void load(Archive& ar, gte::Ellipsoid3<double>& g, const unsigned int version)
		{
			std::array<double, 10> params;
			ar& params;
			g.FromCoefficients(params);
		}
		template<class Archive>
		void save(Archive& ar, const gte::Ellipsoid3<double>& g, const unsigned int version)
		{
			std::array<double, 10> params;
			g.ToCoefficients(params);
			ar& params;
		}

	} // namespace serialization
} // namespace boost

BOOST_SERIALIZATION_ASSUME_ABSTRACT(Shape3D)