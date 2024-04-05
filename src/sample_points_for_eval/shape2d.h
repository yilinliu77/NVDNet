#pragma once

#include "common_util.h"
#include "cgal_tools.h"
#include "ellipse_fit.h"

#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>

#include <gp_Pln.hxx>
#include <Geom2dAPI_ProjectPointOnCurve.hxx>
#include <Geom2dAPI_ExtremaCurveCurve.hxx>
#include <ElSLib.hxx>
#include <ElCLib.hxx>

#include <boost/serialization/export.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <GeomAPI_ProjectPointOnCurve.hxx>
#include <Geom_Ellipse.hxx>
#include <IntAna_Curve.hxx>
#include <CGAL/grid_simplify_point_set.h>

#include "kd_tree_helper.h"

typedef CGAL::Alpha_shape_vertex_base_2<K> Vb_2;
typedef CGAL::Alpha_shape_face_base_2<K> Fb_2;
typedef CGAL::Triangulation_data_structure_2<Vb_2, Fb_2> Tds_2;
typedef CGAL::Delaunay_triangulation_2<K, Tds_2> Dt_2;
typedef CGAL::Alpha_shape_2<Dt_2> Alpha_shape_2;

class Cluster
{
public:
    int type = -1; // 0: point; 1: curve; 2: surface
    std::vector<Eigen::Vector3i> coords;

    std::vector<Eigen::Vector3d> query_points;
    std::vector<Eigen::Vector3d> surface_points;
    std::vector<Eigen::Vector3d> surface_normals;

    Cluster()
    {
    }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& type;
        ar& coords;
        ar& query_points;
        ar& surface_points;
        ar& surface_normals;
    }
};

class Shape
{
public:
    Cluster cluster;
    std::string type;
    std::string detail_type;

    std::vector<Eigen::Vector3d> inliers;
    std::vector<Eigen::Vector3d> boundary_points;

    Shape() {}
    Shape(
        const Cluster& v_cluster,
        const std::vector<Eigen::Vector3d>& v_inliers,
        const std::string& v_type,
        const std::string& v_detail_type)
        : cluster(v_cluster), inliers(v_inliers), type(v_type), detail_type(v_detail_type)
    {

    }

    virtual void find_boundary() = 0;

    virtual double distance(const Eigen::Vector3d& v_point) const = 0;

    virtual double inlier_distance(const Eigen::Vector3d& v_point) const = 0;

    void get_inliers(const std::vector<Eigen::Vector3d>& v_candidates, const double epsilin)
    {
        inliers.clear();

        Point_set p;
        for (int i = 0; i < v_candidates.size(); ++i)
            p.insert(eigen_2_cgal_point(v_candidates[i]));

        auto iterator_to_first_to_remove = CGAL::grid_simplify_point_set(p, 0.001); // optional
        p.remove(iterator_to_first_to_remove, p.end());
        p.collect_garbage();

        inliers.resize(p.size());
        for (int i = 0; i < inliers.size(); ++i)
            inliers[i] = cgal_2_eigen_point<double>(p.point(i));

        std::vector<bool> flags(inliers.size(), false);
    	for (int i = 0; i < inliers.size(); ++i)
            if (distance(inliers[i]) > epsilin)
                flags[i] = true;

        inliers.erase(std::remove_if(inliers.begin(), inliers.end(), [&](const auto& item)
            {
                return flags[&item - inliers.data()];
            }), inliers.end());
    }

    static void fit(const std::vector<Eigen::Vector3d>& v_candidates)
    {
	    
    }

	virtual Point_set sample_parametric(const int num_samples = 10000) const = 0;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& cluster;
        ar& type;
        ar& detail_type;
        ar& inliers;
        ar& boundary_points;
    }
};


class Shape1D : public Shape
{
public:
    Eigen::Vector3d vertex;

    Shape1D(){}
    Shape1D(const Eigen::Vector3d& v_vertex, const Cluster& v_cluster, const std::vector<Eigen::Vector3d>& v_inliers):
    Shape(v_cluster, v_inliers, "vertex", "vertex"), vertex(v_vertex)
    {
	    
    }

    void find_boundary() {}

    double distance(const Eigen::Vector3d& v_point) const
    {
	    return (vertex - v_point).norm();
    }

    double inlier_distance(const Eigen::Vector3d& v_point) const override
    {
	    return distance(v_point);
    }

    Point_set sample_parametric(const int num_samples = 10000) const
    {
        Point_set point_set;
		point_set.insert(eigen_2_cgal_point(vertex));
        return point_set;
    }

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& boost::serialization::base_object<Shape>(*this);
        ar& vertex;
    }
};

class Shape2D: public Shape
{
public:
    gp_Pln m_plane;
    double min_t, max_t;

    virtual double get_parametric(const Eigen::Vector3d& v_point) const = 0;

    virtual Eigen::Vector3d get_cartesian(const double t) const = 0;

    void find_boundary() override
    {
        boundary_points.clear();
        std::vector<double> m_parametrics(inliers.size());
        std::vector<double> errors(inliers.size());
        for (int i = 0; i < inliers.size(); ++i)
        {
            m_parametrics[i] = get_parametric(inliers[i]);
            errors[i] = distance(inliers[i]);
        }

        min_t = *std::min_element(m_parametrics.begin(), m_parametrics.end());
        max_t = *std::max_element(m_parametrics.begin(), m_parametrics.end());

        boundary_points.emplace_back(get_cartesian(min_t));
        boundary_points.emplace_back(get_cartesian(max_t));
    }

    Point_set sample_parametric(const int num_samples = 10000) const override
    {
        std::mt19937 rng;
        std::uniform_real_distribution<double> unif(min_t, max_t);

        Point_set point_set;
        point_set.resize(num_samples);
        for (int i = 0; i < num_samples; ++i)
        {
            const double t = unif(rng);
            point_set.point(i) = eigen_2_cgal_point(get_cartesian(t));
        }
        return point_set;
    }

    // double distance(const Eigen::Vector3d& v_point) const override
    // {
    //     return (get_cartesian(get_parametric(v_point)) - v_point).norm();
    // }

    virtual double distance(const Eigen::Vector3d& v_point) const override = 0;

    double inlier_distance(const Eigen::Vector3d& v_point) const override
    {
        return distance(v_point);
    }

    Shape2D(){}
    Shape2D(
        const Cluster& v_cluster, 
        const gp_Pln& v_plane, 
        const std::vector<Eigen::Vector3d>& v_inliers,
        const std::string& v_type,
        const std::string& v_detail_type)
	    : Shape(v_cluster, v_inliers, v_type, v_detail_type), m_plane(v_plane)
	{}

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& boost::serialization::base_object<Shape>(*this);
        ar& m_plane;
        ar& min_t;
        ar& max_t;
    }
};

class MyLine : public Shape2D
{
public:
    gp_Lin line;

    double get_parametric(const Eigen::Vector3d& v_point) const override
    {
        gp_Pnt p_3d(v_point.x(), v_point.y(), v_point.z());
        double u = ElCLib::LineParameter(line.Position(), p_3d);
        return u;
    }

    Eigen::Vector3d get_cartesian(const double t) const override
    {
        gp_Pnt p_3d = ElCLib::LineValue(t, line.Position());
        return Eigen::Vector3d(p_3d.X(), p_3d.Y(), p_3d.Z());
    }

    double distance(const Eigen::Vector3d& v_point) const override
    {
	    gp_Pnt p_3d(v_point.x(), v_point.y(), v_point.z());
		return line.Distance(p_3d);
    }

    MyLine(){}
    MyLine(
        const Cluster& v_cluster,
        const Eigen::Vector3d& position,
        const Eigen::Vector3d& direction,
        const gp_Pln& v_plane, const std::vector<Eigen::Vector3d>& v_inliers) :
        Shape2D(v_cluster, v_plane, v_inliers, "curve", "line")
    {
        line = gp_Lin(gp_Pnt(position.x(), position.y(), position.z()), gp_Dir(direction.x(), direction.y(), direction.z()));
        find_boundary();
    }

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& boost::serialization::base_object<Shape2D>(*this);
        ar& line;
    }
};

class MyCircle: public Shape2D
{
public:
    int num_bins = 36;
    std::vector<std::vector<double>> bins;
    double bin_range = (10. / 180. * M_PI);

    gp_Circ circle;

    double get_parametric(const Eigen::Vector3d& v_point) const override
    {
        gp_Pnt p_3d(v_point.x(), v_point.y(), v_point.z());
        return ElCLib::CircleParameter(circle.Position(), p_3d);
    }

    Eigen::Vector3d get_cartesian(const double t) const override
    {
        gp_Pnt p_3d = ElCLib::CircleValue(t, circle.Position(), circle.Radius());
		return Eigen::Vector3d(p_3d.X(), p_3d.Y(), p_3d.Z());
    }

    double distance(const Eigen::Vector3d& v_point) const override
    {
        gp_Pnt p_3d(v_point.x(), v_point.y(), v_point.z());
        return circle.Distance(p_3d);
    }

    void find_boundary() override;
    Point_set sample_parametric(const int num_samples = 10000) const override;

    MyCircle(){}
    MyCircle(const Cluster& v_cluster, const double ca, const double cb, const double radius, const gp_Pln& v_plane, const std::vector<Eigen::Vector3d>& v_inliers):
        Shape2D(v_cluster, v_plane, v_inliers, "curve", "circle")
    {
        gp_Pnt center;
        ElSLib::D0(ca, cb, v_plane, center);
		circle = gp_Circ(gp_Ax2(center, v_plane.Axis().Direction()), radius);
	    find_boundary();
    }

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& boost::serialization::base_object<Shape2D>(*this);
        ar& circle;
        ar& bins;
    }
};

class MyEllipse : public Shape2D
{
public:
    int num_bins = 108;
    std::vector<std::vector<double>> bins;
    double bin_range = (360. / num_bins / 180. * M_PI);

    gp_Elips ellipse;

    double get_parametric(const Eigen::Vector3d& v_point) const override
    {
        gp_Pnt p_3d(v_point.x(), v_point.y(), v_point.z());
        return ElCLib::EllipseParameter(ellipse.Position(), ellipse.MajorRadius(), ellipse.MinorRadius(), p_3d);
    }

    Eigen::Vector3d get_cartesian(const double t) const override
    {
        gp_Pnt p_3d = ElCLib::EllipseValue(t, ellipse.Position(), ellipse.MajorRadius(), ellipse.MinorRadius());
        return Eigen::Vector3d(p_3d.X(), p_3d.Y(), p_3d.Z());
    }

    void find_boundary() override;
    Point_set sample_parametric(const int num_samples = 10000) const override;

    double distance(const Eigen::Vector3d& v_point) const override
    {
        gp_Pnt p_3d(v_point.x(), v_point.y(), v_point.z());
        // The distance from the point to the ellipse
        Handle(Geom_Ellipse) ellipse_ = new Geom_Ellipse(ellipse);
        GeomAPI_ProjectPointOnCurve aProjector(p_3d, ellipse_);
        if (aProjector.NbPoints() <= 0)
            throw;
        return p_3d.Distance(aProjector.NearestPoint());
    }

    MyEllipse(){}
    MyEllipse(
        const Cluster& v_cluster,
        const gp_Ax2& axis, 
        const double min_r, const double max_r,
        const gp_Pln& v_plane, const std::vector<Eigen::Vector3d>& v_inliers) :
        Shape2D(v_cluster,v_plane, v_inliers, "curve", "ellipse")
    {
        
        ellipse = gp_Elips(axis, max_r, min_r);
        find_boundary();
    }

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& boost::serialization::base_object<Shape2D>(*this);
        ar& ellipse;
    }
};

class MyIntCurve : public Shape2D
{
public:
    std::vector<Eigen::Vector3d> points;
    matrix_t kdtree_data;
    std::shared_ptr<my_kd_tree_t> kdtree;

    double get_parametric(const Eigen::Vector3d& v_point) const override
    {
        throw "Not implemented";
        return 0.;
    }

    Eigen::Vector3d get_cartesian(const double t) const override
    {
        throw "Not implemented";
        return Eigen::Vector3d(0.,0.,0.);
    }

    void find_boundary() override
    {
        throw "Not implemented";
    }
    Point_set sample_parametric(const int num_samples = 10000) const override
    {
        throw "Not implemented";
        Point_set p;
        return p;
    }

    double distance(const Eigen::Vector3d& v_point) const override
    {
        return search_k_neighbour(*kdtree, v_point.cast<float>(), 1).second[0];
    }

    MyIntCurve() {}
    MyIntCurve(
        const std::vector<Eigen::Vector3d>& v_points): points(v_points)
    {
        kdtree_data = initialize_kd_data(v_points);
        kdtree = initialize_kd_tree(kdtree_data);
    }
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(Shape)
BOOST_SERIALIZATION_ASSUME_ABSTRACT(Shape2D)

BOOST_SERIALIZATION_SPLIT_FREE(gp_XYZ)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Dir2d)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Dir)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Pnt)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Pnt2d)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Ax1)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Ax22d)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Ax3)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Ax2)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Pln)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Lin)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Circ2d)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Circ)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Elips2d)
BOOST_SERIALIZATION_SPLIT_FREE(gp_Elips)

namespace boost {
    namespace serialization {

        template<class Archive>
        void load(Archive& ar, gp_XYZ& g, const unsigned int version)
        {
            double x, y, z;
            ar& x;
            ar& y;
            ar& z;
            g = gp_XYZ(x, y, z);
        }
        template<class Archive>
        void save(Archive& ar, const gp_XYZ& g, const unsigned int version)
        {
            ar& g.X();
            ar& g.Y();
            ar& g.Z();
        }

        template<class Archive>
        void load(Archive& ar, gp_Dir& g, const unsigned int version)
        {
            gp_XYZ xyz;
            ar& xyz;
            g = gp_Dir(xyz);
        }
        template<class Archive>
        void save(Archive& ar, const gp_Dir& g, const unsigned int version)
        {
            ar& g.XYZ();
        }

        template<class Archive>
        void load(Archive& ar, gp_Pnt& g, const unsigned int version)
        {
            gp_XYZ xyz;
            ar& xyz;
            g = gp_Pnt(xyz);
        }
        template<class Archive>
        void save(Archive& ar, const gp_Pnt& g, const unsigned int version)
        {
            ar& g.XYZ();
        }

        template<class Archive>
        void load(Archive& ar, gp_Dir2d& g, const unsigned int version)
        {
            double x, y;
            ar& x;
            ar& y;
            g = gp_Dir2d(x, y);
        }
        template<class Archive>
        void save(Archive& ar, const gp_Dir2d& g, const unsigned int version)
        {
            ar& g.X();
            ar& g.Y();
        }

        template<class Archive>
        void load(Archive& ar, gp_Pnt2d& g, const unsigned int version)
        {
            double x, y;
            ar& x;
            ar& y;
            g = gp_Pnt2d(x, y);
        }
        template<class Archive>
        void save(Archive& ar, const gp_Pnt2d& g, const unsigned int version)
        {
            ar& g.X();
            ar& g.Y();
        }

        template<class Archive>
        void load(Archive& ar, gp_Ax1& g, const unsigned int version)
        {
            gp_Pnt loc;
            gp_Dir vdir;
            ar& loc;
            ar& vdir;
            g = gp_Ax1(loc, vdir);
        }
        template<class Archive>
        void save(Archive& ar, const gp_Ax1& g, const unsigned int version)
        {
            ar& g.Location();
            ar& g.Direction();
        }

        template<class Archive>
        void load(Archive& ar, gp_Ax3& g, const unsigned int version)
        {
            gp_Pnt pos;
            gp_Dir dir, xdir;
            ar& pos;
            ar& dir;
            ar& xdir;
            g = gp_Ax3(pos, dir, xdir);
        }
        template<class Archive>
        void save(Archive& ar, const gp_Ax3& g, const unsigned int version)
        {
            ar& g.Location();
            ar& g.Direction();
            ar& g.XDirection();
        }

        template<class Archive>
        void load(Archive& ar, gp_Pln& g, const unsigned int version)
        {
            gp_Ax3 pos;
            ar& pos;
            g = gp_Pln(pos);
        }
        template<class Archive>
        void save(Archive& ar, const gp_Pln& g, const unsigned int version)
        {
            ar& g.Position();
        }

        template<class Archive>
        void load(Archive& ar, gp_Lin& g, const unsigned int version)
        {
            gp_Ax1 pos;
            ar& pos;
            g = gp_Lin(pos);
        }
        template<class Archive>
        void save(Archive& ar, const gp_Lin& g, const unsigned int version)
        {
            ar& g.Position();
        }

        template<class Archive>
        void load(Archive& ar, gp_Ax22d& g, const unsigned int version)
        {
            gp_Pnt2d pos;
            gp_Dir2d x, y;
            ar& pos;
            ar& x;
            ar& y;
            g = gp_Ax22d(pos, x, y);
        }
        template<class Archive>
        void save(Archive& ar, const gp_Ax22d& g, const unsigned int version)
        {
            ar& g.Location();
            ar& g.XDirection();
            ar& g.YDirection();
        }

        template<class Archive>
        void load(Archive& ar, gp_Circ2d& g, const unsigned int version)
        {
            gp_Ax22d pos;
            double radius;
            ar& pos;
            ar& radius;
            g = gp_Circ2d(pos, radius);
        }
        template<class Archive>
        void save(Archive& ar, const gp_Circ2d& g, const unsigned int version)
        {
            ar& g.Position();
            ar& g.Radius();
        }

        template<class Archive>
        void load(Archive& ar, gp_Ax2& g, const unsigned int version)
        {
            gp_Pnt pos;
            gp_Dir z, x;
            ar& pos;
            ar& z;
            ar& x;
            g = gp_Ax2(pos, z, x);
        }
        template<class Archive>
        void save(Archive& ar, const gp_Ax2& g, const unsigned int version)
        {
            ar& g.Location();
            ar& g.Direction();
            ar& g.XDirection();
        }

        template<class Archive>
        void load(Archive& ar, gp_Circ& g, const unsigned int version)
        {
            gp_Ax2 pos;
            double radius;
            ar& pos;
            ar& radius;
            g = gp_Circ(pos, radius);
        }
        template<class Archive>
        void save(Archive& ar, const gp_Circ& g, const unsigned int version)
        {
            ar& g.Position();
            ar& g.Radius();
        }

        template<class Archive>
        void load(Archive& ar, gp_Elips& g, const unsigned int version)
        {
            gp_Ax2 pos;
            double r1, r2;
            ar& pos;
            ar& r1;
            ar& r2;
            g = gp_Elips(pos, r1, r2);
        }
        template<class Archive>
        void save(Archive& ar, const gp_Elips& g, const unsigned int version)
        {
            ar& g.Position();
            ar& g.MajorRadius();
            ar& g.MinorRadius();
        }

        template<class Archive>
        void load(Archive& ar, gp_Elips2d& g, const unsigned int version)
        {
            gp_Ax22d pos;
            double majorradius, minnorradius;
            ar& pos;
            ar& majorradius;
            ar& minnorradius;
            g = gp_Elips2d(pos, majorradius, minnorradius);
        }
        template<class Archive>
        void save(Archive& ar, const gp_Elips2d& g, const unsigned int version)
        {
            ar& g.Axis();
            ar& g.MajorRadius();
            ar& g.MinorRadius();
        }

    } // namespace serialization
} // namespace boost

void colorize_point_set(Point_set& v_points, const std::string& v_name="index");

void colorize_output_points(const std::vector<std::shared_ptr<Shape>>& v_shapes);
