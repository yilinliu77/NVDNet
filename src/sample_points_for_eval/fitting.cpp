#include "fitting.h"

#include <gp_Pln.hxx>
#include <ElSLib.hxx>

#include <Mathematics/ApprEllipse2.h>
#include <Mathematics/ApprTorus3.h>
#include <Mathematics/ApprCylinder3.h>

#include <Mathematics/ApprEllipsoid3.h>
#include <Mathematics/ApprParaboloid3.h>
#include <Mathematics/ApprCone3.h>
#include <Mathematics/ApprSphere3.h>
#include <Mathematics/ApprQuadratic3.h>
#include <Mathematics/ApprCircle2.h>

#include "kd_tree_helper.h"

#include <tbb/tbb.h>

// #pragma optimize ("", off)

void prepare_gte_data(
	const std::vector<Eigen::Vector3d>& v_points,
	std::vector<gte::Vector3<double>>& data1,
	const int max_samples
)
{
	Point_set p;
	p.resize(v_points.size());
	for (int i = 0; i < p.size(); ++i)
		p.point(i) = eigen_2_cgal_point(v_points[i]);
	// CGAL::IO::write_point_set("temp/before_sample.ply", p);

	auto iterator_to_first_to_remove = CGAL::grid_simplify_point_set(p, 0.001); // optional
	p.remove(iterator_to_first_to_remove, p.end());
	p.collect_garbage();
	// CGAL::IO::write_point_set("temp/after_sample.ply", p);

	const int num_points = std::min((int)v_points.size(), max_samples);

	std::vector<int> indexes(v_points.size());
	std::iota(indexes.begin(), indexes.end(), 0);
	std::mt19937 rng(0);
	std::shuffle(indexes.begin(), indexes.end(), rng);

	data1.resize(num_points);
	for (int i = 0; i < num_points; ++i)
		data1[i] = gte::Vector3<double>({ v_points[indexes[i]][0], v_points[indexes[i]][1], v_points[indexes[i]][2] });

	return;
}

std::pair<Plane_3, double> fit_plane(const Cluster& v_cluster)
{
	const long long num_points = v_cluster.surface_points.size();
	Eigen::MatrixXd A(num_points, 3);
	Eigen::VectorXd b(num_points);

	for (int i = 0; i < num_points; ++i)
	{
		A.row(i) = v_cluster.surface_points[i];
		b[i] = 1;
	}

	Eigen::Vector3d solution = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

	double error = (b - A * solution).cwiseAbs().mean();
	// Append the last term to the normalized solution
	Plane_3 p(solution[0], solution[1], solution[2], -1);

	return { p , error };
}

std::pair<Plane_3, double> fit_plane(const std::vector<gte::Vector3<double>>& gte_data)
{
	const long long num_points = gte_data.size();
	Eigen::MatrixXd A(num_points, 3);
	Eigen::VectorXd b(num_points);

	for (int i = 0; i < num_points; ++i)
	{
		A.row(i) = Eigen::Vector3d(gte_data[i][0], gte_data[i][1], gte_data[i][2]);
		b[i] = 1;
	}

	Eigen::Vector3d solution = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

	double error = (b - A * solution).cwiseAbs().mean();
	// Append the last term to the normalized solution
	Plane_3 p(solution[0], solution[1], solution[2], -1);

	return { p , error };
}


std::pair<std::shared_ptr<Shape>, double> fit_vertex(
	const std::vector<Eigen::Vector3d>& v_points,
	const Cluster& v_cluster,
	const double v_epsilon
)
{
	const int num_points = v_points.size();

	Eigen::MatrixXd v_data(num_points, 3);
	for (int i = 0; i < num_points; ++i)
		v_data.row(i) = v_points[i].transpose();

	Eigen::Vector3d center_point = v_data.leftCols(3).colwise().sum().transpose();
	center_point = center_point / num_points;

	std::vector<Eigen::Vector3d> inliers;
	std::shared_ptr<Shape> shape = std::make_shared<Shape1D>(center_point, v_cluster, inliers);

	std::vector<double> errors(num_points);
	for (int i = 0; i < num_points; ++i)
		errors[i] = shape->distance(v_points[i]);
	shape->get_inliers(v_cluster.surface_points, v_epsilon);
	double error = std::accumulate(errors.begin(), errors.end(), 0.) / num_points;
	return { shape, error };
}


std::pair<std::shared_ptr<Shape>, double> fit_curve(
	const std::vector<Eigen::Vector3d>& v_points,
	const Cluster& v_cluster,
	const gp_Pln& v_plane,
	const std::string& v_type
)
{
	const int num_points = v_points.size();

	std::shared_ptr<Shape> shape;
	if (v_type == "line")
	{
		Eigen::MatrixXd v_data(num_points, 3);
		for (int i = 0; i < num_points; ++i)
			v_data.row(i) = v_points[i].transpose();
		Eigen::Vector3d center_point = v_data.leftCols(3).colwise().sum().transpose();
		center_point = center_point / num_points;

		Eigen::MatrixXd OP = v_data.leftCols(3).rowwise() - center_point.transpose();
		Eigen::Vector3d direction = OP.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).matrixV().col(0).normalized();

		double error = 0;
		Line_3 line(eigen_2_cgal_point(center_point), eigen_2_cgal_vector(direction));
		for (int i = 0; i < num_points; i++)
		{
			Eigen::Vector3d original_point_ = v_data.row(i).leftCols(3);
			Point_3 original_point = eigen_2_cgal_point(original_point_);
			const Point_3 projected_point = line.projection(original_point);
			error += std::sqrt(CGAL::squared_distance(original_point, projected_point));
		}
		error /= num_points;
		shape = std::make_shared<MyLine>(v_cluster, center_point, direction, v_plane, v_points);
	}
	else if (v_type == "circle")
	{
		std::vector<gte::Vector2<double>> data(num_points);
		#pragma omp parallel for
		for (int i = 0; i < num_points; ++i)
		{
			ElSLib::PlaneParameters(v_plane.Position(),
				gp_Pnt(
					v_points[i][0],
					v_points[i][1],
					v_points[i][2]
				), data[i][0], data[i][1]);
		}

		gte::ApprCircle2<double> fitter;
		gte::Circle2<double> circle_;
		fitter.FitUsingSquaredLengths(num_points, data.data(), circle_);

		shape = std::make_shared<MyCircle>(v_cluster, circle_.center[0], circle_.center[1], circle_.radius, v_plane, v_points);
	}
	else if (v_type == "ellipse")
	{
		std::vector<gte::Vector2<double>> data(num_points);
		#pragma omp parallel for
		for (int i = 0; i < num_points; ++i)
		{
			ElSLib::PlaneParameters(v_plane.Position(),
				gp_Pnt(
					v_points[i][0],
					v_points[i][1],
					v_points[i][2]
				), data[i][0], data[i][1]);
		}

		/*std::vector<std::vector<double>> data(num_points, std::vector<double>{0., 0.});
#pragma omp parallel for
		for (int i = 0; i < num_points; ++i)
		{
			ElSLib::PlaneParameters(v_plane.Position(),
				gp_Pnt(
					v_cluster.surface_points[i][0],
					v_cluster.surface_points[i][1],
					v_cluster.surface_points[i][2]
				), data[i][0], data[i][1]);
		}
		Ellipse_fit ellipse_fit;
		ellipse_fit.set(data);
		double h, k, phi, a, b;
		ellipse_fit.fit(h, k, phi, a, b);

		gp_Pnt2d p(h, k);
		gp_Pnt center = ElSLib::PlaneValue(h, k, v_plane.Position());
		gp_Pnt2d p1(p.X() + std::cos(phi), p.Y() + std::sin(phi));
		gp_Pnt2d p2(p.X() + std::sin(phi), p.Y() + std::cos(phi));
		gp_Vec v1(center, ElSLib::PlaneValue(p1.X(), p1.Y(), v_plane.Position()));
		gp_Vec v2(center, ElSLib::PlaneValue(p2.X(), p2.Y(), v_plane.Position()));
		*/

		gte::ApprEllipse2<double> fitter;
		gte::Ellipse2<double> ellipse2;
		fitter(data, 10000, false, ellipse2);
		if (isinf(ellipse2.extent[0]) || isnan(ellipse2.extent[0]))
			return { nullptr, 0. };
		gp_Ax2 axis;
		gp_Pnt2d radius;

		gp_Pnt2d p(ellipse2.center[0], ellipse2.center[1]);
		gp_Pnt center = ElSLib::PlaneValue(p.X(), p.Y(), v_plane.Position());
		gp_Pnt2d p1(p.X() + ellipse2.extent[0] * ellipse2.axis[0][0], p.Y() + ellipse2.extent[0] * ellipse2.axis[0][1]);
		gp_Pnt2d p2(p.X() + ellipse2.extent[1] * ellipse2.axis[1][0], p.Y() + ellipse2.extent[1] * ellipse2.axis[1][1]);
		gp_Vec v1(center, ElSLib::PlaneValue(p1.X(), p1.Y(), v_plane.Position()));
		gp_Vec v2(center, ElSLib::PlaneValue(p2.X(), p2.Y(), v_plane.Position()));
		gp_Vec n = v1;
		n.Cross(v2);

		if (ellipse2.extent[0] < ellipse2.extent[1])
		{
			axis = gp_Ax2(center, n, v2);
			radius = gp_Pnt2d(ellipse2.extent[0], ellipse2.extent[1]);
		}
		else
		{
			axis = gp_Ax2(center, n, v1);
			radius = gp_Pnt2d(ellipse2.extent[1], ellipse2.extent[0]);
		}

		shape = std::make_shared<MyEllipse>(v_cluster,
			axis,
			radius.X(), radius.Y(),
			v_plane, v_points);
	}
	else
		throw;

	std::vector<double> errors(num_points);
	// #pragma omp parallel for
	for (int i = 0; i < num_points; ++i)
	{
		errors[i] = shape->distance(
			Eigen::Vector3d(
				v_points[i][0],
				v_points[i][1],
				v_points[i][2]));
		// std::cout << errors[i] << std::endl;
	}
	double error = std::accumulate(errors.begin(), errors.end(), 0.) / num_points;
	return { shape, error };
}

std::pair<std::shared_ptr<Shape>, double> fit_surface(
	const std::vector<gte::Vector3<double>>& gte_data,
	const Cluster& v_cluster,
	const std::string& v_type
)
{
	std::vector<Eigen::Vector3d> inliers;
	std::shared_ptr<Shape> shape;

	if (gte_data.size() < 10)
		return { nullptr, 999999. };

	if (v_type == "cylinder")
	{
		gte::Cylinder3<double> cylinder_fit;
		gte::ApprCylinder3<double> fitter(20, 1024, 512);
		fitter(gte_data.size(), gte_data.data(), cylinder_fit);
		const double r = cylinder_fit.radius;

		if (r > 2.) {
			auto plane_result = fit_plane(gte_data);
			gp_Pln plane(plane_result.first.a(), plane_result.first.b(), plane_result.first.c(), plane_result.first.d());

			shape = std::make_shared<MyPlane>(v_cluster,
				plane,
				inliers
			);
		}
		else
		{

			Eigen::Vector3d origin = Eigen::Vector3d(
				cylinder_fit.axis.origin[0],
				cylinder_fit.axis.origin[1],
				cylinder_fit.axis.origin[2]);
			Eigen::Vector3d direction = Eigen::Vector3d(
				cylinder_fit.axis.direction[0],
				cylinder_fit.axis.direction[1],
				cylinder_fit.axis.direction[2]).normalized();

			gp_Ax3 ax3(gp_Pnt(origin.x(), origin.y(), origin.z()), gp_Vec(direction.x(), direction.y(), direction.z()));
			gp_Cylinder cylinder_(ax3, r);
			shape = std::make_shared<MyCylinder>(v_cluster, cylinder_, inliers);
		}
	}
	else if (v_type == "cone")
	{
		gte::ApprCone3<double> fitter;
		gte::Vector3<double> coneVertex, coneAxis;
		double coneAngle;
		auto result = fitter(gte_data.size(), gte_data.data(),
			20000, 0., 0., 
			false,
			coneVertex, coneAxis, coneAngle);
		// auto result = fitter(gte_data.size(), gte_data.data(),
		// 	20000, 0., 0.,
		// 	0.001, 1, 1,
		// 	false,
		// 	coneVertex, coneAxis, coneAngle);
		if (coneAngle < 1e-8)
			return { shape, 999999. };
		gp_Cone cone_(gp_Ax3(
			gp_Pnt(coneVertex[0], coneVertex[1], coneVertex[2]),
			gp_Vec(coneAxis[0], coneAxis[1], coneAxis[2])
		),
			coneAngle, 0.001
		);

		shape = std::make_shared<MyCone> (v_cluster,
			cone_,
			inliers
		);
	}
	else if (v_type == "sphere")
	{
		gte::Sphere3<double> sphere__;
		gte::ApprSphere3<double> fitter;
		auto result = fitter.FitUsingLengths(gte_data.size(), gte_data.data(),
			10000, false, sphere__);

		gp_Sphere sphere_(gp_Ax3(
			gp_Pnt(sphere__.center[0], sphere__.center[1], sphere__.center[2]),
			gp_Dir(1, 0, 0)
		),
			sphere__.radius
		);

		shape = std::make_shared<MySphere> (v_cluster,
			sphere_,
			inliers
		);
	}
	else if (v_type == "torus")
	{
		gte::Vector3<double> C, N;
		double n1, n2;
		gte::ApprTorus3<double> fitter;
		auto result = fitter(gte_data.size(), gte_data.data(), 
			// 5000,0, 0, false,
			C, N, n1, n2
		);

		if (n1 <= 0 || n2 <= 0)
			return { nullptr, 999999. };

		gp_Torus torus_(gp_Ax3(
			gp_Pnt(C[0], C[1], C[2]),
			gp_Dir(N[0], N[1], N[2])
		),
			n1,
			n2
		);

		shape = std::make_shared<MyTorus> (v_cluster,
			torus_,
			inliers
		);
	}
	else if (v_type == "ellipsoid")
	{
		gte::ApprEllipsoid3<double> fitter;
		gte::Ellipsoid3<double> ellipsoid_;
		fitter(gte_data, 10000, false, ellipsoid_);

		shape = std::make_shared<MyEllipsoid> (v_cluster,
			ellipsoid_,
			inliers
		);
	}
	else if (v_type == "plane")
	{
		
		auto plane_result = fit_plane(gte_data);
		gp_Pln plane(plane_result.first.a(), plane_result.first.b(), plane_result.first.c(), plane_result.first.d());

		shape = std::make_shared<MyPlane> (v_cluster,
			plane,
			inliers
		);
	}
	else
		throw;
	std::vector<double> errors(v_cluster.surface_points.size());
	#pragma omp parallel for
	for (int i = 0; i < v_cluster.surface_points.size(); ++i)
	{
		errors[i] = shape->distance(
			Eigen::Vector3d(
				v_cluster.surface_points[i][0],
				v_cluster.surface_points[i][1],
				v_cluster.surface_points[i][2]));
	}
	double error = std::accumulate(errors.begin(), errors.end(), 0.) / errors.size();
	return { shape, error };
}

std::pair<std::shared_ptr<Shape>, double> fit_surface(
	const std::vector<Eigen::Vector3d>& v_points,
	const Cluster& v_cluster,
	const std::string& v_type
)
{
	std::vector<Eigen::Vector3d> inliers;
	std::vector<gte::Vector3<double>> gte_data;

	Point_set p;
	p.resize(v_points.size());
	for (int i = 0; i < p.size(); ++i)
		p.point(i) = eigen_2_cgal_point(v_points[i]);

	const auto iterator_to_first_to_remove = CGAL::grid_simplify_point_set(p, 0.001); // optional
	p.remove(iterator_to_first_to_remove, p.end());
	p.collect_garbage();

	gte_data.resize(p.size());
	for (int i = 0; i < gte_data.size(); ++i)
		gte_data[i] = gte::Vector3<double>({ p.point(i)[0], p.point(i)[1], p.point(i)[2] });

	return fit_surface(gte_data, v_cluster, v_type);
}

std::pair<std::shared_ptr<Shape>, double> find_best_fitting(
	const double epsilon,
	const std::vector<gte::Vector3<double>>& gte_data,
	const Cluster& v_cluster)
{
	std::shared_ptr<Shape> shape;
	double best_error = std::numeric_limits<double>::max();

	std::unordered_map<std::string, double> SHAPE_EPSILONS;
	SHAPE_EPSILONS.insert({ "plane", epsilon });
	SHAPE_EPSILONS.insert({ "cylinder", epsilon });
	SHAPE_EPSILONS.insert({ "cone", epsilon });
	SHAPE_EPSILONS.insert({ "sphere", epsilon });
	SHAPE_EPSILONS.insert({ "torus", 5e-3 });

	std::vector<std::string> types = { "plane", "cylinder", "cone", "sphere", "torus" };
	std::string best_type;
	for (const auto& type : types)
	{
		auto result = fit_surface(gte_data, v_cluster, type);
		if (result.second < best_error && result.second < SHAPE_EPSILONS[type])
		{
			shape = (result.first);
			best_error = result.second;
			best_type = type;
		}
		// CGAL::IO::write_point_set("temp/1.ply", result.first->sample_parametric(10000));
	}
	double radius = 0;
	if (best_type == "cylinder")
		radius = std::dynamic_pointer_cast<MyCylinder>(shape)->cylinder.Radius();
	else if (best_type == "sphere")
		radius = std::dynamic_pointer_cast<MySphere>(shape)->sphere.Radius();
	if (radius > 5.) {
		auto result = fit_surface(gte_data, v_cluster, "plane");
		shape = result.first;
		best_error = result.second;
	}
	return { shape, best_error };
}


std::shared_ptr<Shape> fall_back_ransac(const Cluster& v_cluster, const double epsilon, const std::string& v_type, const double radius)
{
	const int max_samples = 100;

	Point_set p;
	p.resize(v_cluster.surface_points.size());
	for (int i = 0; i < p.size(); ++i)
		p.point(i) = eigen_2_cgal_point(v_cluster.surface_points[i]);
	// CGAL::IO::write_point_set("temp/before_sample.ply", p);

	auto iterator_to_first_to_remove = CGAL::grid_simplify_point_set(p, 0.001); // optional
	p.remove(iterator_to_first_to_remove, p.end());
	p.collect_garbage();
	// CGAL::IO::write_point_set("temp/after_sample.ply", p);

	if (p.size() < max_samples)
		return nullptr;

	std::vector<Eigen::Vector3d> non_duplicate(p.size());
	for (int i = 0; i < p.size(); ++i)
		non_duplicate[i] = Eigen::Vector3d(p.point(i)[0], p.point(i)[1], p.point(i)[2]);

	std::unique_ptr<my_kd_tree_t> kdtree;
	kdtree.reset(initialize_kd_tree(p));
	std::vector<double> sample_radiuses{ 0.01, 0.02, 0.03, 0.05 };
	if (radius > 0)
		sample_radiuses = { radius };

	const int num_iters = 100;
	std::vector<int> best_inliers(num_iters, 0);
	std::vector<std::shared_ptr<Shape>> best_shapes(num_iters, nullptr);
	
	tbb::parallel_for(tbb::blocked_range<int>(0, 100), [&](const auto& r0)
		{
			for (int i_iter = r0.begin(); i_iter < r0.end(); ++i_iter)
			{
				std::mt19937 rng(i_iter);
				std::uniform_int_distribution<int> unif(0, p.size() - 1);
				const int id_seed = unif(rng);
				Eigen::Vector3f seed_p(cgal_2_eigen_point<float>(p.point(id_seed)));
				// const auto result = search_range(*kdtree.get(), seed_p, sample_radius);
				const auto result = search_range(*kdtree.get(), seed_p, sample_radiuses[i_iter% sample_radiuses.size()]);
				std::vector<int> indices;
				for (const auto& item : result)
					indices.push_back(item.first);
				std::shuffle(indices.begin(), indices.end(), rng);

				Cluster cluster;

				std::vector<gte::Vector3<double>> gte_data(std::min((int)indices.size(), max_samples));
				cluster.surface_points.resize(gte_data.size());
				for (int i = 0; i < gte_data.size(); ++i)
				{
					cluster.surface_points[i] = { p.point(indices[i])[0], p.point(indices[i])[1], p.point(indices[i])[2] };
					gte_data[i] = gte::Vector3<double>({ p.point(indices[i])[0], p.point(indices[i])[1], p.point(indices[i])[2] });
				}

				// First plane
				double best_error = std::numeric_limits<double>::max();
				std::shared_ptr<Shape> shape = nullptr;
				// auto fitting_result = fit_surface(gte_data, cluster, "plane");
				// if (fitting_result.second < 5 * epsilon)
				// {
				// 	shape = fitting_result.first;
				// 	best_error = fitting_result.second;
				// }

				if (shape == nullptr) // Then quadric primitives
				{
					std::vector<std::string> types = { "plane", "cylinder", "cone", "sphere", };
					if (!v_type.empty())
						types = { v_type };
					for (const auto& type : types)
					{
						auto fitting_result = fit_surface(gte_data, cluster, type);
						if (fitting_result.second < best_error && fitting_result.second < epsilon)
						{
							shape = fitting_result.first;
							best_error = fitting_result.second;
						}
						// CGAL::IO::write_point_set("temp/1.ply", result.first->sample_parametric(10000));
					}
				}

				if (shape != nullptr)
				{
					std::vector<int> errors(p.size());
					for (int i = 0; i < errors.size(); ++i)
					{
						if (shape->distance(cgal_2_eigen_point<double>(p.point(i))) < epsilon)
							errors[i] = 1;
						else
							errors[i] = 0;
					}
					int num_inliers_local = std::accumulate(errors.begin(), errors.end(), 0);
					if (num_inliers_local > 20)
					{
						best_inliers[i_iter] = num_inliers_local;
						best_shapes[i_iter] = shape;
					}
				}
			}
		});

	const int id_best = std::max_element(best_inliers.begin(), best_inliers.end()) - best_inliers.begin();
	if (best_shapes[id_best]==nullptr)
		return nullptr;

	best_shapes[id_best]->get_inliers(non_duplicate, epsilon);
	best_shapes[id_best] = fit_surface(best_shapes[id_best]->inliers, best_shapes[id_best]->cluster, best_shapes[id_best]->detail_type).first;
	return best_shapes[id_best];
}

std::vector<std::shared_ptr<Shape>> fitting(const std::vector<Cluster>& v_input,
	const double epsilon,
	const int num_fitting_points
)
{
	std::vector<std::shared_ptr<Shape>> shapes;
	// Start clustering
	std::mutex locker;
	tbb::parallel_for(tbb::blocked_range<int>(0, v_input.size()), [&](const auto& r0)
	{
		for (int i_cluster = r0.begin(); i_cluster < r0.end(); ++i_cluster)
		{
			if (v_input[i_cluster].surface_points.size() < 10) // Not enough points
				continue;

			Point_set p;
			p.resize(v_input[i_cluster].surface_points.size());
			for (int i = 0; i < p.size(); ++i)
				p.point(i) = eigen_2_cgal_point(v_input[i_cluster].surface_points[i]);
			auto iterator_to_first_to_remove = CGAL::grid_simplify_point_set(p, 0.001); // optional
			p.remove(iterator_to_first_to_remove, p.end());
			p.collect_garbage();

			if (p.size() < 10)  // Not enough points
				continue;

			std::vector<Eigen::Vector3d> non_duplicate_points(p.size());
			for (int i = 0; i < p.size(); ++i)
				non_duplicate_points[i] = cgal_2_eigen_point<double>(p.point(i));

			// CGAL::IO::write_point_set("temp/temp.ply", p);

			bool is_successful = false;
			std::shared_ptr<Shape> shape;

			// Fit point
			auto point_result = fit_vertex(non_duplicate_points, v_input[i_cluster], epsilon);
			if (point_result.second < epsilon)
			{
				shape = point_result.first;
				is_successful = true;
				LOG(INFO) << "Found point";
			}
			else
			{
				const int max_samples = num_fitting_points;
				const int num_points = std::min((int)p.size(), max_samples);

				std::vector<int> indexes(p.size());
				std::iota(indexes.begin(), indexes.end(), 0);
				std::mt19937 rng(0);
				std::shuffle(indexes.begin(), indexes.end(), rng);

				std::vector<gte::Vector3<double>> gte_data(num_points);
				for (int i = 0; i < num_points; ++i)
					gte_data[i] = gte::Vector3<double>({ p.point(indexes[i])[0], p.point(indexes[i])[1], p.point(indexes[i])[2] });

				// Debug
				if (false)
				{
					std::vector<Eigen::Vector3d> points(gte_data.size());
					for(int i=0;i<gte_data.size();++i)
						points[i] = Eigen::Vector3d(gte_data[i][0], gte_data[i][1], gte_data[i][2]);
					export_points("temp/temp1.ply", points);

					points.resize(v_input[i_cluster].surface_points.size());
					for (int i = 0; i < v_input[i_cluster].surface_points.size(); ++i)
						points[i] = Eigen::Vector3d(v_input[i_cluster].surface_points[i][0], v_input[i_cluster].surface_points[i][1], v_input[i_cluster].surface_points[i][2]);
					export_points("temp/temp2.ply", points);
				}

				// Fit Curves
				auto plane_result = fit_surface(gte_data, v_input[i_cluster], "plane");
				// auto plane_result = fit_plane(v_input[i_cluster]);
				// Check if it is a curve
				if (plane_result.second < epsilon)
				{
					gp_Pln plane = std::dynamic_pointer_cast<MyPlane>(plane_result.first)->plane;

					double best_error = std::numeric_limits<double>::max();

					std::vector<std::string> types = { "line", "circle"};
					// std::vector<std::string> types = { "line", "circle", "ellipse" };
					for (const auto& type : types)
					{
						auto result = fit_curve(non_duplicate_points, v_input[i_cluster], plane, type);
						if (result.second < best_error && result.second < epsilon)
						{
							shape = (result.first);
							best_error = result.second;
							is_successful = true;
							LOG(INFO) << ffmt("Found %s in %d") % type % i_cluster;
							break;
						}
						// CGAL::IO::write_point_set("temp/1.ply", result.first->sample_parametric(10000));
					}
					// Now we can sure that it is a plane
					if (!is_successful)
					{
						// shape = find_best_fitting(epsilon, gte_data, v_input[i_cluster]).first;
						shape = plane_result.first;
						// if (shape != nullptr)
						{
							// CGAL::IO::write_point_set("test.ply", my_plane.sample_parametric(1000));
							is_successful = true;
							LOG(INFO) << "Found " << shape->detail_type << " in " << i_cluster;
						}
					}
				}
				else // Fit Surfaces
				{
					// shape = find_best_fitting(epsilon, gte_data, v_input[i_cluster]).first;
					// if (shape != nullptr)
					// {
					// 	// CGAL::IO::write_point_set("test.ply", my_plane.sample_parametric(1000));
					// 	is_successful = true;
					// 	LOG(INFO) << "Found " << shape->detail_type << " in " << i_cluster;
					// }
					
					double best_error = std::numeric_limits<double>::max();
					
					std::unordered_map<std::string, double> SHAPE_EPSILONS;
					SHAPE_EPSILONS.insert({ "cylinder", epsilon });
					SHAPE_EPSILONS.insert({ "cone", epsilon });
					SHAPE_EPSILONS.insert({ "sphere", epsilon });
					SHAPE_EPSILONS.insert({ "torus", 5e-3 });
					
					std::vector<std::string> types = { "cylinder", "cone", "sphere", "torus" };
					for (const auto& type : types)
					{
						auto result = fit_surface(gte_data, v_input[i_cluster], type);
						if (result.second < best_error && result.second < SHAPE_EPSILONS[type])
						{
							shape = (result.first);
							best_error = result.second;
							is_successful = true;
							LOG(INFO) << ffmt("Found %s in %d") % type % i_cluster;
						}
						// CGAL::IO::write_point_set("temp/1.ply", result.first->sample_parametric(10000));
					}
				}
			}

			if (is_successful && shape != nullptr)
			{
				locker.lock();
				shapes.push_back(shape);
				locker.unlock();
			}
			else
			{
				LOG(INFO) << "ERROR: Cannot find suitable primitive in " << i_cluster;
				Cluster cluster = v_input[i_cluster];

				while (cluster.surface_points.size() > 20)
				{
					std::shared_ptr<Shape> shape = fall_back_ransac(cluster, epsilon);
					if (shape == nullptr)
						break;
					else
					{
						Cluster remain;
						shape->cluster.surface_points.clear();
						for (int i = 0; i < cluster.surface_points.size(); ++i)
						{
							if (shape->distance(cluster.surface_points[i]) < epsilon)
							{
								shape->cluster.query_points.push_back(cluster.query_points[i]);
								shape->cluster.coords.push_back(cluster.coords[i]);
								shape->cluster.surface_points.push_back(cluster.surface_points[i]);
							}
							else
							{
								remain.query_points.push_back(cluster.query_points[i]);
								remain.coords.push_back(cluster.coords[i]);
								remain.surface_points.push_back(cluster.surface_points[i]);
							}
						}
						if (shape->cluster.surface_points.size() < 10)
							break;
						shape->get_inliers(shape->cluster.surface_points, epsilon);
						shape->find_boundary();

						cluster = remain;

						locker.lock();
						LOG(INFO) << ffmt("Fallback ransac found %s of %d points in cluster %d") % shape->detail_type % shape->cluster.surface_points.size() % i_cluster;
						shapes.push_back(shape);
						locker.unlock();
					}
				}
			}
		}
	});

	// Refit using all the points
	std::vector<bool> should_be_deleted(shapes.size(), false);
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, shapes.size()), [&](const auto& r0)
			{
				for(int i_shape=r0.begin(); i_shape<r0.end(); ++i_shape)
				{
					if (shapes[i_shape]->type == "surface" && shapes[i_shape]->detail_type != "cone") // Cone fitting is not stable for too many points
						shapes[i_shape] = fit_surface(
							shapes[i_shape]->cluster.surface_points, shapes[i_shape]->cluster, shapes[i_shape]->detail_type).first;
					else if (shapes[i_shape]->type == "curve")
						shapes[i_shape] = fit_curve(
							shapes[i_shape]->cluster.surface_points, 
							shapes[i_shape]->cluster,
							dynamic_pointer_cast<Shape2D>(shapes[i_shape])->m_plane,
							shapes[i_shape]->detail_type).first;
					if (shapes[i_shape] == nullptr)
					{
						LOG(ERROR) << "Re fit failed; Skip";
						should_be_deleted[i_shape] = true;
						continue;
					}

					shapes[i_shape]->get_inliers(shapes[i_shape]->cluster.surface_points, epsilon);
					if (shapes[i_shape]->inliers.empty())
					{
						LOG(ERROR) << "Re fit failed; Skip";
						should_be_deleted[i_shape] = true;
						continue;
					}
					shapes[i_shape]->find_boundary();
				}
			});
	}
	shapes.erase(std::remove_if(shapes.begin(), shapes.end(),[&should_be_deleted, &shapes](const auto& item)
	{
		return should_be_deleted[&item - &shapes[0]];
	}), shapes.end());

	return shapes;
}

std::shared_ptr<Shape> check_valid_ellipse(std::shared_ptr<Shape>& v_shape, const std::vector<Eigen::Vector3d>& inliers,
	const double epsilon)
{
	// Test if we can use line to approximate the ellipse
	if (v_shape->detail_type != "ellipse")
		return v_shape;

	const std::shared_ptr<MyEllipse> ellipse = dynamic_pointer_cast<MyEllipse>(v_shape);

	if (ellipse->ellipse.MajorRadius() > 1. || ellipse->ellipse.MajorRadius() / ellipse->ellipse.MinorRadius() > 30)
	{
		Handle(Geom_Ellipse) ellipse_ = new Geom_Ellipse(ellipse->ellipse);

		std::vector<Eigen::Vector3d> points;
		for (int i = 0; i < inliers.size(); ++i)
		{
			gp_Pnt p_3d(inliers[i].x(), inliers[i].y(), inliers[i].z());
			GeomAPI_ProjectPointOnCurve aProjector(p_3d, ellipse_);
			if (aProjector.NbPoints() <= 0)
				throw;
			gp_Pnt p = aProjector.NearestPoint();
			points.emplace_back(p.X(), p.Y(), p.Z());
		}

		auto line_shape = fit_curve(
			points, ellipse->cluster,
			ellipse->m_plane, "line");
		if (line_shape.second < epsilon)
		{
			line_shape.first->inliers = inliers;
			return line_shape.first;
		}
		else
			return nullptr;
	}
	return v_shape;
}
