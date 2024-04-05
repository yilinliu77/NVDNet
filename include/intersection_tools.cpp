#include "intersection_tools.h"

#include <algorithm>

std::tuple<cv::Mat, cv::Mat, cv::Mat>
get_depth_map_through_meshes(
  const std::vector<Surface_mesh>& v_meshes,
  const int                        v_width,
  const int                        v_height,
  const Eigen::Matrix3f&           v_intrinsic
)
{
    std::vector<Surface_mesh> meshes(v_meshes);

    Tree tree;
    for (int i = 0; i < v_meshes.size(); ++i) {
        meshes[i].add_property_map<Surface_mesh::face_index, int>(
          "instance_id", i
        );
        tree.insert(
          CGAL::faces(meshes[i]).first, CGAL::faces(meshes[i]).second, meshes[i]
        );
    }
    tree.build();

    cv::Mat img_instance(v_height, v_width, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat img_distance_perspective(
      v_height, v_width, CV_32FC1, cv::Scalar(0.f)
    );
    cv::Mat img_distance_planar(v_height, v_width, CV_32FC1, cv::Scalar(0.f));

#pragma omp parallel for
    for (int y = 0; y < v_height; y++) {
        for (int x = 0; x < v_width; x++) {
            Eigen::Vector3f p(x, y, 1);
            p = v_intrinsic.inverse() * p;
            Ray  ray(Point_3(0, 0, 0), Vector_3(p.x(), p.y(), p.z()));
            auto result  = tree.first_intersection(ray);
            auto result1 = tree.first_intersected_primitive(ray);
            if (result1) {
                const Surface_mesh::face_index face_index = result1->first;
                const Surface_mesh*            mesh       = result1->second;
                if (mesh) {
                    const Primitive_id& primitive_id
                      = boost::get<Primitive_id>(result->second);
                    for (int id_mesh = 0; id_mesh < meshes.size(); id_mesh++) {
                        Surface_mesh::
                          Property_map<Surface_mesh::face_index, int>
                             gnus_iter;
                        bool found;
                        boost::tie(gnus_iter, found)
                          = meshes[id_mesh]
                              .property_map<Surface_mesh::face_index, int>(
                                "instance_id"
                              );

                        Surface_mesh::
                          Property_map<Surface_mesh::face_index, int>
                            gnus;
                        boost::tie(gnus, found)
                          = mesh->property_map<Surface_mesh::face_index, int>(
                            "instance_id"
                          );

                        if (gnus_iter[*(meshes[id_mesh].faces_begin())] == gnus[face_index])
                            img_instance.at<cv::Vec3b>(y, x)[2] = id_mesh;
                    }
                    img_distance_perspective.at<float>(y, x)
                      = std::sqrt(CGAL::squared_distance(
                        boost::get<Point_3>(result->first), Point_3(0, 0, 0)
                      ));

                    img_distance_planar.at<float>(y, x)
                      = boost::get<Point_3>(result->first)[2];
                }
            }
        }
    }

    // cv::imshow("1", img_instance);
    // cv::waitKey();

    return { img_instance, img_distance_planar, img_distance_perspective };
}

const Point_cloud
remove_points_inside(
  const Surface_mesh&         v_mesh,
  const std::vector<Point_3>& v_points
)
{
    Tree tree;
    tree.insert(CGAL::faces(v_mesh).first, CGAL::faces(v_mesh).second, v_mesh);
    tree.build();

    // Define scanner camera's position
    std::vector<Point_3> camera;
    float                min_x = std::numeric_limits<float>::max();
    float                min_y = std::numeric_limits<float>::max();
    float                min_z = std::numeric_limits<float>::max();
    float                max_z = std::numeric_limits<float>::min();
    float                max_y = std::numeric_limits<float>::min();
    float                max_x = std::numeric_limits<float>::min();

    for (auto vertex : v_mesh.vertices()) {
        min_x
          = v_mesh.point(vertex).x() < min_x ? v_mesh.point(vertex).x() : min_x;
        min_y
          = v_mesh.point(vertex).y() < min_y ? v_mesh.point(vertex).y() : min_y;
        min_z
          = v_mesh.point(vertex).z() < min_z ? v_mesh.point(vertex).z() : min_z;
        max_x
          = v_mesh.point(vertex).x() > max_x ? v_mesh.point(vertex).x() : max_x;
        max_y
          = v_mesh.point(vertex).y() > max_y ? v_mesh.point(vertex).y() : max_y;
        max_z
          = v_mesh.point(vertex).z() > max_z ? v_mesh.point(vertex).z() : max_z;
    }

    float diff_x = max_x - min_x;
    float diff_y = max_y - min_y;
    float diff_z = max_z - min_z;

    float max_diff = std::max({ diff_x, diff_y, diff_z });
    float pad      = max_diff * 5;

    // Back
    camera.emplace_back(min_x - pad, min_y - pad, min_z - pad);
    camera.emplace_back(min_x - pad, (min_y + max_y) / 2, min_z - pad);
    camera.emplace_back(min_x - pad, max_y + pad, min_z - pad);

    camera.emplace_back(min_x - pad, min_y - pad, (min_z + max_z) / 2);
    camera.emplace_back(min_x - pad, (min_y + max_y) / 2, (min_z + max_z) / 2);
    camera.emplace_back(min_x - pad, max_y + pad, (min_z + max_z) / 2);

    camera.emplace_back(min_x - pad, min_y - pad, max_z + pad);
    camera.emplace_back(min_x - pad, (min_y + max_y) / 2, max_z + pad);
    camera.emplace_back(min_x - pad, max_y + pad, max_z + pad);

    // Mid
    camera.emplace_back((min_x + max_x) / 2, min_y - pad, min_z - pad);
    camera.emplace_back((min_x + max_x) / 2, (min_y + max_y) / 2, min_z - pad);
    camera.emplace_back((min_x + max_x) / 2, max_y + pad, min_z - pad);

    camera.emplace_back((min_x + max_x) / 2, min_y - pad, (min_z + max_z) / 2);
    camera.emplace_back((min_x + max_x) / 2, max_y + pad, (min_z + max_z) / 2);

    camera.emplace_back((min_x + max_x) / 2, min_y - pad, max_z + pad);
    camera.emplace_back((min_x + max_x) / 2, (min_y + max_y) / 2, max_z + pad);
    camera.emplace_back((min_x + max_x) / 2, max_y + pad, max_z + pad);

    // Front
    camera.emplace_back(max_x + pad, min_y - pad, min_z - pad);
    camera.emplace_back(max_x + pad, (min_y + max_y) / 2, min_z - pad);
    camera.emplace_back(max_x + pad, max_y + pad, min_z - pad);

    camera.emplace_back(max_x + pad, min_y - pad, (min_z + max_z) / 2);
    camera.emplace_back(max_x + pad, (min_y + max_y) / 2, (min_z + max_z) / 2);
    camera.emplace_back(max_x + pad, max_y + pad, (min_z + max_z) / 2);

    camera.emplace_back(max_x + pad, min_y - pad, max_z + pad);
    camera.emplace_back(max_x + pad, (min_y + max_y) / 2, max_z + pad);
    camera.emplace_back(max_x + pad, max_y + pad, max_z + pad);

    Point_cloud out_points;

    //#pragma omp parallel for
    for (int i = 0; i < v_points.size(); i++) {
        int visible_flag = 0;
        for (size_t i_camera = 0; i_camera < camera.size(); ++i_camera) {
            Ray ray(v_points[i], camera[i_camera]);
            int result = tree.number_of_intersected_primitives(ray);
            if (result <= 1) { visible_flag += 1; }
        }
        if (visible_flag > 0) {
            //#pragma omp critical
            out_points.insert(v_points[i]);
        }
    }
    return out_points;
}


bool inside_frustum(
    const Eigen::Vector3f& v_point_pos,
    const Eigen::Vector3f& v_view_pos,
    const Eigen::Matrix3f& v_intrinsic_matrix,
    const Eigen::Isometry3f& v_projection_matrix,
    const float v_max_distance
)
{
    const float view_to_origin_point = (v_point_pos - v_view_pos).norm();
    // Distance Check
    if (v_max_distance != -1.f)
        if (v_max_distance < view_to_origin_point) 
            return false;

    // Orientation Check
    Eigen::Vector3f pixel_pos = v_intrinsic_matrix * v_projection_matrix * v_point_pos;
    if (pixel_pos.z() < 0)
        return false;
    pixel_pos /= pixel_pos.z();
    // Frustum Check
    if (pixel_pos.x() < 0 || pixel_pos.x() > v_intrinsic_matrix(0, 2) * 2 ||
        pixel_pos.y() < 0 || pixel_pos.y() > v_intrinsic_matrix(1, 2) * 2)
        return false;
    return true;
}

bool
is_visible(
  const Eigen::Matrix3f&   v_intrinsic_matrix,
  const Eigen::Isometry3f& v_camera_matrix,
  const Eigen::Vector3f&   v_view_pos,
  const Eigen::Vector3f&   v_point_pos,
  const Tree&              v_tree,
  float                    max_distance
)
{

    const Point_3 point_pos(v_point_pos.x(), v_point_pos.y(), v_point_pos.z());
    const Point_3 view_pos(v_view_pos.x(), v_view_pos.y(), v_view_pos.z());

    Vector_3 view_to_point = point_pos - view_pos;
    Vector_3 view_to_point_norm
      = view_to_point / std::sqrt(view_to_point.squared_length());
    double view_to_origin_point = view_to_point.squared_length();
    if (max_distance != -1.f)
        if (max_distance * max_distance < view_to_origin_point) return false;

    // Horizontal
    Eigen::Vector3f pixel_pos
      = v_intrinsic_matrix * v_camera_matrix * v_point_pos;
    pixel_pos /= pixel_pos.z();
    if(pixel_pos.x()<0||pixel_pos.x()>v_intrinsic_matrix(0,2)*2||pixel_pos.y()<0||pixel_pos.y()>v_intrinsic_matrix(1,2)*2)
        return false;

    // Intersection Test
    Ray    ray_query(view_pos, view_to_point_norm);
    auto   intersection = v_tree.first_intersection(ray_query);
    double intersect_point_to_origin_point;
    double view_to_intersect_point;
    ;
    if (intersection && boost::get<Point_3>(&(intersection->first))) {
        const Point_3* p        = boost::get<Point_3>(&(intersection->first));
        view_to_intersect_point = CGAL::squared_distance(view_pos, *p);
        intersect_point_to_origin_point = CGAL::squared_distance(point_pos, *p);
    } else
        return false;
    return view_to_intersect_point >= view_to_origin_point
        || intersect_point_to_origin_point <= 1e-8;
}
