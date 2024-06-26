#include "cgal_tools.h"

void write_ply(const fs::path& v_path, const std::vector<Triangle_3>& v_mesh)
{
	std::ofstream ofs(v_path.string(), std::ios::binary);
	ofs << "ply\n";
	ofs << "format ascii 1.0\n";
	// ofs << "format binary_big_endian 1.0\n";
	ofs << "element vertex " << v_mesh.size() * 3 << "\n";
	ofs << "property float x\n";
	ofs << "property float y\n";
	ofs << "property float z\n";
	ofs << "element face " << v_mesh.size() << "\n";
	ofs << "property list uchar int vertex_index\n";
	ofs << "end_header\n";

	for (int i = 0; i < v_mesh.size(); ++i)
	{
		ofs << v_mesh[i][0].x() << " " << v_mesh[i][0].y() << " " << v_mesh[i][0].z() << "\n";
		ofs << v_mesh[i][1].x() << " " << v_mesh[i][1].y() << " " << v_mesh[i][1].z() << "\n";
		ofs << v_mesh[i][2].x() << " " << v_mesh[i][2].y() << " " << v_mesh[i][2].z() << "\n";
	}
	for (int i = 0; i < v_mesh.size(); ++i)
	{
		ofs << "3 " << i * 3 + 0 << " " << i * 3 + 1 << " " << i * 3 + 2 << "\n";
	}
	ofs.close();
}

std::vector<int> fps_sampling(const Point_set& v_point_set, const int num_points)
{
	int point_size = v_point_set.size();
	//======================================================================================
	std::vector<int> centre_indexes; // Set A
	v_point_set;                     // Set B
	std::vector<float> squared_distance_cache(
		point_size, std::numeric_limits<float>::max()
	); // max distance to set A

	std::vector<int> used_index(point_size, 0);

	// 0st
	int centre_index = Global_helper::get_uniform_int(0, point_size - 1);
	used_index[centre_index] = 1;
	centre_indexes.push_back(centre_index);

	for (int i_iter = 1; i_iter < num_points; ++i_iter) 
	{
		for (int i_point = 0; i_point < v_point_set.size(); ++i_point) 
		{
			float distance;
			if (used_index[i_point] != 0)
				distance = 0;
			else
				distance = CGAL::squared_distance(
					v_point_set.point(centre_indexes.back()),
					v_point_set.point(i_point)
				);
			if (distance < squared_distance_cache[i_point])
				squared_distance_cache[i_point] = distance;
		}
		centre_index
			= std::max_element(
				squared_distance_cache.begin(), squared_distance_cache.end()
			)
			- squared_distance_cache.begin();
		if (squared_distance_cache[centre_index] < 1e-6) break;
		centre_indexes.push_back(centre_index);
		used_index[centre_index] = 1;
	}
	return centre_indexes;
}

// @brief: 
// @notice: Currently only transfer vertices to the cgal Surface mesh
// @param: `attrib_t, shape_t, material_t`
// @ret: Surface_mesh
Surface_mesh convert_obj_from_tinyobjloader_to_surface_mesh(
	const std::tuple<tinyobj::attrib_t, std::vector<tinyobj::shape_t>, std::vector<tinyobj::material_t>> v_obj_in){
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::tie(attrib, shapes, materials)=v_obj_in;
	
	Surface_mesh out;

    std::unordered_map<int, Surface_mesh::vertex_index> vertex_map;

    std::vector<Point_3> points;
    std::vector<std::vector<size_t>> polygon;

	for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t face_id = 0; face_id < shapes[s].mesh.num_face_vertices.size(); face_id++) {
            if (shapes[s].mesh.num_face_vertices[face_id] != 3)
                throw;

            Surface_mesh::vertex_index vi[3];
            for (size_t v = 0; v < 3; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
            	
                tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                points.push_back(Point_3(vx, vy, vz));
                if (vertex_map.find(idx.vertex_index) == vertex_map.end())
                {
                    vertex_map.insert(std::make_pair(idx.vertex_index, out.add_vertex(Point_3(vx, vy, vz))));
                    vi[v] = vertex_map.at(idx.vertex_index);
                }
                else
                    vi[v] = vertex_map.at(idx.vertex_index);
            }
            out.add_face(vi[0], vi[1], vi[2]);
            polygon.push_back(std::vector<size_t>{points.size() - 2, points.size() - 1, points.size()});
            index_offset += 3;
        }
    }
    Surface_mesh out1;
   //CGAL::Polygon_mesh_processing::repair_polygon_soup(points, polygon);
    //CGAL::Polygon_mesh_processing::orient_polygon_soup(points, polygon);
    //CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points,polygon,out1);
    return out;
}

Eigen::AlignedBox3f get_bounding_box(const Point_set& v_point_set)
{
    float xmin = 1e8, ymin = 1e8, zmin = 1e8;
    float xmax = -1e8, ymax = -1e8, zmax = -1e8;

	for(Point_set::Index idx:v_point_set)
	{
        float vx = v_point_set.point(idx).x();
        float vy = v_point_set.point(idx).y();
        float vz = v_point_set.point(idx).z();
        xmin = xmin < vx ? xmin : vx;
        ymin = ymin < vy ? ymin : vy;
        zmin = zmin < vz ? zmin : vz;

        xmax = xmax > vx ? xmax : vx;
        ymax = ymax > vy ? ymax : vy;
        zmax = zmax > vz ? zmax : vz;
	}


    return Eigen::AlignedBox3f(Eigen::Vector3f(xmin, ymin, zmin), Eigen::Vector3f(xmax, ymax, zmax));
}

Eigen::AlignedBox3f get_bounding_box(const std::vector<Point_3>& v_point_set)
{
    float xmin = 1e8, ymin = 1e8, zmin = 1e8;
    float xmax = -1e8, ymax = -1e8, zmax = -1e8;

    for (const Point_3& point : v_point_set)
    {
        float vx = point.x();
        float vy = point.y();
        float vz = point.z();
        xmin = xmin < vx ? xmin : vx;
        ymin = ymin < vy ? ymin : vy;
        zmin = zmin < vz ? zmin : vz;

        xmax = xmax > vx ? xmax : vx;
        ymax = ymax > vy ? ymax : vy;
        zmax = zmax > vz ? zmax : vz;
    }
    return Eigen::AlignedBox3f(Eigen::Vector3f(xmin, ymin, zmin), Eigen::Vector3f(xmax, ymax, zmax));
}

Rotated_box get_bounding_box_rotated(const Point_set& v_point_set)
{
    float xmin = 1e8, ymin = 1e8, zmin = 1e8;
    float xmax = -1e8, ymax = -1e8, zmax = -1e8;

    std::vector<cv::Point2f> points;
	
    for (Point_set::Index idx : v_point_set)
    {
        float vx = v_point_set.point(idx).x();
        float vy = v_point_set.point(idx).y();
        float vz = v_point_set.point(idx).z();
        xmin = xmin < vx ? xmin : vx;
        ymin = ymin < vy ? ymin : vy;
        zmin = zmin < vz ? zmin : vz;

        xmax = xmax > vx ? xmax : vx;
        ymax = ymax > vy ? ymax : vy;
        zmax = zmax > vz ? zmax : vz;
        points.emplace_back(vx, vy);
    }

    cv::RotatedRect rotated_rect = cv::minAreaRect(points);
    
    Rotated_box box(Eigen::AlignedBox3f(
        Eigen::Vector3f(rotated_rect.center.x - rotated_rect.size.width / 2, rotated_rect.center.y - rotated_rect.size.height / 2, zmin),
        Eigen::Vector3f(rotated_rect.center.x + rotated_rect.size.width / 2, rotated_rect.center.y + rotated_rect.size.height / 2, zmax)));

    box.angle = rotated_rect.angle/180.f * 3.1415926f;
    box.cv_box = rotated_rect;
    return box;
}

Eigen::Vector3d cgal_point_2_eigend(const Point_3& p)
{
    return Eigen::Vector3d(p.x(), p.y(), p.z());
}

Eigen::Vector3f cgal_point_2_eigen(const Point_3& p)
{
	return Eigen::Vector3f(p.x(), p.y(), p.z());
}

Eigen::Vector3d cgal_vector_2_eigend(const Vector_3& p)
{
    return Eigen::Vector3d(p.x(), p.y(), p.z());
}

Eigen::Vector3f cgal_vector_2_eigen(const Vector_3& p)
{
    return Eigen::Vector3f(p.x(), p.y(), p.z());
}

Point_3 eigen_2_cgal_point(const Eigen::Vector3f& p)
{
	return Point_3(p.x(), p.y(), p.z());
}
Vector_3 eigen_2_cgal_vector(const Eigen::Vector3f& p)
{
    return Vector_3(p.x(), p.y(), p.z());
}

Point_3 eigen_2_cgal_point(const Eigen::Vector3d& p)
{
    return Point_3(p.x(), p.y(), p.z());
}
Vector_3 eigen_2_cgal_vector(const Eigen::Vector3d& p)
{
    return Vector_3(p.x(), p.y(), p.z());
}