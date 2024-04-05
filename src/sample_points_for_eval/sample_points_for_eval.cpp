#include "common_util.h"

#include "boundary_growing.h"
#include "tools.h"
#include "assemble.h"
#include "filling_holes.h"
#include "fitting.h"
#include "merge_shape.h"
#include "shape2d.h"
#include "shape3d.h"

#include <numeric>
#include <sstream>
#include <istream>
#include <fstream>
#include <omp.h>

#include <argparse/argparse.hpp>

#include <CGAL/edge_aware_upsample_point_set.h>
#include <CGAL/pca_estimate_normals.h>

#include <gp_Pln.hxx>
#include <CGAL/Polygon_mesh_processing/measure.h>

#include "npy.hpp"
#include "cgal_tools.h"

#include <tbb/tbb.h>

#include "assemble_loops.h"

#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <TopoDS_Wire.hxx>
#include <TopoDS_Face.hxx>
#include <BRepLib.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepBuilderAPI_MakeShell.hxx>
#include <BRep_Builder.hxx>
#include <StlAPI_Writer.hxx>
#include <CGAL/Polygon_mesh_processing/repair_degeneracies.h>

// #pragma optimize("", off)

int main(int argc, char* argv[])
{

	fs::path root = "/mnt/d/GSP_test/ours/0121_total_mesh/";
	fs::path output_root = "/mnt/d/GSP_test/ours/0121_total_mesh_eval";
	std::vector<std::string> tasks;
	for (fs::directory_iterator cur_it(root); cur_it != fs::directory_iterator(); ++cur_it)
	{
		const auto str = cur_it->path().filename().string();
		if (str.substr(str.size() - 4) != ".txt")
			continue;
		tasks.push_back(cur_it->path().filename().stem().string().substr(0, 8));
	}

	tbb::parallel_for(tbb::blocked_range<int>(0, tasks.size()), [&](const auto& r0)
		{
			for (int i_task = r0.begin(); i_task != r0.end(); ++i_task)
			{
				std::string task = tasks[i_task];
				std::vector<std::shared_ptr<Shape>> shapes;
				checkFolder(output_root / task / "eval");
				std::ifstream ifs((root / task / "shape_cache").string(), std::ios::binary | std::ios::in);
				if (!ifs.is_open())
					LOG(FATAL) << "Cannot open file " << (root / task / "shape_cache").string();
				boost::archive::binary_iarchive oa(ifs);
				oa >> shapes;
				ifs.close();

				// LOG(INFO) << "Start to save results";

				Point_set surfaces;
				auto surfaces_index_map = surfaces.add_property_map("primitive_index", 0).first;

				int num_surfaces = 0;
				std::vector<int> id_surface(shapes.size(), -1);
				for (int i_shape = 0; i_shape < shapes.size(); ++i_shape)
				{
					if (shapes[i_shape]->type == "surface")
					{
						shapes[i_shape]->find_boundary();
						const auto shape = dynamic_pointer_cast<Shape3D>(shapes[i_shape]);
						Surface_mesh mesh;
						for (const auto& item : shape->m_boundary)
						{
							mesh.add_face(
								mesh.add_vertex(
									eigen_2_cgal_point(shape->get_cartesian(cgal_2_eigen_point<double>(item.vertex(0))))),
								mesh.add_vertex(
									eigen_2_cgal_point(shape->get_cartesian(cgal_2_eigen_point<double>(item.vertex(1))))),
								mesh.add_vertex(
									eigen_2_cgal_point(shape->get_cartesian(cgal_2_eigen_point<double>(item.vertex(2)))))
							);
						}
						double area = CGAL::Polygon_mesh_processing::area(mesh);
						const int num_points = std::ceil(area * 10000);
						Point_set p = shape->sample_parametric(num_points);
						// std::cout << task << ": " << num_points << ", " << p.size() << std::endl;

						for (const auto& item : p.points())
							surfaces_index_map[*surfaces.insert(item)] = num_surfaces;
						id_surface[i_shape] = num_surfaces;
						num_surfaces += 1;
					}
				}

				CGAL::IO::write_point_set((output_root / task / "eval/surfaces.ply").string(), surfaces);
			}
		});


	return 0;
}
