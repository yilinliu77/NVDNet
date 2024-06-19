import time
import open3d as o3d
import os,sys
import numpy as np
import torch

from functools import partial

from tqdm import tqdm
import cutils

sys.path.append(os.path.join(os.path.dirname(__file__), "NDC"))
import datasetpc
import modelpc
import torch.multiprocessing as mp

num_gpu = -1
resolution=128


def predict(i_gpu, files, root):
    receptive_padding = 3 #for grid input
    pooling_radius = 2 #for pointcloud input
    KNN_num = modelpc.KNN_num
    device = torch.device("cuda", i_gpu)

    print('loading net...')

    network_bool =  modelpc.local_pointnet(out_bool=True, out_float=False)
    network_bool.load_state_dict(torch.load("weights_undc_pointcloud_bool.pth"))
    network_float = modelpc.local_pointnet(out_bool=False, out_float=True)
    network_float.load_state_dict(torch.load("weights_undc_pointcloud_float.pth"))

    network_bool.to(device)
    network_float.to(device)
    network_bool.eval()
    network_float.eval()

    print('loading net... complete')

    coords = np.meshgrid(np.arange(256), np.arange(256), np.arange(256), indexing="ij")
    coords = np.stack(coords, axis=3) / (256 - 1)
    coords = (coords * 2 - 1).astype(np.float32).reshape(-1,3)

    if i_gpu == 0:
        bar = tqdm(total=len(files))
    time_statics = [0]*10
    for idx, file in enumerate(files):
        if i_gpu == 0:
            # print(time_statics)
            bar.update(1)
        if idx % num_gpu != i_gpu:
            continue
        cur_time = time.time()
        prefix = file[:-4]
        pcd = o3d.io.read_point_cloud(os.path.join(root, "poisson",file))
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)/2)
        o3d.io.write_point_cloud("{}.ply".format(i_gpu), pcd)
        time_statics[0]+=time.time()-cur_time
        cur_time = time.time()
        dataset_test = datasetpc.single_shape_pointcloud("{}.ply".format(i_gpu), 10000, resolution, KNN_num, pooling_radius, normalize=False)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)  #batch_size must be 1

        for _, data in enumerate(dataloader_test, 0):
            pc_KNN_idx_,pc_KNN_xyz_, voxel_xyz_int_,voxel_KNN_idx_,voxel_KNN_xyz_ = data

            pc_KNN_idx = pc_KNN_idx_[0].to(device)
            pc_KNN_xyz = pc_KNN_xyz_[0].to(device)
            voxel_xyz_int = voxel_xyz_int_[0].to(device)
            voxel_KNN_idx = voxel_KNN_idx_[0].to(device)
            voxel_KNN_xyz = voxel_KNN_xyz_[0].to(device)

            with torch.no_grad():
                pred_output_bool = network_bool(pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz)
                pred_output_float = network_float(pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz)

                pred_output_bool_grid = torch.zeros([resolution+1,resolution+1,resolution+1,3], dtype=torch.int32).to(device)
                pred_output_float_grid = torch.full([resolution+1,resolution+1,resolution+1,3], 0.5).to(device)

                pred_output_bool_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = (pred_output_bool>0.5).int()
                pred_output_float_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = pred_output_float

                pred_output_bool_grid = modelpc.postprocessing(pred_output_bool_grid)

                pred_output_bool_numpy = pred_output_bool_grid.detach().cpu().numpy()
                pred_output_float_numpy = pred_output_float_grid.detach().cpu().numpy()

        pred_output_float_numpy = np.clip(pred_output_float_numpy,0,1)
        vertices, triangles = cutils.dual_contouring_undc(np.ascontiguousarray(pred_output_bool_numpy, np.int32), np.ascontiguousarray(pred_output_float_numpy, np.float32))
        vertices = (vertices / resolution * 2 - 1)
        
        time_statics[1]+=time.time()-cur_time
        cur_time = time.time()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)

        # Adjust the normal
        if False:
            t_mesh = o3d.io.read_triangle_mesh(os.path.join(root, "mesh",file))
            # dense_poisson = t_mesh.sample_points_poisson_disk(100000,use_triangle_normal=True)
            dense_poisson = t_mesh.sample_points_uniformly(100000,use_triangle_normal=True)

            pcd = o3d.io.read_point_cloud(os.path.join(root, "poisson",file))
            face_center = vertices[triangles].mean(axis=1)

            pcd_tree = o3d.geometry.KDTreeFlann(dense_poisson)
            idx = []
            for point in face_center:
                [_, idx_local, _] = pcd_tree.search_knn_vector_3d(point, 1)
                idx.append(idx_local)
            idx = np.asarray(idx)
            # dist = cdist(face_center, np.asarray(dense_poisson.points))
            # idx = np.argmin(dist, axis=1)
            center_normals = np.asarray(dense_poisson.normals)[idx]

            # pc = o3d.geometry.PointCloud()
            # pc.points = o3d.utility.Vector3dVector(face_center)
            # pc.normals = o3d.utility.Vector3dVector(center_normals)
            # o3d.io.write_point_cloud(os.path.join(root,"ndc_mesh","{}_face_center.ply".format(prefix)), pc)

            for i in range(triangles.shape[0]):
                triangle = vertices[triangles[i]]
                triangle_normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[1])
                triangle_normal = triangle_normal / np.linalg.norm(triangle_normal)
                if np.dot(center_normals[i], triangle_normal) < 0:
                    triangles[i] = triangles[i][::-1]

        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        # mesh.triangle_normals = o3d.utility.Vector3dVector(center_normals)
        mesh.compute_triangle_normals()
        mesh.orient_triangles()
        o3d.io.write_triangle_mesh(os.path.join(root,"ndc_mesh","{}.ply".format(prefix)), mesh)
        time_statics[5]+=time.time()-cur_time



if __name__=="__main__":
    root = sys.argv[1]
    num_gpu = int(sys.argv[2])

    files = sorted(os.listdir(os.path.join(root, "poisson")))
    print("Start to produce NDC mesh of {} point clouds using {} GPUs".format(len(files), num_gpu))

    os.makedirs(os.path.join(root, "ndc_mesh"),exist_ok=True)
    mp.set_start_method("spawn")
    p = mp.Pool(num_gpu)
    predict_ = partial(predict, files=files, root=root)
    p.map(predict_, range(0,num_gpu))
