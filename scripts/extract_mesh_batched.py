import os,time
import subprocess
from tqdm import tqdm
import sys

#root=r"/mnt/d/GSP_test/ours/0102_v15++_aug_voronoi/"
root=r"/mnt/d/GSP_test/ours/0102_v15++_aug_noise_full_voronoi/"
exe="./build/src/extract_mesh/extract_mesh"
output_dir="/mnt/d/GSP_test/ours/0121_noise_mesh"
test_id_file = "/mnt/d/GSP_test/test_ids2.txt"
#test_id_file = "G:/Dataset/GSP/List/test_ids_small.txt"


if __name__ == "__main__":
    root = sys.argv[1]
    output_dir = sys.argv[2]
    test_id_file = sys.argv[3]

    tasks = [item[:8] for item in os.listdir(root) if item.endswith("_feat.npy")]

    status = []
    outs = []
    files = sorted([file.strip() for file in open(test_id_file).readlines()])
    bar = tqdm(total=len(files))
    for prefix in files:
        while len(status) >= 8:
            for idx, item in enumerate(status):
                if item.poll() is not None:
                    status.remove(item)
                    outs[idx].close()
                    outs.remove(outs[idx])
                    bar.update(1)
            time.sleep(1)
        if os.path.exists(
            os.path.join(output_dir, prefix, "eval/vertices.ply")) and os.path.exists(
                os.path.join(output_dir, prefix, "eval/curves.ply")) and os.path.exists(
                    os.path.join(output_dir, prefix, "eval/surfaces.ply")):
            continue
        os.makedirs(os.path.join(output_dir, prefix), exist_ok=True)
        out = open(os.path.join(output_dir, "{}.txt".format(prefix)), "w")
        process = subprocess.Popen([
            exe, 
            root,
            "--prefix", prefix,
            "--output_dir", output_dir,
            "--only_evaluate",
            "--flag_mode", "1",
            "--alpha_value", "0.01",
            "--dilate_radius", "1",
            "--fitting_epsilon", "0.001",
            "--restricted",
            "--common_points_threshold", "0.02",
            "--shape_epsilon", "0.02",
            "--output_inlier_epsilon", "0.01",
            "--output_alpha_value", "0.0004",
            "--check_curve_before_adding",
            "--vertex_threshold",
            "0.02",
        ], stdout=out, stderr=out)
        status.append(process)
        outs.append(out)
        # break
    
    while len(status) > 0:
        for idx, item in enumerate(status):
            if item.poll() is not None:
                status.remove(item)
                outs[idx].close()
                outs.remove(outs[idx])
                bar.update(1)
        time.sleep(1)

    for item in status:
        item.wait()
        bar.update(1)
    pass
