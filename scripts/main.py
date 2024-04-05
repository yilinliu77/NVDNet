import os,time
import subprocess
from tqdm import tqdm

root=r"/mnt/d/GSP/GSP_test/test_data_whole/pred_voronoi_quadrics"
exe="/root/repo/C/build/src/GSP_Field/extract_mesh/extract_mesh"
output_dir="/mnt/d/GSP/GSP_test/test_data_whole/extracted_mesh_ndc_quadrics"


if __name__ == "__main__":
    tasks = [item[:8] for item in os.listdir(root) if item.endswith("_feat.npy")]

    status = []
    outs = []

    for prefix in tqdm(tasks):
        while len(status) >= 8:
            for idx, item in enumerate(status):
                if item.poll() is not None:
                    status.remove(item)
                    outs[idx].close()
                    outs.remove(outs[idx])
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
            "--flag_mode", "1",
            "--alpha_value", "0.01",
            "--dilate_radius", "0",
            "--fitting_epsilon", "0.001",
            "--output_dir", output_dir,
            "--only_evaluate"
        ], stdout=out, stderr=out)
        status.append(process)
        outs.append(out)
        # break
    
    for item in status:
        item.wait()
    pass
