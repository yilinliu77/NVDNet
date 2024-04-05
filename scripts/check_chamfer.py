import os

# root = r"/mnt/d/GSP_test/test_data_whole2/1219_v15+_parsenet_unet_base16_focal75_wonormal_channel4_float32_eval"
root = r"/mnt/d/GSP_test/test_data_whole2/1220_v16+_quadrics_unet_base16_focal75_wonormal_channel4_float32_eval"

files = sorted([file for file in os.listdir(root) if file.endswith("_cd.txt")])

for file in files:
    with open(os.path.join(root, file), "r") as f:
        lines = f.readline().strip().split(" ")
        print("{}: {}".format(file[:8], lines[0]))