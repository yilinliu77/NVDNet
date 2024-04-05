import os,time,sys
import subprocess
from tqdm import tqdm



if __name__ == "__main__":
    root_dir=sys.argv[1]
    print("Check under ", root_dir)

    tasks = sorted([item[:8] for item in os.listdir(root_dir) if not item.endswith(".txt")])

    status = []
    outs = []

    for prefix in tasks:
        files = os.listdir(os.path.join(root_dir, prefix, "eval"))
        if len(files) != 4:
            print(prefix)
        for file in files:
            if file == "surfaces.ply":
                size = os.path.getsize(os.path.join(root_dir, prefix, "eval", file))
                if size < 500:
                    print(prefix ,size)
        # break
    
    pass
