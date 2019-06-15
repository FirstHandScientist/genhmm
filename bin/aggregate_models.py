import sys
sys.path.append("..")

import os
import glob
from src.genHMM import GenHMMclassifier, save_model

if __name__ == "__main__":
    usage = "Aggregate models from several classes.\n" \
            "Usage: python bin/aggregate_models.py models/epoch1.mdl"

    if len(sys.argv) != 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage)
        sys.exit(1)

    out_mdl_file = sys.argv[1]

    # Find the class digit
    get_sort_key = lambda x: os.path.basename(x).split(".mdlc")[0].split("_")[1].replace("class", "")

    # Find all trained classes submodels
    in_mdlc_files = sorted(glob.glob(out_mdl_file.replace(".mdl", "_class*.mdlc")), key=get_sort_key)

    mdl = GenHMMclassifier(mdlc_files=in_mdlc_files)

    save_model(mdl, out_mdl_file)
    sys.exit(0)