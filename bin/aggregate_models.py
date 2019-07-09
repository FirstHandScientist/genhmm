import sys
sys.path.append("..")

import os
import glob
from src.genHMM import GenHMMclassifier, save_model
from src.utils import find_stat_pt
from parse import parse

def find_submodels(out_mdl_file):
    folder = os.path.dirname(out_mdl_file)
    
    # Find the class digit
    get_sort_key = lambda x: os.path.basename(x).split(".mdlc")[0].split("_")[1].replace("class", "")

    # Find all trained classes submodels and all cvgd file
    all_submodels = glob.glob(out_mdl_file.replace(".mdl", "_class*.mdlc")) + glob.glob(os.path.join(folder, "class*.cvgd"))
    
    # replace the cvgd files with stat point
    print(all_submodels)

    all_submodels = [find_stat_pt(x) if ".cvgd" in x else x for x in all_submodels ]
    
    # in_mdlc_files = sorted(all_submodels, key=get_sort_key)
    
    print(all_submodels)
    sys.exit(1)
    return in_mdlc_files

if __name__ == "__main__":
    usage = "Aggregate models from several classes.\n" \
            "Usage: python bin/aggregate_models.py models/epoch1.mdl"

    if len(sys.argv) != 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage)
        sys.exit(1)

    out_mdl_file = sys.argv[1]
    
    in_mdlc_files = find_submodels(out_mdl_file)

    sys.exit(1)
    mdl = GenHMMclassifier(mdlc_files=in_mdlc_files)

    save_model(mdl, out_mdl_file)
    sys.exit(0)
