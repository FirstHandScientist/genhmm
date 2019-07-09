import sys
sys.path.append("..")

import os
import glob
from src.genHMM import GenHMMclassifier, save_model
from src.utils import find_stat_pt
from parse import parse
from datetime import datetime

def find_submodels(out_mdl_file):
    folder = os.path.dirname(out_mdl_file)
    
    # Guess the number of classes, we know that all mdlc files are created at least for the first epoch.
    nclasses = len(glob.glob(os.path.join(folder, "epoch1_class*.mdlc")))

    # Find all trained classes submodels and all cvgd file
    all_submodels = glob.glob(out_mdl_file.replace(".mdl", "_class*.mdlc"))
    all_cvgd = glob.glob(os.path.join(folder, "class*.cvgd"))
    
    converged = False
    if len(all_cvgd) == nclasses:
            converged = True
    
    # list(set(...)) removes the duplicate in a list.
    # if we are at an apoch where a class converged, then both the mdlc file and cvgd file exist

    in_mdlc_files = list(set([find_stat_pt(x) if ".cvgd" in x else x for x in all_submodels + all_cvgd ]))
    
    return in_mdlc_files, converged


if __name__ == "__main__":
    usage = "Aggregate models from several classes.\n" \
            "Usage: python bin/aggregate_models.py [.mdl model file]\n"\
            "Example: python bin/aggregate_models.py models/epoch1.mdl"


    if len(sys.argv) != 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage)
        sys.exit(1)

    out_mdl_file = sys.argv[1]
    cvgd_file = os.path.join(os.path.dirname(out_mdl_file),"class_all.cvgd")
    
    if os.path.isfile(cvgd_file):
        print("(should not be here) cvgd file found",file=sys.stderr)
        # SHOULD NOT BE HERE, THE CASE IS MANAGED IN THE MAKEFILE
        sys.exit(0)

    
    in_mdlc_files, converged = find_submodels(out_mdl_file)
                
    mdl = GenHMMclassifier(mdlc_files=in_mdlc_files)

    save_model(mdl, out_mdl_file)

    # If all classes had converged at the previous epoch, then no need to re-compute
    if converged:
        with open(cvgd_file,"w") as f:
            print("{} : converged to {}".format(datetime.now(), out_mdl_file), file=f)

    sys.exit(0)
