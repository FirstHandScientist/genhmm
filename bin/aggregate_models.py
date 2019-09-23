import sys
import pickle as pkl
import os
import glob
from parse import parse
from gm_hmm.src.genHMM import GenHMMclassifier, save_model
from gm_hmm.src.ref_hmm import GaussianHMMclassifier

if __name__ == "__main__":
    usage = "Aggregate models from several classes.\n" \
            "Usage: python bin/aggregate_models.py models/epoch1.mdl"

    if len(sys.argv) != 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage)
        sys.exit(1)

    out_mdl_file = sys.argv[1]

    # Find the class digit
    get_sort_key = lambda x: parse("{}class{:d}.mdlc", x)[1]
    # find the model used, 'gen' or 'gaus'
    model_type = parse("{}/models/{}/{}.mdl", out_mdl_file)[1]

    # Find all trained classes submodels
    in_mdlc_files = sorted(glob.glob(out_mdl_file.replace(".mdl", "_class*.mdlc")), key=get_sort_key)
    if model_type == 'gaus':
        mdl = GaussianHMMclassifier(mdlc_files=in_mdlc_files)
        assert(all([int(h.iclass) == int(i)+1 for i,h in enumerate(mdl.hmms)]))
        with open(out_mdl_file, "wb") as handle:
            pkl.dump(mdl, handle)
    
    elif model_type == 'gen':
        mdl = GenHMMclassifier(mdlc_files=in_mdlc_files)
        assert(all([int(h.iclass) == int(i)+1 for i,h in enumerate(mdl.hmms)]))
        save_model(mdl, out_mdl_file)
    sys.exit(0)
