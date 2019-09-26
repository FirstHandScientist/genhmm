import sys
import glob
from parse import parse
from gm_hmm.src.genHMM import GenHMMclassifier
from gm_hmm.src.ref_hmm import GaussianHMMclassifier
from gm_hmm.src.utils import save_model
import json

if __name__ == "__main__":
    usage = "Aggregate models from several classes.\n" \
            "Usage: python bin/aggregate_models.py models/epoch1.mdl default.json"

    if len(sys.argv) < 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage)
        sys.exit(1)

    out_mdl_file = sys.argv[1]
    try:
        param_file = sys.argv[2]
    except:
        param_file = "default.json"

    with open(param_file) as f_in:
        options = json.load(f_in)

    # Find the class digit
    get_sort_key = lambda x: parse("{}class{:d}.mdlc", x)[1]

    # Find the model used, 'gen' or 'gaus'
    model_type_ = parse("{}/models/{}/{}.mdl", out_mdl_file)[1]

    if "gaus" in model_type_:
        model_type = "gaus"
    elif "gen" in model_type_:
        model_type = "gen"
    else:
        print("No known model type found in {}".format(out_mdl_file),file=sys.stderr)
        sys.exit(1)

    # Find all trained classes submodels
    in_mdlc_files = sorted(glob.glob(out_mdl_file.replace(".mdl", "_class*.mdlc")), key=get_sort_key)
    if model_type == 'gaus':
        mdl = GaussianHMMclassifier(mdlc_files=in_mdlc_files)
        assert(all([int(h.iclass) == int(i)+1 for i, h in enumerate(mdl.hmms)]))
    
    elif model_type == 'gen':
        mdl = GenHMMclassifier(mdlc_files=in_mdlc_files)
        assert(all([int(h.iclass) == int(i)+1 for i, h in enumerate(mdl.hmms)]))
        if options["Train"]["fine_tune"]:
            abs_path = "/home/antoine/Documents/projects/deep_news/proj/pre_infectious_detection/exp/split1/data/"
            mdl = mdl.fine_tune(use_gpu=options["use_gpu"], Mul_gpu=options["Mul_gpu"])
            mdl.save_members()

    else:
        print("(should have been caught earlier) Unknown model type: {}".format(model_type), file=sys.stderr)
        sys.exit(1)

    save_model(mdl, out_mdl_file)
    sys.exit(0)
