import os
import sys
import json
from parse import parse
import pickle as pkl
from gm_hmm.src.genHMM import load_model
from gm_hmm.src.utils import append_class, accuracy_fun, accuracy_fun_torch, divide, parse_, to_device


if __name__ == "__main__":
    usage = "Usage:\nbin/compute_accuracy_class.py exp/gaus/13feats/models/epoch2_class1.mdlc exp/gaus/13feats/data/train.13.pkl exp/gaus/13feats/data/test.13.pkl"
    if len(sys.argv) != 4 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage, file=sys.stderr)
        sys.exit(1)

    models_dir, epoch, iclass = parse("{}/epoch{:d}_class{:d}.mdlc", sys.argv[1])

    # read the model type, 'gen' or 'gaus'
    model_type_ = parse("{}/models/{}", models_dir)[1]

    if "gaus" in model_type_:
        model_type = "gaus"
    elif "gen" in model_type_:
        model_type = "gen"
    else:
        print("No known model type found in {}".format(models_dir),file=sys.stderr)
        sys.exit(1)

    
    # Todo: maybe give choice here
    # load the parameters
    with open("default.json") as f_in:
        options = json.load(f_in)

    training_data_file = sys.argv[2]
    testing_data_file = sys.argv[3]

    mdl_file = os.path.join(models_dir, "epoch{}.mdl".format(epoch))
    # Builds an array of string containing the train and test data sets for each class
    # size: nclass x 2 (train, test)
    data_files = [append_class(training_data_file, iclass), append_class(testing_data_file, iclass)]
    # Load Model
    device = 'cpu'
    if model_type == 'gaus':
        with open(mdl_file, "rb") as handle:
            mdl = pkl.load(handle)
        mdl.device = 'cpu'
        f = lambda x: accuracy_fun(x, mdl=mdl)
    elif model_type == 'gen':
        mdl = load_model(mdl_file)
        mdl = to_device(mdl, use_gpu=options["use_gpu"], Mul_gpu=options["Mul_gpu"])

        # set model into eval mode
        mdl.eval()
        
        f = lambda x: accuracy_fun_torch(x, mdl=mdl, batch_size_=options["Train"]["eval_batch_size"])


    # print("[Acc:] epoch:{}\tclass:{}\tPush model to {}. Done.".format(epoch,iclass, mdl.device), file=sys.stdout)
    # f = lambda x: divide(parse_(accuracy_fun(x, mdl=mdl)))
    
    results_ = list(map(f, data_files))
    results = [r[0] for r in results_]
    userdata = [r[1] for r in results_]

    print("epoch: {} class: {} accc train: {} test: {}".format(epoch, iclass, results[0], results[1]), file=sys.stdout)
    print("epoch: {} class: {} llh train: {} test: {}".format(epoch, iclass, userdata[0], userdata[1]), file=sys.stdout)

    sys.exit(0)
