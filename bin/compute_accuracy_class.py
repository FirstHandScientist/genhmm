import os
import sys
import json
import torch
from parse import parse
import pickle as pkl
from gm_hmm.src.genHMM import load_model
from gm_hmm.src.utils import append_class, accuracy_fun, accuracy_fun_torch, divide, parse_, get_freer_gpu
from functools import partial
import time
import numpy as np

if __name__ == "__main__":
    usage = "bin/compute_accuracy_class.py exp/gaus/13feats/models/epoch2_class1.mdlc exp/gaus/13feats/data/train.13.pkl exp/gaus/13feats/data/test.13.pkl"
    if len(sys.argv) != 4 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage, file=sys.stderr)
        sys.exit(1)

    models_dir, epoch, iclass = parse("{}/epoch{:d}_class{:d}.mdlc", sys.argv[1])
    # read the model type, 'gen' or 'gaus'
    model_type = parse("{}/exp/{}/{:d}feats/{}", os.getcwd())[1]
    
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
        if torch.cuda.is_available():
            if not options["Mul_gpu"]:
                # default case, only one gpu
                device = torch.device('cuda')
                mdl.device = device
                mdl.pushto(mdl.device)   
            else:
                for i in range(4):
                    try:
                        time.sleep(np.random.randint(10))
                        device = torch.device('cuda:{}'.format(int(get_freer_gpu()) ))
                        # print("Try to push to device: {}".format(device))
                        mdl.device = device
                        mdl.pushto(mdl.device)   
                        break
                    except:
                        # if push error (maybe memory overflow, try again)
                        # print("Push to device cuda:{} fail, try again ...")
                        continue
        # set model into eval mode
        mdl.eval()
        f = lambda x: accuracy_fun_torch(x, mdl=mdl)


    # print("[Acc:] epoch:{}\tclass:{}\tPush model to {}. Done.".format(epoch,iclass, mdl.device), file=sys.stdout)
    
   
    #f = lambda x: divide(parse_(accuracy_fun(x, mdl=mdl)))
    
    results = list(map(f, data_files))

    print("epoch: {} class: {} accc train: {} test: {}".format(epoch, iclass, results[0], results[1]), file=sys.stdout)
    sys.exit(0)
