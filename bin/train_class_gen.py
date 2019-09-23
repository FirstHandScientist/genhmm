import os
import sys
from parse import parse
import pickle as pkl
from gm_hmm.src.genHMM import GenHMM, save_model, load_model
from gm_hmm.src.utils import pad_data, TheDataset, get_freer_gpu, data_read_parse
import torch
from torch.utils.data import DataLoader
import json
import numpy as np
import time

if __name__ == "__main__":
    usage = "python bin/train_class.py data/train13.pkl models/epoch1_class1.mdlc param.json"
    if len(sys.argv) < 3 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage)
        sys.exit(1)

    # Parse
    train_inputfile = sys.argv[1]
    out_mdl = sys.argv[2]

    # Test for a third parameter
    try:
        param_file = sys.argv[3]
    except IndexError:
        param_file = "default.json"


    epoch_str, iclass_str = parse('epoch{}_class{}.mdlc',os.path.basename(out_mdl))
    train_class_inputfile = train_inputfile.replace(".pkl", "_class{}.pkl".format(iclass_str))


    #  Load data
    xtrain = data_read_parse(train_class_inputfile)
    if xtrain[0].shape[1] % 2 != 0:
        xtrain = [np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1) for x in xtrain]

    # xtrain = xtrain[:100]
    # Get the length of all the sequences
    l = [x.shape[0] for x in xtrain]

    # load the parameters
    with open(param_file) as f_in:
        options = json.load(f_in)

    # adoptive to set number of states
    options["Net"]["n_states"] = np.clip(int(np.floor(np.mean(l)/2)),
                                         options["Train"]["n_states_min"],
                                         options["Train"]["n_states_max"])

    options["Net"]["net_D"] = xtrain[0].shape[1]
    #  Load or create model
    if epoch_str == '1':
        mdl = GenHMM(**options["Net"])

    else:
        # Load previous model
        mdl = load_model(out_mdl.replace("epoch" + epoch_str, "epoch" + str(int(epoch_str)-1)))

    mdl.iepoch = epoch_str
    mdl.iclass = iclass_str


    mdl.device = 'cpu'
    if 0 and torch.cuda.is_available():
        if not options["Mul_gpu"]:
            # default case, only one gpu
            device = torch.device('cuda')
            mdl.device = device
            mdl.pushto(mdl.device)   

        else:
            for i in range(4):
                try:
                    time.sleep(np.random.randint(20))
                    device = torch.device('cuda:{}'.format(int(get_freer_gpu()) ))
                    print("Try to push to device: {}".format(device))
                    mdl.device = device
                    mdl.pushto(mdl.device)   
                    break
                except:
                    # if push error (maybe memory overflow, try again)
                    print("Push to device cuda:{} fail, try again ...")
                    continue
    print("epoch:{}\tclass:{}\tPush model to {}. Done.".format(epoch_str,iclass_str, mdl.device), file=sys.stdout)
    
    # zero pad data for batch training
    max_len_ = max([x.shape[0] for x in xtrain])
    xtrain_padded = pad_data(xtrain, max_len_)

    traindata = DataLoader(dataset=TheDataset(xtrain_padded, lengths=l, device=mdl.device), batch_size=options["Train"]["batch_size"], shuffle=True)

    # niter counts the number of em steps before saving a model checkpoint
    niter = options["Train"]["niter"]
    
    # add number of training data in model
    mdl.number_training_data = len(xtrain)
    
    # set model into train mode
    mdl.train()
    for iiter in range(niter):
        mdl.fit(traindata)

    # Push back to cpu for compatibility when GPU unavailable.
    mdl.pushto('cpu')
    save_model(mdl, fname=out_mdl)
    sys.exit(0)


