import os
import sys
sys.path.append("..")

from datetime import datetime
from parse import parse
import pickle as pkl
from src.genHMM import GenHMM, save_model, load_model
from src.utils import pad_data, TheDataset, find_stat_pt
import torch
from torch.utils.data import DataLoader
import json

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
    train_class_inputfile = train_inputfile.replace(".pkl", "_{}.pkl".format(iclass_str))
    
    # Check if model has converged
    class_cvgd_file = os.path.join(os.path.dirname(out_mdl),"class{}.cvgd".format(iclass_str))

    # If the class has converged.
    if os.path.isfile(class_cvgd_file):
        # Read cvgd file 
        #with open(class_cvgd_file, "r") as f:
        #    content_str = f.read().split("\n")[0]
        #print(content_str)
        #stat_point = parse("{}converged to {}",content_str)[1]
        
        # Create a symbolic link to the converged model.
        #os.symlink(os.path.basename(stat_point), out_mdl)
        
        sys.exit(0)


    #  Load data
    xtrain = pkl.load(open(train_class_inputfile, "rb"))
    #xtrain = xtrain[:100]
    # Get the length of all the sequences
    l = [x.shape[0] for x in xtrain]

    #  Load or create model
    if epoch_str == '1':
        with open(param_file) as f_in:
            options = json.load(f_in)

        mdl = GenHMM(**options)

    else:
        # Load previous model
        mdl = load_model(out_mdl.replace("epoch" + epoch_str, "epoch" + str(int(epoch_str)-1)))

    mdl.iepoch = epoch_str
    mdl.iclass = iclass_str


    mdl.device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        mdl.device = device
    
    print("epoch:{}\tclass:{}\tPush model to {}...".format(epoch_str,iclass_str, mdl.device), file=sys.stdout)
    mdl.pushto(mdl.device)   
    
    
    # zero pad data for batch training
    max_len_ = max([x.shape[0] for x in xtrain])
    xtrain_padded = pad_data(xtrain, max_len_)
    traindata = DataLoader(dataset=TheDataset(xtrain_padded, lengths=l, device=mdl.device), batch_size=1024, shuffle=True)
    

    # niter counts the number of em steps before saving a model checkpoint
    niter = 2
    
    # em_skip determines the number of back-props before an EM step is performed
    mdl.em_skip = 5

    # TODO: pass lr as a param
    mdl.lr = 1e-3

    # add number of training data in model
    mdl.number_training_data = len(xtrain)

    for iiter in range(niter):
        mdl.fit(traindata)
    
    if int(epoch_str) == 2:
        mdl.converged = True


    # Write a class%.cvgd file to indicate that the class has converged
    if mdl.converged:
        with open(class_cvgd_file, "w") as f:
            print("{} : class {}, converged to {}".format(datetime.now(), iclass_str, out_mdl), file=f)
    
    # Push back to cpu for compatibility when GPU unavailable.
    mdl.pushto('cpu')
    save_model(mdl, fname=out_mdl)


    sys.exit(0)


