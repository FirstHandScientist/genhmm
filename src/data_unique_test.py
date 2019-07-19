import pickle as pkl
import numpy as np
import sys

def uniq_test(ref_file, sel_file):
    # selection some sample in sel_file, compare each of them to every sample in ref_file, print the count of same sample
    
    train = pkl.load(open(ref_file, "rb"))
    test = pkl.load(open(sel_file, "rb"))

    n_samples = 10
    idx_n_sample = np.random.randint(low=0, high=test.shape[0],size=n_samples)
    sample_in_test = [ test[i] for i in idx_n_sample]

    count = 0
    for ref_sample in train:
        for sample in sample_in_test:
            count += np.array_equal(sample, ref_sample)

    print("[{}] and [{}] uniqueness Assert with count: {}".format(ref_file, sel_file, count))
    return count


if __name__ == "__main__":
    usage = "This script is to test uniqueness of each sequence in train??.pkl and test??.pkl \n" \
            "There should not be any sample in test??.pkl that is the same as in train??.pkl \n" \
            "Example python src/data_unique_test.py 39"
    #1. assert train39.pkl and test.pkl
    feats = sys.argv[1]
    if feats=='39':
        train_files = ["data/train39_{}.pkl".format(i) for i in range(1, 62)]
        test_files = ["data/test39_{}.pkl".format(i) for i in range(1, 62)]
    elif feats=='13':
        train_files = ["data/train13_{}.pkl".format(i) for i in range(1, 62)]
        test_files = ["data/test13_{}.pkl".format(i) for i in range(1, 62)]
    else:
        print("Wrong feats input!!!")
        print(usage)
        sys.exit(1)
    
    count=0
    for ref_file in train_files:
        for s_file in test_files:
           count += uniq_test(ref_file, s_file)

    print("Total repeat sample in train/test 39 subsets:{}".format(count))

    
