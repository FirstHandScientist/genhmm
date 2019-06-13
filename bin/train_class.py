import sys


if __name__ == "__main__":
    usage = "python bin/train_class.py data/train13.pkl models/epoch0_class1.mdlc"
    if len(sys.argv) != 3:
        print(usage)
        sys.exit(1)

