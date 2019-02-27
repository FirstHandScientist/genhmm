"""Train script.

Usage:
    train.py <hparams> <dataset> <dataset_root>
"""
import os
import torchvision.datasets as dset
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.trainer import Trainer
from glow.config import JsonConfig


if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset = args["<dataset>"]

    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))

    dataset_root = args["<dataset_root>"]
    assert os.path.exists(dataset_root), (
        "Failed to find root dir `{}` of dataset.".format(dataset_root))
    hparams = JsonConfig(hparams)

    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.Resize(hparams.Data.resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        dataset = dset.MNIST(root=dataset_root,
                         download=True,
                         transform=transform)
    elif dataset == "fashion-mnist":
        transform = transforms.Compose([
            transforms.Resize(hparams.Data.resize),
            transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        dataset = dset.FashionMNIST(root=dataset_root,
                         download=True,
                         transform=transform)
    # build graph and dataset
    built = build(hparams, True)
    trainer = Trainer(**built, dataset=dataset, hparams=hparams)
    trainer.train()
