# two_branch_networks
Pytorch implementation of the two branch network described in [these](http://slazebni.cs.illinois.edu/publications/cvpr16_structure.pdf) [papers](https://arxiv.org/pdf/1704.03470.pdf).

This code was tested on an Ubuntu 16.04 system using Pytorch version 0.1.12.  It is based on the [MNIST Pytorch tutorial](http://pytorch.org/).

## Usage

This code assumes you have cached the VGG (or ResNet) and PCA-reduced HGLMM text features in matlab .mat files (see `data_loader.py`).  Also, it currently assumes each image has 5 sentences, so will need to be modified for datasets with variable number of sentences like MSCOCO.

To use the repo, first clone it,

```sh
    git clone https://github.com/BryanPlummer/two_branch_networks.git
```

Place the features in a directory named `data`. See `data_loader.py` for expected filenames.  Then, you can train a new model using:

```sh
    python main.py --name {your experiment name}
```

This would give you a model trained on the VGG features, to use ResNet instead just use the `--resnet` flag and have the proper features cached.  In my experiments I got the folowing performance on the Flickr30K test set used in the referenced papers:

ResNet:

Test set - Total: 409.4 im2sent: 52.1 80.2 88.2 sent2im: 39.8 69.8 79.3

VGG:

test set - Total: 355.0 im2sent: 40.1 69.5 79.1 sent2im: 31.5 62.2 72.6
