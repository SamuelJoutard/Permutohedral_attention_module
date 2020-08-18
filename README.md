# Permutohedral_attention_module

This repository contains an implementation of the Permutohedral Attention Module (http://arxiv.org/abs/1907.00641) in Pytorch. We first used the Niftynet CRF as RNN implementation as model (https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/crf.html#CRFAsRNNLayer) for the code.

This repository contains two versions of a HashTable in Pytorch, one in plain Pytorch (used in http://arxiv.org/abs/1907.00641) and one with a custom CUDA kernel that needs to be compiled and binded to Pytorch (this is the latest version that should be used now). In addition to those features, the repository also contains an implementation of the CRF-as-RNN widely used for segmentation regularization especially in medical imaging.

The repository also contains all the files to reproduce the experimental results presented in the "Permutohedral Attention Module for Efficient Non-Local Neural Networks" paper. 
In case of any issue to reproduce the results, miss-understanding or mistake you might find, please do not hesitate to contact us at: samuel.joutard@kcl.ac.uk.

