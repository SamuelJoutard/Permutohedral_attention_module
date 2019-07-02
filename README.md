# Permutohedral_attention_module

This repository contains an implementation of the Permutohedral Attention Module (http://arxiv.org/abs/1907.00641) in Pytorch. 
This implementation uses in particular an implementation of a HashTable on GPU (at least something mimicking such behavior). 
This part could be highly improved by implementing the corresponding cuda kernel (hence parallelizing some tests).
The repository also contains all the files to reproduce the experimental results presented in the "Permutohedral Attention Module for Efficient Non-Local Neural Networks" paper. 
In case of any issue to reproduce the results, miss-understanding or mistake you might find, please do not hesitate to contact us at: samuel.joutard@kcl.ac.uk.
