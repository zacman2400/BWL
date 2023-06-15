# BWL
[Block wise learning](https://iopscience.iop.org/article/10.1088/1361-6560/acc003/meta) is a memory efficient method that allows for model based reconstruction for high dimensional non-Cartesian data. Any model that has trouble fitting a single unroll on the GPU can be addressed by block-wise learning. This includes MODL as well as recently proposed deep equilibrium models. The scripts included here make several notable changes from the original paper including replacing gradient descent with conjugate gradient steps to improve convergence and using self-supervised learning in place of training with a supervised proxy.



