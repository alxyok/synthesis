# synthesis

***This repo is a compilation of my understanding of *Deep Learning* (DL) applied to numerical simulation. NOT MEANT AS A REFERENCE. More of an always-on memory.***

Simulations are a key component of contemporary Physics ; it is used to elaborate, design and validate innovative ideas for a range of products which design and engineering heavily depend on our understanding Physics—cars, aircrafts, pipelines, nuclear reactors, and more. It drives the product evolution from genesis to prototyping to actual sellable implementation. 

They usually rely on solving complex differential equations (ODEs and PDEs), which is notoriously difficult because of a few limitations in a pure computer-science perspective, notably: probably intractable, probably continuous, probably no analytical solution. The equations are well known and well understood, but the resolution has been a challenge in many cases for decades, and still an open problem. The are simplified with hypothesis 

The somewhat recent emergence of DL, and their phenomenal generalization properties, has allowed for cross-polinization in a variety of different fields: computer vision, natural language understanding, robotics, cybersecurity, biology, and now physics. 

## Discretization

The equations are continuous, and our use of finite available resources requires discretization over a specific domain of computation: the mesh is a network of inter-connected points. In our current best perception of reality, we can wrap the world in three spatial dimensions and a single temporal dimension. Discretization affects all of these dimension. 

The complexity of the equations is not equaly diffused around space and time, and thus requires a 

## Problem statement flavors

The objective can always be stated as follows: we need to compute a good estimator for the solution of a particular set of XDEs. Neural nets are black-boxed models that take inputs and produce outputs. Depending on the nature of the I/Os, the problem can be categorized like this:
* Point-to-point: fields seen as independent, unrelated points
* Map-to-map: fields seen as inter-correlated data points

## Neural formulation

They are currently trained with backpropagation. 

### Deep learning areas of interest

* MLPs. Need no introduction.
* CNNs + Unets, convnets that extract local correlations with small-sized convolutional kernels over fixed-grid items.
* PINNs or pysics-informed neural networks, that incorporate physics knowledge in the optimization problem for guided feature extraction. Captures the dynamics better.
* GNNs. GraphNets, a special flavor of convnets defined on graphs, a more suitable, more flexible data representation strategy. 
* CloudNets for points clouds problems that treat every element independently, but still compiles volume information in a meaningful manner for local exploitation.
* FNOs, fourier neural operators 
* Bayesian nets
* Data assimilation

### Strategies
* Bayesian

## A few stuff

### Recommended follows
2

### Current tech-stack
PyTorch Geometric largely recommended for PyTorch users. TF-GNN, maybe? for TensorFlow developers.

### Readings

This section compiles a few of the interesting papers on the subject in litterature. Shallow, self-made selection.

Read papers:
* [Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning](https://arxiv.org/abs/1803.04765)
* [Graph Networks as Learnable Physics Engines for Inference and Control](https://arxiv.org/abs/1806.01242)
* [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261)
* [Differentiable Physics-informed Graph Networks](https://arxiv.org/abs/1902.02950)
* [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405)

Still in the FIRO list (first-in random-out):
* [Geometric deep learning: going beyond Euclidean data](https://arxiv.org/abs/1611.08097)
* [Interaction Networks for Learning about Objects, Relations and Physics](https://arxiv.org/abs/1612.00222)
* [A simple neural network module for relational reasoning](https://arxiv.org/abs/1706.01427)
* [Graph Signal Processing: Overview, Challenges and Applications](https://arxiv.org/abs/1712.00468)
* [Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images](https://arxiv.org/abs/1804.01654)
* [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/abs/1811.05320)
* [Mesh R-CNN](https://arxiv.org/abs/1906.02739)
* 
