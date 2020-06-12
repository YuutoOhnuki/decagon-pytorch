#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


#
# This module implements a single layer of the Decagon
# model. This is going to be already quite complex, as
# we will be using all the graph convolutional building
# blocks.
#
# h_{i}^(k+1) = ϕ(∑_r ∑_{j∈N{r}^{i}} c_{r}^{ij} * \
#   W_{r}^(k) h_{j}^{k} + c_{r}^{i} h_{i}^(k))
#
# N{r}^{i} - set of neighbors of node i under relation r
# W_{r}^(k) - relation-type specific weight matrix
# h_{i}^(k) - hidden state of node i in layer k
#   h_{i}^(k)∈R^{d(k)} where d(k) is the dimensionality
#   of the representation in k-th layer
# ϕ - activation function
# c_{r}^{ij} - normalization constants
#   c_{r}^{ij} = 1/sqrt(|N_{r}^{i}| |N_{r}^{j}|)
# c_{r}^{i} - normalization constants
#   c_{r}^{i} = 1/|N_{r}^{i}|
#


from .layer import *
from .input import *
from .convolve import *
from .decode import *
