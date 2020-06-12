#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


"""
This module implements the basic convolutional blocks of Decagon.
Just as a quick reminder, the basic convolution formula here is:

y = A * (x * W)

where:

W is a weight matrix
A is an adjacency matrix
x is a matrix of latent representations of a particular type of neighbors.

As we have x here twice, a trick is obviously necessary for this to work.
A must be previously normalized with:

c_{r}^{ij} = 1/sqrt(|N_{r}^{i}| |N_{r}^{j}|)

or

c_{r}^{i} = 1/|N_{r}^{i}|

Let's work through this step by step to convince ourselves that the
formula is correct.

x = [
    [0, 1, 0, 1],
    [1, 1, 1, 0],
    [0, 0, 0, 1]
]

W = [
    [0, 1],
    [1, 0],
    [0.5, 0.5],
    [0.25, 0.75]
]

A = [
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
]

so the graph looks like this:

(0) -- (1) -- (2)

and therefore the representations in the next layer should be:

h_{0}^{k+1} = c_{r}^{0,1} * h_{1}^{k} * W + c_{r}^{0} * h_{0}^{k}
h_{1}^{k+1} = c_{r}^{0,1} * h_{0}^{k} * W + c_{r}^{2,1} * h_{2}^{k} +
    c_{r}^{1} * h_{1}^{k}
h_{2}^{k+1} = c_{r}^{2,1} * h_{1}^{k} * W + c_{r}^{2} * h_{2}^{k}

In actual Decagon code we can see that that latter part propagating directly
the old representation is gone. I will try to do the same for now.

So we have to only take care of:

h_{0}^{k+1} = c_{r}^{0,1} * h_{1}^{k} * W
h_{1}^{k+1} = c_{r}^{0,1} * h_{0}^{k} * W + c_{r}^{2,1} * h_{2}^{k}
h_{2}^{k+1} = c_{r}^{2,1} * h_{1}^{k} * W

If A is square the Decagon's EdgeMinibatchIterator preprocesses it as follows:

A = A + eye(len(A))
rowsum = A.sum(1)
deg_mat_inv_sqrt = diags(power(rowsum, -0.5))
A = dot(A, deg_mat_inv_sqrt)
A = A.transpose()
A = A.dot(deg_mat_inv_sqrt)

Let's see what gives in our case:

A = A + eye(len(A))

[
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 1]
]

rowsum = A.sum(1)

[2, 3, 2]

deg_mat_inv_sqrt = diags(power(rowsum, -0.5))

[
    [1./sqrt(2), 0,  0],
    [0, 1./sqrt(3),  0],
    [0,  0, 1./sqrt(2)]
]

A = dot(A, deg_mat_inv_sqrt)

[
    [ 1/sqrt(2), 1/sqrt(3),         0 ],
    [ 1/sqrt(2), 1/sqrt(3), 1/sqrt(2) ],
    [         0, 1/sqrt(3), 1/sqrt(2) ]
]

A = A.transpose()

[
    [ 1/sqrt(2), 1/sqrt(2),         0 ],
    [ 1/sqrt(3), 1/sqrt(3), 1/sqrt(3) ],
    [         0, 1/sqrt(2), 1/sqrt(2) ]
]

A = A.dot(deg_mat_inv_sqrt)

[
    [ 1/sqrt(2) * 1/sqrt(2),   1/sqrt(2) * 1/sqrt(3),                       0 ],
    [ 1/sqrt(3) * 1/sqrt(2),   1/sqrt(3) * 1/sqrt(3),   1/sqrt(3) * 1/sqrt(2) ],
    [                     0,   1/sqrt(2) * 1/sqrt(3),   1/sqrt(2) * 1/sqrt(2) ],
]

thus:

[
    [0.5       , 0.40824829, 0.        ],
    [0.40824829, 0.33333333, 0.40824829],
    [0.        , 0.40824829, 0.5       ]
]

This checks out with the 1/sqrt(|N_{r}^{i}| |N_{r}^{j}|) formula.

Then, we get back to the main calculation:

y = x * W
y = A * y

y = x * W

[
    [ 1.25, 0.75 ],
    [ 1.5 , 1.5  ],
    [ 0.25, 0.75 ]
]

y = A * y

[
    0.5 * [ 1.25, 0.75 ] + 0.40824829 * [ 1.5, 1.5 ],
    0.40824829 * [ 1.25, 0.75 ] + 0.33333333 * [ 1.5, 1.5 ] + 0.40824829 * [ 0.25, 0.75 ],
    0.40824829 * [ 1.5, 1.5 ] + 0.5 * [ 0.25, 0.75 ]
]

that is:

[
    [1.23737243, 0.98737244],
    [1.11237243, 1.11237243],
    [0.73737244, 0.98737244]
].

All checks out nicely, good.
"""

from .dense import *
from .sparse import *
from .universal import *
