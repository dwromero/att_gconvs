# Based on implementation from Bekkers (2020) - B-Spline CNNs on Lie Groups.

# Class implementation of the Lie group SE(2)
import torch
import numpy as np


# Rules for setting up a group class:
# A group element is always stored as a 1D vector, even if the elements consist
# only of a scalar (in which case the element is a list of length 1). Here we
# also assume that you can parameterize your group with a set of n parameters,
# with n the dimension of the group. The group elements are thus always lists of
# length n.
#
# This file requires the definition of the base/normal sub-group R^n and the
# sub-group H. Together they will define G = R^n \rtimes H.
#
# In order to derive G (it's product and inverse) we need for the group H to be
# known the group product, inverse and left action on R^n.
#
# Finally we need a way to sample the group. Therefore also a function "grid" is
# defined which samples the group as uniform as possible given a specified
# number of elements N. Not all groups allow a uniform sampling given an
# aribitrary N, in which case the sampling can be made approximately uniform by
# maximizing the distance between all sampled elements in H (e.g. via a
# repulsion model).


## The normal sub-group R^n:
# This is just the vector space R^n with the group product and inverse defined
# via the + and -.
class Rn:
    # Label for the group
    name = 'R^2'
    # Dimension of the base manifold N=R^n
    n = 2
    # The identity element
    e = torch.tensor([0., 0.], dtype=torch.float32)



## The sub-group H:
class H:
    # Label for the group
    name = 'SO(2)'
    # Dimension of the sub-group H
    n = 1  # Each element consists of 1 parameter
    # The identify element
    e = torch.tensor([0.], dtype=torch.float32)
    # Haar measure
    haar_meas = None

    ## Essential for constructing the group G = R^n \rtimes H
    # Define how H acts transitively on R^n
    ## TODO: So far just for multiples of 90 degrees. No interpolation required
    def left_representation_on_Rn(h, xx):
        if h == H.e:
            return xx
        else:
            xx_new = torch.rot90(xx, k=int(torch.round((1./(np.pi/2.)*h)).item()), dims=[-2, -1])
            return xx_new

    def left_representation_on_G(h, fx):
        h_inv_weight = H.left_representation_on_Rn(h, fx)
        h_inv_weight = torch.roll(h_inv_weight, shifts=int(torch.round((1. / (np.pi / 2.) * h)).item()), dims=2)
        return h_inv_weight

    ## Essential in the group convolutions
    # Define the determinant (of the matrix representation) of the group element
    def absdet(h):
        return 1.

    ## Grid class
    class grid_global:  # For a global grid
        # Should a least contain:
        #	N     - specifies the number of grid points
        #	scale - specifies the (approximate) distance between points, this will be used to scale the B-splines
        # 	grid  - the actual grid
        #	args  - such that we always know how the grid was constructed
        # Construct the grid
        def __init__(self, N):
            # This rembembers the arguments used to construct the grid (this is to make it a bit more future proof, you may want to define a grid using specific parameters and later in the code construct a similar grid with the same parameters, but with N changed for example)
            self.args = locals().copy()
            self.args.pop('self')
            # Store N
            self.N = N
            # Define the scale (the spacing between points)
            self.scale = [2 * np.pi / N]
            # Generate the grid
            if self.N == 0:
                h_list = torch.tensor([], dtype=torch.float32)
            else:
                h_list = torch.from_numpy(np.array([np.linspace(0, 2*np.pi - 2*np.pi/N,N)], dtype=np.float32).transpose())
            self.grid = h_list
            # -------------------
            # Update haar measure
            H.haar_meas = (2 * np.pi / self.N)


## The derived group G = R^n \rtimes H.
# The above translation group and the defined group H together define the group G
# The following is automatically constructed and should not be changed unless
# you may have some speed improvements, or you may want to add some functions such
# as the logarithmic and exponential map.
# A group element in G should always be a vector of length Rn.n + H.n
class G:
    # Label for the group G
    name = 'SE(2)'
    # Dimension of the group G
    n = Rn.n + H.n
    # The identity element
    e = torch.cat([Rn.e, H.e], dim=-1)

    # Function that returns the classes for R^n and H
    @staticmethod
    def Rn_H():
        return Rn, H
