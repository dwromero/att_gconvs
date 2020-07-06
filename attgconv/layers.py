# Based on implementation from Bekkers (2020) - B-Spline CNNs on Lie Groups.

import torch
import numpy as np
import math

# Start of (Parent Class)
class layers(torch.nn.Module):
    def __init__(self, group):
        super(layers, self).__init__()
        self.group = group
        self.G = group.G
        self.Rn = group.Rn
        self.H = group.H

    # TODO include dilation everywhere
    # Creates an spatial_layer object
    def ConvRnRn(self,
                 # Required arguments
                 N_in,              # Number of input channels
                 N_out,             # Number of output channels
                 kernel_size,       # Kernel size (integer)
                 # Optional generic arguments
                 stride=1,          # Spatial stride in the convolution
                 padding=1,         # Padding type
                 dilation=1,        # Dilation
                 conv_groups = 1,
                 wscale=1.0):       # White scaling
        return ConvRnRnLayer(self.group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups, wscale)

    # Creates a lifting_layer object
    def ConvRnG(
            self,
            # Required arguments
            N_in,                   # Number of input channels
            N_out,                  # Number of output channels
            kernel_size,            # Kernel size (integer)
            h_grid,                 # The grid of H on which to compute the output (see H.grid for more details)
            # Optional generic arguments
            stride=1,               # Spatial stride in the convolution
            padding=1,              # Padding type
            dilation=1,             # Dilation
            conv_groups = 1,        # Name of generated tensorflow variables
            wscale = 1.0):          # White scaling
        return ConvRnGLayer(self.group, N_in, N_out, kernel_size, h_grid, stride, padding, dilation, conv_groups, wscale)

    # Creates a group convolution layer object
    def ConvGG(
            self,
            # Required arguments
            N_in,                   # Number of input channels
            N_out,                  # Number of output channels
            kernel_size,            # Kernel size (integer)
            h_grid,                 # The grid of H on which to compute the output (see H.grid for more details)
            # Optional grid re-sampling related arguments
            input_h_grid=None,      # In case the input grid is not the same as the intended output grid (see h_grid parameter)
            # Optional generic arguments
            stride=1,               # Spatial stride in the convolution
            padding=1,              # Padding type
            dilation=1,             # Dilation
            conv_groups=1,          # Name of generated tensorflow variables
            wscale = 1.0):          # White scaling
        return ConvGGLayer(self.group, N_in, N_out, kernel_size, h_grid, input_h_grid, stride, padding, dilation, conv_groups, wscale)

    # Creates an attentive lifting_layer object
    def AttConvRnG( # TODO add dilation to these layers as well.
            self,
            # Required arguments
            N_in,                   # Number of input channels
            N_out,                  # Number of output channels
            kernel_size,            # Kernel size (integer)
            h_grid,                 # The grid of H on which to compute the output (see H.grid for more details)
            channel_attention=None, # The form of attention utilized channel-wise
            spatial_attention=None, # The form of attention utilized spatial-wise
            # Optional generic arguments
            stride=1,               # Spatial stride in the convolution
            padding=1,              # Padding type
            dilation=1,             # Dilation
            wscale = 1.0):          # White scaling
        return AttConvRnGLayer(self.group, N_in, N_out, kernel_size, h_grid, channel_attention, spatial_attention, stride, padding, dilation, wscale)

    # Creates an attentive group convolution layer object
    def AttConvGG(
            self,
            # Required arguments
            N_in,                   # Number of input channels
            N_out,                  # Number of output channels
            kernel_size,            # Kernel size (integer)
            h_grid,                 # The grid of H on which to compute the output (see H.grid for more details)
            channel_attention=None, # The form of attention utilized channel-wise
            spatial_attention=None, # The form of attention utilized spatial-wise
            # Optional grid re-sampling related arguments
            input_h_grid=None,      # In case the input grid is not the same as the intended output grid (see h_grid parameter)
            # Optional generic arguments
            stride=1,               # Spatial stride in the convolution
            padding=1,              # Padding type
            dilation=1,             # Dilation
            wscale = 1.0):
        return AttConvGGLayer(self.group, N_in, N_out, kernel_size, h_grid, channel_attention, spatial_attention, input_h_grid, stride, padding, dilation, wscale)

    # Creates a feature map attentive lifting_layer object
    def fAttConvRnG(
            self,
            # Required arguments
            N_in,                   # Number of input channels
            N_out,                  # Number of output channels
            kernel_size,            # Kernel size (integer)
            h_grid,                 # The grid of H on which to compute the output (see H.grid for more details)
            channel_attention=None, # The form of attention utilized channel-wise
            spatial_attention=None, # The form of attention utilized spatial-wise
            # Optional generic arguments
            stride=1,               # Spatial stride in the convolution
            padding=1,              # Padding type
            wscale = 1.0):          # White scaling
        return fAttConvRnGLayer(self.group, N_in, N_out, kernel_size, h_grid, channel_attention, spatial_attention, stride, padding, wscale)

    # Creates a feature map attentive group convolution layer object
    def fAttConvGG(
            self,
            # Required arguments
            N_in,                   # Number of input channels
            N_out,                  # Number of output channels
            kernel_size,            # Kernel size (integer)
            h_grid,                 # The grid of H on which to compute the output (see H.grid for more details)
            channel_attention=None, # The form of attention utilized channel-wise
            spatial_attention=None, # The form of attention utilized spatial-wise
            # Optional grid re-sampling related arguments
            input_h_grid=None,      # In case the input grid is not the same as the intended output grid (see h_grid parameter)
            # Optional generic arguments
            stride=1,               # Spatial stride in the convolution
            padding=1,              # Padding type
            wscale = 1.0):
        return fAttConvGGLayer(self.group, N_in, N_out, kernel_size, h_grid, channel_attention, spatial_attention, input_h_grid, stride, padding, wscale)

    def max_pooling_Rn(self, input, kernel_size, stride, padding = 1):
        input_size = input.size()
        out = input.view(input_size[0], input_size[1] * input_size[2], input_size[3], input_size[4])
        out = torch.max_pool2d(out, kernel_size=kernel_size, stride=stride, padding=padding)
        out = out.view(input_size[0], input_size[1], input_size[2], out.size()[2], out.size()[3])
        return out

    def average_pooling_Rn(self, input, kernel_size, stride, padding = 1):
        input_size = input.size()
        out = input.view(input_size[0], input_size[1] * input_size[2], input_size[3], input_size[4])
        out = torch.nn.functional.avg_pool2d(out, kernel_size=kernel_size, stride=stride, padding=padding)
        out = out.view(input_size[0], input_size[1], input_size[2], out.size()[2], out.size()[3])
        return out


##########################################################################
############################## ConvRnRnLayer #############################
##########################################################################
class ConvRnRnLayer(torch.nn.Module):
    def __init__(self,
                 group,
                 N_in,
                 N_out,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 conv_groups,
                 wscale):
        super(ConvRnRnLayer, self).__init__()
        ## Assert and set inputs
        self.kernel_type = 'Rn'
        self._assert_and_set_inputs(group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups)
        ## Construct the trainable weights and initialize them
        self.weight = torch.nn.Parameter(torch.Tensor(self.N_out, self.N_in, kernel_size, kernel_size))
        self._reset_parameters(wscale=wscale)

########################### Assert and set inputs ########################

    def _assert_and_set_inputs(self, group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups):
        self._assert_and_set_inputs_RnRn(group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups)

    def _assert_and_set_inputs_RnRn(self, group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups):
        ## Check (and parse) all the inputs
        # Include the dictionary of the used parent class
        self.group = group
        self.G = group.G
        self.H = group.H
        self.Rn = group.Rn
        # Mandatory inputs
        self.N_in = self._assert_N_in(N_in)
        self.N_out = self._assert_N_out(N_out)
        self.kernel_size = self._assert_kernel_size(kernel_size)
        # Optional arguments
        self.conv_groups = self._assert_conv_groups(conv_groups)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def _assert_N_in(self, N_in):
        assert isinstance(N_in, int), "The specified argument \"N_in\" should be an integer."
        return N_in

    def _assert_N_out(self, N_out):
        assert isinstance(N_out, int), "The specified argument \"N_out\" should be an integer."
        return N_out

    def _assert_kernel_size(self, kernel_size):
        assert isinstance(kernel_size, int), "The specified argument \"kernel_size\" should be an integer."
        return kernel_size

    def _assert_conv_groups(self, conv_groups):
        assert isinstance(conv_groups, int), "The specified argument \"conv_groups\" should be an integer."
        return conv_groups

############################ Compute the output ##########################

    ## Public functions
    def kernel(self, h=None):
        # The transformation to apply
        if h is None:
            h = self.H.e
        # Sample the kernel on the (transformed) grid
        return (1 / self.H.absdet(h)) * self.H.left_representation_on_Rn(h, self.weight)

    def forward(self, input):
        return  self.conv_Rn_Rn(input)

    def conv_Rn_Rn(self, input):
        output = torch.conv2d(input=input,
                              weight=self.kernel(self.H.e),
                              bias= None,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation,
                              groups=self.conv_groups)
        return output

    def _reset_parameters(self, wscale):
        n = self.N_in
        k = self.kernel_size ** 2
        n *= k
        stdv = wscale * (1. / math.sqrt(n))
        self.stdv = stdv
        self.weight.data.uniform_(-stdv, stdv)
        #if self.bias is not None: # TODO bias
        #    self.bias.data.uniform_(-stdv, stdv)


##########################################################################
############################## ConvRnGLayer ##############################
##########################################################################
# Start of lifting_layer class
class ConvRnGLayer(ConvRnRnLayer, torch.nn.Module):
    def __init__(self,
                 group,
                 N_in,
                 N_out,
                 kernel_size,
                 h_grid,
                 stride,
                 padding,
                 dilation,
                 conv_groups,
                 wscale):
        torch.nn.Module.__init__(self)
        ## Assert and set inputs
        self.kernel_type = 'Rn'
        self._assert_and_set_inputs(group, N_in, N_out, kernel_size, h_grid, stride, padding, dilation, conv_groups)
        ## Construct the trainable weights and initialize them
        self.weight = torch.nn.Parameter(torch.Tensor(self.N_out, self.N_in, kernel_size, kernel_size))
        self._reset_parameters(wscale=wscale)

########################### Assert and set inputs ########################

    # Method overriding:
    def _assert_and_set_inputs(self, group, N_in, N_out, kernel_size, h_grid, stride, padding, dilation, conv_groups):
        # Default Rn assertions
        self._assert_and_set_inputs_RnRn(group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups)
        # Specific initialization/assertion
        self.h_grid = self._assert_h_grid(h_grid)
        self.N_h = int(self.h_grid.grid.shape[0])

    def _assert_h_grid(self, h_grid ):
        assert (len(h_grid.grid.shape) == 2), "The \"h_grid\" option value should be a grid object with h_grid.grid a tensor of dimension 2 (a list of group elements)."
        assert (h_grid.grid.shape[-1] == self.H.n), "The group element specification in \"h_grid\" is not correct. For the current group \"{}\" each group element should be a vector of length {}.".format(self.H.name,self.H.n)
        return h_grid

############################ Compute the output ##########################

    # Method overriding:
    def forward(self, input):
        return self.conv_Rn_G(input)

    def conv_Rn_G(self, input):
        # Generate the full stack of convolution kernels (all transformed copies)
        kernel_stack = torch.cat([self.kernel(self.h_grid.grid[i]) for i in range(self.N_h)], dim=0) # [N_out x N_h, N_in, X, Y]
        # And apply them all at once
        output = torch.conv2d(
            input=input,
            weight=kernel_stack,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.conv_groups)
        # Reshape the last channel to create a vector valued RnxH feature map
        output = torch.stack(torch.split(output, self.N_out, 1), 2)

        #kernel_stack = torch.stack([self.kernel(self.h_grid.grid[i]) for i in range(self.N_h)], dim=1)
        # ks = kernel_stack.shape
        # kernel_stack = torch.reshape(kernel_stack, [ks[0] * ks[1], ks[2], ks[-2], ks[-1]])
        # output_2 = torch.conv2d(
        #     input=input,
        #     weight=kernel_stack,
        #     bias=None,
        #     stride=self.stride,
        #     padding=self.padding,
        #     dilation=self.dilation,
        #     groups=self.conv_groups)
        # output_2=output_2.reshape(output_2.shape[0], self.N_out, self.N_h, output_2.shape[-2], output_2.shape[-1])
        # Return the output
        return output


##########################################################################
############################### ConvGGLayer ##############################
##########################################################################
# Start of group_conv class
class ConvGGLayer(ConvRnGLayer, torch.nn.Module):
    def __init__(
            self,
            group,
            N_in,
            N_out,
            kernel_size,
            h_grid,
            input_h_grid,
            stride,
            padding,
            dilation,
            conv_groups,
            wscale
            ):
        torch.nn.Module.__init__(self)
        ## Assert and set inputs
        self.kernel_type = 'G'
        self._assert_and_set_inputs(group, N_in, N_out, kernel_size, h_grid, input_h_grid, stride, padding, dilation, conv_groups)
        ## Construct the trainable weights and initialize them
        self.weight = torch.nn.Parameter(torch.Tensor(self.N_out, self.N_in, input_h_grid.grid.shape[0], self.kernel_size, self.kernel_size))
        self._reset_parameters(wscale=wscale)

########################### Assert and set inputs ########################

    # Method overriding:
    def _assert_and_set_inputs(self, group, N_in, N_out, kernel_size, h_grid, input_h_grid, stride, padding, dilation, conv_groups):
        # Default Rn assertions
        self._assert_and_set_inputs_GG(group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups)
        # Specific initialization/assertion
        self.h_grid = self._assert_h_grid(h_grid)
        self.input_h_grid = self._assert_input_h_grid(input_h_grid)
        self.N_h = int(self.h_grid.grid.shape[0])  # Target sampling
        self.N_h_in = int(self.input_h_grid.grid.shape[0])

    def _assert_and_set_inputs_GG(self, group, N_in, N_out, kernel_size, stride, padding, dilation, conv_groups):
        ## Check (and parse) all the inputs
        # Include the dictionary of the used parent class
        self.group = group
        self.G = group.G
        self.H = group.H
        self.Rn = group.Rn
        # Mandatory inputs
        self.N_in = self._assert_N_in(N_in)
        self.N_out = self._assert_N_out(N_out)
        self.kernel_size = self._assert_kernel_size(kernel_size)
        # Optional arguments
        self.conv_groups = self._assert_conv_groups(conv_groups)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def _assert_input_h_grid( self, input_h_grid ):
        if (input_h_grid is None):
            return self.h_grid
        else:
            assert (len(input_h_grid.grid.shape) == 2), "The \"input_h_grid\" option value should be a grid object with input_h_grid.grid a tensorflow tensor of dimension 2 (a list of group elements)."
            assert (input_h_grid.grid.shape[-1] == self.H.n), "The group element specification in \"input_h_grid\" is not correct. For the current group \"{}\" each group element should be a vector of length {}.".format(self.H.name,self.H.n)
            return input_h_grid

############################ Compute the output ##########################

    # Method overriding:
    def forward(self, input):
        return self.conv_G_G(input)

    # Method overriding:
    def kernel(self, h=None):
        # The transformation to apply
        if h is None:
            h = self.H.e
        # Sample the kernel on the (transformed) grid
        if not False in (h == self.H.e):
            return self.weight

        h_weight = self.H.left_representation_on_G(h, self.weight)
        return (1 / self.H.absdet(h)) * h_weight

    def conv_G_G(self, input):
        # Generate the full stack of convolution kernels (all transformed copies)
        kernel_stack = torch.cat([self.kernel(self.h_grid.grid[i]) for i in range(self.N_h)], dim=0)  # [N_out x N_h, N_in, N_h_in, Nxy_x, Nxy_y]
        # Reshape input tensor and kernel as if they were Rn tensors
        kernel_stack_as_if_Rn = torch.reshape(kernel_stack, [self.N_h * self.N_out, self.N_in * self.N_h_in, self.kernel_size, self.kernel_size])
        input_tensor_as_if_Rn = torch.reshape(input, [input.shape[0], self.N_in * self.N_h_in, input.shape[-2], input.shape[-1]])
        # And apply them all at once
        output = torch.conv2d(
            input=input_tensor_as_if_Rn,
            weight=kernel_stack_as_if_Rn,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.conv_groups)
        # Reshape the last channel to create a vector valued RnxH feature map
        output = torch.stack(torch.split(output, self.N_out, 1), 2)
        # The above includes integration over S1, take discretization into account
        output = self.group.H.haar_meas * output
        # # Return the output
        return output


###############################################################################
########################### AttentiveConvRnGLayer #############################
###############################################################################
# Start of lifting_layer class
class AttConvRnGLayer(ConvRnGLayer):
    # The initialization of both layers is equal up to the additional parameters
    def __init__(self,
                 group,
                 N_in,
                 N_out,
                 kernel_size,
                 h_grid,
                 channel_attention,
                 spatial_attention,
                 stride,
                 padding,
                 dilation,
                 wscale):
        super(AttConvRnGLayer, self).__init__(group, N_in, N_out, kernel_size, h_grid, stride, padding, dilation, conv_groups=N_in, wscale=wscale)
        self.channel_attention = channel_attention
        self.spatial_attention = spatial_attention

############################ Compute the output ##########################

    # Method overriding:
    def forward(self, input):
        return self.att_conv_Rn_G(input)

    def att_conv_Rn_G(self, input):
        # Generate the full stack of convolution kernels (all transformed copies)
        kernel_stack = torch.stack([self.kernel(self.h_grid.grid[i]) for i in range(self.N_h)], dim=1)
        kernel_stack = torch.reshape(kernel_stack, [self.N_out * self.N_h, self.N_in, kernel_stack.shape[-2], kernel_stack.shape[-1]])
        kernel_stack = torch.transpose(kernel_stack, 0, 1)  # Required for convolutions with parameter group to work accurately.
        kernel_stack_for_conv_groups = torch.reshape(kernel_stack, [self.N_out * self.N_in * self.N_h, 1, self.weight.shape[-2], self.weight.shape[-1]])
        # And apply them all at once
        output = torch.conv2d(
            input=input,
            weight=kernel_stack_for_conv_groups,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.conv_groups)
        # Reshape the last channel to create a vector valued RnxH feature map
        output = torch.reshape(output, [output.shape[0], self.N_in, self.N_out * self.N_h, output.shape[-2], output.shape[-1]])
        output = torch.transpose(output, 1, 2)  # Required for convolutions with parameter group to work accurately.
        output = torch.reshape(output, [output.shape[0], self.N_out, self.N_h, self.N_in, output.shape[-2], output.shape[-1]])
        # Perform attention
        # Check if there's attention mechanism. Otherwise, omit.
        if (self.channel_attention is not None) or (self.spatial_attention is not None):
            # Do channel-wise
            if (self.channel_attention is not None):
                output_att = self._perform_channel_attention(output)
            else: output_att = output
            # Do spatial-wise
            if (self.spatial_attention is not None):
                output_att = self._perform_spatial_attention(output_att)
            # Checkout with responses -- To ensure good gradient propagation we model output * (1 - opposite_attention)
            output = output - output_att
        # Sum over the attented dimension N_in
        output = output.sum(dim=-3)
        # Return the output
        return output

    def _perform_channel_attention(self, input):
        output = self.channel_attention(input)
        output = output * input
        return output

    def _perform_spatial_attention(self, input):
        output = self.spatial_attention(input)
        # Visualize TODO
        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.imshow(output.detach().cpu().reshape([128, -1, 26, 26]).mean(-3).numpy()[1, :, :])
        #plt.show()
        output = output * input
        return output


##########################################################################
########################## AttentiveConvGGLayer ##########################
##########################################################################
# Start of group_conv class
class AttConvGGLayer(ConvGGLayer, torch.nn.Module):
    # The initialization of both layers is equal up to the additional parameters
    def __init__(
            self,
            group,
            N_in,
            N_out,
            kernel_size,
            h_grid,
            channel_attention,
            spatial_attention,
            input_h_grid,
            stride,
            padding,
            dilation,
            wscale):
        super(AttConvGGLayer, self).__init__(group, N_in, N_out, kernel_size, h_grid, input_h_grid,stride, padding, dilation, conv_groups=N_in, wscale=wscale)
        self.channel_attention = channel_attention
        self.spatial_attention = spatial_attention

############################ Compute the output ##########################

    # Method overriding:
    def forward(self, input):
        return self.att_conv_G_G(input)

    def att_conv_G_G(self, input):
        # Generate the full stack of convolution kernels (all transformed copies)
        kernel_stack = torch.stack([self.kernel(self.h_grid.grid[i]) for i in range(self.N_h)], dim=1)
        kernel_stack = torch.reshape(kernel_stack, [self.N_h * self.N_out, self.N_in * self.N_h_in, self.kernel_size, self.kernel_size])
        kernel_stack = kernel_stack.transpose(0, 1)     # Required for convolutions with parameter group to work accurately.
        # Reshape input tensor and kernel as if they were Rn tensors
        kernel_stack_as_if_Rn = torch.reshape(kernel_stack, [self.N_h * self.N_out * self.N_in * self.N_h_in, 1, self.kernel_size, self.kernel_size])
        input_tensor_as_if_Rn = torch.reshape(input, [input.shape[0], self.N_in * self.N_h_in, input.shape[-2], input.shape[-1]])
        # And apply them all at once
        output = torch.conv2d(
            input=input_tensor_as_if_Rn,
            weight=kernel_stack_as_if_Rn,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.N_in * self.N_h_in)
        # Reshape the last channel to create a vector valued RnxH feature map
        output = output.reshape([output.shape[0], self.N_in * self.N_h_in, self.N_out * self.N_h, output.shape[-2], output.shape[-1]])
        output = output.transpose(1, 2)     # Required for convolutions with parameter group to work accurately.
        output = output.reshape([output.shape[0], self.N_out, self.N_h, self.N_in, self.N_h_in, output.shape[-2], output.shape[-1]])
        # Perform attention
        # Check if there's attention mechanism. Otherwise, omit.
        if (self.channel_attention is not None) or (self.spatial_attention is not None):
            # Do channel-wise
            if (self.channel_attention is not None):
                output_att = self._perform_channel_attention(output)
            else: output_att = output
            # Do spatial-wise
            if (self.spatial_attention is not None):
                output_att = self._perform_spatial_attention(output_att)
            # Checkout with responses -- To ensure good gradient propagation we model output * (1 - opposite_attention)
            output = output - output_att
        # Sum over the attented dimensions N_in, N_h_in
        output = output.sum(dim=[-3, -4])
        # The above includes integration over S1, take discretization into account
        output = (2 * np.pi / self.N_h) * output
        # Return the output
        return output

    def _perform_channel_attention(self, input):
        output = self.channel_attention(input)
        output = output * input
        return output

    def _perform_spatial_attention(self, input):
        output = self.spatial_attention(input)
        output = output * input
        return output


###############################################################################
######################### FeatAttentiveConvRnGLayer ###########################
###############################################################################
# Start of lifting_layer class
class fAttConvRnGLayer(ConvRnGLayer):
    # The initialization of both layers is equal up to the additional parameters
    def __init__(self,
                 group,
                 N_in,
                 N_out,
                 kernel_size,
                 h_grid,
                 channel_attention,
                 spatial_attention,
                 stride,
                 padding,
                 wscale):
        super(fAttConvRnGLayer, self).__init__(group, N_in, N_out, kernel_size, h_grid, stride, padding, dilation=1, conv_groups=1, wscale=wscale)
        self.channel_attention = channel_attention
        self.spatial_attention = spatial_attention

    # Method overriding:
    def forward(self, input):
        return self.feat_att_conv_Rn_G(input)

    def feat_att_conv_Rn_G(self, input):
        # Apply attention on the input feature maps
        # Check if there's attention mechanism. Otherwise, omit.
        if (self.channel_attention is not None) or (self.spatial_attention is not None):
            # Do channel-wise
            if (self.channel_attention is not None):
                input_att = self._perform_channel_attention(input)
            else: input_att = input
            # Do spatial-wise
            if (self.spatial_attention is not None):
                input_att = self._perform_spatial_attention(input_att)
            # Checkout with responses -- To ensure good gradient propagation we model output * (1 - opposite_attention)
            input = input - input_att
        # Perform convolution
        output = self.conv_Rn_G(input)
        # Return the output
        return output

    def _perform_channel_attention(self, input):
        output = self.channel_attention(input)
        output = output * input
        return output

    def _perform_spatial_attention(self, input):
        output = self.spatial_attention(input)
        output = output * input
        return output


##########################################################################
######################## FeatAttentiveConvGGLayer ########################
##########################################################################
# Start of group_conv class
class fAttConvGGLayer(ConvGGLayer):
    # The initialization of both layers is equal up to the additional parameters
    def __init__(
            self,
            group,
            N_in,
            N_out,
            kernel_size,
            h_grid,
            channel_attention,
            spatial_attention,
            input_h_grid,
            stride,
            padding,
            wscale):
        super(fAttConvGGLayer, self).__init__(group, N_in, N_out, kernel_size, h_grid, input_h_grid,stride, padding, dilation=1, conv_groups=1, wscale=wscale)
        self.channel_attention = channel_attention
        self.spatial_attention = spatial_attention

############################ Compute the output ##########################

    # Method overriding:
    def forward(self, input):
        return self.feat_att_conv_G_G(input)

    def feat_att_conv_G_G(self, input):
        # Apply attention on the input feature maps
        # Check if there's attention mechanism. Otherwise, omit.
        if (self.channel_attention is not None) or (self.spatial_attention is not None):
            # Do channel-wise
            if (self.channel_attention is not None):
                input_att = self._perform_channel_attention(input)
            else: input_att = input
            # Do spatial-wise
            if (self.spatial_attention is not None):
                input_att = self._perform_spatial_attention(input_att)
            # Checkout with responses -- To ensure good gradient propagation we model output * (1 - opposite_attention)
            input = input - input_att
        # Perform convolution
        output = self.conv_G_G(input)
        # Return the output
        return output

    def _perform_channel_attention(self, input):
        output = self.channel_attention(input)
        output = output * input
        return output

    def _perform_spatial_attention(self, input):
        output = self.spatial_attention(input)
        output = output * input
        return output
