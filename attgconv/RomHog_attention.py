# Implementation of attention by Romero and Hoogendoorn (2020)

# In this class the attention layers are defined, which are used in the attentive group convolutions.
import torch
import math
import numpy as np
# layers
from attgconv.attention_layers import ConvRnGLayer, ConvGGLayer

'''
Small Definition and Comparison with Romero and Hoogendoorn (2020)
-------------------------------------------------------------------

In Romero and Hoogendoorn (2020), attention is performed under rotations in the co-domain space of the convolution 
operation. In other words, this is conceptually equivalent to instantiate a fChannelAttention layer with the following
parameters:
- kernel_size = 1
- no statistics gathering before attention --> N_statistics = N_in
- Softmax-like operation over rotations

For the sake of comparison, we utilize the same non-linearity (Sigmoid) in this implementation. And avoid the usage of
softmax + normalization. 
'''

class fSpatialAttention(ConvRnGLayer):
    def __init__(self,
                 group,
                 N_in,
                 h_grid,
                 stride=1,
                 dilation=1,
                 groups=1,
                 wscale=1.0
                 ):
        # Set parameters and save in self
        kernel_size = 1
        padding = 0
        N_out = 1           #  One channel that describes attention spatially.
        # ------------------------------
        super(fSpatialAttention, self).__init__(group, N_in, N_out, kernel_size, h_grid, stride, padding, dilation, groups, wscale)
        # self.N_in = N_in
        # self.N_out = N_out
        # self.stride = stride
        # self.kernel_size = kernel_size
        # self.padding = (kernel_size // 2)
        # self.dilation = dilation
        # self.groups = groups
        # self.wscale = wscale
        # self.weight = torch.nn.Parameter(torch.Tensor(self.N_out, self.N_in, kernel_size, kernel_size))
        # # Initialize
        # self._reset_parameters(wscale)

    # Method overriding:
    def forward(self, input, visualize=False):
        return self.f_att_conv2d(input, visualize)

    def f_att_conv2d(self, input, visualize):
        # Get statistics
        # avg_in = torch.mean(input, dim=-3, keepdim=True)
        # max_in, _ = torch.max(input, dim=-3, keepdim=True)
        # input = torch.cat([avg_in, max_in], dim=-3)
        # Apply convolution
        output = self.conv_Rn_G(input)
        # Do pooling over the group
        output, _ = output.max(dim=2)
        # Apply sigmoid
        output = torch.sigmoid(output)
        # Return output
        return output


# Corresponds to G-convolution followed by a normalization (sigmoid) step.
class fSpatialAttentionGG(ConvGGLayer):
    def __init__(self,
                 group,
                 N_in,
                 input_h_grid,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 wscale=1.0
                 ):
        kernel_size = 1
        N_out = 1
        padding = 0
        super(fSpatialAttentionGG, self).__init__(group, N_in, N_out, kernel_size, input_h_grid, input_h_grid, stride, padding, dilation, groups, wscale)

    # Method overriding:
    def forward(self, input, visualize=False):
        return self.f_att_conv_GG(input, visualize)

    def f_att_conv_GG(self, input, visualize):
        # Get input statistics
        # avg_in = torch.mean(input, dim=-4, keepdim=True)
        # max_in, _ = torch.max(input, dim=-4, keepdim=True)
        # input = torch.cat([avg_in, max_in], dim=-4)
        # Apply group convolution
        output = self.conv_G_G(input)
        # Apply sigmoid
        output = torch.sigmoid(output)
        # Return the output
        return output
