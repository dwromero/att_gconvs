# torch
import torch
import torch.nn as nn
# built-in
from math import sqrt
import functools


# Based on implementation from Cohen & Welling (2016)
class ResNet(nn.Module):
    def __init__(self, num_blocks=7, nc32=32, nc16=64, nc8=128):
        """
        :param num_blocks: the number of resnet blocks per stage. There are 3 stages, for feature map width 32, 16, 8.
        Total number of layers is 6 * num_blocks + 2
        :param nc32: the number of feature maps in the first stage (where feature maps are 32x32)
        :param nc16: the number of feature maps in the second stage (where feature maps are 16x16)
        :param nc8: the number of feature maps in the third stage (where feature maps are 8x8)
        """
        super(ResNet, self).__init__()

        # Parameters of the model
        padding = 1
        stride = 1
        kernel_size = 3
        eps = 2e-5
        bias = False

        # Initialization parameters
        wscale = sqrt(2.)  # This makes the initialization equal to that of He et al.

        # The first layer is always a convolution.
        self.c1 = nn.Conv2d(in_channels=3, out_channels=nc32, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        # Add num_blocks ResBlocks (2 * num_blocks layers) for the size 32x32 feature maps
        layers_nc32 = []
        for i in range(num_blocks):
            layers_nc32.append(ResBlock2D(in_channels=nc32, out_channels=nc32, kernel_size=kernel_size, fiber_map='id', stride=stride, padding=padding))
        self.layers_nc32 = nn.Sequential(*layers_nc32)

        # Add num_blocks ResBlocks (2 * num_blocks layers) for the size 16x16 feature maps
        # The first convolution uses stride 2
        layers_nc16 = []
        for i in range(num_blocks):
            stride_block = 1 if i > 0 else 2
            fiber_map = 'id' if i > 0 else 'linear'
            nc_in = nc16 if i > 0 else nc32
            layers_nc16.append(ResBlock2D(in_channels=nc_in, out_channels=nc16, kernel_size=kernel_size, fiber_map=fiber_map, stride=stride_block, padding=padding))
        self.layers_nc16 = nn.Sequential(*layers_nc16)

        # Add num_blocks ResBlocks (2 * num_blocks layers) for the size 8x8 feature maps
        # The first convolution uses stride 2
        layers_nc8 = []
        for i in range(num_blocks):
            stride_block = 1 if i > 0 else 2
            fiber_map = 'id' if i > 0 else 'linear'
            nc_in = nc8 if i > 0 else nc16
            layers_nc8.append(ResBlock2D(in_channels=nc_in, out_channels=nc8, kernel_size=kernel_size, fiber_map=fiber_map, stride=stride_block, padding=padding))
        self.layers_nc8 = nn.Sequential(*layers_nc8)

        # Add BN and final layer
        # We do ReLU and average pooling between BN and final layer,
        # but since these are stateless they don't require a Link.
        self.bn_out = nn.BatchNorm2d(num_features=nc8, eps=eps)
        self.c_out = nn.Conv2d(in_channels=nc8, out_channels=10, kernel_size=1, stride=1, padding=0, bias=bias)

        # Initialization:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, wscale * torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))

    def forward(self, x):
        h = x
        # First conv layer
        h = self.c1(h)
        # Residual blocks
        h = self.layers_nc32(h)
        h = self.layers_nc16(h)
        h = self.layers_nc8(h)
        # BN, relu, pool, final layer
        h = self.bn_out(h)
        h = torch.relu(h)
        h = torch.nn.functional.avg_pool2d(h, kernel_size=h.shape[-1])
        h = self.c_out(h)
        h = h.view(h.size(0), 10)
        return h


# New style residual block
class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, fiber_map='id', stride=1, padding=1):
        super(ResBlock2D, self).__init__()

        # Asserts
        assert kernel_size % 2 == 1
        if not padding == (kernel_size - 1) // 2:
            raise NotImplementedError()

        # Parameters of the model
        eps = 2e-5
        bias = False

        if stride != 1:
            self.really_equivariant = True
            self.pooling = torch.max_pool2d
        else:
            self.really_equivariant = False

        self.bn1 = nn.BatchNorm2d(num_features=in_channels, eps=eps)
        self.c1 = nn.Conv2d(in_channels=in_channels , out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if self.really_equivariant:
            self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, eps=eps)
        self.c2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1     , padding=padding, bias=bias)

        if fiber_map == 'id':
            if not in_channels == out_channels:
                raise ValueError('fiber_map cannot be identity when channel dimension is changed.')
            self.fiber_map = nn.Sequential() # Identity
        elif fiber_map == 'zero_pad':
            raise NotImplementedError()
        elif fiber_map == 'linear':
            self.fiber_map = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=bias)
            if self.really_equivariant:
                self.fiber_map = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        else:
            raise ValueError('Unknown fiber_map: ' + str(type))

    def forward(self, x):
        h = self.c1(torch.relu(self.bn1(x)))
        if self.really_equivariant:
            h = self.pooling(h, kernel_size=2, stride=2, padding=0)
        h = self.c2(torch.relu(self.bn2(h)))
        hx = self.fiber_map(x)
        if self.really_equivariant:
            hx = self.pooling(hx, kernel_size=2, stride=2, padding=0)
        return hx + h


# Based on implementation from Cohen & Welling (2016)
class P4MResNet(nn.Module):
    def __init__(self, num_blocks=7, nc32=11, nc16=23, nc8=45):
        """
        :param num_blocks: the number of resnet blocks per stage. There are 3 stages, for feature map width 32, 16, 8.
        Total number of layers is 6 * num_blocks + 2
        :param nc32: the number of feature maps in the first stage (where feature maps are 32x32)
        :param nc16: the number of feature maps in the second stage (where feature maps are 16x16)
        :param nc8: the number of feature maps in the third stage (where feature maps are 8x8)
        """
        super(P4MResNet, self).__init__()

        #Parameters of the group

        # Import the group structure
        import importlib
        group_name = 'E2'
        group = importlib.import_module('attgconv.group.' + group_name)
        # Import the gsplintes package and the layers
        import attgconv
        e2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
        # Create H grid for p4 group
        self.h_grid = e2_layers.H.grid_global(8) # 2*p4

        # Parameters of the model
        stride = 1
        padding = 1
        kernel_size = 3
        eps = 2e-5

        # Initialization parameters
        wscale = sqrt(2.)  # This makes the initialization equal to that of He et al.

        # Pooling layer
        self.avg_pooling = e2_layers.average_pooling_Rn

        # The first layer is always a convolution.
        self.c1 = e2_layers.ConvRnG(N_in=3, N_out=nc32, kernel_size=kernel_size, h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)

        # Add num_blocks ResBlocks (2 * num_blocks layers) for the size 32x32 feature maps
        layers_nc32 = []
        for i in range(num_blocks):
            layers_nc32.append(P4MResBlock2D(in_channels=nc32, out_channels=nc32, kernel_size=kernel_size, fiber_map='id', stride=stride, padding=padding, wscale=wscale))
        self.layers_nc32 = nn.Sequential(*layers_nc32)

        # Add num_blocks ResBlocks (2 * num_blocks layers) for the size 16x16 feature maps
        # The first convolution uses stride 2
        layers_nc16 = []
        for i in range(num_blocks):
            stride_block = 1 if i > 0 else 2
            fiber_map = 'id' if i > 0 else 'linear'
            nc_in = nc16 if i > 0 else nc32
            layers_nc16.append(P4MResBlock2D(in_channels=nc_in, out_channels=nc16, kernel_size=kernel_size, fiber_map=fiber_map, stride=stride_block, padding=padding, wscale=wscale))
        self.layers_nc16 = nn.Sequential(*layers_nc16)

        # Add num_blocks ResBlocks (2 * num_blocks layers) for the size 8x8 feature maps
        # The first convolution uses stride 2
        layers_nc8 = []
        for i in range(num_blocks):
            stride_block = 1 if i > 0 else 2
            fiber_map = 'id' if i > 0 else 'linear'
            nc_in = nc8 if i > 0 else nc16
            layers_nc8.append(P4MResBlock2D(in_channels=nc_in, out_channels=nc8, kernel_size=kernel_size, fiber_map=fiber_map, stride=stride_block, padding=padding,  wscale=wscale))
        self.layers_nc8 = nn.Sequential(*layers_nc8)

        # Add BN and final layer
        # We do ReLU and average pooling between BN and final layer,
        # but since these are stateless they don't require a Link.
        self.bn_out = nn.BatchNorm3d(num_features=nc8, eps=eps)
        self.c_out = e2_layers.ConvGG(N_in=nc8, N_out=10, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=1, padding=0, wscale=wscale)

    def forward(self, x):
        #x = torch.flip(x, dims=[-2])
        #x = torch.rot90(x, k=1, dims=[-2, -1])
        h = x
        # First conv layer
        h = self.c1(h)
        # Residual blocks
        h = self.layers_nc32(h)
        h = self.layers_nc16(h)
        h = self.layers_nc8(h)
        # BN, relu, pool, final layer
        h = self.bn_out(h)
        h = torch.relu(h)
        h = self.avg_pooling(h, kernel_size=h.shape[-1], stride=1, padding=0) # TODO check!
        h = self.c_out(h)
        h = h.mean(dim=2)
        h = h.view(h.size(0), 10)
        return h


# New style residual block
class P4MResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, fiber_map='id', stride=1, padding=1, wscale=1.0):
        super(P4MResBlock2D, self).__init__()

        # Asserts
        assert kernel_size % 2 == 1
        if not padding == (kernel_size - 1) // 2:
            raise NotImplementedError()

        # Parameters of the group

        # Import the group structure
        import importlib
        group_name = 'E2'
        group = importlib.import_module('attgconv.group.' + group_name)
        # Import the gsplintes package and the layers
        import attgconv
        e2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
        # Create H grid for p4 group
        self.h_grid = e2_layers.H.grid_global(8)  # 2*p4

        # Parameters of the model
        eps = 2e-5

        if stride != 1:
            self.really_equivariant = True
            self.pooling = e2_layers.max_pooling_Rn
        else:
            self.really_equivariant = False

        self.bn1 = nn.BatchNorm3d(num_features=in_channels, eps=eps)
        self.c1 = e2_layers.ConvGG(N_in=in_channels , N_out=out_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)
        if self.really_equivariant:
            self.c1 = e2_layers.ConvGG(N_in=in_channels, N_out=out_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=1, padding=padding, wscale=wscale)

        self.bn2 = nn.BatchNorm3d(num_features=out_channels, eps=eps)
        self.c2 = e2_layers.ConvGG(N_in=out_channels, N_out=out_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=1     , padding=padding, wscale=wscale)

        if fiber_map == 'id':
            if not in_channels == out_channels:
                raise ValueError('fiber_map cannot be identity when channel dimension is changed.')
            self.fiber_map = nn.Sequential() # Identity
        elif fiber_map == 'zero_pad':
            raise NotImplementedError()
        elif fiber_map == 'linear':
            self.fiber_map = e2_layers.ConvGG(N_in=in_channels, N_out=out_channels, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=0, wscale=wscale)
            if self.really_equivariant:
                self.fiber_map = e2_layers.ConvGG(N_in=in_channels, N_out=out_channels, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=1, padding=0, wscale=wscale)
        else:
            raise ValueError('Unknown fiber_map: ' + str(type))

    def forward(self, x):
        h = self.c1(torch.relu(self.bn1(x)))
        if self.really_equivariant:
            h = self.pooling(h, kernel_size=2, stride=2, padding=0)
        h = self.c2(torch.relu(self.bn2(h)))
        hx = self.fiber_map(x)
        if self.really_equivariant:
            hx = self.pooling(hx, kernel_size=2, stride=2, padding=0)
        return hx + h


# Based on implementation from Cohen & Welling (2016)
class fA_P4MResNet(nn.Module):
    def __init__(self, num_blocks=7, nc32=11, nc16=23, nc8=45):
        """
        :param num_blocks: the number of resnet blocks per stage. There are 3 stages, for feature map width 32, 16, 8.
        Total number of layers is 6 * num_blocks + 2
        :param nc32: the number of feature maps in the first stage (where feature maps are 32x32)
        :param nc16: the number of feature maps in the second stage (where feature maps are 16x16)
        :param nc8: the number of feature maps in the third stage (where feature maps are 8x8)
        """
        super(fA_P4MResNet, self).__init__()

        #Parameters of the group

        # Import the group structure
        import importlib
        group_name = 'E2'
        group = importlib.import_module('attgconv.group.' + group_name)
        # Import the gsplintes package and the layers
        import attgconv
        e2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
        # Create H grid for p4m group
        n_grid = 8
        h_grid = e2_layers.H.grid_global(n_grid)
        # ----------------------
        # Parameters of the model
        stride = 1
        padding = 1
        kernel_size = 3
        eps = 2e-5
        # --------------------------------------------------------
        # Store in self
        self.group_name = group_name
        self.group = group
        self.layers = e2_layers
        self.n_grid = n_grid
        self.h_grid = h_grid
        # ----------------------
        # Initialization parameters
        wscale = sqrt(2.)  # This makes the initialization equal to that of He et al.
        # ----------------------
        # Parameters of attention
        ch_ratio = 16
        sp_kernel_size = 7
        sp_padding = (sp_kernel_size // 2)

        from attgconv.attention_layers import fChannelAttention as ch_RnG
        from attgconv.attention_layers import fChannelAttentionGG  # as ch_GG
        from attgconv.attention_layers import fSpatialAttention  # as sp_RnG
        from attgconv.attention_layers import fSpatialAttentionGG

        ch_GG = functools.partial(fChannelAttentionGG, N_h_in=n_grid, group=group_name)
        sp_RnG = functools.partial(fSpatialAttention, wscale=wscale)
        sp_GG = functools.partial(fSpatialAttentionGG, group=group, input_h_grid=self.h_grid, wscale=wscale)

        # Pooling layer
        self.avg_pooling = e2_layers.average_pooling_Rn

        # The first layer is always a convolution.
        self.c1 = e2_layers.fAttConvRnG(N_in=3, N_out=nc32, kernel_size=kernel_size, h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                    channel_attention=ch_RnG(N_in=3, ratio=1),
                                    spatial_attention=sp_RnG(group=group, kernel_size=sp_kernel_size, h_grid=self.h_grid)
                                    )
        # Add num_blocks ResBlocks (2 * num_blocks layers) for the size 32x32 feature maps
        layers_nc32 = []
        for i in range(num_blocks):
            layers_nc32.append(fA_P4MResBlock2D(in_channels=nc32, out_channels=nc32, kernel_size=kernel_size, fiber_map='id', stride=stride, padding=padding, wscale=wscale))
        self.layers_nc32 = nn.Sequential(*layers_nc32)

        # Add num_blocks ResBlocks (2 * num_blocks layers) for the size 16x16 feature maps
        # The first convolution uses stride 2
        layers_nc16 = []
        for i in range(num_blocks):
            stride_block = 1 if i > 0 else 2
            fiber_map = 'id' if i > 0 else 'linear'
            nc_in = nc16 if i > 0 else nc32
            layers_nc16.append(fA_P4MResBlock2D(in_channels=nc_in, out_channels=nc16, kernel_size=kernel_size, fiber_map=fiber_map, stride=stride_block, padding=padding, wscale=wscale))
        self.layers_nc16 = nn.Sequential(*layers_nc16)

        # Add num_blocks ResBlocks (2 * num_blocks layers) for the size 8x8 feature maps
        # The first convolution uses stride 2
        layers_nc8 = []
        for i in range(num_blocks):
            stride_block = 1 if i > 0 else 2
            fiber_map = 'id' if i > 0 else 'linear'
            nc_in = nc8 if i > 0 else nc16
            layers_nc8.append(fA_P4MResBlock2D(in_channels=nc_in, out_channels=nc8, kernel_size=kernel_size, fiber_map=fiber_map, stride=stride_block, padding=padding,  wscale=wscale))
        self.layers_nc8 = nn.Sequential(*layers_nc8)

        # Add BN and final layer
        # We do ReLU and average pooling between BN and final layer,
        # but since these are stateless they don't require a Link.
        self.bn_out = nn.BatchNorm3d(num_features=nc8, eps=eps)
        self.c_out = e2_layers.fAttConvGG(N_in=nc8, N_out=10, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=1, padding=0, wscale=wscale,
                                          channel_attention=ch_GG(N_in=nc8, ratio=nc8 // 2),
                                          spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                          )

    def forward(self, x):
        x = torch.flip(x, dims=[-1])
        h = x
        # First conv layer
        h = self.c1(h)
        # Residual blocks
        h = self.layers_nc32(h)
        h = self.layers_nc16(h)
        h = self.layers_nc8(h)
        # BN, relu, pool, final layer
        h = self.bn_out(h)
        h = torch.relu(h)
        h = self.avg_pooling(h, kernel_size=h.shape[-1], stride=1, padding=0) # TODO check!
        h = self.c_out(h)
        h = h.mean(dim=2)
        h = h.view(h.size(0), 10)
        return h


# New style residual block
class fA_P4MResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, fiber_map='id', stride=1, padding=1, wscale=1.0):
        super(fA_P4MResBlock2D, self).__init__()

        # Asserts
        assert kernel_size % 2 == 1
        if not padding == (kernel_size - 1) // 2:
            raise NotImplementedError()

        # Parameters of the group

        # Import the group structure
        import importlib
        group_name = 'E2'
        group = importlib.import_module('attgconv.group.' + group_name)
        # Import the gsplintes package and the layers
        import attgconv
        e2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
        # Create H grid for p4 group
        n_grid = 8
        self.h_grid = e2_layers.H.grid_global(n_grid)  # 2*p4
        # ----------------------
        # Parameters of the model
        eps = 2e-5
        # ----------------------
        # Parameters of attention
        #ch_ratio = 16
        sp_kernel_size = 7
        sp_padding = (sp_kernel_size // 2)
        # --------------------------------------------------------

        from attgconv.attention_layers import fChannelAttention as ch_RnG
        from attgconv.attention_layers import fChannelAttentionGG  # as ch_GG
        from attgconv.attention_layers import fSpatialAttention  # as sp_RnG
        from attgconv.attention_layers import fSpatialAttentionGG

        ch_GG = functools.partial(fChannelAttentionGG, N_h_in=n_grid, group=group_name)
        sp_RnG = functools.partial(fSpatialAttention, wscale=wscale)
        sp_GG = functools.partial(fSpatialAttentionGG, group=group, input_h_grid=self.h_grid, wscale=wscale)

        if stride != 1:
            self.really_equivariant = True
            self.pooling = e2_layers.max_pooling_Rn
        else:
            self.really_equivariant = False

        self.bn1 = nn.BatchNorm3d(num_features=in_channels, eps=eps)
        self.c1 = e2_layers.fAttConvGG(N_in=in_channels , N_out=out_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                       channel_attention=ch_GG(N_in=in_channels, ratio=in_channels // 2),
                                       spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                       )
        if self.really_equivariant:
            self.c1 = e2_layers.fAttConvGG(N_in=in_channels, N_out=out_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=1, padding=padding, wscale=wscale,
                                           channel_attention=ch_GG(N_in=in_channels, ratio=in_channels // 2),
                                           spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                           )

        self.bn2 = nn.BatchNorm3d(num_features=out_channels, eps=eps)
        self.c2 = e2_layers.fAttConvGG(N_in=out_channels, N_out=out_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=1     , padding=padding, wscale=wscale,
                                       channel_attention=ch_GG(N_in=out_channels, ratio=out_channels // 2),
                                       spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                       )
        if fiber_map == 'id':
            if not in_channels == out_channels:
                raise ValueError('fiber_map cannot be identity when channel dimension is changed.')
            self.fiber_map = nn.Sequential() # Identity
        elif fiber_map == 'zero_pad':
            raise NotImplementedError()
        elif fiber_map == 'linear':
            self.fiber_map = e2_layers.fAttConvGG(N_in=in_channels, N_out=out_channels, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=0, wscale=wscale,
                                              channel_attention=ch_GG(N_in=in_channels, ratio=in_channels // 2),
                                              spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                              )
            if self.really_equivariant:
                self.fiber_map = e2_layers.fAttConvGG(N_in=in_channels, N_out=out_channels, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=1, padding=0, wscale=wscale,
                                                      channel_attention=ch_GG(N_in=in_channels, ratio=in_channels // 2),
                                                      spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                                      )
        else:
            raise ValueError('Unknown fiber_map: ' + str(type))

    def forward(self, x):
        h = self.c1(torch.relu(self.bn1(x)))
        if self.really_equivariant:
            h = self.pooling(h, kernel_size=2, stride=2, padding=0)
        h = self.c2(torch.relu(self.bn2(h)))
        hx = self.fiber_map(x)
        if self.really_equivariant:
            hx = self.pooling(hx, kernel_size=2, stride=2, padding=0)
        return hx + h


if __name__ == '__main__':
    from experiments.utils import num_params

    model = ResNet()
    model(torch.rand([1, 3, 32, 32]))  # Sanity check
    num_params(model)

    model = P4MResNet()
    model(torch.rand([1, 3, 32, 32]))  # Sanity check
    num_params(model)

    model = fA_P4MResNet()
    model(torch.rand([1, 3, 32, 32]))  # Sanity check
    num_params(model)


