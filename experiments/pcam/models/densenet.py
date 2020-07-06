# torch
import torch
import torch.nn as nn
# built-in
from math import sqrt
import functools


# Based on implementation from Veeling et. al. (2018)
class DenseNet(nn.Module):
    def __init__(self, num_blocks=5, n_channels=24):
        super(DenseNet, self).__init__()

        # Parameters of the model
        padding = 0
        stride = 1
        kernel_size = 3
        eps = 2e-5
        bias = False
        grow_ch = n_channels

        # Save num_blocks
        self.num_blocks = num_blocks

        # Initialization parameters
        wscale = sqrt(2.) # This makes the initialization equal to that of He et. al.

        # First layer is a convolution
        self.c1 = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        # Add num_blocks - 1 DenseBlocks and TransitionBlocks
        block_layers = []
        trans_layers = []
        for i in range(num_blocks - 1):
            block_layers.append(DenseBlock2D(in_channels=n_channels, out_channels=grow_ch, kernel_size=kernel_size, stride=stride, padding=padding))
            n_channels = n_channels + grow_ch
            trans_layers.append(TransitionBlock2D(in_channels=n_channels, out_channels=n_channels, stride=stride, padding=padding))
        # Add last layer to DenseBlock
        block_layers.append(DenseBlock2D(in_channels=n_channels, out_channels=grow_ch, kernel_size=kernel_size, stride=stride, padding=padding))
        n_channels = n_channels + grow_ch
        # Add layers to self
        self.block_layers = nn.Sequential(*block_layers)
        self.trans_layers = nn.Sequential(*trans_layers)

        # Add BN for final DenseBlock
        self.bn_out = nn.BatchNorm2d(num_features=n_channels, eps=eps)
        # Reduce to 2 channels (IMPORTANT! THIS IS DIFFERENT TO VEELING ET. AL. (2018), They use only one output channel with sigmoid)
        self.c_out = nn.Conv2d(in_channels= n_channels, out_channels= 2, kernel_size=1, stride=stride, padding=padding, bias=bias)
        # Define last pooling layer
        self.pooling = torch.nn.functional.avg_pool2d

        # Initialization: (He initialization)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, wscale * torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))

    def forward(self, x):
        h=x
        # First conv layer
        h = self.c1(h)
        # Blocks till num_blocks - 1
        for i in range(self.num_blocks - 1):
            h = self.block_layers[i](h)
            h = self.trans_layers[i](h)
        # Last block
        h = torch.relu(self.bn_out(self.block_layers[-1](h)))
        h = self.c_out(h)
        # Pooling layer
        h = self.pooling(h, kernel_size=h.shape[-1])
        h = h.view(h.size(0), 2)
        return h


class DenseBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DenseBlock2D, self).__init__()

        # Parameters of the model
        eps = 2e-5
        bias = False
        # Layers
        self.bn = nn.BatchNorm2d(num_features=in_channels, eps=eps)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        # 'ConvBlock'
        h = self.conv(torch.relu(self.bn(x)))
        # Crop (remove one at each side)
        xh = x[:, :, 1:-1, 1:-1]
        # Concatenate along channel axis
        return torch.cat([h, xh], dim=1)


class TransitionBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(TransitionBlock2D, self).__init__()

        # Parameters of the model
        eps = 2e-5
        bias = False
        # Layers
        self.bn = nn.BatchNorm2d(num_features=in_channels, eps=eps)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding, bias=bias)
        self.pooling = torch.nn.functional.avg_pool2d

    def forward(self, x):
        # 'ConvBlock'
        h = self.conv(torch.relu(self.bn(x)))
        # AvgPooling
        h = self.pooling(h, kernel_size=2, stride=2, padding=0)
        return h


class P4DenseNet(nn.Module):
    def __init__(self, num_blocks=5, n_channels=24):
        super(P4DenseNet, self).__init__()

        #Parameters of the group

        # Import the group structure
        import importlib
        group_name = 'SE2'
        group = importlib.import_module('attgconv.group.' + group_name)
        # Import the gsplintes package and the layers
        import attgconv
        se2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
        # Create H grid for p4 group
        self.h_grid = se2_layers.H.grid_global(4) # p4

        # Parameters of the model
        padding = 0
        stride = 1
        kernel_size = 3
        eps = 2e-5
        bias = False
        grow_ch = n_channels

        # Save num_blocks
        self.num_blocks = num_blocks

        # Initialization parameters
        wscale = sqrt(2.) # This makes the initialization equal to that of He et. al.

        # First layer is a convolution
        self.c1 = se2_layers.ConvRnG(N_in=3, N_out=n_channels, kernel_size=kernel_size, h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)

        # Add num_blocks - 1 DenseBlocks and TransitionBlocks
        block_layers = []
        trans_layers = []
        for i in range(num_blocks - 1):
            block_layers.append(P4DenseBlock2D(in_channels=n_channels, out_channels=grow_ch, kernel_size=kernel_size, stride=stride, padding=padding, wscale=wscale))
            n_channels = n_channels + grow_ch
            trans_layers.append(P4TransitionBlock2D(in_channels=n_channels, out_channels=n_channels, stride=stride, padding=padding, wscale=wscale))
        # Add last layer to DenseBlock
        block_layers.append(P4DenseBlock2D(in_channels=n_channels, out_channels=grow_ch, kernel_size=kernel_size, stride=stride, padding=padding, wscale=wscale))
        n_channels = n_channels + grow_ch
        # Add layers to self
        self.block_layers = nn.Sequential(*block_layers)
        self.trans_layers = nn.Sequential(*trans_layers)

        # Add BN for final DenseBlock
        self.bn_out = nn.BatchNorm3d(num_features=n_channels, eps=eps)
        # Reduce to 2 channels (IMPORTANT! THIS IS DIFFERENT TO VEELING ET. AL. (2018), They use only one output channel with sigmoid)
        self.c_out = se2_layers.ConvGG(N_in=n_channels, N_out=2, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)
        # Define last pooling layer
        self.pooling = se2_layers.average_pooling_Rn

    def forward(self, x):
        h=x
        # First conv layer
        h = self.c1(h)
        # Blocks till num_blocks - 1
        for i in range(self.num_blocks - 1):
            h = self.block_layers[i](h)
            h = self.trans_layers[i](h)
        # Last block
        h = torch.relu(self.bn_out(self.block_layers[-1](h)))
        h = self.c_out(h)
        # Pooling layer
        h = self.pooling(h, kernel_size=h.shape[-1], stride=2, padding=0)
        h = h.mean(dim=2)
        h = h.view(h.size(0), 2)
        return h


class P4DenseBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, wscale=1.0):
        super(P4DenseBlock2D, self).__init__()

        # Import the group structure
        import importlib
        group_name = 'SE2'
        group = importlib.import_module('attgconv.group.' + group_name)
        # Import the gsplintes package and the layers
        import attgconv
        se2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
        # Create H grid for p4 group
        self.h_grid = se2_layers.H.grid_global(4)  # p4

        # Parameters of the model
        eps = 2e-5
        bias = False
        # Layers
        self.bn = nn.BatchNorm3d(num_features=in_channels, eps=eps)
        self.conv = se2_layers.ConvGG(N_in=in_channels, N_out=out_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)

    def forward(self, x):
        # 'ConvBlock'
        h = self.conv(torch.relu(self.bn(x)))
        # Crop (remove one at each side)
        xh = x[:, :, :, 1:-1, 1:-1]
        # Concatenate along channel axis
        return torch.cat([h, xh], dim=1)


class P4TransitionBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, wscale=1.0):
        super(P4TransitionBlock2D, self).__init__()

        # Import the group structure
        import importlib
        group_name = 'SE2'
        group = importlib.import_module('attgconv.group.' + group_name)
        # Import the gsplintes package and the layers
        import attgconv
        se2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
        # Create H grid for p4 group
        self.h_grid = se2_layers.H.grid_global(4)  # p4

        # Parameters of the model
        eps = 2e-5
        bias = False
        # Layers
        self.bn = nn.BatchNorm3d(num_features=in_channels, eps=eps)
        self.conv = se2_layers.ConvGG(N_in=in_channels, N_out=out_channels, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)
        self.pooling = se2_layers.average_pooling_Rn

    def forward(self, x):
        # 'ConvBlock'
        h = self.conv(torch.relu(self.bn(x)))
        # AvgPooling
        h = self.pooling(h, kernel_size=2, stride=2, padding=0)
        return h


class P4MDenseNet(nn.Module):
    def __init__(self, num_blocks=5, n_channels=24):
        super(P4MDenseNet, self).__init__()

        #Parameters of the group

        # Import the group structure
        import importlib
        group_name = 'E2'
        group = importlib.import_module('attgconv.group.' + group_name)
        # Import the gsplintes package and the layers
        import attgconv
        e2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
        # Create H grid for p4 group
        self.h_grid = e2_layers.H.grid_global(8) # 2xp4

        # Parameters of the model
        padding = 0
        stride = 1
        kernel_size = 3
        eps = 2e-5
        bias = False
        grow_ch = n_channels

        # Save num_blocks
        self.num_blocks = num_blocks

        # Initialization parameters
        wscale = sqrt(2.) # This makes the initialization equal to that of He et. al.

        # First layer is a convolution
        self.c1 = e2_layers.ConvRnG(N_in=3, N_out=n_channels, kernel_size=kernel_size, h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)

        # Add num_blocks - 1 DenseBlocks and TransitionBlocks
        block_layers = []
        trans_layers = []
        for i in range(num_blocks - 1):
            block_layers.append(P4MDenseBlock2D(in_channels=n_channels, out_channels=grow_ch, kernel_size=kernel_size, stride=stride, padding=padding, wscale=wscale))
            n_channels = n_channels + grow_ch
            trans_layers.append(P4MTransitionBlock2D(in_channels=n_channels, out_channels=n_channels, stride=stride, padding=padding, wscale=wscale))
        # Add last layer to DenseBlock
        block_layers.append(P4MDenseBlock2D(in_channels=n_channels, out_channels=grow_ch, kernel_size=kernel_size, stride=stride, padding=padding, wscale=wscale))
        n_channels = n_channels + grow_ch
        # Add layers to self
        self.block_layers = nn.Sequential(*block_layers)
        self.trans_layers = nn.Sequential(*trans_layers)

        # Add BN for final DenseBlock
        self.bn_out = nn.BatchNorm3d(num_features=n_channels, eps=eps)
        # Reduce to 2 channels (IMPORTANT! THIS IS DIFFERENT TO VEELING ET. AL. (2018), They use only one output channel with sigmoid)
        self.c_out = e2_layers.ConvGG(N_in=n_channels, N_out=2, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)
        # Define last pooling layer
        self.pooling = e2_layers.average_pooling_Rn

    def forward(self, x):
        h=x
        # First conv layer
        h = self.c1(h)
        # Blocks till num_blocks - 1
        for i in range(self.num_blocks - 1):
            h = self.block_layers[i](h)
            h = self.trans_layers[i](h)
        # Last block
        h = torch.relu(self.bn_out(self.block_layers[-1](h)))
        h = self.c_out(h)
        # Pooling layer
        h = self.pooling(h, kernel_size=h.shape[-1], stride=2, padding=0)
        h = h.mean(dim=2)
        h = h.view(h.size(0), 2)
        return h


class P4MDenseBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, wscale=1.0):
        super(P4MDenseBlock2D, self).__init__()

        # Import the group structure
        import importlib
        group_name = 'E2'
        group = importlib.import_module('attgconv.group.' + group_name)
        # Import the gsplintes package and the layers
        import attgconv
        e2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
        # Create H grid for p4 group
        self.h_grid = e2_layers.H.grid_global(8)  # 2xp4

        # Parameters of the model
        eps = 2e-5
        bias = False
        # Layers
        self.bn = nn.BatchNorm3d(num_features=in_channels, eps=eps)
        self.conv = e2_layers.ConvGG(N_in=in_channels, N_out=out_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)

    def forward(self, x):
        # 'ConvBlock'
        h = self.conv(torch.relu(self.bn(x)))
        # Crop (remove one at each side)
        xh = x[:, :, :, 1:-1, 1:-1]
        # Concatenate along channel axis
        return torch.cat([h, xh], dim=1)


class P4MTransitionBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, wscale=1.0):
        super(P4MTransitionBlock2D, self).__init__()

        # Import the group structure
        import importlib
        group_name = 'E2'
        group = importlib.import_module('attgconv.group.' + group_name)
        # Import the gsplintes package and the layers
        import attgconv
        e2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
        # Create H grid for p4 group
        self.h_grid = e2_layers.H.grid_global(8)  # p4

        # Parameters of the model
        eps = 2e-5
        bias = False
        # Layers
        self.bn = nn.BatchNorm3d(num_features=in_channels, eps=eps)
        self.conv = e2_layers.ConvGG(N_in=in_channels, N_out=out_channels, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)
        self.pooling = e2_layers.average_pooling_Rn

    def forward(self, x):
        # 'ConvBlock'
        h = self.conv(torch.relu(self.bn(x)))
        # AvgPooling
        h = self.pooling(h, kernel_size=2, stride=2, padding=0)
        return h


# --------- Attention Versions -------------
class fA_P4DenseNet(nn.Module):
    def __init__(self, num_blocks=5, n_channels=24):
        super(fA_P4DenseNet, self).__init__()

        #Parameters of the group

        # Import the group structure
        import importlib
        group_name = 'SE2'
        group = importlib.import_module('attgconv.group.' + group_name)
        # Import the gsplintes package and the layers
        import attgconv
        se2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
        # Create H grid for p4 group
        n_grid = 4
        self.h_grid = se2_layers.H.grid_global(n_grid) # p4

        # Parameters of the model
        padding = 0
        stride = 1
        kernel_size = 3
        eps = 2e-5
        bias = False
        grow_ch = n_channels

        # Save num_blocks
        self.num_blocks = num_blocks

        # Initialization parameters
        wscale = sqrt(2.) # This makes the initialization equal to that of He et. al.

        # ----------------------
        # Attention Parameters
        from attgconv.attention_layers import fChannelAttention as ch_RnG
        from attgconv.attention_layers import fChannelAttentionGG  # as ch_GG
        from attgconv.attention_layers import fSpatialAttention  # as sp_RnG
        from attgconv.attention_layers import fSpatialAttentionGG
        # Parameters of attention
        ch_ratio = 16
        sp_kernel_size = 7
        sp_padding = (sp_kernel_size // 2)

        ch_GG = functools.partial(fChannelAttentionGG, N_h_in=n_grid, group=group_name)
        sp_RnG = functools.partial(fSpatialAttention, wscale=wscale)
        sp_GG = functools.partial(fSpatialAttentionGG, group=group, input_h_grid=self.h_grid, wscale=wscale)

        # First layer is a convolution
        self.c1 = se2_layers.fAttConvRnG(N_in=3, N_out=n_channels, kernel_size=kernel_size, h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                         channel_attention=ch_RnG(N_in=3, ratio=1),
                                         spatial_attention=sp_RnG(group=group, kernel_size=sp_kernel_size, h_grid=self.h_grid, dilation=4)
                                         )
        # Add num_blocks - 1 DenseBlocks and TransitionBlocks
        block_layers = []
        trans_layers = []
        dilations = [4,4,3,2]
        for i in range(num_blocks - 1):
            block_layers.append(fA_P4DenseBlock2D(in_channels=n_channels, out_channels=grow_ch, kernel_size=kernel_size, stride=stride, padding=padding, wscale=wscale, dilation=dilations[i]))
            n_channels = n_channels + grow_ch
            if i == 3:
                dilations[i] = 1
            trans_layers.append(fA_P4TransitionBlock2D(in_channels=n_channels, out_channels=n_channels, stride=stride, padding=padding, wscale=wscale, dilation=dilations[i]))
        # Add last layer to DenseBlock
        block_layers.append(fA_P4DenseBlock2D(in_channels=n_channels, out_channels=grow_ch, kernel_size=kernel_size, stride=stride, padding=padding, wscale=wscale, sp_kernel_size=5))
        n_channels = n_channels + grow_ch
        # Add layers to self
        self.block_layers = nn.Sequential(*block_layers)
        self.trans_layers = nn.Sequential(*trans_layers)

        # Add BN for final DenseBlock
        self.bn_out = nn.BatchNorm3d(num_features=n_channels, eps=eps)
        # Reduce to 2 channels (IMPORTANT! THIS IS DIFFERENT TO VEELING ET. AL. (2018), They use only one output channel with sigmoid)
        self.c_out = se2_layers.fAttConvGG(N_in=n_channels, N_out=2, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                           channel_attention=ch_GG(N_in=n_channels, ratio=n_channels // 2),
                                           spatial_attention=sp_GG(kernel_size=3)
                                           )
        # Define last pooling layer
        self.pooling = se2_layers.average_pooling_Rn

    def forward(self, x):
        h=x     #.rot90(k=0, dims=[-2, -1])
        # First conv layer
        h = self.c1(h)
        # Blocks till num_blocks - 1
        for i in range(self.num_blocks - 1):
            h = self.block_layers[i](h)
            h = self.trans_layers[i](h)
        # Last block
        h = torch.relu(self.bn_out(self.block_layers[-1](h)))
        h = self.c_out(h)
        # Pooling layer
        h = self.pooling(h, kernel_size=h.shape[-1], stride=2, padding=0)
        h = h.mean(dim=2)
        h = h.view(h.size(0), 2)

        # Visualize
        if False:
            from attgconv.attention_layers import fSpatialAttentionGG
            from attgconv.attention_layers import fSpatialAttention
            import numpy as np
            import matplotlib.pyplot as plt
            inx = 0
            B = 60
            maps = []
            for m in self.modules():
                if isinstance(m, fSpatialAttention):
                    map = m.att_map.cpu().detach()
                    inx = map.shape[-2]
                    map = map.expand(map.shape[0], 4, map.shape[2], map.shape[3]).unsqueeze(1)
                    maps.append(map)
            upsample = torch.nn.UpsamplingBilinear2d(size=inx)
            for m in self.modules():
                if isinstance(m, fSpatialAttentionGG):
                    map = m.att_map.cpu().detach()
                    map = map.reshape(map.shape[0], 4, map.shape[-2], map.shape[-1])
                    map = upsample(map)
                    map = map.reshape(map.shape[0], 1, 4, map.shape[-2], map.shape[-1])
                    maps.append(map)
            map_0 = maps[0]
            for i in range(len(maps) - 1):
                map_0 = map_0 * maps[i + 1]

            # Without arrows
            plt.figure()
            plt.imshow(map_0.sum(-3)[B, 0])
            plt.show()

            # Plot all directions
            cmap = plt.cm.jet
            time_samples = 4
            scale = 10
            z = np.zeros([inx, inx])
            plt.figure(dpi=600)
            for t in range(4):
                plt.imshow(map_0.sum(-3)[B, 0])
                if t == 0:
                    plt.quiver(z, map_0[B, 0, t, :, :], color='red', label=r'$0^{\circ}$', scale=scale)
                if t == 2:
                    plt.quiver(z, -map_0[B, 0, t, :, :], color=cmap(t / time_samples), label=r'$180^{\circ}$', scale=scale)
                if t == 1:
                    plt.quiver(-map_0[B, 0, t, :, :], z, color='cyan', label=r'$90^{\circ}$', scale=scale)
                if t == 3:
                    plt.quiver(map_0[B, 0, t, :, :], z, color=cmap(t / time_samples), label=r'$270^{\circ}$',  scale=scale)
            plt.legend(loc='upper right')
            plt.axis('off')
            plt.tight_layout()
            #plt.savefig('90_rot.png')
            plt.show()

        return h


class fA_P4DenseBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, wscale=1.0, sp_kernel_size=None, dilation=1):
        super(fA_P4DenseBlock2D, self).__init__()

        # Import the group structure
        import importlib
        group_name = 'SE2'
        group = importlib.import_module('attgconv.group.' + group_name)
        # Import the gsplintes package and the layers
        import attgconv
        se2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
        # Create H grid for p4 group
        n_grid = 4
        self.h_grid = se2_layers.H.grid_global(n_grid)  # p4

        # Parameters of the model
        eps = 2e-5
        bias = False
        # ----------------------
        # Parameters of attention
        #ch_ratio = 16
        if sp_kernel_size is None:
            sp_kernel_size = 7
        sp_padding = (sp_kernel_size // 2)
        # --------------------------------------------------------

        from attgconv.attention_layers import fChannelAttentionGG  # as ch_GG
        from attgconv.attention_layers import fSpatialAttention  # as sp_RnG
        from attgconv.attention_layers import fSpatialAttentionGG

        ch_GG = functools.partial(fChannelAttentionGG, N_h_in=n_grid, group=group_name)
        sp_RnG = functools.partial(fSpatialAttention, wscale=wscale)
        sp_GG = functools.partial(fSpatialAttentionGG, group=group, input_h_grid=self.h_grid, wscale=wscale)

        # Layers
        self.bn = nn.BatchNorm3d(num_features=in_channels, eps=eps)
        self.conv = se2_layers.fAttConvGG(N_in=in_channels, N_out=out_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                          channel_attention=ch_GG(N_in=in_channels, ratio=in_channels // 2),
                                          spatial_attention=sp_GG(kernel_size=sp_kernel_size, dilation=dilation)
                                          )

    def forward(self, x):
        # 'ConvBlock'
        h = self.conv(torch.relu(self.bn(x)))
        # Crop (remove one at each side)
        xh = x[:, :, :, 1:-1, 1:-1]
        # Concatenate along channel axis
        return torch.cat([h, xh], dim=1)


class fA_P4TransitionBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, wscale=1.0, sp_kernel_size=None, dilation=1):
        super(fA_P4TransitionBlock2D, self).__init__()

        # Import the group structure
        import importlib
        group_name = 'SE2'
        group = importlib.import_module('attgconv.group.' + group_name)
        # Import the gsplintes package and the layers
        import attgconv
        se2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
        # Create H grid for p4 group
        n_grid = 4
        self.h_grid = se2_layers.H.grid_global(n_grid)  # p4

        # Parameters of the model
        eps = 2e-5
        bias = False
        # ----------------------
        # Parameters of attention
        # ch_ratio = 16
        if sp_kernel_size is None:
            sp_kernel_size = 7
        sp_padding = (sp_kernel_size // 2)
        # --------------------------------------------------------

        from attgconv.attention_layers import fChannelAttentionGG  # as ch_GG
        from attgconv.attention_layers import fSpatialAttention  # as sp_RnG
        from attgconv.attention_layers import fSpatialAttentionGG

        ch_GG = functools.partial(fChannelAttentionGG, N_h_in=n_grid, group=group_name)
        sp_RnG = functools.partial(fSpatialAttention, wscale=wscale)
        sp_GG = functools.partial(fSpatialAttentionGG, group=group, input_h_grid=self.h_grid, wscale=wscale)


        # Layers
        self.bn = nn.BatchNorm3d(num_features=in_channels, eps=eps)
        self.conv = se2_layers.fAttConvGG(N_in=in_channels, N_out=out_channels, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                          channel_attention=ch_GG(N_in=in_channels, ratio=in_channels // 2),
                                          spatial_attention=sp_GG(kernel_size=sp_kernel_size, dilation=dilation)
                                          )
        self.pooling = se2_layers.average_pooling_Rn

    def forward(self, x):
        # 'ConvBlock'
        h = self.conv(torch.relu(self.bn(x)))
        # AvgPooling
        h = self.pooling(h, kernel_size=2, stride=2, padding=0)
        return h


class fA_P4MDenseNet(nn.Module):
    def __init__(self, num_blocks=5, n_channels=24):
        super(fA_P4MDenseNet, self).__init__()

        #Parameters of the group

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

        # Parameters of the model
        padding = 0
        stride = 1
        kernel_size = 3
        eps = 2e-5
        bias = False
        grow_ch = n_channels

        # Save num_blocks
        self.num_blocks = num_blocks

        # Initialization parameters
        wscale = sqrt(2.) # This makes the initialization equal to that of He et. al.

        # ----------------------
        # Parameters of attention
        # ch_ratio = 16
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

        # First layer is a convolution
        self.c1 = e2_layers.fAttConvRnG(N_in=3, N_out=n_channels, kernel_size=kernel_size, h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                        channel_attention=ch_RnG(N_in=3, ratio=1),
                                        spatial_attention=sp_RnG(group=group, kernel_size=sp_kernel_size, h_grid=self.h_grid, dilation=4)
                                    )

        # Add num_blocks - 1 DenseBlocks and TransitionBlocks
        block_layers = []
        trans_layers = []
        dilations = [4, 4, 3, 2]
        for i in range(num_blocks - 1):
            block_layers.append(fA_P4MDenseBlock2D(in_channels=n_channels, out_channels=grow_ch, kernel_size=kernel_size, stride=stride, padding=padding, wscale=wscale, dilation=dilations[i]))
            n_channels = n_channels + grow_ch
            if i == 3:
                dilations[i] = 1
            trans_layers.append(fA_P4MTransitionBlock2D(in_channels=n_channels, out_channels=n_channels, stride=stride, padding=padding, wscale=wscale, dilation=dilations[i]))
        # Add last layer to DenseBlock
        block_layers.append(fA_P4MDenseBlock2D(in_channels=n_channels, out_channels=grow_ch, kernel_size=kernel_size, stride=stride, padding=padding, wscale=wscale, sp_kernel_size=5))
        n_channels = n_channels + grow_ch
        # Add layers to self
        self.block_layers = nn.Sequential(*block_layers)
        self.trans_layers = nn.Sequential(*trans_layers)

        # Add BN for final DenseBlock
        self.bn_out = nn.BatchNorm3d(num_features=n_channels, eps=eps)
        # Reduce to 2 channels (IMPORTANT! THIS IS DIFFERENT TO VEELING ET. AL. (2018), They use only one output channel with sigmoid)
        self.c_out = e2_layers.fAttConvGG(N_in=n_channels, N_out=2, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                          channel_attention=ch_GG(N_in=n_channels, ratio=n_channels // 2),
                                          spatial_attention=sp_GG(kernel_size=3)
                                          )
        # Define last pooling layer
        self.pooling = e2_layers.average_pooling_Rn

    def forward(self, x):
        h=x
        # First conv layer
        h = self.c1(h)
        # Blocks till num_blocks - 1
        for i in range(self.num_blocks - 1):
            h = self.block_layers[i](h)
            h = self.trans_layers[i](h)
        # Last block
        h = torch.relu(self.bn_out(self.block_layers[-1](h)))
        h = self.c_out(h)
        # Pooling layer
        h = self.pooling(h, kernel_size=h.shape[-1], stride=2, padding=0)
        h = h.mean(dim=2)
        h = h.view(h.size(0), 2)
        return h


class fA_P4MDenseBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, wscale=1.0, sp_kernel_size=None, dilation=1):
        super(fA_P4MDenseBlock2D, self).__init__()

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

        # Parameters of the model
        eps = 2e-5
        bias = False

        # ----------------------
        # Parameters of attention
        # ch_ratio = 16
        if sp_kernel_size is None:
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

        # Layers
        self.bn = nn.BatchNorm3d(num_features=in_channels, eps=eps)
        self.conv = e2_layers.fAttConvGG(N_in=in_channels, N_out=out_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                         channel_attention=ch_GG(N_in=in_channels, ratio=in_channels // 2),
                                         spatial_attention=sp_GG(kernel_size=sp_kernel_size, dilation=dilation)
                                         )

    def forward(self, x):
        # 'ConvBlock'
        h = self.conv(torch.relu(self.bn(x)))
        # Crop (remove one at each side)
        xh = x[:, :, :, 1:-1, 1:-1]
        # Concatenate along channel axis
        return torch.cat([h, xh], dim=1)


class fA_P4MTransitionBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, wscale=1.0, sp_kernel_size=None, dilation=1):
        super(fA_P4MTransitionBlock2D, self).__init__()

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

        # Parameters of the model
        eps = 2e-5
        bias = False

        # ----------------------
        # Parameters of attention
        # ch_ratio = 16
        if sp_kernel_size is None:
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

        # Layers
        self.bn = nn.BatchNorm3d(num_features=in_channels, eps=eps)
        self.conv = e2_layers.fAttConvGG(N_in=in_channels, N_out=out_channels, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                         channel_attention=ch_GG(N_in=in_channels, ratio=in_channels // 2),
                                         spatial_attention=sp_GG(kernel_size=sp_kernel_size, dilation=dilation)
                                         )
        self.pooling = e2_layers.average_pooling_Rn

    def forward(self, x):
        # 'ConvBlock'
        h = self.conv(torch.relu(self.bn(x)))
        # AvgPooling
        h = self.pooling(h, kernel_size=2, stride=2, padding=0)
        return h

if __name__ == '__main__':
    from experiments.utils import num_params

    model = DenseNet(n_channels=26)
    model(torch.rand([1,3,96,96]))  # Sanity check
    num_params(model)

    model = P4DenseNet(n_channels=13)
    model(torch.rand([1, 3, 96, 96]))  # Sanity check
    num_params(model)

    model = P4MDenseNet(n_channels=9)
    model(torch.rand([1, 3, 96, 96]))  # Sanity check
    num_params(model)

    model = fA_P4DenseNet(n_channels=13)
    model(torch.rand([1, 3, 96, 96]))  # Sanity check
    num_params(model)

    model = fA_P4MDenseNet(n_channels=9)
    model(torch.rand([1, 3, 96, 96]))  # Sanity check
    num_params(model)
