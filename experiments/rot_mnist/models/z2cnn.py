# torch
import torch
import torch.nn as nn
# built-in
import functools


# Typical Convolutional Network
class Z2CNN(nn.Module):
    def __init__(self, use_bias = False):
        super(Z2CNN, self).__init__()
        # Parameters of the model
        p = 0.3
        stride = 1
        padding = 0
        kernel_size = 3
        N_channels = 20
        eps = 2e-5
        # Conv Layers
        self.c1 = nn.Conv2d(in_channels=1         , out_channels=N_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        self.c2 = nn.Conv2d(in_channels=N_channels, out_channels=N_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        self.c3 = nn.Conv2d(in_channels=N_channels, out_channels=N_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        self.c4 = nn.Conv2d(in_channels=N_channels, out_channels=N_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        self.c5 = nn.Conv2d(in_channels=N_channels, out_channels=N_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        self.c6 = nn.Conv2d(in_channels=N_channels, out_channels=N_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        self.c7 = nn.Conv2d(in_channels=N_channels, out_channels=10        , kernel_size=4          , stride=stride, padding=padding, bias=use_bias)
        # Dropout
        self.dp = nn.Dropout(p)
        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(num_features=N_channels, eps=eps)
        self.bn2 = nn.BatchNorm2d(num_features=N_channels, eps=eps)
        self.bn3 = nn.BatchNorm2d(num_features=N_channels, eps=eps)
        self.bn4 = nn.BatchNorm2d(num_features=N_channels, eps=eps)
        self.bn5 = nn.BatchNorm2d(num_features=N_channels, eps=eps)
        self.bn6 = nn.BatchNorm2d(num_features=N_channels, eps=eps)
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        out = self.dp(torch.relu(self.bn1(self.c1(x))))
        out = torch.relu(self.bn2(self.c2(out)))
        out = torch.max_pool2d(out, kernel_size=2, stride=2, padding=0)

        out = self.dp(torch.relu(self.bn3(self.c3(out))))
        out = self.dp(torch.relu(self.bn4(self.c4(out))))
        out = self.dp(torch.relu(self.bn5(self.c5(out))))
        out = self.dp(torch.relu(self.bn6(self.c6(out))))

        out = self.c7(out)
        out = out.view(out.size(0), 10)
        return out


class P4CNN(nn.Module):
    def __init__(self, use_bias=False):
        super(P4CNN, self).__init__()

        #Parameters of the group

        # Import the group structure
        import importlib
        group_name = 'SE2'
        group = importlib.import_module('attgconv.group.' + group_name)
        # Import the gsplintes package and the layers
        import attgconv
        se2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
        # Create H grid for p4 group
        self.h_grid = se2_layers.H.grid_global(4)

        # Parameters of the model
        p = 0.3
        stride = 1
        padding = 0
        kernel_size = 3
        N_channels = 10
        eps = 2e-5

        # Conv Layers
        self.c1 = se2_layers.ConvRnG(N_in=1,         N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, stride=stride, padding=padding)
        self.c2 = se2_layers.ConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding)
        self.c3 = se2_layers.ConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding)
        self.c4 = se2_layers.ConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding)
        self.c5 = se2_layers.ConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding)
        self.c6 = se2_layers.ConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding)
        self.c7 = se2_layers.ConvGG(N_in=N_channels, N_out=10        , kernel_size=4          , h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding)
        # Dropout
        self.dp = nn.Dropout(p)
        # Batch Normalization
        self.bn1 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn2 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn3 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn4 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn5 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn6 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        # Max Pooling
        self.max_pooling = se2_layers.max_pooling_Rn


    def forward(self, x):
        #x = torch.rot90(x, k=1, dims=[-2, -1])  # Sanity check. Equivariance.
        out = self.dp(torch.relu(self.bn1(self.c1(x))))
        out = torch.relu(self.bn2(self.c2(out)))
        out = self.max_pooling(out, kernel_size=2, stride=2, padding=0)

        out = self.dp(torch.relu(self.bn3(self.c3(out))))
        out = self.dp(torch.relu(self.bn4(self.c4(out))))
        out = self.dp(torch.relu(self.bn5(self.c5(out))))
        out = self.dp(torch.relu(self.bn6(self.c6(out))))

        out = self.c7(out)
        out, _ = torch.max(out, dim=-3)
        out = out.view(out.size(0), 10)
        return out


class A_P4CNN(nn.Module):
    def __init__(self, use_bias=False, attention=False):
        super(A_P4CNN, self).__init__()

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
        self.h_grid = se2_layers.H.grid_global(n_grid)

        # Parameters of the model
        p = 0.3
        stride = 1
        padding = 0
        kernel_size = 3
        N_channels = 10
        eps = 2e-5

        # Parameters of attention
        ch_ratio = 2        #(N_channels // 2)    # Hidden layer consists of 2 neurons
        sp_kernel_size = 7

        # Store in self such that all models share it
        self.ch_ratio = ch_ratio
        self.sp_kernel_size = sp_kernel_size

        from attgconv.attention_layers import ChannelAttention as ch_RnG
        from attgconv.attention_layers import ChannelAttentionGG #as ch_GG
        from attgconv.attention_layers import SpatialAttention #as sp_RnG
        from attgconv.attention_layers import SpatialAttentionGG

        ch_GG = functools.partial(ChannelAttentionGG, N_h=n_grid, N_h_in=n_grid)
        sp_RnG = functools.partial(SpatialAttention, group=group, h_grid=self.h_grid, stride=stride)
        sp_GG = functools.partial(SpatialAttentionGG, group=group, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride)

        self.c1 = se2_layers.AttConvRnG(N_in=1        , N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, stride=stride, padding=padding,
                                        channel_attention=ch_RnG(N_out=N_channels, N_in=1        , ratio=1),
                                        spatial_attention=sp_RnG(N_out=N_channels, N_in=1        , kernel_size=sp_kernel_size)
                                        )
        self.c2 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out= N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out= N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c3 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c4 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c5 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c6 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c7 = se2_layers.AttConvGG(N_in=N_channels, N_out=10        , kernel_size=4         , h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out=10        , N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=10        , N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        # Dropout
        self.dp = nn.Dropout(p)
        # Batch Normalization
        self.bn1 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn2 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn3 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn4 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn5 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn6 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        # Max Pooling
        self.max_pooling = se2_layers.max_pooling_Rn

    def forward(self, x):
        #x = torch.rot90(x, k=3, dims=[-2, -1])  # Sanity check. Equivariance.
        out = self.dp(torch.relu(self.bn1(self.c1(x))))
        out = torch.relu(self.bn2(self.c2(out)))
        out = self.max_pooling(out, kernel_size=2, stride=2, padding=0)

        out = self.dp(torch.relu(self.bn3(self.c3(out))))
        out = self.dp(torch.relu(self.bn4(self.c4(out))))
        out = self.dp(torch.relu(self.bn5(self.c5(out))))
        out = self.dp(torch.relu(self.bn6(self.c6(out))))

        out = self.c7(out)
        out, _ = torch.max(out, dim=-3)
        out = out.view(out.size(0), 10)
        return out


class A_Ch_P4CNN(A_P4CNN):
    # Inherits forward
    def __init__(self, use_bias=False, attention=False):
        super(A_Ch_P4CNN, self).__init__()

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
        self.h_grid = se2_layers.H.grid_global(n_grid)

        # Parameters of the model
        p = 0.3
        stride = 1
        padding = 0
        kernel_size = 3
        N_channels = 10
        eps = 2e-5

        # Parameters of attention
        ch_ratio = self.ch_ratio       #(N_channels // 2)    # Hidden layer consists of 2 neurons
        sp_kernel_size = self.sp_kernel_size

        from attgconv.attention_layers import ChannelAttention as ch_RnG
        from attgconv.attention_layers import ChannelAttentionGG #as ch_GG
        from attgconv.attention_layers import SpatialAttention #as sp_RnG
        from attgconv.attention_layers import SpatialAttentionGG

        ch_GG = functools.partial(ChannelAttentionGG, N_h=n_grid, N_h_in=n_grid)
        sp_RnG = functools.partial(SpatialAttention, group=group, h_grid=self.h_grid, stride=stride)
        sp_GG = functools.partial(SpatialAttentionGG, group=group, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride)

        self.c1 = se2_layers.AttConvRnG(N_in=1        , N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, stride=stride, padding=padding,
                                        channel_attention=ch_RnG(N_out=N_channels, N_in=1        , ratio=1),
                                        #spatial_attention=sp_RnG(N_out=N_channels, N_in=1        , kernel_size=sp_kernel_size)
                                        )
        self.c2 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out= N_channels, N_in=N_channels, ratio=ch_ratio),
                                       #spatial_attention=sp_GG(N_out= N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c3 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       #spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c4 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       #spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c5 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       #spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c6 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       #spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c7 = se2_layers.AttConvGG(N_in=N_channels, N_out=10        , kernel_size=4         , h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out=10        , N_in=N_channels, ratio=ch_ratio),
                                       #spatial_attention=sp_GG(N_out=10        , N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        # Dropout
        self.dp = nn.Dropout(p)
        # Batch Normalization
        self.bn1 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn2 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn3 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn4 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn5 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn6 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        # Max Pooling
        self.max_pooling = se2_layers.max_pooling_Rn


class A_Sp_P4CNN(A_P4CNN):
    # Inherits forward
    def __init__(self, use_bias=False, attention=False):
        super(A_Sp_P4CNN, self).__init__()

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
        self.h_grid = se2_layers.H.grid_global(n_grid)

        # Parameters of the model
        p = 0.3
        stride = 1
        padding = 0
        kernel_size = 3
        N_channels = 10
        eps = 2e-5

        # Parameters of attention
        ch_ratio = self.ch_ratio        #(N_channels // 2)    # Hidden layer consists of 2 neurons
        sp_kernel_size = self.sp_kernel_size

        from attgconv.attention_layers import ChannelAttention as ch_RnG
        from attgconv.attention_layers import ChannelAttentionGG #as ch_GG
        from attgconv.attention_layers import SpatialAttention #as sp_RnG
        from attgconv.attention_layers import SpatialAttentionGG

        ch_GG = functools.partial(ChannelAttentionGG, N_h=n_grid, N_h_in=n_grid)
        sp_RnG = functools.partial(SpatialAttention, group=group, h_grid=self.h_grid, stride=stride)
        sp_GG = functools.partial(SpatialAttentionGG, group=group, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride)

        self.c1 = se2_layers.AttConvRnG(N_in=1        , N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, stride=stride, padding=padding,
                                        #channel_attention=ch_RnG(N_out=N_channels, N_in=1        , ratio=1),
                                        spatial_attention=sp_RnG(N_out=N_channels, N_in=1        , kernel_size=sp_kernel_size)
                                        )
        self.c2 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       #channel_attention=ch_GG(N_out= N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out= N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c3 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       #channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c4 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       #channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c5 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       #channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c6 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       #channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c7 = se2_layers.AttConvGG(N_in=N_channels, N_out=10        , kernel_size=4         , h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       #channel_attention=ch_GG(N_out=10        , N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=10        , N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        # Dropout
        self.dp = nn.Dropout(p)
        # Batch Normalization
        self.bn1 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn2 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn3 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn4 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn5 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn6 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        # Max Pooling
        self.max_pooling = se2_layers.max_pooling_Rn


class fA_P4CNN(nn.Module):
    def __init__(self, use_bias=False):
        super(fA_P4CNN, self).__init__()

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
        h_grid = se2_layers.H.grid_global(n_grid)
        # ----------------------
        # Parameters of the model
        p = 0.3
        stride = 1
        padding = 0
        kernel_size = 3
        N_channels = 10
        eps = 2e-5
        # ----------------------
        # Parameters of attention
        ch_ratio = 2        #(N_channels // 2)    # Hidden layer consists of 2 neurons
        sp_kernel_size = 7

        # --------------------------------------------------------
        # Store all parameters such that all (sub)models share it.
        self.group_name = group_name
        self.group = group
        self.layers = se2_layers
        self.n_grid = n_grid
        self.h_grid = h_grid
        self.p = p
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.N_channels = N_channels
        self.eps = eps
        self.ch_ratio = ch_ratio
        self.sp_kernel_size = sp_kernel_size

        from attgconv.attention_layers import fChannelAttention as ch_RnG
        from attgconv.attention_layers import fChannelAttentionGG #as ch_GG
        from attgconv.attention_layers import fSpatialAttention as sp_RnG
        from attgconv.attention_layers import fSpatialAttentionGG

        ch_GG = functools.partial(fChannelAttentionGG, N_h_in=n_grid)
        sp_GG = functools.partial(fSpatialAttentionGG, group=group, input_h_grid=self.h_grid, stride=stride)

        self.c1 = se2_layers.fAttConvRnG(N_in=1        , N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, stride=stride, padding=padding,
                                        channel_attention=ch_RnG(N_in=1        , ratio=1),
                                        spatial_attention=sp_RnG(group=group, kernel_size=sp_kernel_size, h_grid=self.h_grid, dilation=4)
                                        )
        self.c2 = se2_layers.fAttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                       )
        self.c3 = se2_layers.fAttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                       )
        self.c4 = se2_layers.fAttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                       )
        self.c5 = se2_layers.fAttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                       )
        self.c6 = se2_layers.fAttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                       )
        self.c7 = se2_layers.fAttConvGG(N_in=N_channels, N_out=10        , kernel_size=4         , h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                       )
        # Dropout
        self.dp = nn.Dropout(p)
        # Batch Normalization
        self.bn1 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn2 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn3 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn4 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn5 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn6 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        # Max Pooling
        self.max_pooling = se2_layers.max_pooling_Rn


    def forward(self, x):
        #out = torch.rot90(x, k=1, dims=[-2, -1]) # Sanity check. Equivariance.
        out = self.dp(torch.relu(self.bn1(self.c1(x))))
        out = torch.relu(self.bn2(self.c2(out)))
        out = self.max_pooling(out, kernel_size=2, stride=2, padding=0)

        out = self.dp(torch.relu(self.bn3(self.c3(out))))
        out = self.dp(torch.relu(self.bn4(self.c4(out))))
        out = self.dp(torch.relu(self.bn5(self.c5(out))))
        out = self.dp(torch.relu(self.bn6(self.c6(out))))

        out = self.c7(out)
        out, _ = torch.max(out, dim=-3)
        out = out.view(out.size(0), 10)

        # Visualize
        if False:
            from attgconv.attention_layers import fSpatialAttentionGG
            from attgconv.attention_layers import fSpatialAttention
            import numpy as np
            import matplotlib.pyplot as plt
            inx = 0
            B = 15
            maps = []
            for m in self.modules():
                if isinstance(m, fSpatialAttention):
                    map = m.att_map.cpu().detach()
                    iny = map.shape[-1]
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
            plt.figure()
            plt.imshow(map_0.sum(-3)[B, 0])
            plt.show()
            plt.subplots(1, 4)
            for i in range(4):
                plt.subplot(1, 4, i + 1)
                plt.imshow(maps[-2][B, 0, i, :, :])
            plt.show()
            plt.subplots(1, 4)
            for i in range(4):
                plt.subplot(1, 4, i + 1)
                plt.imshow(self.c6.spatial_attention.att_map.detach().cpu().numpy()[B, 0, i, :, :])
            plt.show()

            # Plot all directions
            cmap = plt.cm.jet
            time_samples = 4
            scale = 10
            z = np.zeros([inx, inx])
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

            # # Plot main direction
            # cmap = plt.cm.jet
            # time_samples = 4
            # z = np.zeros([28, 28])
            # map_max = (map_0 == map_0.max(dim=-3, keepdim=True)[0]).float() * map_0
            # scale = 10
            # for t in range(4):
            #     plt.imshow(map_0.sum(-3)[B, 0])
            #     if t == 0:
            #         plt.quiver(z, map_max[B, 0, t, :, :], color='red', label='0', scale=scale)
            #     if t == 2:
            #         plt.quiver(z, -map_max[B, 0, t, :, :], color=cmap(t / time_samples), label='180', scale=scale)
            #     if t == 1:
            #         plt.quiver(-map_max[B, 0, t, :, :], z, color='cyan', label='90', scale=scale)
            #     if t == 3:
            #         plt.quiver(map_max[B, 0, t, :, :], z, color=cmap(t / time_samples), label='270', scale=scale)
            # plt.legend()
            # plt.show()

        return out


if __name__ == '__main__':
    from experiments.utils import num_params
    # Sanity check
    model = Z2CNN()
    num_params(model)

    model = P4CNN()
    num_params(model)

    model = A_P4CNN()
    model(torch.rand([1,1,28,28]))  # Sanity check
    num_params(model)

    model = A_Sp_P4CNN()
    model(torch.rand([1,1,28,28]))  # Sanity check
    num_params(model)

    model = A_Ch_P4CNN()
    num_params(model)

    model = fA_P4CNN()
    model(torch.rand([1,1,28,28]))  # Sanity check
    num_params(model)