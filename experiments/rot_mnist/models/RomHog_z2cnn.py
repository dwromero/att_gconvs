# torch
import torch
import torch.nn as nn
# built-in
import functools

class RomHog_fA_P4CNN(nn.Module):
    def __init__(self, use_bias=False):
        super(RomHog_fA_P4CNN, self).__init__()

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

        from attgconv.RomHog_attention import fSpatialAttention as sp_RnG
        from attgconv.RomHog_attention import fSpatialAttentionGG

        sp_GG = functools.partial(fSpatialAttentionGG, group=group, input_h_grid=self.h_grid, stride=stride)

        self.c1 = se2_layers.fAttConvRnG(N_in=1        , N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, stride=stride, padding=padding,
                                        spatial_attention=sp_RnG(group=group, h_grid=self.h_grid, N_in=1)
                                        )
        self.c2 = se2_layers.fAttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       spatial_attention=sp_GG(N_in=N_channels)
                                       )
        self.c3 = se2_layers.fAttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       spatial_attention=sp_GG(N_in=N_channels)
                                       )
        self.c4 = se2_layers.fAttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       spatial_attention=sp_GG(N_in=N_channels)
                                       )
        self.c5 = se2_layers.fAttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       spatial_attention=sp_GG(N_in=N_channels)
                                       )
        self.c6 = se2_layers.fAttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       spatial_attention=sp_GG(N_in=N_channels)
                                       )
        self.c7 = se2_layers.fAttConvGG(N_in=N_channels, N_out=10        , kernel_size=4         , h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       spatial_attention=sp_GG(N_in=N_channels)
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

    model = RomHog_fA_P4CNN()
    model(torch.rand([1,1,28,28]))  # Sanity check
    num_params(model)