import torch
import torch.nn as nn
import torch.nn.functional as F


def create_network(settings):
    return AggressiveNet(settings)


class Conv1dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, bias=True, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        # Same padding for kernel_size=2, stride=1.
        if self.kernel_size == 2 and self.dilation == 1:
            x = F.pad(x, (0, 1))
        else:
            pad_total = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (pad_total // 2, pad_total - pad_total // 2))
        return self.conv(x)



class AggressiveNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._create(input_size=(config.seq_len, config.min_number_fts, 5))

    def _create(self, input_size, has_bias=True, learn_affine=True):
        """Init.
        Args:
            input_size (float): size of input
            has_bias (bool, optional): Defaults to True. Conv1d bias?
            learn_affine (bool, optional): Defaults to True. InstanceNorm affine?
        """
        f = 2.0
        self.pointnet = nn.Sequential(
            nn.Conv1d(5, int(16 * f), kernel_size=1, stride=1, padding=0, bias=has_bias),
            nn.InstanceNorm1d(int(16 * f), eps=1e-5, affine=learn_affine),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv1d(int(16 * f), int(32 * f), kernel_size=1, stride=1, padding=0, bias=has_bias),
            nn.InstanceNorm1d(int(32 * f), eps=1e-5, affine=learn_affine),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv1d(int(32 * f), int(64 * f), kernel_size=1, stride=1, padding=0, bias=has_bias),
            nn.InstanceNorm1d(int(64 * f), eps=1e-5, affine=learn_affine),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv1d(int(64 * f), int(64 * f), kernel_size=1, stride=1, padding=0, bias=has_bias),
        )

        self.fts_mergenet = nn.Sequential(
            Conv1dSame(int(64 * f), int(64 * f), kernel_size=2, bias=has_bias),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            Conv1dSame(int(64 * f), int(32 * f), kernel_size=2, bias=has_bias),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            Conv1dSame(int(32 * f), int(32 * f), kernel_size=2, bias=has_bias),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            Conv1dSame(int(32 * f), int(32 * f), kernel_size=2, bias=has_bias),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )
        self.fts_merge_fc = nn.Linear(int(32 * f) * self.config.seq_len, int(64 * f))

        g = 2.0
        self.states_conv = nn.Sequential(
            Conv1dSame(self.config.state_dim, int(64 * g), kernel_size=2, bias=has_bias),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            Conv1dSame(int(64 * g), int(32 * g), kernel_size=2, bias=has_bias),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            Conv1dSame(int(32 * g), int(32 * g), kernel_size=2, bias=has_bias),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            Conv1dSame(int(32 * g), int(32 * g), kernel_size=2, bias=has_bias),
        )
        self.states_fc = nn.Linear(int(32 * g) * self.config.seq_len, int(64 * g))

        self.control_module = nn.Sequential(
            nn.Linear(int(64 * g) * (2 if self.config.use_fts_tracks else 1), int(64 * g)),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Linear(int(64 * g), int(32 * g)),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Linear(int(32 * g), int(16 * g)),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Linear(int(16 * g), 4),
        )

    def forward_features(self, fts):
        """Feature branch forward pass.

        Args:
            fts: Tensor shaped (batch, seq_len, num_points, 5).
        Returns:
            Tensor shaped (batch, 128) after temporal merging.
        """
        batch_size, seq_len, num_points, feat_dim = fts.shape
        x = fts.reshape(batch_size * seq_len, num_points, feat_dim)
        x = x.permute(0, 2, 1)
        x = self.pointnet(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        x = x.view(batch_size, seq_len, -1)
        x = x.permute(0, 2, 1)
        x = self.fts_mergenet(x)
        x = x.flatten(start_dim=1)
        x = self.fts_merge_fc(x)
        return x

    def forward_states(self, states):
        """State branch forward pass.

        Args:
            states: Tensor shaped (batch, seq_len, 30).
        Returns:
            Tensor shaped (batch, 128).
        """
        x = states.permute(0, 2, 1)
        x = self.states_conv(x)
        x = x.flatten(start_dim=1)
        x = self.states_fc(x)
        return x

    def forward(self, states, fts=None):
        """Full forward pass.

        Args:
            states: Tensor shaped (batch, seq_len, 30).
            fts: Optional tensor shaped (batch, seq_len, num_points, 5).
        Returns:
            Tensor shaped (batch, 4).
        """
        state_emb = self.forward_states(states)
        if self.config.use_fts_tracks:
            if fts is None:
                raise ValueError("fts is required when use_fts_tracks is True")
            fts_emb = self.forward_features(fts)
            emb = torch.cat((fts_emb, state_emb), dim=1)
        else:
            emb = state_emb
        return self.control_module(emb)


if __name__ == "__main__":
    class _Config:
        seq_len = 8
        min_number_fts = 20
        state_dim = 30
        use_fts_tracks = True

    cfg = _Config()
    model = AggressiveNet(cfg)
    batch_size = 4
    num_points = cfg.min_number_fts

    states = torch.randn(batch_size, cfg.seq_len, cfg.state_dim)
    fts = torch.randn(batch_size, cfg.seq_len, num_points, 5)
    output = model(states, fts=fts)
    print("output shape:", tuple(output.shape))
        
