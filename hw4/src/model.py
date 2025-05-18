import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and residual connection"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn(self.conv2(out))
        out += residual  # residual connection
        return self.relu(out)


class PromptEncoder(nn.Module):
    """Prompt encoder that transforms degradation prompt to feature space"""

    def __init__(self, prompt_dim=64, out_dim=64):
        super().__init__()
        self.prompt_dim = prompt_dim

        # Learnable prompt parameters for rain and snow
        self.rain_prompt = nn.Parameter(torch.randn(1, prompt_dim))
        self.snow_prompt = nn.Parameter(torch.randn(1, prompt_dim))

        # MLP to project prompt to output dimension
        self.projector = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(prompt_dim*2, out_dim)
        )

    def forward(self, degradation_type):
        """
        Args:
            degradation_type: String indicating 'rain' or 'snow'
        """
        if degradation_type == 'rain':
            prompt = self.rain_prompt
        else:  # snow
            prompt = self.snow_prompt

        return self.projector(prompt)


class FeatureExtractor(nn.Module):
    """Feature extraction backbone"""

    def __init__(self, in_channels=3, base_channels=64, num_blocks=6):
        super().__init__()

        # Initial convolution
        self.init_conv = ConvBlock(in_channels, base_channels)

        # Residual blocks
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(ResidualBlock(base_channels))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Final feature projection
        self.final_conv = nn.Conv2d(
            base_channels, base_channels, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.init_conv(x)
        feat = self.res_blocks(feat)
        feat = self.final_conv(feat)
        return feat


class PromptFusion(nn.Module):
    """Fusion module to merge image features with prompt information"""

    def __init__(self, feature_dim=64, prompt_dim=64):
        super().__init__()

        self.prompt_attn = nn.Sequential(
            nn.Conv2d(prompt_dim, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.fusion_conv = nn.Conv2d(
            feature_dim, feature_dim, kernel_size=3, padding=1)

    def forward(self, image_features, prompt_features):
        # Reshape prompt features to apply attention
        batch_size, _, h, w = image_features.shape
        prompt_attn = prompt_features.view(
            batch_size, -1, 1, 1).expand(-1, -1, h, w)

        # Apply prompt-based attention
        attn_map = self.prompt_attn(prompt_attn)
        attended_features = image_features * attn_map

        # Fuse together
        fused_features = self.fusion_conv(attended_features)
        return fused_features


class Decoder(nn.Module):
    """Decoder module to restore the image"""

    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            ResidualBlock(in_channels),
            ConvBlock(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.decoder(x)


class PromptIR(nn.Module):
    """PromptIR model for unified rain and snow removal"""

    def __init__(self, base_channels=64, prompt_dim=64):
        super().__init__()

        # Feature extraction
        self.feature_extractor = FeatureExtractor(
            in_channels=3, base_channels=base_channels)

        # Prompt encoder
        self.prompt_encoder = PromptEncoder(
            prompt_dim=prompt_dim, out_dim=prompt_dim)

        # Fusion module
        self.fusion_module = PromptFusion(
            feature_dim=base_channels, prompt_dim=prompt_dim)

        # Decoder
        self.decoder = Decoder(in_channels=base_channels, out_channels=3)

    def forward(self, x, degradation_type='rain'):
        """
        Args:
            x: Input degraded image
            degradation_type: 'rain' or 'snow'
        """
        # Extract image features
        image_features = self.feature_extractor(x)

        # Get prompt features
        prompt_features = self.prompt_encoder(degradation_type)

        # Fusion
        fused_features = self.fusion_module(image_features, prompt_features)

        # Decode to get restored image
        restored = self.decoder(fused_features)

        return restored
