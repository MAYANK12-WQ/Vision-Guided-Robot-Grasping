"""
Grasp Quality Prediction Network

CNN architecture for predicting grasp quality from RGB-D images.
Outputs grasp pose: [x, y, angle, width, quality_score]
"""

import torch
import torch.nn as nn
import torchvision.models as models


class GraspNet(nn.Module):
    """
    Grasp quality prediction network.

    Architecture:
        ResNet-18 backbone (pretrained on ImageNet)
        → Feature extraction
        → Grasp detection head
        → Output: grasp parameters

    Args:
        input_channels (int): Number of input channels (4 for RGB-D)
        pretrained (bool): Use pretrained ResNet weights
    """

    def __init__(self, input_channels=4, pretrained=True):
        super(GraspNet, self).__init__()

        # ResNet-18 backbone
        resnet = models.resnet18(pretrained=pretrained)

        # Modify first conv layer for RGB-D input
        if input_channels != 3:
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
            # Initialize new channels with small random weights
            if pretrained:
                pretrained_weight = resnet.conv1.weight.data
                self.conv1.weight.data[:, :3, :, :] = pretrained_weight
                self.conv1.weight.data[:, 3:, :, :] = torch.randn(64, input_channels-3, 7, 7) * 0.01
        else:
            self.conv1 = resnet.conv1

        # Use ResNet layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Grasp detection head
        self.grasp_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5)  # [x, y, angle, width, quality]
        )

        # Initialize grasp head
        self._init_grasp_head()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input RGB-D image [B, 4, H, W]

        Returns:
            torch.Tensor: Grasp parameters [B, 5]
                - x, y: normalized grasp center (0-1)
                - angle: grasp rotation (-π to π)
                - width: gripper opening (0-1 normalized)
                - quality: grasp quality score (0-1)
        """
        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Grasp prediction
        grasp_params = self.grasp_head(x)

        # Apply activations
        grasp_params[:, 0:2] = torch.sigmoid(grasp_params[:, 0:2])  # x, y (0-1)
        grasp_params[:, 2] = torch.tanh(grasp_params[:, 2]) * 3.14159  # angle (-π to π)
        grasp_params[:, 3] = torch.sigmoid(grasp_params[:, 3])  # width (0-1)
        grasp_params[:, 4] = torch.sigmoid(grasp_params[:, 4])  # quality (0-1)

        return grasp_params

    def _init_grasp_head(self):
        """Initialize grasp head weights."""
        for m in self.grasp_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_num_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PixelwiseGraspNet(nn.Module):
    """
    Pixelwise grasp quality prediction (dense prediction).

    Predicts grasp quality for each pixel in the image.
    Useful for grasp sampling and heatmap visualization.
    """

    def __init__(self, input_channels=4):
        super(PixelwiseGraspNet, self).__init__()

        # Encoder (downsampling)
        self.enc1 = self._conv_block(input_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Decoder (upsampling)
        self.dec4 = self._upconv_block(512, 256)
        self.dec3 = self._upconv_block(256 + 256, 128)  # +skip connection
        self.dec2 = self._upconv_block(128 + 128, 64)
        self.dec1 = self._upconv_block(64 + 64, 32)

        # Output heads
        self.quality_head = nn.Conv2d(32, 1, kernel_size=1)  # Grasp quality
        self.angle_head = nn.Conv2d(32, 1, kernel_size=1)    # Grasp angle
        self.width_head = nn.Conv2d(32, 1, kernel_size=1)    # Gripper width

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input [B, 4, H, W]

        Returns:
            dict: {
                'quality': [B, 1, H, W],
                'angle': [B, 1, H, W],
                'width': [B, 1, H, W]
            }
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool2d(e1, 2))
        e3 = self.enc3(nn.functional.max_pool2d(e2, 2))
        e4 = self.enc4(nn.functional.max_pool2d(e3, 2))

        # Decoder with skip connections
        d4 = self.dec4(e4)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        # Output predictions
        quality = torch.sigmoid(self.quality_head(d1))
        angle = torch.tanh(self.angle_head(d1)) * 3.14159
        width = torch.sigmoid(self.width_head(d1))

        return {
            'quality': quality,
            'angle': angle,
            'width': width
        }

    def _conv_block(self, in_ch, out_ch):
        """Convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _upconv_block(self, in_ch, out_ch):
        """Upsampling block."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


def test_models():
    """Test grasp networks."""
    print("Testing GraspNet...")
    model1 = GraspNet(input_channels=4, pretrained=False)
    x = torch.randn(2, 4, 224, 224)
    out = model1(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {model1.get_num_parameters():,}")

    print("\nTesting PixelwiseGraspNet...")
    model2 = PixelwiseGraspNet(input_channels=4)
    out_dict = model2(x)
    print(f"Quality map: {out_dict['quality'].shape}")
    print(f"Angle map: {out_dict['angle'].shape}")
    print(f"Width map: {out_dict['width'].shape}")


if __name__ == "__main__":
    test_models()
