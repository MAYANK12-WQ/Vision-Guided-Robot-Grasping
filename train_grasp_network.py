"""
Training script for grasp quality prediction network

Trains on Cornell Grasp Dataset or synthetic data.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from models.grasp_net import GraspNet


class GraspLoss(nn.Module):
    """Multi-task loss for grasp prediction."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        """
        Calculate grasp loss.

        Args:
            pred: [B, 5] - [x, y, angle, width, quality]
            target: [B, 5]
        """
        # Position loss (x, y)
        loss_pos = self.mse(pred[:, :2], target[:, :2])

        # Angle loss
        loss_angle = self.mse(pred[:, 2], target[:, 2])

        # Width loss
        loss_width = self.mse(pred[:, 3], target[:, 3])

        # Quality loss (binary)
        loss_quality = self.bce(pred[:, 4], target[:, 4])

        # Total loss
        total_loss = loss_pos + 0.5 * loss_angle + 0.5 * loss_width + 2.0 * loss_quality

        return total_loss, {
            'pos': loss_pos.item(),
            'angle': loss_angle.item(),
            'width': loss_width.item(),
            'quality': loss_quality.item()
        }


def create_synthetic_data(batch_size=32):
    """Create synthetic grasp data for demonstration."""
    # Random RGB-D images
    images = torch.rand(batch_size, 4, 224, 224)

    # Random grasp labels
    targets = torch.zeros(batch_size, 5)
    targets[:, 0] = torch.rand(batch_size)  # x
    targets[:, 1] = torch.rand(batch_size)  # y
    targets[:, 2] = (torch.rand(batch_size) - 0.5) * 3.14159 * 2  # angle
    targets[:, 3] = torch.rand(batch_size) * 0.5 + 0.3  # width
    targets[:, 4] = (torch.rand(batch_size) > 0.3).float()  # quality

    return images, targets


def train_epoch(model, optimizer, criterion, device, num_batches=100):
    """Train for one epoch on synthetic data."""
    model.train()
    total_loss = 0.0
    loss_components = {'pos': 0, 'angle': 0, 'width': 0, 'quality': 0}

    progress_bar = tqdm(range(num_batches), desc='Training')

    for _ in progress_bar:
        # Generate synthetic batch
        images, targets = create_synthetic_data(batch_size=32)
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)

        # Calculate loss
        loss, components = criterion(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += components[key]

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return avg_loss, loss_components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-path', type=str, default='./checkpoints')
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_path, exist_ok=True)

    # Model
    model = GraspNet(input_channels=4, pretrained=True)
    model = model.to(device)
    print(f"Parameters: {model.get_num_parameters():,}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = GraspLoss()

    # Training loop
    print("\nTraining started...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, components = train_epoch(model, optimizer, criterion, device)
        print(f"Loss: {train_loss:.4f} | Pos: {components['pos']:.4f} | "
              f"Angle: {components['angle']:.4f} | Quality: {components['quality']:.4f}")

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, os.path.join(args.save_path, f'checkpoint_epoch_{epoch}.pth'))

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
