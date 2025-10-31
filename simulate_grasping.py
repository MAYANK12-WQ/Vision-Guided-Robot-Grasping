"""
Simulate robot grasping with trained network

Demonstrates end-to-end grasping pipeline:
1. Capture camera image
2. Predict grasp pose
3. Execute grasp
4. Evaluate success
"""

import argparse
import torch
import numpy as np

from models.grasp_net import GraspNet
from simulation.environment import GraspingEnvironment


def load_model(model_path, device):
    """Load trained grasp network."""
    model = GraspNet(input_channels=4, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def preprocess_rgbd(rgb, depth):
    """
    Preprocess RGB-D image for network input.

    Args:
        rgb: [H, W, 3] numpy array
        depth: [H, W] numpy array

    Returns:
        torch.Tensor: [1, 4, H, W]
    """
    # Normalize RGB
    rgb = rgb.astype(np.float32) / 255.0

    # Normalize depth
    depth = depth.astype(np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # Stack RGB-D
    rgbd = np.concatenate([rgb, depth[:, :, np.newaxis]], axis=2)

    # Convert to tensor
    rgbd = torch.from_numpy(rgbd).permute(2, 0, 1).unsqueeze(0)

    return rgbd


def execute_grasp(env, model, device):
    """
    Execute one grasp attempt.

    Returns:
        bool: Success
    """
    # 1. Capture image
    img_data = env.get_camera_image()
    rgb = img_data['rgb']
    depth = img_data['depth']

    # 2. Preprocess
    rgbd = preprocess_rgbd(rgb, depth)
    rgbd = rgbd.to(device)

    # 3. Predict grasp
    with torch.no_grad():
        grasp_params = model(rgbd)[0].cpu().numpy()

    x, y, angle, width, quality = grasp_params

    print(f"Predicted grasp: pos=({x:.3f}, {y:.3f}), angle={angle:.3f}, "
          f"width={width:.3f}, quality={quality:.3f}")

    # 4. Convert to 3D coordinates
    # Simple mapping (in practice, use camera calibration)
    target_pos = [0.5 + (x - 0.5) * 0.2, (y - 0.5) * 0.2, 0.15]

    # 5. Move to pre-grasp pose
    pre_grasp_pos = target_pos.copy()
    pre_grasp_pos[2] += 0.1  # 10cm above
    env.move_to_pose(pre_grasp_pos)

    # 6. Open gripper
    env.grasp(close=False)

    # 7. Move down to grasp
    env.move_to_pose(target_pos)

    # 8. Close gripper
    env.grasp(close=True)

    # 9. Lift object
    lift_pos = target_pos.copy()
    lift_pos[2] += 0.2
    env.move_to_pose(lift_pos)

    # 10. Check success
    if len(env.object_ids) > 0:
        success = env.check_grasp_success(env.object_ids[0])
    else:
        success = False

    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--robot', type=str, default='panda')
    parser.add_argument('--num-objects', type=int, default=1)
    parser.add_argument('--num-trials', type=int, default=10)
    parser.add_argument('--gui', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    print("Loading grasp network...")
    model = load_model(args.model_path, device)

    # Create environment
    print("Creating simulation environment...")
    env = GraspingEnvironment(robot_type=args.robot, gui=args.gui)

    # Run trials
    successes = 0

    for trial in range(args.num_trials):
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{args.num_trials}")
        print('='*60)

        # Reset environment
        env.reset()

        # Spawn random objects
        for _ in range(args.num_objects):
            pos = [0.5 + np.random.uniform(-0.1, 0.1),
                   np.random.uniform(-0.1, 0.1),
                   0.05]
            shape = np.random.choice(['cube', 'sphere', 'cylinder'])
            env.spawn_object(pos, shape=shape, size=np.random.uniform(0.03, 0.06))

        # Execute grasp
        success = execute_grasp(env, model, device)

        if success:
            successes += 1
            print("✓ Grasp successful!")
        else:
            print("✗ Grasp failed")

        print(f"Success rate: {successes}/{trial+1} ({100*successes/(trial+1):.1f}%)")

    # Final results
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"Successes: {successes}/{args.num_trials}")
    print(f"Success Rate: {100*successes/args.num_trials:.1f}%")
    print('='*60)

    env.close()


if __name__ == "__main__":
    main()
