"""
PyBullet Simulation Environment for Robot Grasping

Provides a realistic physics simulation environment with:
- Robot arms (Panda, UR5)
- Objects to grasp
- Camera sensors
- Physics dynamics
"""

import pybullet as p
import pybullet_data
import numpy as np
import os


class GraspingEnvironment:
    """
    Simulation environment for robot grasping tasks.

    Args:
        robot_type (str): Type of robot ('panda', 'ur5')
        gui (bool): Enable GUI visualization
        img_size (tuple): Camera image size (width, height)
    """

    def __init__(self, robot_type='panda', gui=True, img_size=(224, 224)):
        self.robot_type = robot_type
        self.img_size = img_size

        # Connect to PyBullet
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        # Setup simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Load robot
        self._load_robot()

        # Object IDs
        self.object_ids = []

        # Camera parameters
        self.camera_target = [0.5, 0, 0.2]
        self.camera_distance = 1.0
        self.camera_yaw = 45
        self.camera_pitch = -30

        print(f"Grasping environment initialized with {robot_type} robot")

    def _load_robot(self):
        """Load robot model."""
        if self.robot_type == 'panda':
            # Panda robot (7-DOF + 2 gripper joints)
            self.robot_id = p.loadURDF("franka_panda/panda.urdf",
                                       basePosition=[0, 0, 0],
                                       useFixedBase=True)
            self.end_effector_index = 11
            self.gripper_joints = [9, 10]
            self.arm_joints = list(range(7))

        elif self.robot_type == 'ur5':
            # UR5 robot (6-DOF)
            # Note: UR5 URDF not in default PyBullet data
            # For demo, we'll use Panda
            print("UR5 not available, using Panda instead")
            self.robot_id = p.loadURDF("franka_panda/panda.urdf",
                                       basePosition=[0, 0, 0],
                                       useFixedBase=True)
            self.end_effector_index = 11
            self.gripper_joints = [9, 10]
            self.arm_joints = list(range(7))

        # Set robot to home position
        self.reset_robot()

    def reset_robot(self):
        """Reset robot to home configuration."""
        home_positions = [0, -0.5, 0, -2.0, 0, 1.5, 0.785]  # Panda home
        for i, pos in enumerate(home_positions):
            if i < len(self.arm_joints):
                p.resetJointState(self.robot_id, self.arm_joints[i], pos)

        # Open gripper
        for joint in self.gripper_joints:
            p.resetJointState(self.robot_id, joint, 0.04)

    def spawn_object(self, position, shape='cube', size=0.05, mass=0.1):
        """
        Spawn an object in the environment.

        Args:
            position (list): [x, y, z] position
            shape (str): 'cube', 'sphere', 'cylinder'
            size (float): Object size
            mass (float): Object mass

        Returns:
            int: Object ID
        """
        if shape == 'cube':
            collision_shape = p.createCollisionShape(p.GEOM_BOX,
                                                     halfExtents=[size/2]*3)
            visual_shape = p.createVisualShape(p.GEOM_BOX,
                                               halfExtents=[size/2]*3,
                                               rgbaColor=np.random.rand(4))

        elif shape == 'sphere':
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE,
                                                     radius=size/2)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE,
                                               radius=size/2,
                                               rgbaColor=np.random.rand(4))

        elif shape == 'cylinder':
            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER,
                                                     radius=size/2,
                                                     height=size)
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER,
                                               radius=size/2,
                                               length=size,
                                               rgbaColor=np.random.rand(4))

        # Create multi-body
        obj_id = p.createMultiBody(baseMass=mass,
                                   baseCollisionShapeIndex=collision_shape,
                                   baseVisualShapeIndex=visual_shape,
                                   basePosition=position)

        self.object_ids.append(obj_id)
        return obj_id

    def get_camera_image(self):
        """
        Capture RGB-D image from camera.

        Returns:
            dict: {
                'rgb': numpy array [H, W, 3],
                'depth': numpy array [H, W],
                'seg': numpy array [H, W]
            }
        """
        width, height = self.img_size

        # Compute view and projection matrices
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width/height,
            nearVal=0.1,
            farVal=100.0
        )

        # Capture image
        (_, _, rgb, depth, seg) = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Process images
        rgb = np.array(rgb).reshape(height, width, 4)[:, :, :3]  # Remove alpha
        depth = np.array(depth).reshape(height, width)
        seg = np.array(seg).reshape(height, width)

        return {
            'rgb': rgb,
            'depth': depth,
            'seg': seg
        }

    def move_to_pose(self, target_pos, target_orn=None):
        """
        Move end-effector to target pose using inverse kinematics.

        Args:
            target_pos (list): [x, y, z] target position
            target_orn (list): [x, y, z, w] target orientation (quaternion)

        Returns:
            bool: Success
        """
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([0, np.pi, 0])

        # Inverse kinematics
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_pos,
            target_orn
        )

        # Move joints
        for i, joint_pos in enumerate(joint_poses[:len(self.arm_joints)]):
            p.setJointMotorControl2(
                self.robot_id,
                self.arm_joints[i],
                p.POSITION_CONTROL,
                targetPosition=joint_pos,
                force=500,
                maxVelocity=1.0
            )

        # Step simulation
        for _ in range(240):
            p.stepSimulation()

        return True

    def grasp(self, close=True):
        """
        Open or close gripper.

        Args:
            close (bool): True to close, False to open
        """
        target_pos = 0.0 if close else 0.04

        for joint in self.gripper_joints:
            p.setJointMotorControl2(
                self.robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=200
            )

        # Wait for gripper
        for _ in range(100):
            p.stepSimulation()

    def check_grasp_success(self, obj_id):
        """
        Check if object was successfully grasped.

        Args:
            obj_id (int): Object ID

        Returns:
            bool: True if grasped
        """
        # Check if object is above table
        obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
        return obj_pos[2] > 0.1  # Object lifted > 10cm

    def reset(self):
        """Reset environment."""
        # Remove all objects
        for obj_id in self.object_ids:
            p.removeBody(obj_id)
        self.object_ids = []

        # Reset robot
        self.reset_robot()

    def close(self):
        """Close simulation."""
        p.disconnect()


if __name__ == "__main__":
    # Test environment
    env = GraspingEnvironment(robot_type='panda', gui=True)

    # Spawn objects
    env.spawn_object([0.5, 0, 0.1], shape='cube', size=0.05)
    env.spawn_object([0.5, 0.1, 0.1], shape='sphere', size=0.04)

    # Capture image
    img_data = env.get_camera_image()
    print(f"RGB shape: {img_data['rgb'].shape}")
    print(f"Depth shape: {img_data['depth'].shape}")

    # Wait
    input("Press Enter to close...")
    env.close()
