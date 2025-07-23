import argparse

import cv2

from spelke_net.inference.object_manipulation.edit_model import LRAS3D
from spelke_net.inference.object_manipulation.obj_motion_utils import plot_flow_visualizations
import h5py as h5
import os
import torch
import numpy as np

def euler_to_rotation_matrix_deg(yaw_deg, pitch_deg, roll_deg):
    """
    Convert Euler angles (yaw, pitch, roll) in degrees to a rotation matrix.

    Parameters:
        yaw_deg   : rotation about the Z-axis in degrees
        pitch_deg : rotation about the Y-axis in degrees
        roll_deg  : rotation about the X-axis in degrees

    Returns:
        A 3x3 numpy array representing the rotation matrix.
    """
    # Convert degrees to radians
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    # Compute individual rotation matrices
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Combined rotation matrix: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R



def parse_args():
    """
    Parses command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Custom Object Edits with LRAS3D")
    parser.add_argument(
        "--hdf5_file",
        type=str,
        required=True,
        help="Path to the HDF5 file containing image and annotation data."
    )

    parser.add_argument(
        "--segment_hdf5_file",
        type=str,
        default=None,
        help="Path to the HDF5 file containing precomputed segments."
    )

    #spelke_seg or sam_seg
    parser.add_argument(
        "--segment_type",
        type=str,
        default='sam',
        help="Type of segmentation to use: 'spelkenet' or 'sam' or 'GT'. Default is 'sam'."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the visualization output. Default is 'counterfactual_image.png'."
    )

    # Add a store true argument to the parser
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Store True"
    )

    #azimuth
    parser.add_argument(
        "--azimuth",
        type=float,
        default=0.0,
        help="Azimuth angle in degrees."
    )
    #elevation
    parser.add_argument(
        "--elevation",
        type=float,
        default=0.0,
        help="Elevation angle in degrees."
    )
    #tilt
    parser.add_argument(
        "--tilt",
        type=float,
        default=0.0,
        help="Tilt angle in degrees."
    )

    #translation
    parser.add_argument(
        "--tx",
        type=float,
        default=0.0,
        help="Translation in x direction."
    )
    parser.add_argument(
        "--ty",
        type=float,
        default=0.0,
        help="Translation in y direction."
    )
    parser.add_argument(
        "--tz",
        type=float,
        default=0.0,
        help="Translation in z direction."
    )

    #num runs
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of runs to perform."
    )


    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    rollout_config = {"temperature": 0.9, "top_k": 1000, "top_p": 0.9, "rollout_mode": "sequential", "seed": 48}

    object_editor = LRAS3D()

    annotations = args.hdf5_file

    output_path = args.output_dir

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    for runs in range(args.num_runs):

        object_editor.seed = rollout_config['seed'] + runs*10

        with h5.File(annotations, 'r') as f:
            image0 = f['image1'][:]

            three_points_on_ground = f['ground_points_image1'][:]

            K = f['K'][:]

            if args.segment_type == 'GT':
                gt_segment_map = f['GT_segment'][:]


        if args.segment_type != 'GT':
            with h5.File(args.segment_hdf5_file, 'r') as f:
                if args.segment_type == 'sam':
                    gt_segment_map = f['SAM_segment'][:]
                elif args.segment_type == 'spelkenet':
                    gt_segment_map = f['SpelkeNet_segment'][:]
                else:
                    raise ValueError("Invalid segment type. Choose either 'sam' or 'spelke_seg'.")

        image0_downsampled = cv2.resize(image0, (256, 256), interpolation=cv2.INTER_AREA)

        with torch.no_grad():

            elevation = args.elevation
            azimuth = args.azimuth
            tilt = args.tilt

            R_world = euler_to_rotation_matrix_deg(azimuth, tilt, elevation)

            T_world = np.array([args.tx, args.ty, args.tz])

            rgb1_pred = object_editor.run_forward_with_RT(
                image0, three_points_on_ground, R_world, T_world, K, gt_segment_map, condition_rgb=True,
                new_segment_sampling=True,
                condition_from_nvs=True, full_segment_map=False)

            # make a dir to save result images
            if not os.path.exists(output_path + '/viz/'):
                os.makedirs(output_path + '/viz/')

            plot_flow_visualizations(image0_downsampled,
                                     rgb1_pred,
                                     output_path=output_path + '/viz/' +
                                                 annotations.split('/')[-1].split('.')[0] + '_seed_' + str(
                                         runs) + '_obj_motion_elevation_' + str(elevation) + '_azimuth_' + str(azimuth) + '_tilt_' + str(tilt) + '_' +
                                                 str(T_world[0]) + '_' + str(T_world[1]) + '_' + str(T_world[2]) + '.png')

