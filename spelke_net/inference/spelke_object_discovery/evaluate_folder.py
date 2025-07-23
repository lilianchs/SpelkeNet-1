import argparse
import os
import glob
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from spelke_net.utils.segment import evaluate_AP_AR_single_image, plot_segment_overlay



def evaluate_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    h5_files = glob.glob(os.path.join(input_dir, '*.h5'))

    all_AP, all_AR, all_IOU = [], [], []

    for fn in h5_files:
        with h5.File(fn, 'r') as f:
            img = f['image'][:]
            seg_pred = f['segment_pred'][:]
            seg_gt = f['segment_gt'][:]
            probe_points = f['probe_points'][:] if 'probe_points' in f else None

        result = evaluate_AP_AR_single_image(seg_pred, seg_gt)
        all_AP.append(result['AP'])
        all_AR.append(result['AR'])
        all_IOU.append(np.mean(result['iou_mat'].max(-1)))

        base_name = os.path.splitext(os.path.basename(fn))[0]

        # Create a grid of segment overlays
        N = seg_pred.shape[0]
        fig, axs = plt.subplots(2, N+1, figsize=(5 * N, 5*1.8))
        # axs = np.atleast_1d(axs)
        axs[0, 0].imshow(img)
        axs[0, 0].set_title("Input image")
        axs[0, 0].axis('off')

        axs[1, 0].set_visible(False)

        for i in range(1, N+1):
            poke = tuple(probe_points[i-1]) if probe_points is not None else None
            plot_segment_overlay(axs[0, i], img, seg_pred[i-1], poke_xy=poke)
            axs[0, i].set_title(f"Pred Segment {i+1}")

        for i in range(1, N+1):
            poke = tuple(probe_points[i-1]) if probe_points is not None else None
            axs[1, i].imshow(seg_gt[i-1], cmap='gray')
            plot_segment_overlay(axs[1, i], img, seg_gt[i - 1], poke_xy=poke)
            axs[1, i].set_title(f"GT Segment {i}")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_viz.png"), bbox_inches='tight')
        plt.close(fig)

    print(f"AP: {np.mean(all_AP):.4f}")
    print(f"AR: {np.mean(all_AR):.4f}")
    print(f"IoU: {np.mean(all_IOU):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing .h5 files')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save visualizations and metrics')
    args = parser.parse_args()

    evaluate_directory(args.input_dir, args.output_dir)
