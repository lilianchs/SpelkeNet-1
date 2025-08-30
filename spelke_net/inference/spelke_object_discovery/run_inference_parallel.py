import argparse
import os
import subprocess
import h5py as h5
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Run segmentation inference on multiple images in parallel across GPUs.")
    parser.add_argument("--gpus", nargs="+", type=int, required=True, help="List of GPU IDs to use.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output .h5 files.")
    parser.add_argument("--script_path", type=str, default="spelke_net/inference/spelke_object_discovery/run_inference.py", help="Path to the script to run.")
    parser.add_argument("--model_name", type=str, required=True, help="model class name")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the HDF5 dataset.")
    parser.add_argument("--num_splits", type=int, default=1, help="Total number of splits for chunking the dataset.")
    parser.add_argument("--split_num", type=int, default=0, help="Index of the current split (used for SLURM-style sharding).")
    return parser.parse_args()

def chunk_list(lst, n):
    return np.array_split(lst, n)

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with h5.File(args.dataset_path, 'r') as f:
        keys = sorted(f.keys())
        np.random.seed(42)
        np.random.shuffle(keys)
        keys = chunk_list(keys, args.num_splits)[args.split_num]

    if len(keys) == 0:
        print("No image keys found in dataset.")
        return
    
    gpu_chunks = chunk_list(keys, len(args.gpus))
    processes = []
    
    for gpu_id, chunk in zip(args.gpus, gpu_chunks):
        if len(chunk) == 0:
            continue
        
        cmd = [
            f"CUDA_VISIBLE_DEVICES={gpu_id}",
            "python", args.script_path,
            "--model_name", args.model_name,
            "--dataset_path", args.dataset_path,
            "--output_dir", args.output_dir,
            "--device", f"cuda:{gpu_id}",
            "--img_names", *chunk
        ]
        
        # Join as string for subprocess
        full_cmd = " ".join(f'"{c}"' if ' ' in c else c for c in cmd)
        
        print(f"Launching on GPU {gpu_id} with {len(chunk)} images...")
        p = subprocess.Popen(full_cmd, shell=True)
        processes.append(p)
    
    for p in processes:
        p.wait()

if __name__ == "__main__":
    main()