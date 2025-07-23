import argparse
import os
import h5py
import torch
import numpy as np
import importlib

def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(':')[-1] if 'cuda' in args.device  else ''

    #import model using args.model_name

    # Full import path as a string
    full_path = 'spelke_net.inference.spelke_object_discovery.spelke_bench_class.' + args.model_name

    # Split the path into module and class name
    *module_parts, class_name = full_path.split('.')
    module_path = '.'.join(module_parts)

    # Import the module
    module = importlib.import_module(module_path)

    # Get the class
    model = getattr(module, class_name)()

    os.makedirs(args.output_dir, exist_ok=True)

    with h5py.File(args.dataset_path, 'r') as inp_data:

        for img_path in args.img_names:

            im = inp_data[img_path]['rgb'][:]
            segments = inp_data[img_path]['segment'][:]
            centroids = torch.tensor(inp_data[img_path]['centroid'][:])#.to(torch.int32)

            base_name = os.path.splitext(os.path.basename(img_path))[0]

            out_path = os.path.join(args.output_dir, f"{base_name}.h5")

            #check if file already exists and contains predictions
            if os.path.exists(out_path):

                with h5py.File(out_path, 'r') as f:
                    if "segment_pred" in f.keys():
                        print(f"File {out_path} already exists. Skipping...")
                        continue
                    else:
                        print(f"File {out_path} exists but does not contain predictions. Overwriting...")

            #save in a tmp file first in /tmp dir and then move it
            tmp_out_path = os.path.join("/tmp", f"{base_name}.h5")

            with h5py.File(tmp_out_path, "w") as f:

                f.create_dataset("segments", data=segments, compression="gzip")

                all_pred_segs = model.get_all_segmemts(im, centroids.tolist())


                f.create_dataset("probe_points", data=centroids, compression="gzip")
                #save in the file
                all_pred_segs = np.stack(all_pred_segs)
                f.create_dataset("segment_pred", data=all_pred_segs, compression="gzip")
                #gt seg
                f.create_dataset("segment_gt", data=segments, compression="gzip")
                #image
                f.create_dataset("image", data=im, compression="gzip")

            #move file to final location using os system mv
            os.system(f"mv {tmp_out_path} {out_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default='SpelkeNetModel1B', type=str)

    parser.add_argument("--dataset_path", type=str,default="./datasets/spelke_bench.h5")

    parser.add_argument("--img_names", type=str, nargs='+', default=['entityseg_1_image2926'])

    parser.add_argument("--output_dir", type=str, required=True)
    #device
    parser.add_argument("--device", type=str, default='cuda:0')

    args = parser.parse_args()

    main(args)
