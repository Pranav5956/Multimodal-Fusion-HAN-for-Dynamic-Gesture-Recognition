import argparse
import os
from progress.bar import IncrementalBar
import numpy as np
import cv2
from glob import glob
import shutil
from .utils import utils


def generate_processed_dataset(args: argparse.Namespace, split: str = "train"):
    print()

    output_dir = os.path.join(args.out_dir, split)
    os.mkdir(output_dir)
    print(f"Created \"{split}\" directory!")
    
    shutil.copyfile(
        os.path.join(args.data_dir, f"{split}_gestures.txt"), 
        os.path.join(output_dir, "metadata.txt")
    )

    metadata_list = utils.get_data_list(args.data_dir, split)
    num_metadata = len(metadata_list)
    # total number of images to process
    total_steps = sum([metadata[-1] + 2 for metadata in metadata_list])

    with IncrementalBar("Processing dataset", max=total_steps, suffix="%(percent)d%% | elapsed: %(elapsed)ds | eta: %(eta)ds") as bar:
        for metadata_index, metadata in enumerate(metadata_list):
            # create split folder
            target_dir = utils.format_data_dir(args.data_dir, metadata)
            output_target_dir = os.path.join(output_dir, f"{metadata_index}")
            os.mkdir(output_target_dir)

            # copy skeleton joint files
            skeleton_joints_2d_file = os.path.join(
                target_dir, "skeletons_image.txt")
            skeleton_joints_3d_file = os.path.join(
                target_dir, "skeletons_world.txt")

            bar.message = f'[{metadata_index + 1}/{num_metadata}] Copying "skeletons_image.txt"'
            shutil.copyfile(skeleton_joints_2d_file, os.path.join(
                output_target_dir, "skeletons_2d.txt"))
            bar.next()

            bar.message = f'[{metadata_index + 1}/{num_metadata}] Copying "skeletons_world.txt"'
            shutil.copyfile(skeleton_joints_3d_file, os.path.join(
                output_target_dir, "skeletons_3d.txt"))
            bar.next()

            # get rois for depth maps
            rois = np.loadtxt(os.path.join(
                target_dir, "general_informations.txt"))
            rois = rois[:, 1:].astype(np.uint32)

            # process and store depth and normal maps as RGBA images
            os.mkdir(os.path.join(output_target_dir, "cropped"))
            os.mkdir(os.path.join(output_target_dir, "uncropped"))
            depth_map_filepaths = glob(os.path.join(target_dir, "*_depth.png"))
            total_depth_maps = len(depth_map_filepaths)

            for depth_map_index, depth_map_filepath in enumerate(depth_map_filepaths):
                bar.message = f"[{metadata_index + 1}/{num_metadata}] Processing depth map {depth_map_index + 1}/{total_depth_maps}"

                # uncropped
                depth_map = cv2.imread(depth_map_filepath, cv2.IMREAD_ANYDEPTH)
                depth_features = utils.encode_depth_features(
                    depth_map, rois[depth_map_index])
                cv2.imwrite(os.path.join(output_target_dir, "uncropped",
                            f"{depth_map_index}.png"), depth_features)

                # cropped
                depth_features = utils.encode_depth_features(
                    depth_map, rois[depth_map_index], True)
                cv2.imwrite(os.path.join(output_target_dir, "cropped",
                            f"{depth_map_index}.png"), depth_features)

                bar.next()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir",
                        help="Path to dataset directory", dest="data_dir", default="HandGestureDataset_SHREC2017_dir")
    parser.add_argument("-o", "--out-dir",
                        help="Path to save processed dataset", dest="out_dir", default="HandGestureDataset_SHREC2017_Processed")

    args = parser.parse_args()

    print()
    try:
        os.makedirs(args.out_dir)
        print(f"Created new directory at {args.out_dir}!")
        
        generate_processed_dataset(args, split="train")
        generate_processed_dataset(args, split="test")
    except OSError as e:
        print("Output directory already exists!")
