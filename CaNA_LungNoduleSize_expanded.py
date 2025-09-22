# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn.functional as F
import json
import nibabel as nib
import torch
import os
import argparse
import os
import json
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F

import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import ball
from skimage.measure import label
from scipy.ndimage import label as cc_label
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import ball
from collections import Counter

def multiple_lesions_corrected(label_np, lesion_label=23, lung_labels=[28,29,30,31,32], change_percent=20):
    """
    Modify multiple lesions with improved anatomically-constrained morphological operations.
    
    For shrinking: fills lesion with lung tissue first, then erodes to target size.
    For growing: expands lesion within lung boundaries to target size.
    
    Args:
        label_np: 3D numpy array with segmentation labels
        lesion_label: Label value for lesions/nodules (default: 23)
        lung_labels: List of lung tissue label values (default: [28,29,30,31,32])
        change_percent: Percentage change (positive for growth, negative for shrinking)
    
    Returns:
        Modified label array with adjusted lesion sizes
    """
    label_np = label_np.copy()
    lesion_mask = (label_np == lesion_label)
    lung_mask = np.isin(label_np, lung_labels)
    cc, num_lesions = cc_label(lesion_mask)

    if num_lesions == 0:
        print("No lesions found.")
        return label_np

    for i in range(1, num_lesions + 1):
        single_lesion_mask = (cc == i)
        original_volume = np.sum(single_lesion_mask)

        if original_volume == 0:
            continue

        print(f"Processing lesion {i}: original volume = {original_volume} voxels")

        # Get dominant neighboring lung label
        dilated = binary_dilation(single_lesion_mask, structure=ball(3))
        border = dilated & (~single_lesion_mask)
        neighbors = label_np[border]
        valid_neighbors = neighbors[np.isin(neighbors, lung_labels)]
        fill_label = Counter(valid_neighbors).most_common(1)[0][0] if len(valid_neighbors) > 0 else 30

        target_volume = int(original_volume * (1 + change_percent / 100.0))
        print(f"Target volume for lesion {i}: {target_volume} voxels ({1 + change_percent / 100.0:.2f}x original)")
        
        current_mask = single_lesion_mask.copy()
        struct = ball(1)

        if change_percent < 0:
            # Shrinking: fill first, then shrink
            label_np[single_lesion_mask] = fill_label

            for _ in range(1000):
                next_mask = binary_erosion(current_mask, structure=struct)
                next_mask = next_mask & lung_mask
                if np.array_equal(next_mask, current_mask):
                    break
                current_mask = next_mask
                if np.sum(current_mask) <= target_volume:
                    break

        else:
            # Growing: keep original, then expand with improved logic
            max_iterations = min(1000, target_volume)  # Reasonable limit
            stuck_count = 0
            
            for iteration in range(max_iterations):
                current_volume = np.sum(current_mask)
                
                # Check if target already reached or exceeded
                if current_volume >= target_volume:
                    print(f"✅ Lesion {i}: target volume reached at {current_volume} voxels (target: {target_volume})")
                    break
                
                # Try to dilate
                next_mask = binary_dilation(current_mask, structure=struct)
                
                # Only keep parts that are within lung boundaries
                valid_expansion = next_mask & lung_mask
                new_volume = np.sum(valid_expansion)
                
                # Check if expansion would exceed target by too much
                if new_volume > target_volume * 1.1:  # Allow 10% overshoot maximum
                    print(f"✅ Lesion {i}: stopping to avoid overshoot. Current: {current_volume}, next would be: {new_volume}, target: {target_volume}")
                    break
                
                # Check if we made progress
                if new_volume == current_volume:
                    stuck_count += 1
                    if stuck_count >= 3:  # Allow some attempts before giving up
                        print(f"⚠️ Lesion {i}: growth stopped by boundaries at {current_volume} voxels (target: {target_volume})")
                        break
                else:
                    stuck_count = 0
                    
                current_mask = valid_expansion
            
            # Final validation
            final_volume = np.sum(current_mask)
            if final_volume < original_volume:
                print(f"❌ Error: Lesion {i} shrunk during growth! Using original mask.")
                current_mask = single_lesion_mask

        # Write updated lesion mask
        label_np[current_mask] = lesion_label
        
        # Final results summary
        final_volume = np.sum(current_mask)
        actual_ratio = final_volume / original_volume if original_volume > 0 else 0
        print(f"Lesion {i} final result: {original_volume} → {final_volume} voxels ({actual_ratio:.2f}x, target was {1 + change_percent / 100.0:.2f}x)")

    return label_np

import numpy as np
from scipy.ndimage import distance_transform_edt, label as cc_label, binary_dilation, label
from skimage.morphology import ball
from collections import Counter

def shrink_lesions_preserve_shape_connectivity(label_np, lesion_label=23, lung_labels=[28,29,30,31,32], shrink_percent=50, min_keep_voxels=10):
    """
    Shrinks lesions labeled as 23 by a precise percent using distance transform.
    Preserves shape and keeps only the largest connected component inside lung.

    Args:
        label_np (np.ndarray): 3D label volume.
        lesion_label (int): Label used for lesions (default: 23).
        lung_labels (list): List of lung region labels (default: 28–32).
        shrink_percent (float): Percentage to shrink (e.g., 50).
        min_keep_voxels (int): Minimum voxels to keep in shrunk lesion.

    Returns:
        np.ndarray: Updated label array.
    """
    label_np = label_np.copy()
    lung_mask = np.isin(label_np, lung_labels)
    lesion_mask = (label_np == lesion_label)
    cc, num_lesions = cc_label(lesion_mask)

    if num_lesions == 0:
        print("No lesions found.")
        return label_np

    for i in range(1, num_lesions + 1):
        lesion_i_mask = (cc == i)
        original_voxels = np.argwhere(lesion_i_mask)

        if len(original_voxels) == 0:
            continue

        original_volume = len(original_voxels)
        target_volume = int(original_volume * (1 - shrink_percent / 100.0))
        target_volume = max(target_volume, min_keep_voxels)  # avoid over-shrinking

        # Compute distance map
        dist_map = distance_transform_edt(lesion_i_mask)

        # Sort voxels: inner ones first
        voxel_indices = np.argwhere(lesion_i_mask)
        distances = dist_map[lesion_i_mask]
        sorted_indices = np.argsort(-distances)  # deepest first
        top_voxels = voxel_indices[sorted_indices[:target_volume]]

        # Fill lesion region with nearby lung label
        dilated = binary_dilation(lesion_i_mask, structure=ball(3))
        border = dilated & (~lesion_i_mask)
        neighbors = label_np[border]
        valid_neighbors = neighbors[np.isin(neighbors, lung_labels)]
        fill_label = Counter(valid_neighbors).most_common(1)[0][0] if len(valid_neighbors) > 0 else 30
        label_np[lesion_i_mask] = fill_label

        # Build mask from top N voxels
        shrunk_mask = np.zeros_like(label_np, dtype=bool)
        for x, y, z in top_voxels:
            if lung_mask[x, y, z]:
                shrunk_mask[x, y, z] = True

        # Preserve only largest connected component
        cc_shrunk, num_cc = label(shrunk_mask)
        if num_cc > 0:
            sizes = [(cc_shrunk == idx).sum() for idx in range(1, num_cc + 1)]
            largest_cc = (cc_shrunk == (np.argmax(sizes) + 1))
            if largest_cc.sum() >= min_keep_voxels:
                label_np[largest_cc] = lesion_label
                print(f"✅ Lesion {i}: shrunk from {original_volume} → {largest_cc.sum()} voxels")
            else:
                print(f"⚠️  Lesion {i} shrunk below min threshold, skipped.")
        else:
            print(f"⚠️  Lesion {i} lost all connectivity, skipped.")

    return label_np



import logging
from datetime import datetime

def augment_and_save_masks_from_json(json_path, dict_to_read, data_root, lunglesion_lbl, scale_percent, mode, save_dir, log_file=None, random_seed=None, prefix="aug_"):
    # Set up logging
    logger = logging.getLogger(__name__)
    if log_file:
        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    # Log the start of processing
    start_time = datetime.now()
    logger.info(f"Starting augmentation process at {start_time}")
    logger.info(f"Parameters: json_path={json_path}, dict_to_read={dict_to_read}, scale_percent={scale_percent}%, mode={mode}")

    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded JSON file with {len(data[dict_to_read])} entries")

    for idx, mask_entry in enumerate(data[dict_to_read]):
        logger.info(f"Processing entry {idx + 1}/{len(data[dict_to_read])}: {mask_entry['label']}")

        mask_path = os.path.join(data_root, mask_entry['label'])
        output_size = mask_entry['dim']

        # Load NIfTI
        nii = nib.load(mask_path)
        mask_data = nii.get_fdata()
        affine = nii.affine
        header = nii.header

        if mode == 'shrink':
            augmented_np = shrink_lesions_preserve_shape_connectivity(mask_data, lesion_label=lunglesion_lbl, lung_labels=[28,29,30,31,32], shrink_percent=scale_percent, min_keep_voxels=10)
        elif mode == 'grow':
            augmented_np = multiple_lesions_corrected(mask_data, lesion_label=lunglesion_lbl, lung_labels=[28,29,30,31,32], change_percent=scale_percent)

        # Compute original and augmented lesion volumes
        original_volume = np.sum(mask_data == lunglesion_lbl)
        augmented_volume = np.sum(augmented_np == lunglesion_lbl)
        volume_ratio = 100 * augmented_volume / original_volume if original_volume > 0 else 0

        logger.info(f"Original lesion volume: {original_volume} voxels")
        logger.info(f"Augmented lesion volume: {augmented_volume} voxels")
        logger.info(f"Volume ratio: {volume_ratio:.2f}% of original")

        # Save with new filename
        base_name = os.path.basename(mask_path)
        new_base_name = prefix + base_name
        new_path = os.path.join(save_dir, new_base_name)

        # Create output directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        augmented_nii = nib.Nifti1Image(augmented_np, affine, header)
        nib.save(augmented_nii, new_path)
        logger.info(f"Augmented and saved: {new_path}")

    # Log completion
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Augmentation process completed at {end_time}")
    logger.info(f"Total processing time: {duration}")
    logger.info(f"Successfully processed {len(data[dict_to_read])} files")

def main():
    parser = argparse.ArgumentParser(description="Augment and save masks from JSON config.")
    parser.add_argument("--json_path", required=True, help="Path to the input JSON file.")
    parser.add_argument("--dict_to_read", required=True, help="Dictionary key to read in JSON.")
    parser.add_argument("--data_root", required=True, help="Root directory for mask files.")
    parser.add_argument("--lunglesion_lbl", type=int, required=True, help="Lung label value.")
    parser.add_argument("--scale_percent", type=int, required=True, help="Lobe label value.")
    parser.add_argument('--mode', type=str, choices=['shrink', 'grow'], required=True, help="Operation to perform: 'shrink' or 'grow'.")
    parser.add_argument("--save_dir", required=True, help="Directory to save augmented masks.")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed (optional).")
    parser.add_argument("--prefix", default="aug_", help="Prefix for output files (optional).")
    parser.add_argument("--log_file", default=None, help="Log file path (optional).")
    
    args = parser.parse_args()

    augment_and_save_masks_from_json(
        json_path=args.json_path,
        dict_to_read=args.dict_to_read,
        data_root=args.data_root,
        lunglesion_lbl=args.lunglesion_lbl,
        scale_percent=args.scale_percent,
        mode=args.mode,
        save_dir=args.save_dir,
        log_file=args.log_file,
        random_seed=args.random_seed,
        prefix=args.prefix
    )

if __name__ == "__main__":
    main()

def improved_grow_logic(label_np, lesion_label=23, lung_labels=[28,29,30,31,32], change_percent=50):
    """
    Improved grow logic with better boundary handling and validation.
    """
    label_np = label_np.copy()
    lesion_mask = (label_np == lesion_label)
    lung_mask = np.isin(label_np, lung_labels)
    cc, num_lesions = cc_label(lesion_mask)

    for i in range(1, num_lesions + 1):
        single_lesion_mask = (cc == i)
        original_volume = np.sum(single_lesion_mask)
        
        if original_volume == 0:
            continue

        target_volume = int(original_volume * (1 + change_percent / 100.0))
        current_mask = single_lesion_mask.copy()
        struct = ball(1)
        
        # Growth with better logic
        max_iterations = min(1000, target_volume)  # Reasonable limit
        stuck_count = 0
        
        for iteration in range(max_iterations):
            # Try to dilate
            next_mask = binary_dilation(current_mask, structure=struct)
            
            # Only keep parts that are within lung boundaries
            valid_expansion = next_mask & lung_mask
            
            # Check if we made progress
            if np.sum(valid_expansion) == np.sum(current_mask):
                stuck_count += 1
                if stuck_count >= 3:  # Allow some attempts
                    print(f"⚠️ Lesion {i}: growth stopped by boundaries at {np.sum(current_mask)} voxels")
                    break
            else:
                stuck_count = 0
                
            current_mask = valid_expansion
            
            # Check if target reached
            if np.sum(current_mask) >= target_volume:
                print(f"✅ Lesion {i}: target volume reached at {np.sum(current_mask)} voxels")
                break
        
        # Final validation
        final_volume = np.sum(current_mask)
        volume_ratio = final_volume / original_volume if original_volume > 0 else 0
        
        if final_volume < original_volume:
            print(f"❌ Error: Lesion {i} shrunk during growth! Using original.")
            current_mask = single_lesion_mask
        
        # Update the label map
        label_np[current_mask] = lesion_label
        
        print(f"Lesion {i}: {original_volume} → {final_volume} voxels ({volume_ratio:.2f}x)")

    return label_np