#!/usr/bin/env python3
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

"""
Improved Lung Nodule Size Augmentation Script

This script shrinks lung nodules in segmentation masks to a specified percentage
of their original volume. It processes NIfTI images based on a JSON configuration
and fills removed nodule areas with surrounding lung lobe tissue.
"""

# Standard library imports
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import traceback
import csv

# Third-party imports
import numpy as np
import nibabel as nib
from scipy.ndimage import (
    binary_erosion, 
    generate_binary_structure, 
    label, 
    distance_transform_edt,
    center_of_mass
)
from skimage.measure import label as sk_label
from skimage.measure import regionprops

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_nodule_properties(mask, affine, voxel_volume):
    """
    Extract properties of each nodule in the mask including world coordinates.
    
    Args:
        mask: Binary 3D mask with nodules
        affine: NIfTI affine matrix for world coordinate transformation
        voxel_volume: Volume of a single voxel in mm³
        
    Returns:
        list: List of dictionaries with nodule properties
    """
    labeled_mask, num_features = label(mask, structure=generate_binary_structure(3, 1))
    
    if num_features == 0:
        return []
    
    nodule_props = []
    
    # Get properties for each nodule
    for i in range(1, num_features + 1):
        # Extract this nodule
        nodule = (labeled_mask == i)
        
        # Calculate center of mass (voxel coordinates)
        center = center_of_mass(nodule)
        
        # Convert to world coordinates
        world_coords = nib.affines.apply_affine(affine, center)
        
        # Calculate volume
        volume_voxels = np.sum(nodule)
        volume_mm3 = volume_voxels * voxel_volume
        
        # Get bounding box in voxel coordinates
        z_indices, y_indices, x_indices = np.where(nodule)
        min_z, max_z = np.min(z_indices), np.max(z_indices)
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        
        # Calculate dimensions in voxel coordinates
        size_z = max_z - min_z + 1
        size_y = max_y - min_y + 1
        size_x = max_x - min_x + 1
        
        # Min and max points in world coordinates
        min_point = nib.affines.apply_affine(affine, [min_z, min_y, min_x])
        max_point = nib.affines.apply_affine(affine, [max_z, max_y, max_x])
        
        # Add properties to list
        nodule_props.append({
            'id': i,
            'volume_voxels': int(volume_voxels),
            'volume_mm3': float(volume_mm3),
            'center_voxel': [float(c) for c in center],
            'center_world': [float(c) for c in world_coords],
            'min_voxel': [int(min_z), int(min_y), int(min_x)],
            'max_voxel': [int(max_z), int(max_y), int(max_x)],
            'size_voxel': [int(size_z), int(size_y), int(size_x)],
            'min_world': [float(c) for c in min_point],
            'max_world': [float(c) for c in max_point],
            'dimensions_world': [float(max_point[i] - min_point[i]) for i in range(3)]
        })
    
    return nodule_props


def compute_lesion_volume(mask, voxel_volume, label=1):
    """
    Compute the volume of a lesion in mm³ and its voxel count.
    
    Args:
        mask: 3D numpy array containing the segmentation mask
        voxel_volume: Volume of a single voxel in mm³
        label: Label value to consider as lesion (default: 1)
        
    Returns:
        tuple: (total_volume_mm3, lesion_voxel_count)
    """
    lesion_voxels = np.sum(mask == label)
    total_volume = lesion_voxels * voxel_volume
    return total_volume, lesion_voxels


def save_nodule_csv(original_props, shrunk_props, output_path, case_id):
    """
    Save nodule properties to a CSV file.
    
    Args:
        original_props: List of dictionaries with original nodule properties
        shrunk_props: List of dictionaries with shrunk nodule properties
        output_path: Path to save CSV file
        case_id: Identifier for the case
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create mappings between original and shrunk nodules
        # In this simple version, we assume nodules maintain their order and no new ones appear
        # A more sophisticated version might use spatial overlap or nearest center distance
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'case_id', 'nodule_id', 
                'original_volume_voxels', 'original_volume_mm3',
                'shrunk_volume_voxels', 'shrunk_volume_mm3', 
                'volume_ratio',
                'original_center_x', 'original_center_y', 'original_center_z',
                'shrunk_center_x', 'shrunk_center_y', 'shrunk_center_z',
                'original_min_x', 'original_min_y', 'original_min_z',
                'original_max_x', 'original_max_y', 'original_max_z',
                'shrunk_min_x', 'shrunk_min_y', 'shrunk_min_z',
                'shrunk_max_x', 'shrunk_max_y', 'shrunk_max_z',
                'original_dim_x', 'original_dim_y', 'original_dim_z',
                'shrunk_dim_x', 'shrunk_dim_y', 'shrunk_dim_z'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Process each original nodule
            for i, orig in enumerate(original_props):
                # Find matching shrunk nodule if it exists
                shrunk = shrunk_props[i] if i < len(shrunk_props) else None
                
                # Calculate volume ratio
                if shrunk:
                    volume_ratio = shrunk['volume_mm3'] / orig['volume_mm3'] if orig['volume_mm3'] > 0 else 0
                else:
                    volume_ratio = 0
                
                row = {
                    'case_id': case_id,
                    'nodule_id': orig['id'],
                    'original_volume_voxels': orig['volume_voxels'],
                    'original_volume_mm3': orig['volume_mm3'],
                    'shrunk_volume_voxels': shrunk['volume_voxels'] if shrunk else 0,
                    'shrunk_volume_mm3': shrunk['volume_mm3'] if shrunk else 0,
                    'volume_ratio': volume_ratio,
                    'original_center_x': orig['center_world'][0],
                    'original_center_y': orig['center_world'][1],
                    'original_center_z': orig['center_world'][2],
                    'shrunk_center_x': shrunk['center_world'][0] if shrunk else 0,
                    'shrunk_center_y': shrunk['center_world'][1] if shrunk else 0,
                    'shrunk_center_z': shrunk['center_world'][2] if shrunk else 0,
                    'original_min_x': orig['min_world'][0],
                    'original_min_y': orig['min_world'][1],
                    'original_min_z': orig['min_world'][2],
                    'original_max_x': orig['max_world'][0],
                    'original_max_y': orig['max_world'][1],
                    'original_max_z': orig['max_world'][2],
                    'shrunk_min_x': shrunk['min_world'][0] if shrunk else 0,
                    'shrunk_min_y': shrunk['min_world'][1] if shrunk else 0,
                    'shrunk_min_z': shrunk['min_world'][2] if shrunk else 0,
                    'shrunk_max_x': shrunk['max_world'][0] if shrunk else 0,
                    'shrunk_max_y': shrunk['max_world'][1] if shrunk else 0,
                    'shrunk_max_z': shrunk['max_world'][2] if shrunk else 0,
                    'original_dim_x': orig['dimensions_world'][0],
                    'original_dim_y': orig['dimensions_world'][1],
                    'original_dim_z': orig['dimensions_world'][2],
                    'shrunk_dim_x': shrunk['dimensions_world'][0] if shrunk else 0,
                    'shrunk_dim_y': shrunk['dimensions_world'][1] if shrunk else 0,
                    'shrunk_dim_z': shrunk['dimensions_world'][2] if shrunk else 0
                }
                
                writer.writerow(row)
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving nodule CSV: {str(e)}")
        return False
    # Label connected components
    struct = generate_binary_structure(3, connectivity)
    labeled, num_features = label(mask, structure=struct)
    
    # Create output mask
    out_mask = np.zeros_like(mask)
    
    # Process each component separately
    for i in range(1, num_features + 1):
        component = (labeled == i)
        component_size = np.sum(component)
        
        # Log component processing
        logger.debug(f"Processing nodule component {i}/{num_features}, size: {component_size} voxels")
        
        # Shrink this component
        shrunk = shrink_component(component, percent, connectivity)
        
        # Add to output mask
        out_mask[shrunk > 0] = 1
        
        # Log shrinkage results
        final_size = np.sum(shrunk)
        achieved_percent = (final_size / component_size * 100) if component_size > 0 else 0
        logger.debug(f"Nodule {i}: Original={component_size}, Shrunk={final_size}, " 
                     f"Achieved={achieved_percent:.1f}% (Target={percent}%)")
    
    return out_mask


def shrink_component(mask, target_percent, connectivity=1, min_voxels=5):
    """
    Shrink a single connected component to a target percentage of its original volume.
    
    Args:
        mask: Binary 3D numpy array (single component)
        target_percent: Target percentage of original volume (0-100)
        connectivity: Connectivity for structural element (1=6-connected, 2=18-connected, 3=26-connected)
        min_voxels: Minimum number of voxels to maintain in very small lesions
        
    Returns:
        Binary 3D numpy array with shrunk component
    """
    # Convert to binary mask and get original volume
    mask = (mask > 0).astype(np.uint8)
    orig_vol = np.sum(mask)
    
    # Return immediately if mask is empty or too small
    if orig_vol == 0:
        return mask
        
    # Very small lesions: ensure we keep at least min_voxels if possible
    if orig_vol <= min_voxels:
        logger.warning(f"Very small lesion detected ({orig_vol} voxels). Maintaining original shape.")
        return mask
        
    # Use a small structuring element for fine control
    struct = generate_binary_structure(3, connectivity)
    temp = mask.copy()
    
    # Calculate the target volume
    target_volume = max(int(orig_vol * target_percent / 100), min_voxels)
    
    # Iteratively erode until we reach target percentage or 100 iterations
    for i in range(1, 100):
        eroded = binary_erosion(temp, structure=struct)
        
        # Check if erosion would make the component disappear or go below target
        eroded_vol = np.sum(eroded)
        if eroded_vol == 0 or eroded_vol < target_volume:
            break
            
        temp = eroded
        curr_vol = eroded_vol
        shrink_ratio = curr_vol / orig_vol * 100
        
        # Stop if we've reached or exceeded the target percentage
        if shrink_ratio <= target_percent:
            break
    
    # If we somehow ended up with nothing, revert to original with warning
    if np.sum(temp) == 0 and orig_vol > 0:
        logger.warning(f"Erosion removed entire component of size {orig_vol}. Using minimal component.")
        return mask
        
    return temp


def shrink_mask_multi_nodule(mask, percent, connectivity=1):
    """
    Shrink multiple nodules in a mask, processing each connected component separately.
    
    Args:
        mask: 3D numpy array containing binary mask
        percent: Target percentage of original volume (0-100)
        connectivity: Connectivity for structural element
        
    Returns:
        3D numpy array with shrunk components
    """
    # Label connected components
    struct = generate_binary_structure(3, connectivity)
    labeled, num_features = label(mask, structure=struct)
    
    # Create output mask
    out_mask = np.zeros_like(mask)
    
    # Process each component separately
    for i in range(1, num_features + 1):
        component = (labeled == i)
        component_size = np.sum(component)
        
        # Log component processing
        logger.debug(f"Processing nodule component {i}/{num_features}, size: {component_size} voxels")
        
        # Shrink this component
        shrunk = shrink_component(component, percent, connectivity)
        
        # Add to output mask
        out_mask[shrunk > 0] = 1
        
        # Log shrinkage results
        final_size = np.sum(shrunk)
        achieved_percent = (final_size / component_size * 100) if component_size > 0 else 0
        logger.debug(f"Nodule {i}: Original={component_size}, Shrunk={final_size}, " 
                     f"Achieved={achieved_percent:.1f}% (Target={percent}%)")
    
    return out_mask


def process_single_mask(mask_path, lunglesion_lbl, scale_percent, save_dir, 
                        lobe_values, prefix="aug_", csv_output=None):
    """
    Process a single mask file: shrink nodules and save augmented result.
    
    Args:
        mask_path: Path to the mask file
        lunglesion_lbl: Label value for lung lesion
        scale_percent: Target percentage for shrinking
        save_dir: Directory to save output
        lobe_values: List of label values representing lung lobes
        prefix: Prefix for output filenames
        csv_output: Path to CSV file for nodule coordinates (optional)
        
    Returns:
        dict: Processing results including nodule properties for CSV
    """
    try:
        # Load NIfTI
        nii = nib.load(mask_path)
        mask_data = nii.get_fdata()
        affine = nii.affine
        header = nii.header

        # Compute voxel volume from NIfTI header
        spacing = header.get_zooms()[:3]  # (x, y, z) in mm
        voxel_volume = np.prod(spacing)   # mm^3
        
        # Create binary lesion mask
        lesion_mask = (mask_data == lunglesion_lbl).astype(np.uint8)
        orig_volume, orig_voxels = compute_lesion_volume(lesion_mask, voxel_volume, label=1)
        
        # Check if there are any lesions
        if orig_voxels == 0:
            logger.warning(f"No lesions found in {mask_path}")
            return {
                "status": "warning",
                "message": "No lesions found",
                "orig_voxels": 0,
                "shrunk_voxels": 0,
                "shrink_ratio": 0
            }
        
        # Extract nodule properties from original mask (for CSV output)
        case_id = os.path.splitext(os.path.basename(mask_path))[0]
        original_props = get_nodule_properties(lesion_mask, affine, voxel_volume) if csv_output else []
        
        # Shrink lesion nodules
        shrunk_mask = shrink_mask_multi_nodule(lesion_mask, scale_percent, connectivity=1)

        # Extract properties from shrunk mask (for CSV output)
        shrunk_props = get_nodule_properties(shrunk_mask, affine, voxel_volume) if csv_output else []
        
        # For CSV output, we'll collect properties and return them
        nodule_data = {
            'case_id': case_id,
            'original_props': original_props,
            'shrunk_props': shrunk_props
        }

        # Compute shrunk lesion volume
        shrunk_volume, shrunk_voxels = compute_lesion_volume(shrunk_mask, voxel_volume, label=1)
        
        # Calculate shrink ratio
        shrink_ratio = 100 * shrunk_volume / orig_volume if orig_volume > 0 else 0
        
        # Prepare output: copy and fill, then set shrunken lesion voxels to lesion label
        filled_label = fill_removed_lesion_with_lobe(
            shrunk_mask, lesion_mask, mask_data, lobe_values
        )
        filled_label[shrunk_mask > 0] = lunglesion_lbl

        # Save with new filename
        base_name = os.path.basename(mask_path)
        new_base_name = f"{prefix}{base_name}"
        new_path = os.path.join(save_dir, new_base_name)
        
        # Ensure save directory exists
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        
        # Save augmented mask
        augmented_nii = nib.Nifti1Image(filled_label, affine, header)
        nib.save(augmented_nii, new_path)
        
        return {
            "status": "success",
            "message": f"Saved to {new_path}",
            "orig_voxels": int(orig_voxels),
            "orig_volume_mm3": float(orig_volume),
            "shrunk_voxels": int(shrunk_voxels),
            "shrunk_volume_mm3": float(shrunk_volume),
            "shrink_ratio": float(shrink_ratio),
            "output_path": new_path,
            "nodule_data": nodule_data if csv_output else None
        }
        
    except Exception as e:
        logger.error(f"Error processing {mask_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e),
            "orig_voxels": 0,
            "shrunk_voxels": 0,
            "shrink_ratio": 0
        }


def fill_removed_lesion_with_lobe(shrunk_mask, original_mask, label_img, lobe_values):
    """
    Fill areas where lesion was removed with the nearest lobe label.
    
    Args:
        shrunk_mask: Binary mask of shrunk lesions
        original_mask: Binary mask of original lesions
        label_img: Full segmentation image with all labels
        lobe_values: List of label values representing lung lobes
        
    Returns:
        3D numpy array with filled labels
    """
    # Find voxels that were lesion in original but not in shrunken
    removed = (original_mask > 0) & (shrunk_mask == 0)
    filled_label = label_img.copy()
    
    # Skip if nothing was removed
    if not np.any(removed):
        return filled_label
        
    # Create a mask of all lobe voxels
    lobe_mask = np.isin(label_img, lobe_values)
    
    # Find nearest lobe label for each removed voxel
    try:
        dist, indices = distance_transform_edt(~lobe_mask, return_indices=True)
        
        # Only update the removed lesion voxels
        filled_label[removed] = label_img[tuple(ind[removed] for ind in indices)]
        
        logger.debug(f"Filled {np.sum(removed)} voxels with nearest lobe labels")
    except Exception as e:
        logger.error(f"Error filling removed lesion: {e}")
        # In case of error, keep original labels
    
    return filled_label


def augment_and_save_masks_from_json(json_path, dict_to_read, data_root, lunglesion_lbl, 
                                     scale_percent, save_dir, log_file=None, 
                                     random_seed=None, prefix="aug_", csv_output=None):
    """
    Process multiple masks based on a JSON configuration.
    
    Args:
        json_path: Path to JSON file with mask information
        dict_to_read: Key in JSON dictionary to read
        data_root: Root directory for mask files
        lunglesion_lbl: Label value for lung lesion
        scale_percent: Target percentage for shrinking
        save_dir: Directory to save output
        log_file: Path to log file (optional)
        random_seed: Random seed for reproducibility (optional)
        prefix: Prefix for output filenames
        csv_output: Path to CSV file for nodule coordinates (optional)
        
    Returns:
        dict: Summary of processing results
    """
    # Set up logging
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Log the start of processing
    start_time = datetime.now()
    logger.info(f"Starting augmentation process at {start_time}")
    logger.info(f"Parameters: json_path={json_path}, dict_to_read={dict_to_read}, "
                f"scale_percent={scale_percent}%")
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the JSON file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if dict_to_read not in data:
            raise KeyError(f"Key '{dict_to_read}' not found in JSON file. "
                          f"Available keys: {list(data.keys())}")
            
        logger.info(f"Loaded JSON file with {len(data[dict_to_read])} entries")
    except Exception as e:
        logger.error(f"Error loading JSON file: {str(e)}")
        return {"status": "error", "message": str(e)}
    
    # Process each mask
    results = []
    successful = 0
    warnings = 0
    errors = 0
    total_original_volume = 0
    total_shrunk_volume = 0
    
    # Lung lobe labels
    lobe_values = [28, 29, 30, 31, 32]
    
    # Setup for single CSV output
    all_original_props = []
    all_shrunk_props = []
    all_case_ids = []
    
    for idx, mask_entry in enumerate(data[dict_to_read]):
        try:
            # Get mask path
            mask_path = os.path.join(data_root, mask_entry['label'])
            logger.info(f"Processing entry {idx + 1}/{len(data[dict_to_read])}: {mask_entry['label']}")
            
            # Process this mask
            result = process_single_mask(
                mask_path=mask_path,
                lunglesion_lbl=lunglesion_lbl,
                scale_percent=scale_percent,
                save_dir=save_dir,
                lobe_values=lobe_values,
                prefix=prefix,
                csv_output=csv_output
            )
            
            # Update statistics
            if result["status"] == "success":
                successful += 1
                total_original_volume += result["orig_volume_mm3"]
                total_shrunk_volume += result["shrunk_volume_mm3"]
                
                # Collect nodule data for CSV
                if csv_output and result["nodule_data"]:
                    case_id = result["nodule_data"]["case_id"]
                    orig_props = result["nodule_data"]["original_props"]
                    shrk_props = result["nodule_data"]["shrunk_props"]
                    
                    # Store for later CSV writing
                    for prop in orig_props:
                        all_original_props.append(prop)
                        all_case_ids.append(case_id)
                    
                    for prop in shrk_props:
                        all_shrunk_props.append(prop)
                
                # Log results
                logger.info(f"Original lesion: {result['orig_voxels']} voxels, "
                            f"{result['orig_volume_mm3']:.2f} mm³")
                logger.info(f"Shrunk lesion: {result['shrunk_voxels']} voxels, "
                            f"{result['shrunk_volume_mm3']:.2f} mm³")
                logger.info(f"Shrink ratio: {result['shrink_ratio']:.2f}% of original")
                logger.info(f"Augmented and saved: {result['output_path']}")
                
            elif result["status"] == "warning":
                warnings += 1
                logger.warning(f"Warning processing {mask_path}: {result['message']}")
                
            else:  # status == "error"
                errors += 1
                logger.error(f"Error processing {mask_path}: {result['message']}")
            
            results.append({
                "file": mask_entry['label'],
                **result
            })
            
        except Exception as e:
            logger.error(f"Unexpected error processing entry {idx}: {str(e)}")
            errors += 1
            results.append({
                "file": mask_entry['label'] if 'label' in mask_entry else f"entry_{idx}",
                "status": "error",
                "message": str(e)
            })
    
    # Calculate overall statistics
    overall_shrink_ratio = (
        100 * total_shrunk_volume / total_original_volume 
        if total_original_volume > 0 else 0
    )
    
    # Log completion
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Augmentation process completed at {end_time}")
    logger.info(f"Total processing time: {duration}")
    logger.info(f"Files processed: {len(results)} (Success: {successful}, "
                f"Warnings: {warnings}, Errors: {errors})")
    logger.info(f"Overall volume change: {total_original_volume:.2f} mm³ → "
                f"{total_shrunk_volume:.2f} mm³ ({overall_shrink_ratio:.2f}%)")
    
    # Save combined CSV if requested
    if csv_output and all_original_props:
        try:
            # Make sure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(csv_output)), exist_ok=True)
            
            # Create mappings between original and shrunk nodules
            logger.info(f"Saving combined nodule data to {csv_output}")
            
            # Get matched pairs of original and shrunk nodules
            matched_nodules = []
            for i, (case_id, orig) in enumerate(zip(all_case_ids, all_original_props)):
                # Find matching shrunk nodule if it exists (same index)
                shrunk = all_shrunk_props[i] if i < len(all_shrunk_props) else None
                
                # Add to matched pairs
                matched_nodules.append((case_id, orig, shrunk))
            
            # Save combined CSV
            with open(csv_output, 'w', newline='') as csvfile:
                fieldnames = [
                    'case_id', 'nodule_id', 
                    'original_volume_voxels', 'original_volume_mm3',
                    'shrunk_volume_voxels', 'shrunk_volume_mm3', 
                    'volume_ratio',
                    'original_center_x', 'original_center_y', 'original_center_z',
                    'shrunk_center_x', 'shrunk_center_y', 'shrunk_center_z',
                    'original_min_x', 'original_min_y', 'original_min_z',
                    'original_max_x', 'original_max_y', 'original_max_z',
                    'shrunk_min_x', 'shrunk_min_y', 'shrunk_min_z',
                    'shrunk_max_x', 'shrunk_max_y', 'shrunk_max_z',
                    'original_dim_x', 'original_dim_y', 'original_dim_z',
                    'shrunk_dim_x', 'shrunk_dim_y', 'shrunk_dim_z'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write each nodule
                for case_id, orig, shrunk in matched_nodules:
                    # Calculate volume ratio
                    if shrunk:
                        volume_ratio = shrunk['volume_mm3'] / orig['volume_mm3'] if orig['volume_mm3'] > 0 else 0
                    else:
                        volume_ratio = 0
                    
                    row = {
                        'case_id': case_id,
                        'nodule_id': orig['id'],
                        'original_volume_voxels': orig['volume_voxels'],
                        'original_volume_mm3': orig['volume_mm3'],
                        'shrunk_volume_voxels': shrunk['volume_voxels'] if shrunk else 0,
                        'shrunk_volume_mm3': shrunk['volume_mm3'] if shrunk else 0,
                        'volume_ratio': volume_ratio,
                        'original_center_x': orig['center_world'][0],
                        'original_center_y': orig['center_world'][1],
                        'original_center_z': orig['center_world'][2],
                        'shrunk_center_x': shrunk['center_world'][0] if shrunk else 0,
                        'shrunk_center_y': shrunk['center_world'][1] if shrunk else 0,
                        'shrunk_center_z': shrunk['center_world'][2] if shrunk else 0,
                        'original_min_x': orig['min_world'][0],
                        'original_min_y': orig['min_world'][1],
                        'original_min_z': orig['min_world'][2],
                        'original_max_x': orig['max_world'][0],
                        'original_max_y': orig['max_world'][1],
                        'original_max_z': orig['max_world'][2],
                        'shrunk_min_x': shrunk['min_world'][0] if shrunk else 0,
                        'shrunk_min_y': shrunk['min_world'][1] if shrunk else 0,
                        'shrunk_min_z': shrunk['min_world'][2] if shrunk else 0,
                        'shrunk_max_x': shrunk['max_world'][0] if shrunk else 0,
                        'shrunk_max_y': shrunk['max_world'][1] if shrunk else 0,
                        'shrunk_max_z': shrunk['max_world'][2] if shrunk else 0,
                        'original_dim_x': orig['dimensions_world'][0],
                        'original_dim_y': orig['dimensions_world'][1],
                        'original_dim_z': orig['dimensions_world'][2],
                        'shrunk_dim_x': shrunk['dimensions_world'][0] if shrunk else 0,
                        'shrunk_dim_y': shrunk['dimensions_world'][1] if shrunk else 0,
                        'shrunk_dim_z': shrunk['dimensions_world'][2] if shrunk else 0
                    }
                    
                    writer.writerow(row)
                    
            logger.info(f"Saved {len(matched_nodules)} nodule entries to {csv_output}")
        except Exception as e:
            logger.error(f"Error saving combined CSV: {str(e)}")
    
    # Return summary
    return {
        "status": "completed",
        "total_files": len(results),
        "successful": successful,
        "warnings": warnings,
        "errors": errors,
        "processing_time": str(duration),
        "total_original_volume_mm3": float(total_original_volume),
        "total_shrunk_volume_mm3": float(total_shrunk_volume),
        "overall_shrink_ratio": float(overall_shrink_ratio),
        "results": results
    }


def main():
    """
    Parse command-line arguments and run the augmentation process.
    """
    parser = argparse.ArgumentParser(
        description="Augment and save masks from JSON config by shrinking lung nodules."
    )
    parser.add_argument("--json_path", required=True,
                        help="Path to the input JSON file.")
    parser.add_argument("--dict_to_read", required=True,
                        help="Dictionary key to read in JSON.")
    parser.add_argument("--data_root", required=True,
                        help="Root directory for mask files.")
    parser.add_argument("--lunglesion_lbl", type=int, required=True,
                        help="Lung lesion label value.")
    parser.add_argument("--scale_percent", type=int, required=True,
                        help="Scale percentage for shrinking (0-100).")
    parser.add_argument("--save_dir", required=True,
                        help="Directory to save augmented masks.")
    parser.add_argument("--log_file", required=True,
                        help="Path to the log file.")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed for reproducibility (optional).")
    parser.add_argument("--prefix", default="aug_",
                        help="Prefix for output files (optional).")
    parser.add_argument("--summary_json", default=None,
                        help="Path to save processing summary as JSON (optional).")
    parser.add_argument("--csv_output", default=None,
                        help="Path to CSV file for output of nodule coordinates (optional).")
    
    args = parser.parse_args()
    
    # Validate scale_percent
    if not 0 <= args.scale_percent <= 100:
        logger.error(f"Scale percentage must be between 0 and 100, got {args.scale_percent}")
        return 1
        
    # Run augmentation
    summary = augment_and_save_masks_from_json(
        json_path=args.json_path,
        dict_to_read=args.dict_to_read,
        data_root=args.data_root,
        lunglesion_lbl=args.lunglesion_lbl,
        scale_percent=args.scale_percent,
        save_dir=args.save_dir,
        log_file=args.log_file,
        random_seed=args.random_seed,
        prefix=args.prefix,
        csv_output=args.csv_output
    )
    
    # Save summary if requested
    if args.summary_json:
        try:
            with open(args.summary_json, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Summary saved to {args.summary_json}")
        except Exception as e:
            logger.error(f"Error saving summary: {str(e)}")
    
    # Return success if no critical errors
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
