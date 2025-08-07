import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os
from pathlib import Path

def load_nifti_data(filepath):
    """Load NIfTI file and return the data array."""
    try:
        nii = nib.load(filepath)
        data = nii.get_fdata()
        return data, nii.header
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

def normalize_ct_data(ct_data, window_center=40, window_width=400):
    """Normalize CT data using window/level settings for soft tissue."""
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2
    
    # Clip values to window range
    windowed = np.clip(ct_data, window_min, window_max)
    
    # Normalize to 0-1 range
    normalized = (windowed - window_min) / (window_max - window_min)
    return normalized

def create_segmentation_overlay(ct_slice, seg_slice):
    """Create colored overlay for segmentation on CT image."""
    # Create RGB image from CT
    ct_normalized = normalize_ct_data(ct_slice)
    rgb_image = np.stack([ct_normalized, ct_normalized, ct_normalized], axis=-1)
    
    # Define colors for each tissue type
    colors = {
        1: [1.0, 0.0, 0.0],    # Muscle - Red
        2: [0.0, 1.0, 0.0],    # Subcutaneous fat - Green  
        3: [0.0, 0.0, 1.0],    # Visceral fat - Blue
        4: [1.0, 1.0, 0.0]     # Muscle fat - Yellow
    }
    
    # Apply colors to segmentation regions with transparency
    alpha = 0.5
    for label, color in colors.items():
        mask = seg_slice == label
        for i in range(3):
            rgb_image[mask, i] = alpha * color[i] + (1 - alpha) * rgb_image[mask, i]
    
    return rgb_image

def visualize_segmentation(input_file, segmentation_file, output_dir=None, slice_step=10):
    """
    Visualize CT scan with segmentation overlay.
    
    Args:
        input_file: Path to original CT NIfTI file
        segmentation_file: Path to segmentation NIfTI file
        output_dir: Directory to save visualization images (optional)
        slice_step: Show every nth slice (default: 10)
    """
    print(f"Loading CT data from: {input_file}")
    ct_data, ct_header = load_nifti_data(input_file)
    if ct_data is None:
        return
    
    print(f"Loading segmentation data from: {segmentation_file}")
    seg_data, seg_header = load_nifti_data(segmentation_file)
    if seg_data is None:
        return
    
    # Get image dimensions
    print(f"CT data shape: {ct_data.shape}")
    print(f"Segmentation data shape: {seg_data.shape}")
    
    # Check if dimensions match
    if ct_data.shape != seg_data.shape:
        print("Warning: CT and segmentation dimensions don't match!")
        return
    
    # Get unique labels in segmentation
    unique_labels = np.unique(seg_data)
    print(f"Segmentation labels found: {unique_labels}")
    
    # Create figure for visualization
    num_slices = ct_data.shape[2]
    slice_indices = range(0, num_slices, slice_step)
    
    # Create legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.5, label='Muscle'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.5, label='Subcutaneous Fat'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.5, label='Visceral Fat'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='yellow', alpha=0.5, label='Muscle Fat')
    ]
    
    # Show a few representative slices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('CT Muscle and Fat Segmentation Visualization', fontsize=16)
    
    # Select 6 representative slices
    selected_slices = [
        num_slices // 6,      # Early slice
        num_slices // 3,      # Early-middle
        num_slices // 2,      # Middle
        2 * num_slices // 3,  # Late-middle
        5 * num_slices // 6,  # Late slice
        num_slices - 10       # Near end
    ]
    
    for idx, slice_idx in enumerate(selected_slices):
        if slice_idx >= num_slices:
            slice_idx = num_slices - 1
            
        row = idx // 3
        col = idx % 3
        
        ct_slice = ct_data[:, :, slice_idx]
        seg_slice = seg_data[:, :, slice_idx]
        
        # Create overlay
        overlay = create_segmentation_overlay(ct_slice, seg_slice)
        
        # Display
        axes[row, col].imshow(overlay, origin='lower')
        axes[row, col].set_title(f'Slice {slice_idx}')
        axes[row, col].axis('off')
        
        # Count pixels for each tissue type in this slice
        muscle_pixels = np.sum(seg_slice == 1)
        sfat_pixels = np.sum(seg_slice == 2)
        vfat_pixels = np.sum(seg_slice == 3)
        mfat_pixels = np.sum(seg_slice == 4)
        
        info_text = f"M:{muscle_pixels} S:{sfat_pixels} V:{vfat_pixels} F:{mfat_pixels}"
        axes[row, col].text(0.02, 0.98, info_text, transform=axes[row, col].transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=8)
    
    # Add legend
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5))
    plt.tight_layout()
    
    # Save visualization if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'segmentation_visualization.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
    
    plt.show()
    
    # Print summary statistics
    print("\n=== Segmentation Summary ===")
    total_voxels = ct_data.size
    muscle_voxels = np.sum(seg_data == 1)
    sfat_voxels = np.sum(seg_data == 2)
    vfat_voxels = np.sum(seg_data == 3)
    mfat_voxels = np.sum(seg_data == 4)
    background_voxels = np.sum(seg_data == 0)
    
    print(f"Total voxels: {total_voxels:,}")
    print(f"Background: {background_voxels:,} ({100*background_voxels/total_voxels:.1f}%)")
    print(f"Muscle: {muscle_voxels:,} ({100*muscle_voxels/total_voxels:.1f}%)")
    print(f"Subcutaneous Fat: {sfat_voxels:,} ({100*sfat_voxels/total_voxels:.1f}%)")
    print(f"Visceral Fat: {vfat_voxels:,} ({100*vfat_voxels/total_voxels:.1f}%)")
    print(f"Muscle Fat: {mfat_voxels:,} ({100*mfat_voxels/total_voxels:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Visualize CT muscle and fat segmentation results')
    parser.add_argument('--input', type=str, default='demo/example.nii.gz', 
                       help='Path to input CT NIfTI file')
    parser.add_argument('--segmentation', type=str, default='outputResults.nii.gz',
                       help='Path to segmentation NIfTI file')
    parser.add_argument('--output', type=str, default='outputResults',
                       help='Output directory for saving visualizations')
    parser.add_argument('--slice-step', type=int, default=10,
                       help='Show every nth slice in detailed view')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    if not os.path.exists(args.segmentation):
        print(f"Error: Segmentation file not found: {args.segmentation}")
        return
    
    # Run visualization
    visualize_segmentation(
        input_file=args.input,
        segmentation_file=args.segmentation,
        output_dir=args.output,
        slice_step=args.slice_step
    )

if __name__ == "__main__":
    main()