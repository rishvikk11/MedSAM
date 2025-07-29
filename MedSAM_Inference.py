# -*- coding: utf-8 -*-

"""
usage example:
python MedSAM_Inference.py -i assets/img_demo.png -o ./ --box "[95,255,190,350]"

"""

# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse
import glob
from tqdm import tqdm
import json


# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

def preprocess_image(img_path, device):
    """Preprocess single image for MedSAM inference"""
    # Load numpy array
    if img_path.endswith('.npy'):
        img_np = np.load(img_path)
    else:
        img_np = io.imread(img_path)
    
    # Handle different numpy array formats
    if len(img_np.shape) == 2:
        # Grayscale image - convert to 3 channel
        # Normalize to 0-255 range if needed
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    elif len(img_np.shape) == 3 and img_np.shape[-1] == 1:
        # Single channel with explicit dimension
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        img_3c = np.repeat(img_np, 3, axis=-1)
    else:
        img_3c = img_np
    
    H, W, _ = img_3c.shape
    
    # Resize to 1024x1024
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    
    # Convert to tensor
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )
    
    return img_3c, img_1024_tensor, H, W

def get_default_box(H, W, margin_ratio=0.1):
    """Generate a default bounding box covering most of the image"""
    margin_h = int(H * margin_ratio)
    margin_w = int(W * margin_ratio)
    return np.array([[margin_w, margin_h, W - margin_w, H - margin_h]])

def calculate_metrics(pred_mask, gt_mask):
    """Calculate Dice and IoU metrics"""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    # Intersection and Union
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # Dice coefficient
    dice = (2.0 * intersection) / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0.0
    
    # IoU (Jaccard)
    iou = intersection / union if union > 0 else 0.0
    
    return {
        'dice': float(dice),
        'iou': float(iou)
    }

def load_ground_truth(gt_path):
    """Load ground truth mask from numpy file"""
    if gt_path and os.path.exists(gt_path):
        gt_mask = np.load(gt_path)
        # Ensure binary mask
        if gt_mask.max() > 1:
            gt_mask = (gt_mask > 0).astype(np.uint8)
        return gt_mask.astype(np.uint8)
    return None
import numpy as np

def get_bounding_box(mask):
    """Generate [x0, y0, x1, y1] bounding box from binary mask."""
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return [0, 0, mask.shape[1], mask.shape[0]]  # fallback full image
    x0, x1 = x_indices.min(), x_indices.max()
    y0, y1 = y_indices.min(), y_indices.max()
    return [x0, y0, x1, y1]

def find_matching_gt(img_path, gt_dir):
    """Find matching ground truth file for an image"""
    if not gt_dir or not os.path.exists(gt_dir):
        return None
    
    img_name = Path(img_path).stem

    name = f"{img_name}.npy"
    
    gt_path = os.path.join(gt_dir, name)
    if os.path.exists(gt_path):
        return gt_path
    
    return None
    """Generate a default bounding box covering most of the image"""
    margin_h = int(H * margin_ratio)
    margin_w = int(W * margin_ratio)
    return np.array([[margin_w, margin_h, W - margin_w, H - margin_h]])

def process_batch(model, image_dir, output_dir, gt_dir=None, box_coords=None, save_visualizations=True, max_images=None, random_subset=False):
    """Process a batch of images"""
    
    # Supported image formats (now including .npy)
    image_extensions = ['*.npy', '*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} total images")
    
    # Select subset if requested
    if max_images and max_images < len(image_paths):
        if random_subset:
            np.random.seed(42)  # For reproducible results
            image_paths = np.random.choice(image_paths, max_images, replace=False).tolist()
            print(f"Randomly selected {max_images} images for testing")
        else:
            image_paths = image_paths[:max_images]
            print(f"Using first {max_images} images for testing")
    
    print(f"Processing {len(image_paths)} images")
    
    # Create output directories
    seg_dir = os.path.join(output_dir, 'segmentations')
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(seg_dir, exist_ok=True)
    if save_visualizations:
        os.makedirs(vis_dir, exist_ok=True)
    
    results = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Load and preprocess image
            img_3c, img_1024_tensor, H, W = preprocess_image(img_path, model.device)

            # Load ground truth if available
            gt_path = find_matching_gt(img_path, gt_dir)
            gt_mask = load_ground_truth(gt_path)
            
            # Use provided box or generate default box
            box_np = np.array([get_bounding_box(gt_mask)])
            
            # Scale box to 1024x1024
            box_1024 = box_np / np.array([W, H, W, H]) * 1024
            
            # Get image embedding
            with torch.no_grad():
                image_embedding = model.image_encoder(img_1024_tensor)
            
            # Run inference
            medsam_seg = medsam_inference(model, image_embedding, box_1024, H, W)
            
            # Calculate metrics if ground truth is available
            metrics = {}
            if gt_mask is not None:
                # Resize ground truth to match prediction if needed
                if gt_mask.shape != medsam_seg.shape:
                    gt_mask = transform.resize(gt_mask, medsam_seg.shape, order=0, preserve_range=True).astype(np.uint8)
                metrics = calculate_metrics(medsam_seg, gt_mask)
            
            # Save segmentation mask
            img_name = Path(img_path).stem
            seg_path = os.path.join(seg_dir, f"seg_{img_name}.npy")
            np.save(seg_path, medsam_seg)
            
            # Also save as PNG for visualization
            seg_png_path = os.path.join(seg_dir, f"seg_{img_name}.png")
            io.imsave(seg_png_path, (medsam_seg * 255).astype(np.uint8), check_contrast=False)
            
            # Save visualization
            if save_visualizations:
                num_plots = 4 if gt_mask is not None else 3
                fig, ax = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
                
                # Original image
                ax[0].imshow(img_3c, cmap='gray' if len(img_3c.shape) == 2 else None)
                ax[0].set_title("Original CT Image")
                ax[0].axis('off')
                
                # Image with bounding box
                ax[1].imshow(img_3c, cmap='gray' if len(img_3c.shape) == 2 else None)
                show_box(box_np[0], ax[1])
                ax[1].set_title("Image with Bounding Box")
                ax[1].axis('off')
                
                # Segmentation result
                ax[2].imshow(img_3c, cmap='gray' if len(img_3c.shape) == 2 else None)
                show_mask(medsam_seg, ax[2])
                show_box(box_np[0], ax[2])
                title = "MedSAM Segmentation"
                if metrics:
                    title += f"\nDice: {metrics['dice']:.3f}, IoU: {metrics['iou']:.3f}"
                ax[2].set_title(title)
                ax[2].axis('off')
                
                # Ground truth comparison if available
                if gt_mask is not None:
                    ax[3].imshow(img_3c, cmap='gray' if len(img_3c.shape) == 2 else None)
                    show_mask(gt_mask, ax[3], random_color=True)
                    ax[3].set_title("Ground Truth")
                    ax[3].axis('off')
                
                plt.tight_layout()
                vis_path = os.path.join(vis_dir, f"vis_{img_name}.png")
                plt.savefig(vis_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            # Calculate basic statistics
            seg_area = np.sum(medsam_seg > 0)
            total_area = H * W
            seg_ratio = seg_area / total_area
            
            results.append({
                'image_path': img_path,
                'image_name': img_name,
                'segmentation_path': seg_path,
                'gt_path': gt_path,
                'segmented_pixels': int(seg_area),
                'total_pixels': int(total_area),
                'segmentation_ratio': float(seg_ratio),
                'image_size': [H, W],
                'bounding_box': box_np[0].tolist(),
                'metrics': metrics
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    # Save results summary
    results_path = os.path.join(output_dir, 'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} images successfully")
    print(f"Results saved to: {output_dir}")
    print(f"Summary statistics saved to: {results_path}")
    
    return results
# %% load model and image
parser = argparse.ArgumentParser(
    description="run inference on testing set based on MedSAM"
)
parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    default="data/npy/CT_Abd/imgs/",
    help="path to the data folder",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default="assets/results/",
    help="path to the segmentation folder",
)
parser.add_argument(
    "-gt", 
    "--gt_dir", 
    type=str, 
    default="data/npy/CT_Abd/gts/",
    help="Directory containing ground truth masks (.npy files)"
)
parser.add_argument(
    "--box",
    type=str,
    default=None,
    help="bounding box of the segmentation target",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    required = True,
    help="path to the trained model",
)
parser.add_argument("--no_vis", action="store_true",
                       help="Skip saving visualization images")
parser.add_argument("--max_images", type=int, default=50,
                       help="Maximum number of images to process (default: process all)")
parser.add_argument("--random", action="store_true",
                       help="Select random subset instead of first N images")
args = parser.parse_args()

# Load model
print("Loading MedSAM model...")
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
state_dict = torch.load(args.checkpoint)
model_state_dict = state_dict["model"]
torch.save(model_state_dict, "medsam_model_weights_only.pth")
medsam_model = sam_model_registry["vit_b"](checkpoint="medsam_model_weights_only.pth")
#medsam_model.load_state_dict(checkpoint["model"], strict=False)
medsam_model = medsam_model.to(device)
medsam_model.eval()
print(f"Model loaded on {device}")

# Parse bounding box if provided
box_coords = None
if args.box:
    try:
        box_coords = [int(x) for x in args.box.strip('[]').split(',')]
        if len(box_coords) != 4:
            raise ValueError("Box must have 4 coordinates")
    except:
        print("Invalid box format. Using default box.")
        box_coords = None

# Process batch
results = process_batch(
    medsam_model, 
    args.input_dir, 
    args.output_dir, 
    args.gt_dir,
    box_coords, 
    save_visualizations=not args.no_vis,
    max_images=args.max_images,
    random_subset=args.random
)

# Print summary statistics
if results:
    seg_ratios = [r['segmentation_ratio'] for r in results]
    print("\n=== Summary Statistics ===")
    print(f"Total images processed: {len(results)}")
    print(f"Average segmentation ratio: {np.mean(seg_ratios):.3f}")
    print(f"Segmentation ratio std: {np.std(seg_ratios):.3f}")
    print(f"Min segmentation ratio: {np.min(seg_ratios):.3f}")
    print(f"Max segmentation ratio: {np.max(seg_ratios):.3f}")
    
    # Print metrics summary if ground truth was provided
    results_with_gt = [r for r in results if r['metrics']]
    if results_with_gt:
        metrics_summary = {}
        for metric in ['dice', 'iou']:
            values = [r['metrics'][metric] for r in results_with_gt]
            metrics_summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        print(f"\n=== Quantitative Evaluation ({len(results_with_gt)} images with GT) ===")
        for metric, stats in metrics_summary.items():
            print(f"{metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                  f"(range: {stats['min']:.3f} - {stats['max']:.3f})")
    else:
        print("\nNo ground truth files found for quantitative evaluation.")




'''device = args.device
medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
medsam_model = medsam_model.to(device)
medsam_model.eval()

img_np = io.imread(args.data_path)
if len(img_np.shape) == 2:
    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
else:
    img_3c = img_np
H, W, _ = img_3c.shape
# %% image preprocessing
img_1024 = transform.resize(
    img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
).astype(np.uint8)
img_1024 = (img_1024 - img_1024.min()) / np.clip(
    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
)  # normalize to [0, 1], (H, W, 3)
# convert the shape to (3, H, W)
img_1024_tensor = (
    torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
)

box_np = np.array([[int(x) for x in args.box[1:-1].split(',')]]) 
# transfer box_np t0 1024x1024 scale
box_1024 = box_np / np.array([W, H, W, H]) * 1024
with torch.no_grad():
    image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
io.imsave(
    join(args.seg_path, "seg_" + os.path.basename(args.data_path)),
    medsam_seg,
    check_contrast=False,
)

# %% visualize results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_3c)
show_box(box_np[0], ax[0])
ax[0].set_title("Input Image and Bounding Box")
ax[1].imshow(img_3c)
show_mask(medsam_seg, ax[1])
show_box(box_np[0], ax[1])
ax[1].set_title("MedSAM Segmentation")
plt.show() '''
