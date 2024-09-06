import argparse
import pdb

import torch
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import os
from modules.multi_frame_dataset import MultiFrameDataset
from modules.multi_frame_model import DriveVLMT5
from tqdm import tqdm as progress_bar
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.patches as patches
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





# inference visualization helpers
def extract_coordinates(text):
    pattern = r"<c\d,(CAM_[^,]+),([0-9.]+),([0-9.]+)>"
    matches = re.findall(pattern, text)
    return matches

def plot_images(images, titles, question, generated_answer, true_answer):
    fig, axes = plt.subplots(2, 3, figsize=(25, 20))

    # Map images to specific positions
    positions = {
        'CAM_FRONT': (0, 1),  # CAM FRONT in the center of the top row
        'CAM_FRONT_LEFT': (0, 0),  # CAM FRONT LEFT in the left of the top row
        'CAM_FRONT_RIGHT': (0, 2),  # CAM FRONT RIGHT in the right of the top row
        'CAM_BACK': (1, 1),  # CAM BACK in the center of the bottom row
        'CAM_BACK_LEFT': (1, 0),  # CAM BACK LEFT in the left of the bottom row
        'CAM_BACK_RIGHT': (1, 2),  # CAM BACK RIGHT in the right of the bottom row
    }

    # Extract coordinates from question, generated answer, and true answer
    question_coords = extract_coordinates(question)
    generated_coords = extract_coordinates(generated_answer)
    true_coords = extract_coordinates(true_answer)

    for idx, (img, title) in enumerate(zip(images, titles)):
        row, col = positions[title.replace(' ', '_').upper()]
        ax = axes[row, col]

        # Rescale the image to [0, 1] range
        img = (img - img.min()) / (img.max() - img.min())

        ax.imshow(img.permute(1, 2, 0).cpu().numpy())  # Convert tensor image to numpy and display
        ax.set_title(title)
        ax.axis('off')

        # Draw circles based on the coordinates
        for cam, x, y in question_coords:
            if cam == title.replace(' ', '_').upper():
                ax.add_patch(patches.Circle((float(x), float(y)), 35, edgecolor='lime', facecolor='none', linewidth=4))

    # Add the question and answers below the images
    plt.figtext(0.5, 0.01, f"Question: {question}\nGenerated Answer: {generated_answer}\nTrue Answer: {true_answer}", ha="center", fontsize=12)
    plt.tight_layout()
    plt.show()
    
# memory visualizaiton helpers
def normalize_to_255(weights):
    # Min-max scaling to normalize weights to the range [0, 255]
    min_val = weights.min()
    max_val = weights.max()
    
    # Handle the case where min and max are the same to avoid division by zero
    if min_val == max_val:
        normalized_weights = np.zeros_like(weights)  # All values will be 0
    else:
        normalized_weights = (weights - min_val) / (max_val - min_val) * 255
    
    return normalized_weights.astype(np.uint8)


def visualize_memory(model, frame_number, output_dir="visualizations", mode="hebbian"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model = model.mvp.img_model
    num_memory_networks = len(model.neural_memory_networks)
    num_subnetworks = model.neural_memory_networks[0].num_subnetworks
    num_layers = model.neural_memory_networks[0].num_layers

    fig, axes = plt.subplots(num_memory_networks, num_subnetworks, figsize=(40, 40))
    fig.suptitle(f'Weights Visualization for Frame {frame_number}', fontsize=10)

    for mem_idx, neural_memory_network in enumerate(model.neural_memory_networks):
        for sub_idx, subnetwork in enumerate(neural_memory_network.subnetworks):
            weight_grids = []
            for layer in subnetwork.layers:
                if mode == "hebbian":
                    weights = layer.hebbian_weights.detach().cpu().numpy()
                elif mode == "recurrent":
                    weights = layer.hebbian_recurrent_weights.detach().cpu().numpy()
                else:
                    raise ValueError("visualize parameter must be 'hebbian' or 'recurrent'")
                
                weight_grids.append(weights)
                
            combined_weights = np.concatenate(weight_grids, axis=0)

            # Normalize weights to 0-255
            combined_weights = normalize_to_255(combined_weights)

            ax = axes[mem_idx, sub_idx]
            img = ax.imshow(combined_weights, aspect='auto', cmap='viridis')
            ax.set_title(f'Memory {mem_idx+1}, Subnet {sub_idx+1}', fontsize=8)
            ax.tick_params(axis='both', which='both', labelsize=6)
    
    # # Add color legend
    # cbar = fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.95)
    # cbar.set_label('Weight Intensity (0-255)', fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{mode}_weights_frame_{frame_number}.png')
    plt.savefig(output_path)
    plt.close(fig)
    print(f'Saved visualization for frame {frame_number} to {output_path}')
