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

from visualize import plot_images, visualize_memory
from metrics import metrics, tally_metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# main inderence loop
def inference(dloader):
    model.eval()
    visnum = 0

    with torch.no_grad():
        for idx, (q_texts, a_texts, encodings, imgs, labels, img_paths) in progress_bar(enumerate(dloader), total=len(dloader)):

            outputs = model.generate(encodings, imgs)
            text_outputs = [processor.decode(output, skip_special_tokens=True) for output in outputs]

            # # Plot images with titles, question, generated answer, and true answer
            # for i in range(len(text_outputs)):
            #     image_batch = imgs[i]  # Get the image batch
            #     image_paths = img_paths[i]  # Get the corresponding paths
                
            #     # Load and prepare images for display
            #     images = [image_batch[j] for j in range(image_batch.size(0))]
            #     titles = ['CAM FRONT', 'CAM FRONT LEFT', 'CAM FRONT RIGHT', 'CAM BACK', 'CAM BACK LEFT', 'CAM BACK RIGHT']
                   
            #     plot_images(
            #         images, 
            #         titles, 
            #         question=q_texts[i], 
            #         generated_answer=text_outputs[i], 
            #         true_answer=processor.decode(labels[i], skip_special_tokens=True)
            #     )

            # Visualize Hebbian layer weights
            if (idx + 1) % 20 == 0 or  idx == 0:
                visnum += 1
                visualize_memory(model, frame_number=visnum)

            # call metrics fun
            metrics(text_outputs, a_texts)

            # # just early stop for debug
            # if idx == 10:
            #     break
    
    tally_metrics()


def params():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=4, type=int,
                        help="Batch size per GPU/CPU for training and evaluation, defaults to 4.")
    parser.add_argument("--epochs", default=15, type=int,
                        help="Number of epochs to train for, default is 15")
    parser.add_argument('--gpa-hidden-size', default=128, type=int, help='Hidden dimension for Gated Pooling Attention, '
                                                                         'default is 128')
    parser.add_argument('--freeze-lm', action='store_true', help='Freeze LM during training')
    parser.add_argument('--lm', default='T5-Base', choices=['T5-Base', 'T5-Large'], type=str, help='Backbone LM to use, '
                                                                                        'use \'T5-Base\' for T5-Medium')
    parser.add_argument('--lora', action='store_true', help='Perform LoRA finetuning, recommend if '
                                                            'using T5-Large backbone LM')
    parser.add_argument('--lora-dim', default=64, type=int, help='LoRA dimension')
    parser.add_argument('--lora-alpha', default=32, type=int, help='LoRA alpha')
    parser.add_argument('--lora-dropout', default=0.05, type=float, help='LoRA dropout')
    parser.add_argument('--max-len', default=512, type=int, help='Max length for generating sequence')
    parser.add_argument('--num-workers', default=0, type=int, help='# of Workers used by Dataloader')
    parser.add_argument('--model-name', default='ok1', type=str, help='The checkpoint to load from '
                                                                                 'multi_frame_results directory')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    config = params()

    if config.lm == 'T5-Base':
        processor = T5Tokenizer.from_pretrained('google-t5/t5-base')
    else:
        processor = T5Tokenizer.from_pretrained('google-t5/t5-large')

    processor.add_tokens('<')
    # Add the cls token to the tokenizer
    cls_token = '<cls>'
    special_tokens_dict = {'additional_special_tokens': [cls_token]}
    processor.add_special_tokens(special_tokens_dict)
    # Get the token ID for the new CLS token
    cls_token_id = processor.convert_tokens_to_ids(cls_token)
    
    model = DriveVLMT5(config, tokenizer=processor)  # Pass the tokenizer here
    model.load_state_dict(
        torch.load(os.path.join('multi_frame_results', config.model_name,
                                'latest_model.pth')))
    model.to(device)
    
    # Load dataset and dataloader
    test_dset = MultiFrameDataset(
        input_file=os.path.join('data', 'inference',
                                'filtered_sorted_multi_frame_val.json'),
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])
    )
    test_dloader = DataLoader(test_dset, shuffle=False, batch_size=config.batch_size, drop_last=True,
                              collate_fn=test_dset.test_collate_fn)
    
    # Run inference and visualization
    inference(test_dloader)
