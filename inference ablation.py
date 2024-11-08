import argparse
import pdb

import torch
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import os
from modules.dataset import Dataset
from modules.model_t5 import DriveT5VisionModel
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
                with torch.no_grad():
                    outputs = model.generate(encodings, imgs)
                    text_outputs = [processor.decode(output, skip_special_tokens=True) for output in outputs]

                    
                    # Plot images with titles, question, generated answer, and true answer
                    
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

                    # call metrics fun
                    metrics(text_outputs, a_texts)

                    # # just early stop for debug
                    # if idx == 10:
                    #     break
    
    tally_metrics()


def params():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=8, type=int,
                        help="Batch size per GPU/CPU for training and evaluation, defaults to 4.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    config = params()

    processor = T5Tokenizer.from_pretrained('google-t5/t5-base')
 

    processor.add_tokens('<')
    # Add the cls token to the tokenizer
    cls_token = '<cls>'
    special_tokens_dict = {'additional_special_tokens': [cls_token]}
    processor.add_special_tokens(special_tokens_dict)
    # Get the token ID for the new CLS token
    cls_token_id = processor.convert_tokens_to_ids(cls_token)
    
    # Load the model from the checkpoint
    model= torch.load(
        "multi_frame_results/20241031-101245/latest_model_11.pth"
    )


    # Move the model to the appropriate device
    model.to(device)

    
    # Load dataset and dataloader
    # test_dset = Dataset(
    #     input_file=os.path.join('data2', 'inference',
    #                             'filtered_sorted_multi_frame_val.json'),
    #     tokenizer=processor,
    #     transform=transforms.Compose([
    #         transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
    #     ])
    # )
    test_dset = Dataset(
        input_file="data/multi_frame/multi_frame_test.json",
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])
    )

    # test_dloader = DataLoader(test_dset, shuffle=False, batch_size=config.batch_size, drop_last=True,
    #                           collate_fn=test_dset.test_collate_fn)
    test_dloader = DataLoader(test_dset, shuffle=True, batch_size=8, drop_last=True,
                              collate_fn=test_dset.test_collate_fn)
    
    # Run inference and visualization
    inference(test_dloader)
 