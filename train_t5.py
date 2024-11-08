from transformers import T5Tokenizer, TrainingArguments, Trainer
from transformers import LlamaTokenizer, LlamaForCausalLM

from torchvision import transforms
import json
import os
import time
from torch.utils.data import DataLoader
import torch.nn.init as init
import torch
import torch.nn as nn
import argparse
from modules.dataset import Dataset
from modules.model_t5 import print_trainable_parameters, DriveT5VisionModel
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import gc
from visualize import visualize_memory
import random
random.seed(2002)
torch.cuda.manual_seed(2002)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def save_model(model, model_name):
    # Save the model into the designated folder
    path = os.path.join('multi_frame_results', timestr, model_name + '.pth')
    torch.save(model, path)


def val_model(dloader, val_model):
    val_model.eval()
    val_loss = 0

    for idx, (inputs, imgs, labels) in tqdm(enumerate(dloader), total=len(dloader)):
        outputs = val_model(inputs, imgs, labels)
        val_loss += outputs.loss.item()

    return val_loss / len(val_dataloader)


def save_stats(train_loss, val_loss, epochs, lr):
    stats_dict = {
        'losses': losses,
        'val losses': val_losses,
        'min train loss': train_loss,
        'min val loss': val_loss,
        'epochs': epochs,
        'learning rate': lr,
        'LM': 'T5-Base',
    }

    # Save stats into checkpoint
    with open(os.path.join('multi_frame_results', timestr, 'stats.json'), 'w') as f:
        json.dump(stats_dict, f)


def plot_loss(training_loss, val_loss):
    num_epochs = len(training_loss)

    plt.plot(range(1, num_epochs + 1), training_loss, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Num epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('multi_frame_results', timestr, 'loss.png'))


def init_weights(layer):
    if isinstance(layer, nn.Conv2d):
        init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.Linear):
        init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        if layer.bias is not None:
            init.constant_(layer.bias, 0)


def train(train_loss, val_loss, best_model, epochs, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)

    for epoch in range(epochs, config.epochs):
        print('-------------------- EPOCH ' + str(epoch) + ' ---------------------')
        model.train()
        epoch_loss = 0

        for step, (inputs, imgs, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # if step % 100 == 0:
                # print(step)
                # visualize_memory(model, frame_number=i)
            if step % 50 == 0:
                gc.collect()  # Collect garbage to free CPU memory
                torch.cuda.empty_cache()  # Free up GPU memory

            # Zero out gradients for all optimizers
            optimizer.zero_grad()
            # Forward pass through model
            outputs = model(inputs, imgs, labels)
            # Calculate loss
            loss = outputs.loss
            epoch_loss += loss.item()
            # Back-propogate
            loss.backward()
            optimizer.step()

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_mean = param.grad.mean().item()  # Min of current param's gradient
            #         print(name, grad_mean)
            #     else:
            #         print(name, "None")

            if step % config.checkpoint_frequency == 0:
                print(step)
                print('Loss: ' + str(loss.item()))

                # Get the hidden states (output)
                hidden_states = outputs.logits

                # Perform decoding (e.g., greedy decoding)
                outputs = torch.argmax(hidden_states, dim=-1)

                try:
                    text_outputs = [processor.decode(output.to('cpu'), skip_special_tokens=True) for output in outputs]
                    text_questions = [processor.decode(q.to('cpu'), skip_special_tokens=True) for q in inputs]
                    text_labels = [processor.decode(a.to('cpu'), skip_special_tokens=True) for a in labels]
                    print()
                    print('Questions:')
                    print(text_questions)
                    print()
                    print('Generated Answers:')
                    print(text_outputs)
                    print()
                    print('Ground Truth Answers:')
                    print(text_labels)
                except:
                    print("printout qa example errored out")    

        # Get train and val loss per batch
        epoch_train_loss = epoch_loss / len(train_dataloader)
        losses.append(epoch_train_loss)

        epoch_val_loss = val_model(val_dataloader, model)
        val_losses.append(epoch_val_loss)

        # Adjust learning rate scheduler
        scheduler.step()

        print('Training Loss: ' + str(epoch_train_loss))
        print('Validation Loss: ' + str(epoch_val_loss))
        print('---------------------------------------------')

        # Save model and stats for checkpoints
        save_model(model, f'latest_model_{epoch}')
        epochs += 1
        save_stats(train_loss, val_loss, epochs, scheduler.get_last_lr()[0])
        # plot the loss
        plot_loss(losses, val_losses)
    return train_loss, val_loss

def params():

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", default=1e-4, type=float,
                        help="Model learning rate starting point, default is 1e-4.")
    parser.add_argument("--batch-size", default=6, type=int,
                        help="Batch size per GPU/CPU for training and evaluation, defaults to 8.")
    parser.add_argument("--epochs", default=12, type=int,
                        help="Number of epochs to train for, default is 15")
    parser.add_argument('--checkpoint-frequency', default=500, type=int, help='Frequency of showing example outputs')
    parser.add_argument('--load-checkpoint', action='store_true', help='Whether to load a checkpoint from '
                                                                       'multi_frame_results folder')
    parser.add_argument('--checkpoint-file', default='T5-Medium', type=str, help='The checkpoint to load from '
                                                                                 'multi_frame_results directory')
    parser.add_argument('--num-workers', default=0, type=int, help='# of Workers used by Dataloader')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # torch.autograd.set_detect_anomaly(True)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    config = params()

    losses = []
    val_losses = []
    min_train_loss = None
    min_val_loss = None
    best_model = None
    epochs_ran = 0

    # Load processors and models

    processor = T5Tokenizer.from_pretrained('google-t5/t5-base')


    processor.add_tokens('<')
    # Add the cls token to the tokenizer
    cls_token = '<cls>'
    special_tokens_dict = {'additional_special_tokens': [cls_token]}
    processor.add_special_tokens(special_tokens_dict)
    # Get the token ID for the new CLS token
    cls_token_id = processor.convert_tokens_to_ids(cls_token)
    
    model = DriveT5VisionModel(config, tokenizer=processor)  # Pass the tokenizer here
    model.image_processor.visual_embedding_module.apply(init_weights)

    # If multiple GPUs are available, wrap the model in DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)
    print('Trainable Parameters for full model')
    print_trainable_parameters(model)

    train_dset = Dataset(
        input_file=os.path.join('data', 'multi_frame',
                                'multi_frame_train.json'),
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])
    )
    val_dset = Dataset(
        input_file=os.path.join('data', 'multi_frame',
                                'multi_frame_val.json'),
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])
    )

    # Create Dataloaders
    # train_dataloader = DataLoader(train_dset, shuffle=False, batch_size=config.batch_size,
    #                               num_workers=config.num_workers, collate_fn=train_dset.collate_fn, drop_last=True)
    # val_dataloader = DataLoader(val_dset, shuffle=False, batch_size=config.batch_size,
    #                             num_workers=config.num_workers, collate_fn=train_dset.collate_fn, drop_last=True)
    train_dataloader = DataLoader(train_dset, shuffle=True, batch_size=config.batch_size,
                                  num_workers=config.num_workers, collate_fn=train_dset.collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_dset, shuffle=True, batch_size=config.batch_size,
                                num_workers=config.num_workers, collate_fn=train_dset.collate_fn, drop_last=True)


    # Load checkpoint if neccesary:
    if config.load_checkpoint:

        print('Loading model from ' + config.checkpoint_file)

        # Load the model and stats from the checkpoint
        model = torch.load(os.path.join('multi_frame_results', config.checkpoint_file,
                                                          'latest_model_7.pth'))

        with open("multi_frame_results/20241023-214628/stats.json", 'r') as f:
            stats = json.load(f)

        min_train_loss, min_val_loss, losses, val_losses, epochs_ran = stats['min train loss'], stats[
            'min val loss'], stats['losses'], stats['val losses'], stats['epochs']

        print(f'Minimum Training Loss: {min_train_loss}')
        print(f'Training Losses: {losses}')
        print(f'Minimum Validation Loss: {min_val_loss}')
        print(f'Validation Losses: {val_losses}')
        print(f'Epochs ran: {epochs_ran}')
        timestr = config.checkpoint_file
    else:
        checkpoint_path = os.path.join('multi_frame_results', timestr)
        print(f'All model checkpoints and training stats will be saved in {checkpoint_path}')
        os.mkdir(os.path.join('multi_frame_results', timestr))

    # If loading a checkpoint, use the learning rate from the last epoch
    if config.load_checkpoint:
        lr = stats['learning rate']
    else:
        lr = config.learning_rate

    min_train_loss, min_val_loss = train(min_train_loss, min_val_loss, best_model, epochs_ran, lr)


