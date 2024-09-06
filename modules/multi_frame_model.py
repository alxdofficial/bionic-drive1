from transformers import T5ForConditionalGeneration
from torchvision.models import vit_b_32
import torch.nn as nn
import torch
from peft import LoraConfig, get_peft_model, LoftQConfig

import sys
import os

# Add the parent directory of the modules folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.neuralvismem import Brain
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"Trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class DriveVLMT5(nn.Module):

    def __init__(self, config, tokenizer=None):
        super().__init__()
        
        self.tokenizer = tokenizer  # Store the tokenizer

        # Make tokenizer and text model
        if config.lm == 'T5-Base':
            self.model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-base')
        else:
            self.model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-large')

            # For quantization
            loftq_config = LoftQConfig(loftq_bits=8)

            # Create LoRA model
            lora_config = LoraConfig(
                r=config.lora_dim,
                lora_alpha=config.lora_alpha,
                loftq_config=loftq_config,
                lora_dropout=config.lora_dropout,
                bias='none',
                target_modules=['q', 'v']
            )
            self.model = get_peft_model(self.model, lora_config)

        hidden_size = self.model.config.d_model
        print('Trainable Parameters for LM model:')
        print_trainable_parameters(self.model)

        # Create instance for multi-view processor
        self.mvp = self.MultiViewProcessor(config.gpa_hidden_size, hidden_size, config.lm, self.tokenizer, freeze=True)


    class MultiViewProcessor(nn.Module):
        def __init__(self, gpa_hidden_size, hidden_size, lm, tokenizer, freeze=False):
            super().__init__()

            # Store the tokenizer
            self.tokenizer = tokenizer

            # Use the Brain model for image embeddings
            self.img_model = Brain()
            self.lm = lm

            # Modal embedding to distinguish between image and text
            self.modal_embeddings = nn.Embedding(2, hidden_size)
            self.modal_embeddings.weight.data.normal_(mean=0.0, std=0.02)

        def get_img_embedding(self, imgs, cls_tokens):
            # Ensure imgs and cls_tokens are lists
            # print(f"in mvp get img embedding, imgs shape: ", imgs.shape, f"   cls tokens shape {cls_tokens.shape}")
            imgs = [imgs[:,i,:,:,:] for i in range(imgs.shape[1])] if not isinstance(imgs, list) else imgs
            cls_tokens = [cls_tokens[:,i,:] for i in range(cls_tokens.shape[1])] if not isinstance(cls_tokens, list) else cls_tokens

            # Pass images and CLS tokens through the Brain model
            memory_network_activations = self.img_model(imgs, cls_tokens)  # List of (batch_size, hidden_size) tensors

            # Project to VL dimension if using T5-Large
            if self.lm != 'T5-Base':
                memory_network_activations = self.img_projection_layer(memory_network_activations)

            # Add modal type embedding to merged embedding
            memory_network_activations += self.modal_embeddings(
                torch.ones((memory_network_activations.shape[0], memory_network_activations.shape[1]), dtype=torch.int, device=memory_network_activations.device)
            )

            return memory_network_activations

        def forward(self, text_enc, imgs, text_model):
            # Get the text embeddings (batch_size, seq_length, hidden_size)
            text_embeddings = text_model.get_input_embeddings()(text_enc)
            # print(f"in mvp forward, text_embeddings shape: ", text_embeddings.shape)
            num_cls = 3
            # Add num_cls cls token embeddings to the end of text embeddings
            cls_token_id = self.tokenizer.convert_tokens_to_ids('<cls>')  # Use the tokenizer to get the ID
            cls_token_embeds = text_model.get_input_embeddings()(torch.tensor([cls_token_id]*num_cls, device=text_enc.device))
            cls_token_embeds = cls_token_embeds.unsqueeze(0).expand(text_embeddings.size(0), -1, -1)
            text_embeddings = torch.cat([text_embeddings, cls_token_embeds], dim=1)

            # Calculate self-attention using T5 encoder layers
            attention_mask = torch.ones(text_embeddings.size()[:-1], device=text_enc.device)
            outputs = text_model.encoder(inputs_embeds=text_embeddings, attention_mask=attention_mask)
            cls_tokens = outputs.last_hidden_state[:, -3:, :]  # Extract the last 3 tokens (CLS tokens)
            text_embeddings = outputs.last_hidden_state[:, :-3, :]  # Remove the CLS tokens from the sequence

            # Pass images and CLS tokens through Brain model
            img_embeddings = self.get_img_embedding(imgs, cls_tokens)

            # Add modal embeddings to text
            text_embeddings += self.modal_embeddings(torch.zeros((1, text_embeddings.shape[1]), dtype=torch.int,
                                                                 device=device))
            # Concatenate text and image embeddings
            merged_embedding = torch.cat([text_embeddings, img_embeddings], dim=1)

            return merged_embedding

    def forward(self, text_enc, imgs, labels=None):

        # Get the merged embeddings
        merged_embedding = self.mvp(text_enc, imgs, self.model)

        # If training include the labels
        return self.model(inputs_embeds=merged_embedding, labels=labels)

    def generate(self, text_enc, imgs, lidar=None):

        merged_embedding = self.mvp(text_enc, imgs, self.model)

        attention_mask = torch.ones(merged_embedding.shape[:2], dtype=torch.long, device=device)
        decoder_input_ids = torch.ones((merged_embedding.shape[0], 1), dtype=torch.long, device=device)*self.model.config.decoder_start_token_id
        output_ids = self.model.generate(attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, inputs_embeds=merged_embedding, max_length=512, early_stopping=True)

        return output_ids
