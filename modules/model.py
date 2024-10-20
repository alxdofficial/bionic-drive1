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
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"Trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.2f}"
    )


class DriveT5VisionModel(nn.Module):

    def __init__(self, config, tokenizer=None):
        super().__init__()

        self.tokenizer = tokenizer  # Store the tokenizer
        self.language_model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-base')

        hidden_size = self.language_model.config.d_model
        print('Trainable Parameters for the Language Model:')
        print_trainable_parameters(self.language_model)

        # Initialize ImageProcessor for handling visual inputs
        self.image_processor = self.ImageProcessor(hidden_size, config.lm, self.tokenizer, freeze=True)

    class ImageProcessor(nn.Module):
        """
        Processes image and text embeddings together and manages multi-modal operations.
        """
        def __init__(self, hidden_size, lm_type, tokenizer, freeze=False):
            super().__init__()

            self.tokenizer = tokenizer
            self.visual_embedding_module = Brain()  # Use Brain model for processing visual data
            self.lm_type = lm_type

            # Embedding to distinguish between modalities (text/image)
            self.modality_embeddings = nn.Embedding(2, hidden_size)
            self.modality_embeddings.weight.data.normal_(mean=0.0, std=0.02)

        def extract_visual_features(self, imgs, cls_tokens):
            # Ensure the input data is correctly split into a list of individual views
            imgs = [imgs[:, i, :, :, :] for i in range(imgs.shape[1])] if not isinstance(imgs, list) else imgs
            cls_tokens = [cls_tokens[:, i, :] for i in range(cls_tokens.shape[1])] if not isinstance(cls_tokens, list) else cls_tokens

            # Pass images and CLS tokens through the Brain model
            visual_features = self.visual_embedding_module(imgs, cls_tokens)

            # Apply linear transformation if using a large LM model
            if self.lm_type != 'T5-Base':
                visual_features = self.project_to_language_model(visual_features)

            # Add modality embeddings for image embeddings
            visual_features += self.modality_embeddings(
                torch.ones(visual_features.shape[:2], dtype=torch.int, device=visual_features.device)
            )

            return visual_features

        def forward(self, text_input, imgs, text_model):
            # Extract text embeddings
            text_embeddings = text_model.get_input_embeddings()(text_input)

            # Create and append CLS tokens
            cls_token_id = self.tokenizer.convert_tokens_to_ids('<cls>')
            cls_embeddings = text_model.get_input_embeddings()(torch.tensor([cls_token_id] * 3, device=text_input.device))
            cls_embeddings = cls_embeddings.unsqueeze(0).expand(text_embeddings.size(0), -1, -1)

            text_embeddings = torch.cat([text_embeddings, cls_embeddings], dim=1)

            # Pass through the T5 encoder
            attention_mask = torch.ones(text_embeddings.size()[:-1], device=text_input.device)
            outputs = text_model.encoder(inputs_embeds=text_embeddings, attention_mask=attention_mask)
            
            # Extract CLS tokens and text embeddings
            cls_tokens = outputs.last_hidden_state[:, -3:, :]
            text_embeddings = outputs.last_hidden_state[:, :-3, :]

            # Extract visual features and combine with CLS tokens
            visual_features = self.extract_visual_features(imgs, cls_tokens)

            # Add modality embedding to text embeddings
            text_embeddings += self.modality_embeddings(
                torch.zeros((1, text_embeddings.shape[1]), dtype=torch.int, device=text_input.device)
            )

            # Concatenate text and visual features
            combined_embedding = torch.cat([text_embeddings, visual_features], dim=1)

            return combined_embedding

    def forward(self, text_input, imgs, labels=None):
        # Get the combined embeddings from the ImageProcessor
        combined_embedding = self.image_processor(text_input, imgs, self.language_model)

        # Pass through the T5 model with the combined embedding
        return self.language_model(inputs_embeds=combined_embedding, labels=labels)

    def generate(self, text_input, imgs):
        # Get the combined embeddings for generation
        combined_embedding = self.image_processor(text_input, imgs, self.language_model)

        # Prepare decoder input IDs and attention mask
        attention_mask = torch.ones(combined_embedding.shape[:2], dtype=torch.long, device=device)
        decoder_input_ids = torch.ones((combined_embedding.shape[0], 1), dtype=torch.long, device=device) * self.language_model.config.decoder_start_token_id

        # Generate outputs using the T5 model
        generated_ids = self.language_model.generate(
            inputs_embeds=combined_embedding,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            max_length=512,
            early_stopping=True
        )

        return generated_ids
