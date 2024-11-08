import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(20020109)
torch.cuda.manual_seed(20020109)
torch.backends.cudnn.deterministic = True
# Define the input and output dimensions for internal modules
INTERNAL_DIM = 768
NUM_SUB_NETWORKS = 2
NUM_LAYERS = 4
NUM_MEMORY_NETWORKS = 8
IMG_WIDTH = 1600
IMG_HEIGHT = 900
LONG_SHORT_TERM_PROB = 0.5

# 1. Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, num_layers=5, kernel_size=5, stride=3, dropout=0.3):
        super(FeatureExtractor, self).__init__()
        channels = [16, 64, 128, 256, 512]  # Specified number of channels for each layer
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(input_channels, channels[i], kernel_size=kernel_size, stride=stride))
            if i != num_layers - 1:
                layers.append(nn.BatchNorm2d(channels[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_channels = channels[i]
        self.extractor = nn.Sequential(*layers)

    def forward(self, x):   
        return self.extractor(x)
# 2. Fully Connected Layer
class FullyConnected(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super(FullyConnected, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

# takes in an embedding of peripheral vision and one cls token, predicts x and y coordinate of fovea as percent
# and predicts which image out of 6 to focus on
class FoveaPosPredictor(nn.Module):
    def __init__(self, dropout=0.3):
        super(FoveaPosPredictor, self).__init__()
        hidden_dims = [256, 128, 32]
        self.fc = FullyConnected(INTERNAL_DIM + INTERNAL_DIM, hidden_dims, 8)  # 2 for x,y and 6 for image selection
        self.sigmoid = nn.Sigmoid()  # For x and y coordinates
        self.softmax = nn.Softmax(dim=-1)  # For image selection (6 cameras)

    def forward(self, peripheral_embedding, cls_token):
        # Concatenate peripheral image encoding and CLS token along the last dimension
        combined_input = torch.cat((peripheral_embedding, cls_token), dim=-1)
        output = self.fc(combined_input)
        
        # Separate the output into fovea coordinates and image selection
        fovea_coords = self.sigmoid(output[:, :2])  # 2D coordinates normalized between 0 and 1
        image_selection = self.softmax(output[:, 2:])  # Softmax over 6 cameras

        return fovea_coords, image_selection


# 3. Vision Encoder
class VisionEncoder(nn.Module):
    def __init__(self):
        super(VisionEncoder, self).__init__()
        self.img_width = IMG_WIDTH
        self.img_height = IMG_HEIGHT

        # Feature extractors
        self.fovea_feature_extractor = FeatureExtractor()
        self.peripheral_feature_extractor = FeatureExtractor()
        
        # Fully connected layers
        flattened_dim = 512  # Output from feature extractor after flattening
        hidden_dims = [256]
        output_dim = INTERNAL_DIM
        
        self.fovea_fc = FullyConnected(flattened_dim, hidden_dims, output_dim)
        self.peripheral_fc = FullyConnected(flattened_dim, hidden_dims, output_dim)

    def forward_fovea(self, images, fovea_coords):
        # images: (batch_size, 3, img_height, img_width)
        # fovea_coords: (batch_size, 2) - (x, y) in percentage

        batch_size = images.size(0)
        x_coords = (fovea_coords[:, 0] * self.img_width).int()
        y_coords = (fovea_coords[:, 1] * self.img_height).int()
        # print("x,y: ", x_coords.item(), y_coords.item())
        # Crop the fovea image (centered around the scaled coordinates in the original image)
        crop_size = 512
        half_crop = crop_size // 2

        fovea_images = []
        for i in range(batch_size):
            x, y = x_coords[i], y_coords[i]
            x = min(max(x, half_crop), self.img_width - half_crop)
            y = min(max(y, half_crop), self.img_height - half_crop)
            fovea_image = images[i:i+1, :, y-half_crop:y+half_crop, x-half_crop:x+half_crop]
            # fovea_image: (1, 3, 512, 512)
            fovea_images.append(fovea_image)

        fovea_images = torch.cat(fovea_images, dim=0)
        # fovea_images: (batch_size, 3, 512, 512)

        # Fovea feature extraction
        fovea_features = self.fovea_feature_extractor(fovea_images)
        # fovea_features: (batch_size, channels, h, w) after feature extraction 
        fovea_features = fovea_features.view(fovea_features.size(0), -1)  # Flatten
        # fovea_features: (batch_size, flattened_dim)
        fovea_output = self.fovea_fc(fovea_features)
        # fovea_output: (batch_size, output_dim)

        return fovea_output

    def forward_peripheral(self, image):
        # image: (batch_size, 3, original_height, original_width)
        # print(f"in vision encoder forward_peripheral, image shape: ", image.shape)
        # Resize the entire image
        peripheral_image = F.interpolate(image, size=(512, 512), mode='bilinear', align_corners=False)
        # peripheral_image: (batch_size, 3, 512, 512)

        # Peripheral feature extraction
        peripheral_features = self.peripheral_feature_extractor(peripheral_image)
        # peripheral_features: (batch_size, channels, h, w) after feature extraction, h,w is 1
        peripheral_features = peripheral_features.view(peripheral_features.size(0), -1)  # Flatten
        # peripheral_features: (batch_size, flattened_dim)
        peripheral_output = self.peripheral_fc(peripheral_features)
        # peripheral_output: (batch_size, output_dim)

        return peripheral_output
    
# 10. Attention Module
class AttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query_input, key_value_inputs):
        # Prepare for attention mechanism
        key_value_tensor = torch.stack(key_value_inputs)  # Shape: (num_sources, batch_size, internal_dim)
        # Query input
        query = query_input.unsqueeze(0)  # Shape: (1, batch_size, internal_dim)
        

        # print(key_value_tensor.shape, query.shape)
        # Apply attention with query input, and key_value_inputs as key and value
        attn_output, _ = self.attention(query, key_value_tensor, key_value_tensor)
        
        # Remove the singleton dimension from the output
        attn_output = attn_output.squeeze(0)  # Shape: (batch_size, internal_dim)

        return attn_output

# 11. Brain
class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()

        self.fovea_loc_pred = FoveaPosPredictor()
        self.vision_encoder = VisionEncoder()

        # Attention mechanism
        self.attention = AttentionModule(embed_dim=INTERNAL_DIM, num_heads=4)
        # Linear layer to project positional encoding to match INTERNAL_DIM
        self.positional_encoder = nn.Linear(8, INTERNAL_DIM)  # 8: 2 for coordinates, 6 for one-hot image index
        # LayerNorm for peripheral and fovea encodings after positional encoding
        self.peripheral_norm = nn.LayerNorm(INTERNAL_DIM)
        self.fovea_norm = nn.LayerNorm(INTERNAL_DIM)

    
    def forward(self, imgs, cls_tokens):
        batch_size = imgs[0].size(0)  # Get batch size from the first image

        # Process peripheral vision for all images with positional encoding
        peripheral_encodings = []
        for i, img in enumerate(imgs):
            peripheral_encoding = self.vision_encoder.forward_peripheral(img)

            # Use 0s for fovea coordinates as a placeholder
            zero_fovea_coords = torch.zeros(batch_size, 2, device=img.device)

            # Create a tensor for the selected image index, shaped (batch_size,)
            selected_img_idx = torch.full((batch_size,), i, device=img.device, dtype=torch.long)
            # Add positional encoding and normalize
            pos_encoding = self.add_positional_encoding(zero_fovea_coords, selected_img_idx)
            peripheral_encoding = peripheral_encoding + pos_encoding
            peripheral_encoding = self.peripheral_norm(peripheral_encoding)  # Apply LayerNorm

            peripheral_encodings.append(peripheral_encoding)

        # Combine peripheral encodings with attention
        peripheral_combined = self.attention(peripheral_encodings[0], peripheral_encodings[1:])
        final_outputs = []
        final_outputs.append(peripheral_combined)

        # Process fovea vision using CLS tokens
        for i, cls in enumerate(cls_tokens):
            # Predict fovea coordinates and the target image index
            fovea_coords_logits, img_selector_logits = self.fovea_loc_pred(peripheral_combined, cls)

            img_selector_probs = F.softmax(img_selector_logits, dim=-1)
            selected_img_idx = torch.argmax(img_selector_probs, dim=-1)
            # print("img idx: ", selected_img_idx.item())

            # Select images for each batch element based on predicted index
            selected_img = torch.cat([
                imgs[selected_img_idx[b].item()][b].unsqueeze(0) for b in range(batch_size)
            ], dim=0)
            
            # Process fovea encoding using the selected image
            fovea_encoding = self.vision_encoder.forward_fovea(selected_img, fovea_coords_logits)

            # Add positional encoding and normalize
            fovea_encoding_with_pos = fovea_encoding + self.add_positional_encoding(
                fovea_coords_logits, selected_img_idx
            )
            fovea_encoding_with_pos = self.fovea_norm(fovea_encoding_with_pos)  # Apply LayerNorm

            final_outputs.append(fovea_encoding_with_pos)

        # Stack all memory network outputs along a sequence dimension
        
        final_outputs = torch.stack(final_outputs, dim=1)

        return final_outputs

    def add_positional_encoding(self, fovea_coords, selected_img_idx):
        # Add positional encoding for the fovea coordinates and selected image index
        batch_size = fovea_coords.size(0)

        # Ensure selected_img_idx is a tensor
        selected_img_idx = selected_img_idx.unsqueeze(-1)  # Ensure it has the right dimensions
        
        # One-hot encode the selected image index and remove the extra dimension (from 3D to 2D)
        one_hot_img_idx = F.one_hot(selected_img_idx, num_classes=6).float().squeeze(1)  # Shape: (batch_size, 6)

        # Combine fovea coordinates with the selected image index (one-hot encoded)
        fovea_encoding_pos = torch.cat([
            fovea_coords,  # Fovea coordinates as position (2D)
            one_hot_img_idx  # One-hot encode the selected image index (6 images)
        ], dim=-1)  # Final size: (batch_size, 8)

        # Project positional encoding to the same dimension as INTERNAL_DIM
        positional_encoding_proj = self.positional_encoder(fovea_encoding_pos)  # Shape: (batch_size, INTERNAL_DIM)

        return positional_encoding_proj
