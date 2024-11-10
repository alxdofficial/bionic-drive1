import torch
import torch.nn as nn
import torch.nn.functional as F
import random

IMG_WIDTH = 1600
IMG_HEIGHT = 900
LONG_SHORT_TERM_PROB = 0.5
INTERNAL_DIM = 768
NUM_SUBNETWORKS = 1
NUM_UNITS = 32
TOTAL_DIM = INTERNAL_DIM * NUM_SUBNETWORKS  # Calculate the total dimension
NUM_LAYERS = 4
NUM_MEMORY_NETWORKS = 4
POSITIONAL_ENCODING_SCALE = 0.01
ALPHA_BIAS = 0.1
ALPHA_SCALE = 9.9
DECAY_BIAS = 0.01
DECAY_SCALE = 1.99



# Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, kernel_size=5, stride=3, dropout=0.2):
        super(FeatureExtractor, self).__init__()
        hidden = [16, 64, 256, INTERNAL_DIM]  # Specified number of channels for each layer
        layers = []
        for i in range(len(hidden)):
            layers.append(nn.Conv2d(input_channels, hidden[i], kernel_size=kernel_size, stride=stride, padding=2))
            if i != len(hidden) - 1:
                layers.append(nn.BatchNorm2d(hidden[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_channels = hidden[i]
        self.extractor = nn.Sequential(*layers)

    def forward(self, x):   
        return self.extractor(x)
    
# Fovea Position Predictor
class FoveaPosPredictor(nn.Module):
    def __init__(self, input_dim=INTERNAL_DIM * 2, dropout=0.2):
        super(FoveaPosPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 8*23)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(256)
        self.layer_norm2 = nn.LayerNorm(128)
        self.sigmoid = nn.Sigmoid()  # For generating the importance map

    def forward(self, memory_encoding, cls_token):
        # Concatenate memory encoding and CLS token along the last dimension
        combined_input = torch.cat((memory_encoding, cls_token), dim=-1)
        # Pass through fully connected layers
        x = F.relu(self.layer_norm1(self.fc1(combined_input)))
        x = self.dropout(x)
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        importance_map = self.sigmoid(self.fc3(x))  # Importance map with shape matching flattened_features

        return importance_map
    
# Vision Encoder
class VisionEncoder(nn.Module): 
    def __init__(self, device='cuda'):
        super(VisionEncoder, self).__init__()
        self.device = device
        # Feature extractor with CNN layers to downsample to a feature map
        self.feature_extractor = FeatureExtractor()
        self.fovea_pos_predictor = FoveaPosPredictor()

    def concatenate_images(self, imgs):
        # Resize each image to 300x600
        resized_imgs = [F.interpolate(img, size=(300, 600), mode='bilinear', align_corners=False) for img in imgs]
        
        # Arrange images into 2x3 grid
        top_row = torch.cat([resized_imgs[1], resized_imgs[0], resized_imgs[2]], dim=-1)  # front_left, front, front_right
        bottom_row = torch.cat([resized_imgs[4], resized_imgs[3], resized_imgs[5]], dim=-1)  # back_left, back, back_right
        
        # Concatenate the top and bottom rows to form a 2x3 grid
        full_concat_img = torch.cat([top_row, bottom_row], dim=-2)  # Shape: (batch_size, 3, 600, 1800)
        
        return full_concat_img
    
    def grid_positional_encoding(self, grid_x, grid_y):
        """
        Generate a grid-based positional encoding by normalizing grid_x and grid_y and replicating
        them to match INTERNAL_DIM.

        :param grid_x: x-coordinate of the grid cell.
        :param grid_y: y-coordinate of the grid cell.
        :return: Positional encoding tensor of shape (INTERNAL_DIM,)
        """
        # Normalize grid coordinates between 0 and 1
        norm_x = grid_x / (self.grid_width - 1)
        norm_y = grid_y / (self.grid_height - 1)

        # Use the device of the feature map to ensure compatibility
        device = self.feature_map.device
        
        # Create a tensor for positional encoding by replicating norm_x and norm_y
        pos_encoding = torch.tensor([norm_x, norm_y], device=device, dtype=torch.float32)

        # Replicate to match INTERNAL_DIM
        repeats = INTERNAL_DIM // pos_encoding.shape[0]
        remainder = INTERNAL_DIM % pos_encoding.shape[0]
        pos_encoding = torch.cat([pos_encoding.repeat(repeats), pos_encoding[:remainder]])

        return pos_encoding * POSITIONAL_ENCODING_SCALE  # Shape: (INTERNAL_DIM,)

    def forward_peripheral(self, imgs):
        # Concatenate images in a 2x3 grid format
        concat_image = self.concatenate_images(imgs)  # Shape: (batch_size, 3, IMG_HEIGHT*2, IMG_WIDTH*3)
        # Pass the concatenated image through the CNN to get the initial feature map
        feature_map = self.feature_extractor(concat_image)  # Shape: (batch_size, channels, grid_height, grid_width)
        self.feature_map = feature_map
        # Store the original spatial dimensions before pooling
        original_height, original_width = feature_map.size(2), feature_map.size(3)
        self.grid_height = original_height
        self.grid_width = original_width
        # print(self.feature_map.shape)
        # Apply max pooling to reduce spatial dimensions
        pooled_feature_map = F.adaptive_max_pool2d(feature_map, output_size=(original_height // 4, original_width // 6))
        # pooled_feature_map shape: (batch_size, channels, pooled_height, pooled_width)
        # Calculate the positional encoding for each pooled vector and add it
        batch_size, channels, pooled_height, pooled_width = pooled_feature_map.size()
        pos_encoded_pooled_features = []
    
        for row in range(pooled_height):
            for col in range(pooled_width):
                # Calculate the center of the corresponding region in the original feature map
                avg_row = (row + 0.5) * original_height / pooled_height
                avg_col = (col + 0.5) * original_width / pooled_width

                # Generate the positional encoding for the (avg_row, avg_col) position
                pos_encoding = self.grid_positional_encoding(avg_row, avg_col).to(self.device)
                # Expand to batch size and add to each vector in the batch
                pos_encoding_batch = pos_encoding.unsqueeze(0).expand(batch_size, -1)

                # Extract the pooled vector at (row, col) and add positional encoding
                pooled_vector = pooled_feature_map[:, :, row, col]  # Shape: (batch_size, channels)
                pooled_vector = F.normalize(pooled_vector)
                # print(pooled_vector.mean(), pos_encoding_batch.mean())
                pooled_vector_with_pos = pooled_vector + pos_encoding_batch  # Add positional encoding
                # Store in the list
                pos_encoded_pooled_features.append(pooled_vector_with_pos)
        # print(len(pos_encoded_pooled_features), pos_encoded_pooled_features[0].shape)
        return pos_encoded_pooled_features  # Shape: (batch_size, pooled_height * pooled_width, channels)

    def forward_fovea(self, memory_encoding, cls_token, temperature=0.1):
        # Get importance map from FoveaPosPredictor
        importance_map = self.fovea_pos_predictor(memory_encoding, cls_token)  # Shape: (batch_size, flattened_features_dim)
        # Reshape to match the 2D grid dimensions
        importance_grid = importance_map.view(-1, self.grid_height, self.grid_width)  # Shape: (batch_size, grid_height, grid_width)

        # Apply softmax with temperature scaling to get a peaked importance distribution
        scaled_importance_map = F.softmax(importance_grid.view(-1, self.grid_height * self.grid_width) / temperature, dim=-1)
        scaled_importance_map = scaled_importance_map.view(-1, self.grid_height, self.grid_width)

        # Expand dimensions to match feature map for element-wise multiplication
        scaled_importance_map_expanded = scaled_importance_map.unsqueeze(1)  # Shape: (batch_size, 1, grid_height, grid_width)

        # Apply weighted sum on feature map using the scaled importance map
        weighted_features = (self.feature_map * scaled_importance_map_expanded).sum(dim=(2, 3))  # Sum over spatial dimensions
        weighted_features = F.normalize(weighted_features)
        # Find the max value's position for positional encoding
        max_positions = scaled_importance_map.view(-1, self.grid_height * self.grid_width).argmax(dim=-1)
        max_rows = max_positions // self.grid_width
        max_cols = max_positions % self.grid_width

        # Generate positional encoding for each max position in the batch
        batch_size = weighted_features.size(0)
        pos_encodings = []
        for i in range(batch_size):
            pos_encoding = self.grid_positional_encoding(max_rows[i].item(), max_cols[i].item())
            pos_encodings.append(pos_encoding)

        # Concatenate positional encodings to match batch size and add to weighted features
        pos_encodings = torch.stack(pos_encodings).to(self.device)  # Shape: (batch_size, INTERNAL_DIM)
        weighted_features_with_pos = weighted_features + pos_encodings  # Add positional encoding to weighted features
        return weighted_features_with_pos  # Shape of weighted_features_with_pos: (batch_size, channels)


# Neuromodulator Class
class Neuromodulator(nn.Module):
    def __init__(self, total_dim, num_units):
        super(Neuromodulator, self).__init__()
        self.num_units = num_units
        
        # Fully connected layers with ReLU and normalization
        self.fc1 = nn.Linear(total_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_units)
        
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(256)
        self.output_norm = nn.LayerNorm(num_units)
        self.tanh = nn.Tanh()  # Output dopamine signals in range [-1, 1]

    def forward(self, x):
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = F.relu(self.layer_norm2(self.fc2(x)))
        dopamine_signals = self.tanh(self.output_norm(self.fc3(x)))  # Shape: (batch_size, num_units)
        return dopamine_signals.mean(dim=0)  # Average across batch, final shape: (num_units,)

# Updated Hebbian Layer Class
class HebbianLayer(nn.Module):
    def __init__(self, total_dim, num_units):
        super(HebbianLayer, self).__init__()

        self.total_dim = total_dim
        self.num_units = num_units

        # Hebbian weights and recurrent weights as buffers (not parameters)
        self.register_buffer('hebbian_weights', torch.randn(total_dim, total_dim))
        self.register_buffer('hebbian_recurrent_weights', torch.randn(total_dim, total_dim))

        # Randomized alpha and decay values for each dimension
        self.register_buffer('alpha', torch.rand(total_dim))
        self.register_buffer('decay', torch.rand(total_dim))

        # Layer normalization
        self.layer_norm_activations = nn.LayerNorm(total_dim)
        self.layer_norm_recurrent = nn.LayerNorm(total_dim)

        # Initialize neuromodulator
        self.neuromodulator = Neuromodulator(total_dim, num_units)

        # Store previous activations
        self.register_buffer('previous_activation', None)

    def forward(self, stimulus):
        batch_size = stimulus.size(0)

        # Initialize previous activations if not set
        if self.previous_activation is None:
            self.previous_activation = torch.zeros(batch_size, self.total_dim, device=stimulus.device)

        # Compute recurrent output
        recurrent_input = self.previous_activation
        recurrent_output = torch.matmul(recurrent_input, self.hebbian_recurrent_weights)
        recurrent_output = F.relu(recurrent_output)  # Apply ReLU after recurrent calculation
        recurrent_output_norm = self.layer_norm_recurrent(recurrent_output)

        # Compute activations
        stimulus_output = torch.matmul(stimulus, self.hebbian_weights)
        stimulus_output = F.relu(stimulus_output)  # Apply ReLU after stimulus calculation
        final_output = stimulus_output + recurrent_output_norm

        # Update previous activations
        self.previous_activation = final_output.detach()

        # Apply neuromodulator
        dopamine_signals = self.neuromodulator(final_output)  # Shape: (num_units,)
        expanded_dopamine_signals = dopamine_signals.repeat_interleave(self.total_dim // self.num_units)
        self.alpha = expanded_dopamine_signals * ALPHA_SCALE # Modulate alpha

        # Perform Hebbian update
        self.hebbian_update(recurrent_input, recurrent_output, stimulus, stimulus_output)

        # return final_output, estimated_batch_loss
        return final_output

    def hebbian_update(self, recurrent_input, recurrent_output, stimulus, stimulus_output):
        # Perform the outer product using broadcasting and bmm
        hebbian_updates = stimulus.unsqueeze(-1) * stimulus_output.unsqueeze(-2)  # (batch_size, total_dim, total_dim)
        recurrent_updates = recurrent_input.unsqueeze(-1) * recurrent_output.unsqueeze(-2)  # (batch_size, total_dim, total_dim)
        
        # Sum across the batch dimension
        hebbian_updates = hebbian_updates.sum(dim=0)  # Sum over batch to get (total_dim, total_dim)
        recurrent_updates = recurrent_updates.sum(dim=0)  # Sum over batch
        
        # Modulate updates with alpha (learning rate)
        modulated_hebbian_updates = hebbian_updates * self.alpha.unsqueeze(0)
        modulated_recurrent_updates = recurrent_updates * self.alpha.unsqueeze(0)

        # Apply updates to weights and recurrent weights
        self.hebbian_weights = self.hebbian_weights + modulated_hebbian_updates
        self.hebbian_recurrent_weights = self.hebbian_recurrent_weights + modulated_recurrent_updates

        # Apply decay to weights
        decay_matrix = torch.diag(self.decay)
        self.hebbian_weights = self.hebbian_weights - torch.matmul(decay_matrix, self.hebbian_weights)
        self.hebbian_recurrent_weights = self.hebbian_recurrent_weights - torch.matmul(decay_matrix, self.hebbian_recurrent_weights)

        # Normalize weights
        self.hebbian_weights = F.normalize(self.hebbian_weights, p=2, dim=1)
        self.hebbian_recurrent_weights = F.normalize(self.hebbian_recurrent_weights, p=2, dim=1)

# Neural Memory Class
class NeuralMemory(nn.Module):
    def __init__(self, num_layers):
        super(NeuralMemory, self).__init__()

        # Define hebbian layers in memory network
        self.hebbian_layers = nn.ModuleList([
            HebbianLayer(TOTAL_DIM, NUM_UNITS)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # Step 1: Replicate input to match TOTAL_DIM
        x = x.repeat_interleave(NUM_SUBNETWORKS, dim=-1)  # Shape: (batch_size, TOTAL_DIM)

        # Step 2: Pass through each Hebbian layer and accumulate the estimated loss
        all_layer_outputs = []

        for layer in self.hebbian_layers:
            x = layer(x)
            # Split the output of this Hebbian layer into segments of size INTERNAL_DIM
            layer_outputs = torch.split(x, INTERNAL_DIM, dim=-1)  # List of (batch_size, INTERNAL_DIM) tensors
            all_layer_outputs.extend(layer_outputs)  # Collect all segments

        return all_layer_outputs


# Brain Class
class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()

        # Vision encoder
        self.vision_encoder = VisionEncoder()

        # Initialize neural memory networks
        self.neural_memory_networks = nn.ModuleList([
            NeuralMemory(num_layers=NUM_LAYERS)
            for _ in range(NUM_MEMORY_NETWORKS)
        ])

    def forward(self, imgs, cls_tokens):
        self.clear_gradients()  # Clear gradients before processing each new batch
        self.reset_memory()     # Reset memory and estimated batch losses at the start of each forward pass
        latest_memory_outputs = []
        
        # Step 1: Calculate peripheral encodings
        peripheral_encodings = self.vision_encoder.forward_peripheral(imgs)  # Returns a list of peripheral encodings

        # Step 2: Process each peripheral encoding through neural memory networks
        for peripheral_encoding in peripheral_encodings:
            # print("periph enc in brain: ", peripheral_encoding.shape, peripheral_encoding.mean())
            latest_memory_outputs = []
            for nmn in self.neural_memory_networks:
                nmn_output = nmn(peripheral_encoding)
                latest_memory_outputs.extend(nmn_output)  # Collect all memory outputs for current peripheral encoding
            # print("neural memory periph: ", len(latest_memory_outputs), latest_memory_outputs[0].mean())
        # Step 3: For each CLS token, apply global pooling on memory outputs and pass through forward_fovea
        for cls_token in cls_tokens:
            # Global pooling (average pooling here) over the memory outputs to get a compact representation
            pooled_memory_output = F.normalize(torch.stack(latest_memory_outputs, dim=0)).mean(dim=0)
            # print("pooled memory: ", pooled_memory_output.shape, pooled_memory_output.mean())
            # Step 4: Pass pooled memory output and cls_token into forward_fovea to get the fovea encoding
            fovea_encoding = self.vision_encoder.forward_fovea(pooled_memory_output, cls_token)
            # print("fovea enc in brain: ", fovea_encoding.shape, fovea_encoding.mean())

            # Step 5: Clear the latest memory outputs, replace with the new fovea encoding, and repeat the process
            latest_memory_outputs = []
            for nmn in self.neural_memory_networks:
                nmn_output = nmn(fovea_encoding)
                latest_memory_outputs.extend(nmn_output)

            # print("neural memory fovea: ", len(latest_memory_outputs), latest_memory_outputs[0].mean())

        # Return the final memory outputs after all CLS tokens have been processed
        return torch.stack(latest_memory_outputs, dim=1)

    def reset_memory(self):
        """Reset previous activations in each Hebbian layer across all memory networks."""
        for nmn in self.neural_memory_networks:
            for layer in nmn.hebbian_layers:
                layer.previous_activation = None
    
    def clear_gradients(self):
        """Clear gradients efficiently by zeroing gradients without detaching entire tensors."""
        for nmn in self.neural_memory_networks:
            for layer in nmn.hebbian_layers:
                # Check and clear gradients if they exist, avoiding full detachment
                layer.alpha = layer.alpha.detach()
                layer.hebbian_weights = layer.hebbian_weights.detach()
                layer.hebbian_recurrent_weights = layer.hebbian_recurrent_weights.detach()
