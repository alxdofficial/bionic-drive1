import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.jit

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

# 4. Hebbian Layer
class HebbianLayer(nn.Module):
    def __init__(self, dim, alpha_init=0.1, decay_init=0.01, device='cuda'):
        super(HebbianLayer, self).__init__()
        self.dim = dim
        self.device = device

        # Hebbian weights and recurrent weights (regular tensors)
        self.hebbian_weights = torch.randn(dim, dim).to(device)
        self.hebbian_recurrent_weights = torch.randn(dim, dim).to(device)

        # Hebbian parameters (regular tensors)
        self.alpha = torch.full((dim,), alpha_init).to(device)
        self.decay = torch.full((dim,), decay_init).to(device)

        # Layer normalization layers
        self.layer_norm_activations = nn.LayerNorm(dim)
        self.layer_norm_recurrent = nn.LayerNorm(dim)

        # Store previous activations
        self.previous_activation = None

    def forward(self, stimulus):
        batch_size = stimulus.size(0)

        # Initialize previous activations if not set
        if self.previous_activation is None:
            self.previous_activation = torch.zeros(batch_size, self.dim, device=self.device)

        # Compute recurrent output
        recurrent_input = self.previous_activation
        recurrent_output = torch.matmul(recurrent_input, self.hebbian_recurrent_weights)  # Detach here
        recurrent_output_norm = self.layer_norm_recurrent(recurrent_output)

        # Compute activations
        stimulus_output = torch.matmul(stimulus, self.hebbian_weights)  # Detach here
        final_output = F.relu(stimulus_output + recurrent_output_norm)
        final_output = self.layer_norm_activations(final_output)

        # Update previous activations
        self.previous_activation = final_output.detach()  # Prevent backprop through activations

        # Perform Hebbian update
        self.hebbian_update(recurrent_input, recurrent_output, stimulus, stimulus_output)

        return final_output

    def hebbian_update(self, recurrent_input, recurrent_output, stimulus, stimulus_output):
        # Standard Hebbian update rule
        hebbian_updates = torch.einsum('bi,bj->ij', stimulus, stimulus_output)
        recurrent_updates = torch.einsum('bi,bj->ij', recurrent_input, recurrent_output)

        # Modulate updates with alpha (learning rate)
        modulated_hebbian_updates = hebbian_updates * self.alpha.unsqueeze(0)
        modulated_recurrent_updates = recurrent_updates * self.alpha.unsqueeze(0)

        # Apply updates to weights and recurrent weights
        new_hebbian_weights = self.hebbian_weights + modulated_hebbian_updates
        new_recurrent_weights = self.hebbian_recurrent_weights + modulated_recurrent_updates

        # Apply decay to weights
        decay_matrix = torch.diag(self.decay).to(self.device)
        new_hebbian_weights -= torch.matmul(decay_matrix, self.hebbian_weights)
        new_recurrent_weights -= torch.matmul(decay_matrix, self.hebbian_recurrent_weights)

        # Normalize weights
        new_hebbian_weights = F.normalize(new_hebbian_weights, p=2, dim=1)
        new_recurrent_weights = F.normalize(new_recurrent_weights, p=2, dim=1)

        # Update weights
        self.hebbian_weights = new_hebbian_weights.detach()
        self.hebbian_recurrent_weights = new_recurrent_weights.detach()


# 5. SubNetwork
class SubNetwork(nn.Module):
    def __init__(self, dim, num_layers, memory_type="short"):
        super(SubNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if memory_type == "short":
                scaling = torch.FloatTensor(1).uniform_(0.1, 1).item()  # Higher alpha
                decay = torch.FloatTensor(1).uniform_(0.01, 0.05).item()  # Higher decay
            else:  # long-term memory
                scaling = torch.FloatTensor(1).uniform_(0.01, 0.1).item()  # Lower alpha
                decay = torch.FloatTensor(1).uniform_(0.001, 0.01).item()  # Lower decay
            self.layers.append(HebbianLayer(dim, scaling, decay))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class NeuromodulatorNetwork(nn.Module):
    def __init__(self, input_dim, num_subnetworks):
        super(NeuromodulatorNetwork, self).__init__()
        hidden_dims = [256, 128]
        self.num_subnetworks = num_subnetworks
        self.fc = FullyConnected(input_dim, hidden_dims, num_subnetworks)  # Output 1 dopamine signal per subnetwork
        self.tanh = nn.Tanh()  # Output will be restricted between -1 and 1

    def forward(self, activations):
        dopamine_signals = self.fc(activations.detach())  # Shape: (batch_size, num_subnetworks)
        return self.tanh(dopamine_signals.mean(dim=0))  # Dopamine signals per subnetwork, Averaging across batch dimension, final shape: (num_subnetworks,)

class ExpectationRewardNetwork(nn.Module):
    def __init__(self, input_dim, num_subnetworks, device='cuda'):
        super(ExpectationRewardNetwork, self).__init__()
        # input_dim will include both activations and dopamine signals from all subnetworks
        total_input_dim = input_dim + num_subnetworks  # Add the number of dopamine signals to input dimension
        hidden_dims = [256, 128]  # Can be adjusted based on the use case
        
        # Fully connected layers for the expectation-reward prediction
        self.fc = FullyConnected(total_input_dim, hidden_dims, 1)  # Output a scalar reward prediction
        self.device = device

    def forward(self, activations, dopamine_signals):
        # Concatenate activations and dopamine signals for all subnetworks
        activations = activations.detach()
        batch_size = activations.size(0)

        dopamine_signals = dopamine_signals.repeat(batch_size, 1)  # Shape: (batch_size, num_subnetworks)
        combined_input = torch.cat([activations, dopamine_signals], dim=-1)  # Shape: (batch_size, input_dim + num_subnetworks)
        
        # Pass through the fully connected layers to predict reward
        reward_prediction = self.fc(combined_input)
        return reward_prediction.mean(dim=0)


# 7. Neural Memory Network
class NeuralMemoryNetwork(nn.Module):
    def __init__(self, num_subnetworks, num_layers, device='cuda'):
        super(NeuralMemoryNetwork, self).__init__()
        self.subnetworks = nn.ModuleList()
        self.hidden_size = INTERNAL_DIM // num_subnetworks
        assert INTERNAL_DIM % num_subnetworks == 0, "INTERNAL_DIM must be divisible by num_subnetworks"
        self.num_subnetworks = num_subnetworks
        self.num_layers = num_layers
        for _ in range(num_subnetworks):
            # Randomly decide if the subnetwork is long-term or short-term
            memory_type = "long" if random.random() < LONG_SHORT_TERM_PROB else "short"
            self.subnetworks.append(SubNetwork(self.hidden_size, num_layers, memory_type))

        self.output_layer = nn.Linear(INTERNAL_DIM, INTERNAL_DIM)
        self.layer_norm = nn.LayerNorm(INTERNAL_DIM)  # Apply layer normalization

        self.prev_loss = None
        self.activations = None

        # Initialize neuromodulator and expectation-reward networks
        self.neuromodulator = NeuromodulatorNetwork(INTERNAL_DIM, num_subnetworks)  # One dopamine signal per subnetwork
        self.expectation_reward_network = ExpectationRewardNetwork(INTERNAL_DIM, num_subnetworks)  # Takes activations + dopamine
        self.dopamine_signals = None  # Cache dopamine signals

        self.device = device


    def update_neural_modulators(self):
        """
        This function will be called before the forward pass to modulate alpha values of each subnetwork.
        The dopamine signals are computed using the neuromodulator network based on the activations of the previous step.
        """
        if self.activations is not None:
            # Compute dopamine signals using the neuromodulator network
            self.dopamine_signals = self.neuromodulator(self.activations)  # (batch_size, num_subnetworks)

            # Update alpha values in each subnetwork's Hebbian layers
            for i, subnetwork in enumerate(self.subnetworks):
                for layer in subnetwork.layers:
                    # Use no_grad to perform in-place updates without affecting autograd
                    with torch.no_grad():
                        # layer.alpha += self.dopamine_signals[i]  # Modulate alpha with dopamine signal
                        layer.alpha = layer.alpha + self.dopamine_signals[i]

        else:
            # Initialize dopamine signals as zeros if this is the first step
            self.dopamine_signals = torch.zeros(1, self.num_subnetworks, device=self.device)


    def forward(self, x):
        batch_size = x.size(0)  # (batch_size, INTERNAL_DIM)
        x_splits = x.split(self.hidden_size, dim=-1)  # List of (batch_size, hidden_size) tensors
        activations_of_all_subnetworks = []

        for i, subnetwork in enumerate(self.subnetworks):
            activations = subnetwork(x_splits[i])  # (batch_size, hidden_size)
            activations_of_all_subnetworks.append(activations)

        # Combine final activations for the final output
        combined_activations = torch.cat(activations_of_all_subnetworks, dim=-1)  # (batch_size, INTERNAL_DIM)
        
        # Use combined activations to generate the final output
        final_output = self.output_layer(combined_activations)  # (batch_size, INTERNAL_DIM)
        final_output = self.layer_norm(final_output)  # (batch_size, INTERNAL_DIM)
        self.activations = final_output
        return final_output
    
    def compute_expected_reward(self):
        # Use the expectation-reward network to compute the predicted quality of the dopamine signals
        predicted_reward = self.expectation_reward_network(self.activations, self.dopamine_signals)
        return predicted_reward
    
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

        # Neural memory networks
        self.neural_memory_networks = nn.ModuleList([
            NeuralMemoryNetwork(
                num_subnetworks=NUM_SUB_NETWORKS,
                num_layers=NUM_LAYERS,
            ) for _ in range(NUM_MEMORY_NETWORKS)
        ])

        # Attention mechanism
        self.attention = AttentionModule(embed_dim=INTERNAL_DIM, num_heads=4)
        # Linear layer to project positional encoding to match INTERNAL_DIM
        self.positional_encoder = nn.Linear(8, INTERNAL_DIM)  # 8: 2 for coordinates, 6 for one-hot image index
        # LayerNorm for peripheral and fovea encodings after positional encoding
        self.peripheral_norm = nn.LayerNorm(INTERNAL_DIM)
        self.fovea_norm = nn.LayerNorm(INTERNAL_DIM)

    def update_neural_modulators(self):
        """Update the neuromodulators for each NMN before the forward pass."""
        for nmn in self.neural_memory_networks:
            nmn.update_neural_modulators()

    def get_predicted_dopamine_quality(self):
        """Get the predicted dopamine signal quality for each NMN."""
        dopamine_qualities = []
        for nmn in self.neural_memory_networks:
            dopamine_quality = nmn.compute_expected_reward()  # Returns the predicted quality of dopamine signals
            dopamine_qualities.append(dopamine_quality)
        return torch.stack(dopamine_qualities)
    
    
    def forward(self, imgs, cls_tokens):
        batch_size = imgs[0].size(0)

        # Process peripheral vision for all images
        peripheral_encodings = []
        for i, img in enumerate(imgs):
            peripheral_encoding = self.vision_encoder.forward_peripheral(img)
            zero_fovea_coords = torch.zeros(batch_size, 2, device=img.device)
            selected_img_idx = torch.full((batch_size,), i, device=img.device, dtype=torch.long)
            pos_encoding = self.add_positional_encoding(zero_fovea_coords, selected_img_idx)
            peripheral_encoding = self.peripheral_norm(peripheral_encoding + pos_encoding)
            peripheral_encodings.append(peripheral_encoding)

        # Combine peripheral encodings with attention
        peripheral_combined = self.attention(peripheral_encodings[0], peripheral_encodings[1:])

        # Run NMNs in parallel using torch.jit.fork
        futures = [torch.jit.fork(nmn, peripheral_combined) for nmn in self.neural_memory_networks]
        
        # Collect outputs from all NMNs
        memory_outputs = [torch.jit.wait(future) for future in futures]

        # Process fovea vision with CLS tokens
        fovea_encodings = []
        for i, cls in enumerate(cls_tokens):
            if i == 0:
                fovea_coords_logits, img_selector_logits = self.fovea_loc_pred(peripheral_combined, cls)
            else:
                fovea_coords_logits, img_selector_logits = self.fovea_loc_pred(fovea_encodings[-1], cls)

            img_selector_probs = F.softmax(img_selector_logits, dim=-1)
            selected_img_idx = torch.argmax(img_selector_probs, dim=-1)

            # Select images based on predicted index
            selected_img = torch.cat([
                imgs[selected_img_idx[b].item()][b].unsqueeze(0) for b in range(batch_size)
            ], dim=0)

            fovea_encoding = self.vision_encoder.forward_fovea(selected_img, fovea_coords_logits)
            fovea_encoding_with_pos = self.fovea_norm(
                fovea_encoding + self.add_positional_encoding(fovea_coords_logits, selected_img_idx)
            )
            fovea_encodings.append(fovea_encoding_with_pos)

            # Run fovea encoding through NMNs and collect results
            fovea_futures = [torch.jit.fork(nmn, fovea_encoding_with_pos) for nmn in self.neural_memory_networks]
            fovea_memory_outputs = [torch.jit.wait(future) for future in fovea_futures]
            memory_outputs.extend(fovea_memory_outputs)

        # Stack all memory network outputs along a new sequence dimension
        final_output = torch.stack(memory_outputs, dim=1)

        return final_output

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
