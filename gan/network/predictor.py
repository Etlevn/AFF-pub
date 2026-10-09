# Predictor network module - responsible for evaluating the quality of factor expressions
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import copy

import torch
from torch.utils.tensorboard import SummaryWriter


class NetP(nn.Module):
    """
    Predictor Network - Evaluating the Quality of Factor Expressions Using CNN

    Network architecture:
    1. 2D convolutional layer: extract local features of the token sequence
    2. Fully connected layer: mapping features to quality scores
    3. Output: Single quality score value (IC value)
    """
    def __init__(
        self,
        n_chars,    # Vocabulary size (token number of categories)
        hidden,      # Hidden layer dimension (not used, keep the interface consistent)
        seq_len,     # Sequence length (fixed to 20)
    ):
        super().__init__()
        assert seq_len == 20  # Ensure that the sequence length is 20

        # 2D convolution layer: extract the local features of the token sequence
        self.convs = nn.Sequential(
            nn.Conv2d(n_chars, 96, kernel_size=(1, 3)),  # First layer of convolution: 96 feature maps
            nn.ReLU(),  # Activation function
            nn.MaxPool2d((1, 2)),  # Max pooling: downsampling
            nn.Conv2d(96, 128, kernel_size=(1, 4)),  # The second layer of convolution: 128 feature maps
            nn.ReLU(),  # Activation function
        )  # Output: [batch_size, 128, 1, 6]

        # Fully connected layer: mapping features to quality scores
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 6, 256),  # The flattened features are mapped to the 256 dimension
            nn.Dropout(0.2),  # Dropout prevents overfitting
            nn.ReLU(),  # Activation function
        )
        self.fc2 = nn.Sequential(nn.Linear(256, 1))  # Final output layer: single quality score

    def forward(self, x, latent=False):
        """
        Forward pass - Evaluating the quality of factor expressions

        Args:
            x: one-hot encoding of token sequence (batch_size, seq_len, n_chars)
            latent: Whether to return latent features

        Returns:
            x: Quality score (batch_size, 1) or (x, latent_tensor) if latent=True
        """
        # x (batch_size, 20, 48)-one-hot encoding of the token sequence
        x = x.float()  # Make sure the data type is float
        x = x.permute(0, 2, 1)[:, :, None]  # Reshape into convolution input: (batch_size, 48, 1, 20)
        # x (batch_size, 48, 1, 20) - Add channel dimension for 2D convolution

        # 2D convolution feature extraction
        x = self.convs(x)  # Convolution feature extraction: (batch_size, 128, 1, 6)
        # #[batch_size, 128, 1, 6] - Features after convolution and pooling

        # Flattening features are used in fully connected layers
        x = x.reshape([x.shape[0], 128 * 6])  # Flatten: (batch_size, 128*6)

        # Fully connected layer processing
        latent_tensor = self.fc1(x)  # The first layer of full connection: (batch_size, 256)
        x = self.fc2(latent_tensor)  # Final output layer: (batch_size, 1) - quality score

        if latent:
            return x, latent_tensor  # Return scores and latent features
        else:
            return x  # Return only quality score

    def initialize_parameters(self):
        """Initialize network parameters - use Xavier normal distribution to initialize weights, and bias initialization to 0"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape)>1  :
                nn.init.xavier_normal_(param)  # Weights are initialized using Xavier normal distribution
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  # The bias is initialized to 0

class ResBlock(nn.Module):
    """
    Residual block - used to build deep convolutional networks

    Structure: input -> ReLU -> Conv1d -> ReLU -> Conv1d -> 0.3*output + input
    Use residual connection to alleviate the gradient disappearance problem, and the scaling factor 0.3 is used to stabilize training
    """
    def __init__(self, hidden):
        super(ResBlock, self).__init__()
        # Residual block body: ReLU -> Conv1d -> ReLU -> Conv1d
        self.res_block = nn.Sequential(
            nn.ReLU(True),  # Activation function
            nn.Conv1d(hidden, hidden, 5, padding=2),  # 1D convolution, maintaining dimensions
            nn.ReLU(True),  # Activation function
            nn.Conv1d(hidden, hidden, 5, padding=2),  # 1D convolution, maintaining dimensions
        )

    def forward(self, input):
        """Forward pass - residual connection (scaled version)"""
        output = self.res_block(input)  # Residual block output
        return input + (0.3*output)  # Residual connection: x + 0.3*F (x) - scaling factor stable training



class NetP_CNN(nn.Module):
    """
    CNN Predictor Network - Evaluating the quality of factor expressions using 1D convolution and residual blocks

    Network architecture:
    1. 1D convolutional layer: convert the token sequence into features
    2. Residual block: Use ResBlock for feature extraction
    3. Fully connected layer: mapping features to quality scores
    """
    def __init__(self, n_chars, seq_len, hidden):
        super().__init__()
        # Save network parameters
        self.n_chars = n_chars      # Vocabulary size
        self.seq_len = seq_len      # Sequence length
        self.hidden = hidden        # Hidden layer dimension

        # Residual block sequence: used for feature extraction and conversion
        self.block = nn.Sequential(
            ResBlock(hidden),  # The first residual block
            ResBlock(hidden),  # The second residual block
            # ResBlock(hidden), # Commented out extra residual block
            # ResBlock(hidden),
            # ResBlock(hidden),
        )

        # 1D convolution layer: convert the token sequence into features
        self.conv1d = nn.Conv1d(n_chars, hidden, 1)  # 1x1 convolution, mapped to hidden dimensions

        # Final fully connected layer: generate quality score
        self.linear = nn.Linear(seq_len*hidden, 1)  # Map to a single rating value

    def forward(self, input,latent=False):
        """
        Forward pass - Evaluating the quality of factor expressions

        Args:
            input: one-hot encoding of token sequence (batch_size, seq_len, n_chars)
            latent: Whether to return latent features

        Returns:
            output: Quality score (batch_size, 1) or (output, latent_tensor) if latent=True
        """
        # Adjust dimension order: (batch_size, seq_len, n_chars) -> (batch_size, n_chars, seq_len)
        output = input.transpose(1, 2)  # (batch_size, n_chars, seq_len)

        # 1D convolution: convert the token sequence into features
        output = self.conv1d(output)  # (batch_size, hidden, seq_len)

        # Residual block: feature extraction and conversion
        output = self.block(output)  # (batch_size, hidden, seq_len)

        # Flattening features are used in fully connected layers
        output = output.view(-1, self.seq_len*self.hidden)  # (batch_size, seq_len*hidden)

        # Save latent features
        latent_tensor = output  # (batch_size, seq_len*hidden)

        # Fully connected layer: generate quality score
        output = self.linear(latent_tensor)  # (batch_size, 1)

        if latent:
            return output, latent_tensor  # Return scores and latent features
        else:
            return output  # Return only quality score


    def initialize_parameters(self):
        """Initialize network parameters - use Xavier normal distribution to initialize weights, and bias initialization to 0"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape)>1  :
                nn.init.xavier_normal_(param)  # Weights are initialized using Xavier normal distribution
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  # The bias is initialized to 0


def train_regression_model(
    train_loader,
    valid_loader,
    net,
    loss_fn,
    optimizer,
    num_epochs=10,
    use_tensorboard=True,
    tensorboard_path="logs",
    early_stopping_patience=None,
):
    # Initialize TensorBoard SummaryWriter if requested
    writer = None
    if use_tensorboard:
        writer = SummaryWriter(tensorboard_path)

    # Initialize variables for early stopping
    best_valid_loss = float("inf")
    best_weights = None
    patience_counter = 0

    # Training loop for regression
    for epoch in range(num_epochs):
        net.train()
        total_train_loss = 0
        total_samples_train = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = net(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_y.size(0)
            total_samples_train += batch_y.size(0)

        average_train_loss = total_train_loss / total_samples_train

        # Validation
        net.eval()
        with torch.no_grad():
            total_valid_loss = 0
            total_samples_valid = 0
            for batch_x, batch_y in valid_loader:
                outputs = net(batch_x)
                loss = loss_fn(outputs, batch_y)
                total_valid_loss += loss.item() * batch_y.size(0)
                total_samples_valid += batch_y.size(0)

            average_valid_loss = total_valid_loss / total_samples_valid

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Validation Loss: {average_valid_loss:.4f}"
            )

            # Write to TensorBoard if requested
            if use_tensorboard:
                writer.add_scalar("Train Loss", average_train_loss, epoch)
                writer.add_scalar("Validation Loss", average_valid_loss, epoch)

            # Early Stopping
            if (
                early_stopping_patience is not None
                and average_valid_loss < best_valid_loss
            ):
                best_valid_loss = average_valid_loss
                patience_counter = 0
                best_weights = copy.deepcopy(net.state_dict())  # Record the best weights
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load the best weights back to the net
    if best_weights is not None:
        net.load_state_dict(best_weights)

    # Close the TensorBoard SummaryWriter if used
    if use_tensorboard:
        writer.close()



def train_regression_model_with_weight(
    train_loader,
    valid_loader,
    net,
    loss_fn,
    optimizer,
    device = 'cpu',
    num_epochs=10,
    use_tensorboard=True,
    tensorboard_path="logs",
    early_stopping_patience=None,
    plot_loss=False,  # New parameter: whether to draw a loss graph
):
    print(f"  [Predictor Training] Maximum epoch: {num_epochs}, early stop patience: {early_stopping_patience}")

    # Initialize TensorBoard SummaryWriter if requested
    writer = None
    if use_tensorboard:
        writer = SummaryWriter(tensorboard_path)

    # Initialize variables for early stopping
    best_valid_loss = float("inf")
    best_weights = None
    patience_counter = 0

    # Record loss history (for plotting)
    train_losses_history = []
    valid_losses_history = []

    # Training loop for regression
    for epoch in range(num_epochs):
        # Create a temporary progress bar of a single epoch
        from tqdm import tqdm
        epoch_progress = tqdm(total=len(train_loader) + len(valid_loader),
                             desc=f"Epoch {epoch+1}/{num_epochs}",
                             unit="batch", leave=False, ncols=80)

        net.train()
        total_train_loss = 0
        total_samples_train = 0
        for batch_x, batch_y, batch_w in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)

            optimizer.zero_grad()
            outputs = net(batch_x)

            # Use the debug version of the loss function in the first batch of the first epoch
            if epoch == 0 and total_samples_train == 0:
                print()
                print(f"  [DEBUG] =============== Loss function ===============")
                print(f"  [DEBUG] _NetP_: mean={outputs.mean().item():.6f}, std={outputs.std().item():.6f}, range=[{outputs.min().item():.6f}, {outputs.max().item():.6f}]")
                print(f"  [DEBUG] Target: mean={batch_y.mean().item():.6f}, std={batch_y.std().item():.6f}, range=[{batch_y.min().item():.6f}, {batch_y.max().item():.6f}]")
                # Using the debug version of the loss function
                if hasattr(loss_fn, '__name__') and 'adaptive' in loss_fn.__name__:
                    print(f"  [DEBUG] Use adaptive loss function: {loss_fn.__name__ if hasattr(loss_fn, '__name__') else str(loss_fn)}")
                    loss = adaptive_weighted_mse_loss_debug(outputs, batch_y, batch_w)
                else:
                    loss = loss_fn(outputs, batch_y, batch_w)
                    print(f"  [DEBUG] Using the original loss function: {loss_fn.__name__ if hasattr(loss_fn, '__name__') else str(loss_fn)}")
                print(f"  [DEBUG] ========================================")
            else:
                loss = loss_fn(outputs, batch_y, batch_w)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_y.size(0)
            total_samples_train += batch_y.size(0)

            # Update progress bar
            epoch_progress.update(1)

        average_train_loss = total_train_loss / total_samples_train

        # Validation
        net.eval()
        with torch.no_grad():
            total_valid_loss = 0
            total_samples_valid = 0
            for batch_x, batch_y, batch_w in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_w = batch_w.to(device)
                outputs = net(batch_x)
                loss = loss_fn(outputs, batch_y, batch_w)
                total_valid_loss += loss.item() * batch_y.size(0)
                total_samples_valid += batch_y.size(0)

                # Update progress bar
                epoch_progress.update(1)

            average_valid_loss = total_valid_loss / total_samples_valid

            # Close temporary progress bar
            epoch_progress.close()

            # Record loss history
            train_losses_history.append(average_train_loss)
            valid_losses_history.append(average_valid_loss)

            # Restore detailed printing of the original version
            print(f"  Epoch {epoch + 1:3d}/{num_epochs}: Training Loss: {average_train_loss:.5f}, verify Loss: {average_valid_loss:.5f}")

            # Write to TensorBoard if requested
            if use_tensorboard:
                writer.add_scalar("Train Loss", average_train_loss, epoch)
                writer.add_scalar("Validation Loss", average_valid_loss, epoch)

            # Early Stopping
            if (
                early_stopping_patience is not None
                and average_valid_loss < best_valid_loss - 1e-5
            ):
                best_valid_loss = average_valid_loss
                patience_counter = 0
                best_weights = copy.deepcopy(net.state_dict())  # Record the best weights
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"  Early stop triggered at epoch {epoch + 1}, the best validation loss: {best_valid_loss:.5f}")
                break

    # Load the best weights back to the net
    if best_weights is not None:
        net.load_state_dict(best_weights)
        print(f"  [Predictor Training Complete] The best weight has been loaded, verify Loss: {best_valid_loss:.5f}")
    else:
        print(f"  [Predictor Training Complete] Training completed")

    # Plot loss (if enabled)
    if plot_loss and train_losses_history:
        try:
            from gan.utils.plot_utils import plot_loss_ascii
            plot_loss_ascii(train_losses_history, valid_losses_history,
                           "Predictor Training Loss", width=70, height=12)
        except ImportError:
            print("  [Plot] Unable to import drawing tool, skipping loss graph drawing")
        except Exception as e:
            print(f"  [Plot] An error occurred while drawing the loss graph: {e}")

    # Close the TensorBoard SummaryWriter if used
    if use_tensorboard:
        writer.close()


import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ============================================================================
# Loss function definition module - uniformly defines the loss function for import and use by other modules
# ============================================================================

def weighted_mse_loss(input, target, weights):
    """
    Original weighted MSE loss function
    """
    out = (input - target)**2
    out = out * weights.expand_as(out)
    loss = out.mean()
    return loss

def adaptive_weighted_mse_loss(input, target, weights, debug=False):
    """
    Adaptive normalized loss function
    During loss calculation, input is normalized to the distribution of target to solve the scale mismatch problem

    This is a recommended loss function that can automatically handle the scale difference between the network output and the target value.
    """
    # Calculate the statistical information of target
    target_mean = target.mean()
    target_std = target.std()

    # Calculate the statistical information of input
    input_mean = input.mean()
    input_std = input.std() + 1e-8  # Avoid division by zero

    # Check whether the standard deviation of target is too small (border case processing)
    if target_std < 1e-6:
        if debug:
            print(f"    [ADAPTIVE_LOSS_DEBUG] Target std too small ({target_std.item():.8f}) -> Original Loss")
        # Fall back to the original loss function
        return weighted_mse_loss(input, target, weights)

    # Distribution of normalizing input to target: (input - input_mean) / input_std * target_std + target_mean
    input_normalized = (input - input_mean) / input_std * target_std + target_mean

    # Debugging information: only keep key information
    if debug:
        scale_ratio = abs(input_std.item() / (target_std.item() + 1e-8))
        print(f"    [ADAPTIVE_LOSS] Scale: {scale_ratio:.1f} x -> 1.0x")

    # Calculate the loss after normalization
    out = (input_normalized - target)**2
    out = out * weights.expand_as(out)
    loss = out.mean()

    return loss

# Debug version of the adaptive loss function, used for debugging during training
def adaptive_weighted_mse_loss_debug(input, target, weights):
    """Debug version of the adaptive normalized loss function, always enable debug output"""
    return adaptive_weighted_mse_loss(input, target, weights, debug=True)

# ============================================================================

def train_net_p_with_weight(cfg,net,x,y,weights,lr=0.001):
    # Example usage
    x_train, x_valid, y_train, y_valid,weights_train,weights_valid = train_test_split(x, y,weights, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = DataLoader(TensorDataset(x_train, y_train,weights_train),
                                batch_size=cfg.batch_size_p, shuffle=True,
                            )
    valid_loader = DataLoader(TensorDataset(x_valid, y_valid,weights_valid),
                              batch_size=cfg.batch_size_p, shuffle=False)

    # Select loss function
    # loss_fn = torch.nn.MSELoss()
    loss_fn = adaptive_weighted_mse_loss  # Recommendation: Adaptive normalized loss function
    # loss_fn = weighted_mse_loss  # Alternative: original loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)


    # Check whether loss plot drawing is enabled
    plot_loss = getattr(cfg, 'plot_loss', False)  # Default is False

    train_regression_model_with_weight(train_loader, valid_loader, net,
                           loss_fn, optimizer, device=cfg.device,
                           num_epochs=cfg.num_epochs_p, use_tensorboard=False,
                           tensorboard_path='logs', early_stopping_patience=cfg.es_p,
                           plot_loss=plot_loss)
