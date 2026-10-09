# Generator network module - responsible for generating the token sequence of factor expressions
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gan.utils.builder import Builders
from gan.utils import save_blds

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import variable
from gan.network.loss import get_losses
import copy

class NetG_DCGAN(nn.Module):
    """
    DCGAN style generator network - using transposed convolution to generate the factor-expression token sequence from the latent space

    Network architecture:
    1. Linear layer: Map latent vectors to initial features
    2. Transposed convolution layer: gradually upsampling to generate sequence features
    3. Convolution layer: finally generate token probability distribution
    """
    def __init__(
            self,
            n_chars:int,        # Vocabulary size (token number of categories)
            latent_size: int,    # Latent space dimensions
            seq_len: int,        # Sequence length (fixed to 20)
            hidden: int,         # Hidden layer dimension (not used, keep the interface consistent)

        ):
        super().__init__()
        assert seq_len == 20  # Ensure that the sequence length is 20
        use_bias=True

        # Linear layer: map latent vectors to initial feature space
        self.linear=nn.Linear(latent_size,6*384)  # Output dimension: 6*384

        # Transposed convolution layer: stepwise upsampling to generate sequence features
        self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(384,256,(6,1),(1,1),bias=use_bias),#[batch, 256, 47, 1]
                    nn.BatchNorm2d(256),nn.ReLU(),
                    nn.ConvTranspose2d(256,192,(5,1),(1,1),bias=use_bias),#[batch, 192, 100, 1]
                    nn.BatchNorm2d(192),nn.ReLU(),
                    nn.ConvTranspose2d(192,128,(6,1),(1,1),bias=use_bias),#[batch, 128, 205, 1]
                    nn.BatchNorm2d(128),nn.ReLU(),
                )

        # Convolution layer: finally generate token probability distribution
        self.conv=nn.Sequential(
                    nn.ZeroPad2d((0,0,4,3)),#[batch, 128, 212, 1] - Add padding
                    nn.Conv2d(128,128,(8,1),(1,1),0,bias=use_bias),#[batch, 128, 205, 1]
                    nn.BatchNorm2d(128),nn.ReLU(),
                    nn.ZeroPad2d((0,0,4,3)),#[batch, 128, 212, 1] - Add padding
                    nn.Conv2d(128,64,(8,1),(1,1),0,bias=use_bias),#[batch, 64, 205, 1]
                    nn.BatchNorm2d(64),nn.ReLU(),
                    nn.ZeroPad2d((0,0,4,3)),#[batch, 64, 212, 1] - Add padding
                    nn.Conv2d(64,n_chars,(8,1),(1,1),0,bias=use_bias),#[batch, n_chars, 205, 1] - Output token probability
        #             nn.BatchNorm2d(4)
        )

    def initialize_parameters(self):
        """Initialize network parameters - use Xavier normal distribution to initialize weights, and bias initialization to 0"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape)>1:
                nn.init.xavier_normal_(param)  # Weights are initialized using Xavier normal distribution
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  # The bias is initialized to 0

    def forward(self, x):
        """
        Forward pass - generate token sequence from latent vectors

        Args:
            x: latent vector (batch_size, latent_size)

        Returns:
            x: token logits (batch_size, seq_len, n_chars)
        """
        x = self.linear(x)  # Linear transformation: (batch_size, 6*384)
        x = x.view(x.shape[0],384,6,1)  # Reshape into convolution input: (batch_size, 384, 6, 1)
        x = self.deconv(x)  # Transposed convolution upsampling: (batch_size, 128, 205, 1)
        x = self.conv(x)  # Convolution generates final features: (batch_size, n_chars, 205, 1)
        x = x.view(x.shape[0],x.shape[2],x.shape[1])  # Reshape into sequence: (batch_size, 205, n_chars)
        return x  # Output: (batch_size, seq_len, n_chars) - token logits

class NetG_Lstm(nn.Module):
    """
    LSTM Generator Network - Use LSTM to generate the factor-expression token sequence from the latent space

    Network architecture:
    1. Latent vector mapping: Map the latent vector to the initial state of LSTM
    2. Embedding layer: Convert token ID into vector representation
    3. LSTM layer: Generate sequence features
    4. Output layer: Generate token probability distribution
    """
    def __init__(
        self,
        n_chars:int,        # Vocabulary size (token number of categories)
        n_layers: int,       # LSTM layer number
        d_model: int,        # Model dimensions (hidden layer size)
        dropout: float,      # Dropout ratio
        seq_len: int,        # Sequence length
        potential_size: int, # Latent space dimensions
    ):
        super().__init__()
        self.n_chars = n_chars
        self.max_len = seq_len
        self.n_layers = n_layers
        self.d_model = d_model

        # Latent vector mapping layer: Map the latent vector to the initial hidden state of LSTM
        self.fc_h = nn.Sequential(
            nn.Linear(potential_size,n_layers*d_model),nn.ReLU()  # Map to hidden state
        )
        # Latent vector mapping layer: Map the latent vector to the initial cell state of LSTM
        self.fc_c = nn.Sequential(
            nn.Linear(potential_size,n_layers*d_model),nn.ReLU()  # Map to cell state
        )
        # Embedding layer: Convert token ID into vector representation (including padding token)
        self.emb = nn.Embedding(n_chars+1,d_model,0)  # +1 is for padding token
        # LSTM layer: generate sequence features
        self.rnn = nn.LSTM(
            input_size = d_model,      # Input dimensions
            hidden_size = d_model,     # Hidden layer dimension
            num_layers = n_layers,     # LSTM layer number
            batch_first = True,        # Batch dimension first
            dropout = dropout          # Dropout ratio
        )
        # Output layer: generate token probability distribution
        self.fc = nn.Linear(d_model,n_chars)  # Map to vocabulary size
    def initialize_parameters(self):
        """Initialize network parameters - use Xavier normal distribution to initialize weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)  # Weights are initialized using Xavier normal distribution
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        # for name, param in self.named_parameters():
        #     nn.init.xavier_normal_(param.data)  # Initialize weight matrices using Xavier initialization

    def forward(self,z):
        """
        Forward pass - generate token sequence from latent vectors (version with syntax constraints)

        Args:
            z: latent vector (batch_size, potential_size)

        Returns:
            tuple: ((onehot_s, logit_s, mask_s), builders)
                - onehot_s: generated token sequence (batch_size, max_len)
                - logit_s: token logits (batch_size, max_len, n_chars)
                - mask_s: syntax constraint mask (batch_size, max_len, n_chars)
                - builders: expression builder object
        """
        # z: (batch_size, potential_size)
        # h,c:(n_layers, batch_size, hidden_size)
        device,bs = z.device,z.shape[0]  # Get device and batch size

        # Map the latent vector to the initial state of LSTM
        h = self.fc_h(z).view(bs,self.n_layers,self.d_model).permute(1,0,2)  # Hidden state
        c = self.fc_c(z).view(bs,self.n_layers,self.d_model).permute(1,0,2)  # Cell status

        # Create expression builder - for syntax constraints
        builders = Builders(bs, self.max_len, self.n_chars)  # One builder per sample
        result = []

        # Initialize input token (use padding token as the starting point)
        input_step = torch.full((bs,),fill_value=self.n_chars,dtype=torch.long)

        # Initialize output sequence and mask
        onehot_s = np.zeros([bs,self.max_len])  # The generated token sequence
        logit_s =torch.zeros(bs,self.max_len,self.n_chars,device=device)  # token logits
        mask_s = torch.zeros(bs,self.max_len,self.n_chars,dtype=torch.bool,device=device)  # Syntax constraint mask

        # Autoregressive generation sequence
        for t in range(self.max_len):

            # Embed current token
            embedded = self.emb(input_step)[:,None]  # (batch_size, 1, d_model)

            # LSTM forward propagation
            if h is None:
                output,(h,c) = self.rnn(embedded)  # First call
            else:
                output,(h,c) = self.rnn(embedded,(h,c))  # Subsequent calls

            # Get the current syntax constraint mask
            mask = builders.get_valid_op()  # (batch_size, n_action) - valid token

            # Generate token logits
            logit = self.fc(output).squeeze(1)  # (batch_size, n_chars)

            # Apply syntax constraints - set logits of invalid token to a minimum value
            tmp = logit.detach().cpu().numpy().copy()
            # Ensure that the mask matches the logits dimension
            if mask.shape[1] > tmp.shape[1]:
                # If the mask dimension is greater than the logits dimension, intercept the previous part
                mask = mask[:, :tmp.shape[1]]
            elif mask.shape[1] < tmp.shape[1]:
                # If the mask dimension is less than the logits dimension, fill it with False
                mask_padded = np.zeros((mask.shape[0], tmp.shape[1]), dtype=bool)
                mask_padded[:, :mask.shape[1]] = mask
                mask = mask_padded
            tmp[~mask]=-1e8  # Invalid token is set to a minimum value

            # Create mask_tensor matching logits dimensions
            mask_tensor = torch.from_numpy(mask).to(device)
            onehot = tmp.argmax(1)  # Greedy decoding selects the best token

            # Verify whether the selected token is valid
            assert (mask[:,onehot]*1.).mean()  # Ensure that the selected token is in a valid mask

            # Update expression builder status
            builders.add_token(onehot)  # Add the selected token to the builder

            # Save the generated token and mask
            onehot_s[:,t] = onehot  # Save the generated token
            logit_s[:,t,:] = logit  # Save logits
            mask_s[:,t,:] = mask_tensor  # Save mask

            # Update input token for next step
            input_step = torch.from_numpy(onehot).to(device)

        return (onehot_s,logit_s,mask_s),builders  # Return the generated sequence and builder

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

class NetG_CNN(nn.Module):
    """
    CNN Generator Network - Generate factor-expression token sequence from latent space using 1D convolution and residual blocks

    Network architecture:
    1. Linear layer: mapping latent vectors to sequence features
    2. Residual block: Use ResBlock for feature extraction
    3. Convolutional layer: Generate token probability distribution
    """
    def __init__(self, n_chars, latent_size,seq_len , hidden):
        super( ).__init__()
        # Linear layer: maps latent vectors to sequence feature space
        self.fc1 = nn.Linear(latent_size, hidden*seq_len)  # Output dimension: hidden*seq_len

        # Residual block sequence: used for feature extraction and conversion
        self.block = nn.Sequential(
            ResBlock(hidden),  # The first residual block
            ResBlock(hidden),  # The second residual block
            # ResBlock(hidden), # Commented out extra residual block
            # ResBlock(hidden),
            # ResBlock(hidden),
        )

        # Final convolution layer: generate token probability distribution
        self.conv1 = nn.Conv1d(hidden, n_chars, 1)  # 1x1 convolution, mapped to vocabulary size

        # Save network parameters
        self.n_chars = n_chars      # Vocabulary size
        self.seq_len = seq_len      # Sequence length
        self.hidden = hidden        # Hidden layer dimension

    def initialize_parameters(self):
        """Initialize network parameters - use Xavier normal distribution to initialize weights, and bias initialization to 0"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape)>1:
                nn.init.xavier_normal_(param)  # Weights are initialized using Xavier normal distribution
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  # The bias is initialized to 0

    def forward(self, noise):
        """
        Forward pass - generate token sequence from latent vectors

        Args:
            noise: latent vector (batch_size, latent_size)

        Returns:
            output: token logits (batch_size, seq_len, n_chars)
        """
        batch_size = noise.size(0)  # Get batch size

        # Linear transformation: mapping latent vectors to sequence features
        output = self.fc1(noise)  # (batch_size, hidden*seq_len)

        # Reshape into convolution input format
        output = output.view(-1, self.hidden, self.seq_len)  # (batch_size, hidden, seq_len)

        # Feature extraction through residual blocks
        output = self.block(output)  # (batch_size, hidden, seq_len)

        # Final convolution layer: generate token probability distribution
        output = self.conv1(output)  # (batch_size, n_chars, seq_len)

        # Adjust dimension order: (batch_size, seq_len, n_chars)
        output = output.transpose(1, 2)  # Exchange seq_len and n_chars dimensions

        # Ensure memory continuity and reshape
        shape = output.size()  # Save original shape
        output = output.contiguous()  # Ensure memory continuity
        output = output.view(batch_size*self.seq_len, -1)  # Flatten to 2D

        return output.view(shape)  # Restore original shape: (batch_size, seq_len, n_chars)

def train_network_generator(netG, netM, netP, cfg, data, target,current_round,random_method,metric,lr,n_actions):
    """
    Training generator network - Optimizing generators using adversarial training to produce high-quality factor expressions

    Training strategy:
    1. Generator generates the factor-expression token sequence
    2. Masker ensures syntax correctness
    3. Predictor evaluation expression quality
    4. Optimizing the generator through multi-component loss functions

    Args:
        netG: generator network
        netM: Masker network
        netP: predictor network
        cfg: Configuration object
        data: Stock data
        target: target variable
        current_round: Current training round
        random_method: Random noise generation method
        metric: Evaluation index function
        lr: learning rate
        n_actions: Action space size
    """
    print(f"  [Generator Training] round {current_round+1}, maximum epoch: {cfg.num_epochs_g}, early stop threshold: {cfg.g_es}")

    # Initialize optimizer
    opt = torch.optim.Adam(netG.parameters(),lr=lr)  # Adam optimizer

    # Early stop related variables
    best_weights = None      # Optimal weight
    best_score = -float('inf')  # Best score
    patience_counter = 0     # Patience counter

    # Initialize latent vector (used to generate two different expressions)
    z1 = torch.zeros([cfg.batch_size,cfg.potential_size]).to(cfg.device)  # The first latent vector
    z2 = torch.zeros([cfg.batch_size,cfg.potential_size]).to(cfg.device)  # The second latent vector

    # Set masker and predictor to evaluation mode
    netM.eval()  # The masker does not participate in gradient calculation
    netP.eval()  # Predictor does not participate in gradient calculation

    # Training state variables
    empty_blds = None        # Empty builder (used to save the best results)
    best_str_to_print = ''   # Best expression string

    # Loss history (for plotting)
    loss_history = []

    for epoch in range(cfg.num_epochs_g):
        """Training loop - each epoch optimized generator"""
        # Display temporary progress bar at the beginning of epoch
        from tqdm import tqdm
        import time

        # Create a single epoch progress bar, divided into more processing steps
        total_steps = 12  # Increase the number of steps to provide more granular progress
        epoch_progress = tqdm(total=total_steps, desc=f"Epoch {epoch+1}/{cfg.num_epochs_g}",
                             unit="step", leave=False, ncols=120)

        epoch_start_time = time.time()  # Record epoch start time
        netG.train()  # Set the generator to training mode
        opt.zero_grad()  # Clear gradient

        # Step 1: Generate random noise - provide input to the generator
        epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - generate noise")
        z1 = random_method(z1)  # Generate the first latent vector
        z2 = random_method(z2)  # Generate a second latent vector (for similarity loss)
        epoch_progress.update(1)

        # Step 2: Generator forward propagation - generating factor expression
        epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - Generator")
        netG_output_1 = netG(z1)  # Generate the first expression
        netG_output_2 = netG(z2)  # Generate the second expression
        epoch_progress.update(1)

        # Step 3: Masker processing - ensure grammatical correctness
        epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - Masker")

        # Handling the output formats of different generators
        if isinstance(netG_output_1, tuple):
            # LSTM generator: return ((onehot_s, logit_s, mask_s), builders)
            (onehot_s_1, logit_s_1, mask_s_1), blds_1 = netG_output_1
            (onehot_s_2, logit_s_2, mask_s_2), blds_2 = netG_output_2
            # LSTM generator already contains syntax constraints, use its output directly
            # logit_s and mask_s are already tensor, just make sure the equipment is correct
            masked_x_1 = logit_s_1.to(cfg.device)
            masked_x_2 = logit_s_2.to(cfg.device)
            masks_1 = mask_s_1.to(cfg.device)
            masks_2 = mask_s_2.to(cfg.device)
            logit_raw_1 = masked_x_1
            logit_raw_2 = masked_x_2
        else:
            # DCGAN/CNN generator: return logits
            logit_raw_1 = netG_output_1
            logit_raw_2 = netG_output_2
            masked_x_1, masks_1, blds_1 = netM(logit_raw_1)  # The masker processes the first expression
            masked_x_2, masks_2, blds_2 = netM(logit_raw_2)  # The masker processes the second expression
        epoch_progress.update(1)

        # Step 4: Predictor forward propagation
        epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - Predictor")
        onehot_tensor_1 = F.gumbel_softmax(masked_x_1,hard=True)
        pred_1,latent_1 = netP(onehot_tensor_1,latent=True)

        onehot_tensor_2 = F.gumbel_softmax(masked_x_2,hard=True)
        pred_2,latent_2 = netP(onehot_tensor_2,latent=True)
        epoch_progress.update(1)

        # Step 5: Loss calculation
        epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - Calculate loss")
        loss_inputs = {
            'logit_raw_1':logit_raw_1,#(batch_size,MAX_EXPR_LENGTH,SIZE_ACTION)
            'logit_raw_2':logit_raw_2,
            'masked_x_1':masked_x_1,#(batch_size,MAX_EXPR_LENGTH,SIZE_ACTION)
            'masked_x_2':masked_x_2,
            'masks_1':masks_1,#(batch_size,MAX_EXPR_LENGTH,SIZE_ACTION)
            'masks_2':masks_2,
            'blds_1':blds_1,
            'blds_2':blds_2,
            'z1':z1,#(batch_size,latent_size)
            'z2':z2,
            'onehot_tensor_1':onehot_tensor_1,#(batch_size,MAX_EXPR_LENGTH,SIZE_ACTION)
            'onehot_tensor_2':onehot_tensor_2,
            'pred_1':pred_1,#(batch_size,1)
            'pred_2':pred_2,
            'latent_1':latent_1,#(batch_size,256)
            'latent_2':latent_2,
        }
        loss, loss_components = get_losses(loss_inputs,cfg)
        # Record loss history
        loss_history.append(loss.item())
        epoch_progress.update(1)

        # Step 6: Merge Builder
        epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - merge builder")
        blds:Builders = blds_1+blds_2
        epoch_progress.update(1)

        # Step 7: Check validity
        epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - Check validity")
        idx = [i for i in range(blds.batch_size) if blds.builders[i].is_valid()]
        epoch_progress.update(1)

        # Step 8: Filter invalid expressions
        epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - filter invalid expressions")
        pre_filter_size = blds.batch_size
        blds.drop_invalid()
        post_filter_size = blds.batch_size
        if pre_filter_size > 100:  # Display filter information for large batches
            epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - Filter: {pre_filter_size}→{post_filter_size}")
        epoch_progress.update(1)

        # Step 9: Prepare evaluation data
        if blds.batch_size > 0:
            epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - Prepare for assessment {blds.batch_size} expressions")
        else:
            epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - no valid expression")
        epoch_progress.update(1)

        # Step 10: Evaluate expression performance
        epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - Assessment {blds.batch_size} expressions")
        # If the number of expressions is large, use verbose mode to show evaluation progress
        eval_verbose = blds.batch_size > 50
        if eval_verbose:
            epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - Start evaluation {blds.batch_size} expressions...")
        blds.evaluate(data,target,metric,verbose=eval_verbose)
        epoch_progress.update(1)

        # Step 11: Calculate statistical information
        epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - Calculate statistics")
        # Move statistical calculations here from behind
        mean_score = np.mean(blds.scores)
        max_score = np.max(blds.scores) if len(blds.scores)>0 else 0
        std_score = np.std(blds.scores) if len(blds.scores)>0 else 0
        epoch_progress.update(1)

        # Step 12: Backpropagation and optimization
        if epoch > 0:
            epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - Backpropagation")
            loss.backward()
            opt.step()
        else:
            epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - Complete initialization")
        epoch_progress.update(1)

        # Display epoch summary before closing the progress bar
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        if blds.batch_size > 0:
            epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - Complete ({epoch_duration:.1f} s)")
        else:
            epoch_progress.set_description(f"Epoch {epoch+1}/{cfg.num_epochs_g} - no valid expression ({epoch_duration:.1f} s)")
        epoch_progress.close()

        # Restore detailed printing of the original (using calculated statistics)
        str_to_print = f"  Epoch {epoch+1:3d}/{cfg.num_epochs_g}: valid expression {len(idx):3d}→{len(blds.scores):3d}"
        str_to_print += f", the highest score: {max_score:.4f}, average: {mean_score:.4f}, standard deviation: {std_score:.4f}"
        blds.drop_duplicated()
        str_to_print += f", after deduplication: {blds.batch_size}"
        print(str_to_print)

        # Detailed printing loss composition
        loss_str = f"    Loss: {loss:.4f} = "
        loss_parts = []
        # Display all possible loss components in a fixed order
        component_order = ['simi', 'pred', 'potential', 'entropy']
        for comp_name in component_order:
            if comp_name in loss_components:
                comp_data = loss_components[comp_name]
                # Only the components whose weight is not 0 are displayed, even if the weighted loss is 0, it will be displayed
                if comp_data['weight'] != 0:
                    loss_parts.append(f"{comp_data['weighted']:.4f}({comp_name})")
        if loss_parts:
            loss_str += " + ".join(loss_parts)
        else:
            loss_str += "0.0000"
        print(loss_str)
        if max_score>0:
            exprs = blds.exprs_str[np.argmax(blds.scores)]
            print(f"    Optimal expression (score {max_score:.4f}): {exprs}")

        if empty_blds is None:
            empty_blds = blds
        else:
            empty_blds = empty_blds + blds


        if cfg.g_es_score == 'mean':
            es_score = mean_score
        elif cfg.g_es_score == 'max':
            es_score = max_score
        elif cfg.g_es_score == 'combined':
            es_score = max_score + 2. *  std_score
        else:
            raise NotImplementedError

        if es_score > best_score:
            best_score = es_score
            best_weights = copy.deepcopy(netG.state_dict())
            best_str_to_print = str_to_print
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > cfg.g_es:
                print(f'  Early stop triggered at epoch {epoch+1}, best score: {best_score:.4f}')
                break

    if best_weights is not None:
        print('  Load optimal weight')
        netG.load_state_dict(best_weights)
        print(f"  Best result: {best_str_to_print}")

    empty_blds.drop_duplicated()
    print(f"  [Generator Training Complete] generated {empty_blds.batch_size} unique expressions")

    # Plot loss (if enabled)
    plot_loss = getattr(cfg, 'plot_loss', False)  # Default is False
    if plot_loss and loss_history:
        try:
            from gan.utils.plot_utils import plot_generator_loss_ascii
            plot_generator_loss_ascii(loss_history,
                                    "Generator Training Loss", width=70, height=12)
        except ImportError:
            print("  [Plot] Unable to import drawing tool, skipping loss graph drawing")
        except Exception as e:
            print(f"  [Plot] An error occurred while drawing the loss graph: {e}")

    return empty_blds
