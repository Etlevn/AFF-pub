# Masker network module - responsible for ensuring that the generated factor expression complies with grammatical rules
import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from gan.utils.builder import Builders


class NetM(nn.Module):
    """
    Masker network - ensures that the generated factor expression conforms to the grammar rules

    Function:
    1. Generate a valid token mask based on the current expression state
    2. Apply mask to ensure only valid token is selected
    3. Maintain expression builder state

    Working principle:
    - token is generated position by position, and each position determines valid token according to the current syntax status
    - Use Builders object to track expression build status
    - Apply mask to set logits of invalid token to minimum value
    """
    def __init__(
        self,
        max_len = 20,      # Maximum sequence length
        size_action = 48,   # Action space size (token number of types)

    ):
        super().__init__()
        self.max_len = max_len      # Save the maximum sequence length
        self.size_action = size_action  # Save action space size

    def forward(self, x: torch.Tensor):
        """
        Forward pass - applying syntax constraint mask

        Args:
            x: logits (batch_size, seq_len, n_actions) output by the generator

        Returns:
            masked_x: logits (batch_size, seq_len, n_actions) after applying mask
            masks: syntax constraint mask (batch_size, seq_len, n_actions)
            blds: expression builder object
        """
        # x: (batch_size, seq_len, n_actions) - logits output by the generator

        device = x.device  # Obtain device information
        bs,seq_len,n_actions = x.shape  # Get tensor dimensions
        blds = Builders(bs,max_len=seq_len,n_actions=n_actions)  # Create expression builder

        # Initialize output tensor
        masks = torch.zeros(bs,seq_len,n_actions).to(device)  # Syntax constraint mask
        masked_x = torch.zeros(bs,seq_len,n_actions).to(device)  # logits after applying mask
        # prev_select = None  # Commented out previous position selection

        # Position-by-position processing sequence
        for i in range(seq_len):

            # Obtain the syntax constraint mask of the current position
            if i<=self.max_len:
                mask = blds.get_valid_op()  # (batch_size, n_actions) - Get valid token according to the current status
            else:
                # Exceeds the maximum length, only the terminator is allowed to be selected
                mask = np.zeros([bs, n_actions],dtype=bool)  # Initialize full False mask
                mask[:,n_actions-1] = True  # Only the last token (terminator) is allowed to be selected

            # Ensure that the mask matches the desired output dimensions
            if mask.shape[1] != n_actions:
                if mask.shape[1] > n_actions:
                    # If the mask dimension is greater than the expected dimension, intercept the previous part
                    mask = mask[:, :n_actions]
                else:
                    # If the mask dimension is smaller than the desired dimension, fill it with False
                    mask_padded = np.zeros((mask.shape[0], n_actions), dtype=bool)
                    mask_padded[:, :mask.shape[1]] = mask
                    mask = mask_padded

            # Convert the mask to tensor and store it
            mask_tensor = torch.from_numpy(mask).to(device)  # Convert to tensor
            masks[:,i,:] = mask_tensor  # Store the mask of the current location

            # Apply mask and select token
            logit = x[:,i,:]  # logits (batch_size, n_actions) at the current position
            tmp = logit.detach().cpu().numpy()  # Convert to numpy array
            # Make sure the mask is consistent with the logits dimension (check again, just in case)
            if mask.shape[1] != tmp.shape[1]:
                if mask.shape[1] > tmp.shape[1]:
                    mask = mask[:, :tmp.shape[1]]
                else:
                    mask_padded = np.zeros((mask.shape[0], tmp.shape[1]), dtype=bool)
                    mask_padded[:, :mask.shape[1]] = mask
                    mask = mask_padded
            tmp[~mask] = -1e8  # Set logits of invalid token to a minimum value
            select = tmp.argmax(axis=1)  # Greedy decoding selects the best token (batch_size,)
            # prev_select = select  # Commented out previous position selection

            # Verify whether the selected token is valid (repaired assertion)
            assert all(mask[i, select[i]] for i in range(bs)), "The selected token must be in a valid mask"

            # Update expression builder status
            blds.add_token(select)  # Add the selected token to the builder

            # Generate logits after applying mask
            masked_x[:,i,:] = x[:,i,:]  # Copy original logits
            # Apply mask to output to ensure dimensions match
            if mask_tensor.shape[1] == masked_x.shape[2]:
                masked_x[:,i,:][~mask_tensor] = -1e8  # Set logits of invalid token to a minimum value
            else:
                # If dimensions do not match, apply mask only to matching parts
                min_dim = min(mask_tensor.shape[1], masked_x.shape[2])
                temp_mask = mask_tensor[:, :min_dim]
                masked_x[:,i,:min_dim][~temp_mask] = -1e8

        return masked_x,masks,blds  # Return the masked logits, mask and builder
