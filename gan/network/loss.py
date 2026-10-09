
# Generator loss function module - defines a multi-component loss function for optimizing the generator
import torch

def loss_simi(loss_inputs,cfg):
    """
    Similarity loss function - penalizes generators for producing expressions that are too similar

    Goal: Encourage generators to generate diverse factor expressions and avoid generating duplicate or too similar expressions

    Args:
        loss_inputs: A dictionary containing two expressions one-hot encoding
        cfg: Configuration object, including l_simi_thresh threshold

    Returns:
        simi: similarity loss value (scalar)
    """
    # Obtain the one-hot encoding of two expressions
    onehot_tensor_1 = loss_inputs['onehot_tensor_1']  # (batch_size, seq_len, n_actions)
    onehot_tensor_2 = loss_inputs['onehot_tensor_2']  # (batch_size, seq_len, n_actions)

    # Calculate similarity: sum after element-wise multiplication
    simi = torch.sum(onehot_tensor_1*onehot_tensor_2,dim=-1).sum(dim = -1)  # (batch_size,)
    simi = simi / onehot_tensor_1.shape[1]  # Normalization: divided by sequence length

    # Apply threshold: Penalize only when similarity exceeds threshold
    simi = simi - cfg.l_simi_thresh  # Subtract threshold
    simi = torch.relu(simi)  # Only keep positive values (ReLU)
    # simi = simi**2  # Commented out square penalty
    simi = simi.mean()  # Take the average
    return simi

def loss_pred(loss_inputs,cfg):
    """
    Predictor loss function - encourages the generator to produce high-quality expressions

    Goal: Maximize the score of the predictor to the generated expression, indirectly optimizing the factor quality
    Note: The negative sign is used because we want to maximize the predictor score, but the optimizer is to minimize the loss

    Args:
        loss_inputs: Dictionary containing predictor scores
        cfg: Configuration object

    Returns:
        loss: Predictor loss value (scalar, negative predictor score)
    """
    # Get the score of the predictor for the first expression
    pred_1 = loss_inputs['pred_1'][:,0]  # (batch_size,) - the first element of the predictor output

    return - pred_1.mean()  # Return a negative average rating (because we want to maximize the rating)

def loss_potential(loss_inputs,cfg):
    """
    Latent space loss function - encourages the generator to produce distinctive features in the latent space

    Goal: Ensure that the generator produces discriminative feature representations in the latent space of the predictor,
          Avoid all expressions mapping to the same underlying representation

    Args:
        loss_inputs: A dictionary containing latent features of two expressions
        cfg: Configuration object, including epsilon and threshold parameters

    Returns:
        similarity: Latent-space similarity loss value (scalar)
    """

    epsilong=cfg.l_potential_epsilon  # Clipping boundaries to avoid numerical instability
    u1,u2=loss_inputs['latent_1'],loss_inputs['latent_2']  # (batch_size, latent_size_netP)

    # Cut latent features to a safe range to avoid numerical instability
    u1=u1.clip(epsilong,1-epsilong)  # Crop to [epsilon, 1-epsilon]
    u2=u2.clip(epsilong,1-epsilong)  # Crop to [epsilon, 1-epsilon]

    # Calculate cosine similarity
    similarity=(u1*u2).sum(axis=1)/ (
        ((u1**2).sum(axis=1))**0.5 * ((u2**2).sum(axis=1))**0.5
                            ) -cfg.l_potential_thresh  # Subtract threshold

    similarity=similarity*(similarity>0)  # Only keep positive values (ReLU)
    return similarity.mean()  # Return the average loss

def loss_entropy(loss_inputs,cfg):
    """
    Entropy loss function - a distribution of expressions that encourages the generator to generate uncertainty

    Goal: Increase the uncertainty of the generator output and prevent the generator from converging to a deterministic output
    Note: This loss function is not used in the current version (l_entropy=0)

    Args:
        loss_inputs: A dictionary containing two expressions one-hot encoding
        cfg: Configuration object

    Returns:
        entropy: entropy loss value (scalar)
    """
    # Obtain the one-hot encoding of two expressions
    onehot_tensor_1 = loss_inputs['onehot_tensor_1']  # (batch_size, seq_len, n_actions)
    onehot_tensor_2 = loss_inputs['onehot_tensor_2']  # (batch_size, seq_len, n_actions)

    # Calculate the entropy of the first expression
    entropy_1 = -torch.sum(onehot_tensor_1*torch.log(onehot_tensor_1),dim=-1).sum(dim = -1)  # (batch_size,)
    entropy_1 = entropy_1 / onehot_tensor_1.shape[1]  # Normalization

    # Calculate the entropy of the second expression
    entropy_2 = -torch.sum(onehot_tensor_2*torch.log(onehot_tensor_2),dim=-1).sum(dim = -1)  # (batch_size,)
    entropy_2 = entropy_2 / onehot_tensor_2.shape[1]  # Normalization

    # Total entropy loss
    entropy = entropy_1 + entropy_2  # The sum of the entropies of two expressions
    entropy = entropy.mean()  # Take the average
    return entropy
def get_losses(loss_inputs,cfg):
    """
    Multi-component loss function - combining multiple loss components for generator optimization

    Loss component:
    1. loss_simi: Similarity loss - encourage diversity
    2. loss_pred: Predictor loss - encourage high quality
    3. loss_potential: Latent space loss - Encourage feature discrimination
    4. loss_entropy: Entropy loss - Encourage uncertainty (not currently used)

    Args:
        loss_inputs: A dictionary containing all inputs required for loss calculations
        cfg: Configuration object, including the weight of each loss component

    Returns:
        loss: Total loss value (scalar)
        loss_components: Loss component dictionary (for detailed printing)
    """

    loss = 0  # Initialization total loss
    loss_components = {}  # Initialize loss component dictionary

    # Similarity loss: punish expressions that are too similar
    if cfg.l_simi != 0 :
        simi_loss = loss_simi(loss_inputs,cfg)
        weighted_simi_loss = cfg.l_simi * simi_loss
        loss += weighted_simi_loss
        loss_components['simi'] = {
            'raw': simi_loss.item(),
            'weighted': weighted_simi_loss.item(),
            'weight': cfg.l_simi
        }

    # Predictor loss: Encourage the generation of high-quality expressions
    if cfg.l_pred != 0 :
        pred_loss = loss_pred(loss_inputs,cfg)
        weighted_pred_loss = cfg.l_pred * pred_loss
        loss += weighted_pred_loss
        loss_components['pred'] = {
            'raw': pred_loss.item(),
            'weighted': weighted_pred_loss.item(),
            'weight': cfg.l_pred
        }

    # Latent space loss: Encourage distinctive features in the latent space
    if cfg.l_potential !=0 :
        potential_loss = loss_potential(loss_inputs,cfg)
        weighted_potential_loss = cfg.l_potential * potential_loss
        loss += weighted_potential_loss
        loss_components['potential'] = {
            'raw': potential_loss.item(),
            'weighted': weighted_potential_loss.item(),
            'weight': cfg.l_potential
        }

    # Entropy loss: Encourage uncertainty (not currently used)
    if cfg.l_entropy !=0 :
        entropy_loss = loss_entropy(loss_inputs,cfg)
        weighted_entropy_loss = cfg.l_entropy * entropy_loss
        loss += weighted_entropy_loss
        loss_components['entropy'] = {
            'raw': entropy_loss.item(),
            'weighted': weighted_entropy_loss.item(),
            'weight': cfg.l_entropy
        }

    return loss, loss_components  # Return the total loss and loss components
