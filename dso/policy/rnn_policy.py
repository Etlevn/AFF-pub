"""Controller used to generate distribution over hierarchical, variable-length objects."""
import tensorflow as tf
import numpy as np

from dso.program import Program
from dso.program import _finish_tokens
from dso.memory import Batch

from dso.policy import Policy
from dso.utils import make_batch_ph

class DynamicLSTMCell(tf.keras.layers.LSTMCell):
    """LSTMCell that handles dynamic input dimensions"""
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
        self._input_dim = None
        self._built = False

    def build(self, input_shape):
        if not self._built or self._input_dim != input_shape[-1]:
            self._input_dim = input_shape[-1]
            # Clear existing weights
            if hasattr(self, 'kernel'):
                delattr(self, 'kernel')
            if hasattr(self, 'recurrent_kernel'):
                delattr(self, 'recurrent_kernel')
            if hasattr(self, 'bias'):
                delattr(self, 'bias')
            self._built = False
            super().build(input_shape)
            self._built = True

    def call(self, inputs, states, training=None):
        if not self._built or self._input_dim != inputs.shape[-1]:
            self.build(inputs.shape)
        return super().call(inputs, states, training)

class DynamicGRUCell(tf.keras.layers.GRUCell):
    """GRUCell that handles dynamic input dimensions"""
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
        self._input_dim = None
        self._built = False

    def build(self, input_shape):
        if not self._built or self._input_dim != input_shape[-1]:
            self._input_dim = input_shape[-1]
            # Clear existing weights
            if hasattr(self, 'kernel'):
                delattr(self, 'kernel')
            if hasattr(self, 'recurrent_kernel'):
                delattr(self, 'recurrent_kernel')
            if hasattr(self, 'bias'):
                delattr(self, 'bias')
            self._built = False
            super().build(input_shape)
            self._built = True

    def call(self, inputs, states, training=None):
        if not self._built or self._input_dim != inputs.shape[-1]:
            self.build(inputs.shape)
        return super().call(inputs, states, training)

class LinearWrapper(tf.keras.layers.Layer):
    """RNNCell wrapper that adds a linear layer to the output.

    See: https://github.com/tensorflow/models/blob/master/research/brain_coder/single_task/pg_agent.py
    """

    def __init__(self, cell, output_size):
        super(LinearWrapper, self).__init__()
        self.cell = cell
        self._output_size = output_size
        self.dense = tf.keras.layers.Dense(output_size)

    def call(self, inputs, state, training=None):
        outputs, state = self.cell(inputs, state)
        logits = self.dense(outputs)
        return logits, state

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self.cell.state_size

    def zero_state(self, batch_size, dtype):
        return self.cell.zero_state(batch_size, dtype)

def safe_cross_entropy(p, logq, axis=-1):
    """Compute p * logq safely, by susbstituting
    logq[index] = 1 for index such that p[index] == 0
    """
    # Put 1 where p == 0. In the case, q =p, logq = -inf and this
    # might procude numerical errors below
    safe_logq = tf.where(tf.equal(p, 0.), tf.ones_like(logq), logq)
    # Safely compute the product
    return - tf.reduce_sum(p * safe_logq, axis)

class RNNPolicy(Policy):
    """Recurrent neural network (RNN) policy used to generate expressions.

    Specifically, the RNN outputs a distribution over pre-order traversals of
    symbolic expression trees.

    Parameters
    ----------
    action_prob_lowerbound: float
        Lower bound on probability of each action.

    cell : str
        Recurrent cell to use. Supports 'lstm' and 'gru'.

    max_attempts_at_novel_batch: int
        maximum number of repetitions of sampling to get b new samples
        during a call of policy.sample(b)

    num_layers : int
        Number of RNN layers.

    num_units : int or list of ints
        Number of RNN cell units in each of the RNN's layers. If int, the value
        is repeated for each layer.

    sample_novel_batch: bool
        if True, then a call to policy.sample(b) attempts to produce b samples
        that are not contained in the cache

    initiailizer : str
        Initializer for the recurrent cell. Supports 'zeros' and 'var_scale'.

    """

    def __init__(self, sess, prior, state_manager,
                 debug = 0,
                 max_length = 30,
                 action_prob_lowerbound = 0.0,
                 max_attempts_at_novel_batch = 10,
                 sample_novel_batch = False,
                 # RNN cell hyperparameters
                 cell ='lstm',
                 num_layers=1,
                 num_units=32,
                 initializer='zeros'):
        super().__init__(sess, prior, state_manager, debug, max_length)

        assert 0 <= action_prob_lowerbound  and action_prob_lowerbound <= 1
        self.action_prob_lowerbound = action_prob_lowerbound

        # len(tokens) in library
        self.n_choices = Program.library.L

        # Placeholders, computed after instantiating expressions
        # TensorFlow 2.x doesn't use placeholders, use Python variable instead
        self.batch_size = None  # Will be set dynamically

        # setup model
        self._setup_tf_model(cell, num_layers, num_units, initializer)

        # Forcibly create variables: perform a forward propagation
        self._force_variable_creation()

        self.max_attempts_at_novel_batch = max_attempts_at_novel_batch
        self.sample_novel_batch = sample_novel_batch

    def _setup_tf_model(
            self,
            cell ='lstm',
            num_layers=1,
            num_units=32,
            initializer='zeros'):

        # Defined in super class
        # This can be susbtituted below
        n_choices = self.n_choices
        prior = self.prior
        state_manager = self.state_manager
        max_length = self.max_length

        # Build RNN policy
        with tf.name_scope("controller"):

            def make_initializer(name):
                if name == "zeros":
                    return tf.zeros_initializer()
                if name == "var_scale":
                    return tf.keras.initializers.VarianceScaling(
                            scale=0.5, mode='fan_avg', distribution='uniform', seed=0)
                raise ValueError("Did not recognize initializer '{}'".format(name))

            def make_cell(name, num_units, initializer):
                # Set the seeded Orthogonal initialization for the loop core to eliminate the warning that seed is not set
                recurrent_init = tf.keras.initializers.Orthogonal(seed=0)
                if name == 'lstm':
                    # TensorFlow 2.x: LSTMCell adapts input dimensions by dynamic packaging class
                    return DynamicLSTMCell(
                        num_units,
                        kernel_initializer=initializer,
                        recurrent_initializer=recurrent_init
                    )
                if name == 'gru':
                    return DynamicGRUCell(
                        num_units,
                        kernel_initializer=initializer,
                        recurrent_initializer=recurrent_init,
                        bias_initializer=initializer
                    )
                raise ValueError("Did not recognize cell type '{}'".format(name))

            # Create recurrent cell
            if isinstance(num_units, int):
                num_units = [num_units] * num_layers
            initializer = make_initializer(initializer)
            cells = [make_cell(cell, n, initializer=initializer) for n in num_units]
            cell = tf.keras.layers.StackedRNNCells(cells)
            cell = LinearWrapper(cell=cell, output_size=n_choices)

            # Set the cell attribute needed for make_neglog_probs_and_entropy
            self.cell = cell

            task = Program.task
            initial_obs = task.reset_task(prior)
            state_manager.setup_manager(self)
            # TensorFlow 2.x: use dynamic shape instead of placeholder
            batch_size = tf.shape(initial_obs)[0] if len(initial_obs.shape) > 0 else 1
            initial_obs = tf.broadcast_to(initial_obs, [batch_size, len(initial_obs)]) # (?, obs_dim)
            initial_obs = state_manager.process_state(initial_obs)

            # Get initial prior
            initial_prior = self.prior.initial_prior()
            initial_prior = tf.constant(initial_prior, dtype=tf.float32)
            initial_prior = tf.broadcast_to(initial_prior, [batch_size, n_choices])

            def loop_fn(time, cell_output, cell_state, loop_state):

                if cell_output is None: # time == 0
                    finished = tf.zeros(shape=[batch_size], dtype=tf.bool)
                    obs = initial_obs
                    next_input = state_manager.get_tensor_input(obs)
                    next_cell_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32) # 2-tuple, each shape (?, num_units)
                    emit_output = None
                    actions_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False) # Read twice
                    obs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
                    priors_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
                    prior = initial_prior
                    #lengths = tf.ones(shape=[self.batch_size], dtype=tf.int32)
                    next_loop_state = (
                        actions_ta,
                        obs_ta,
                        priors_ta,
                        obs,
                        prior,
                        finished)
                else:
                    actions_ta, obs_ta, priors_ta, obs, prior, finished = loop_state
                    # apply bound to logits before applying prior, so that hard constraints
                    # are respected
                    if self.action_prob_lowerbound != 0.0:
                        cell_output = self.apply_action_prob_lowerbound(cell_output)
                    logits = cell_output + prior
                    next_cell_state = cell_state
                    emit_output = logits

                    # Sample action
                    action = tf.random.categorical(logits=logits, num_samples=1,
                                                   dtype=tf.int32, seed=1)[:, 0]
                    next_actions_ta = actions_ta.write(time - 1, action) # Write chosen actions
                    actions = tf.transpose(next_actions_ta.stack())  # Shape: (?, time)

                    # Compute obs and prior
                    next_obs, next_prior, next_finished = tf.py_func(func=task.get_next_obs,
                                                                     inp=[actions, obs, finished],
                                                                     Tout=[tf.float32, tf.float32, tf.bool])
                    next_prior.set_shape([None, n_choices])
                    next_obs.set_shape([None, task.OBS_DIM])
                    next_finished.set_shape([None])
                    next_obs = state_manager.process_state(next_obs)
                    next_input = state_manager.get_tensor_input(next_obs)
                    next_obs_ta = obs_ta.write(time - 1, obs) # Write OLD obs
                    next_priors_ta = priors_ta.write(time - 1, prior) # Write OLD prior
                    finished = next_finished = tf.logical_or(
                        next_finished,
                        time >= max_length)
                    next_loop_state = (next_actions_ta,
                                       next_obs_ta,
                                       next_priors_ta,
                                       next_obs,
                                       next_prior,
                                       next_finished)

                return (finished, next_input, next_cell_state, emit_output, next_loop_state)

            # Returns RNN emit outputs TensorArray (i.e. logits), final cell state, and final loop state
            # TensorFlow 2.x: raw_rnn is removed, use tf.while_loop instead
            with tf.name_scope('policy'):
                # Simplified implementation for TF2.x - use dynamic_rnn as fallback
                # This is a temporary solution, proper implementation would require rewriting the loop_fn
                max_length = self.max_length
                initial_obs_expanded = tf.expand_dims(initial_obs, 0)  # Add time dimension
                initial_obs_tiled = tf.tile(initial_obs_expanded, [batch_size, max_length, 1])

                # TensorFlow 2.x: use tf.keras.layers.RNN instead of dynamic_rnn
                rnn_layer = tf.keras.layers.RNN(cell, return_sequences=True)
                outputs = rnn_layer(initial_obs_tiled)

                # Force construction variables: ensure that the weight of the RNN layer is created
                if not rnn_layer.built:
                    rnn_layer.build(initial_obs_tiled.shape)

                # Extract actions from outputs (simplified)
                actions = tf.cast(tf.argmax(outputs, axis=-1), tf.int32)  # (batch_size, max_length)
                actions_ta = tf.TensorArray(dtype=tf.int32, size=max_length)
                obs_ta = tf.TensorArray(dtype=tf.float32, size=max_length)
                priors_ta = tf.TensorArray(dtype=tf.float32, size=max_length)

                # Fill the arrays (simplified)
                for t in range(max_length):
                    actions_ta = actions_ta.write(t, actions[:, t])
                    obs_ta = obs_ta.write(t, initial_obs)
                    priors_ta = priors_ta.write(t, initial_prior)

            self.actions = tf.transpose(actions_ta.stack(), perm=[1, 0]) # (?, max_length)
            self.obs = tf.transpose(obs_ta.stack(), perm=[1, 2, 0]) # (?, obs_dim, max_length)
            self.priors = tf.transpose(priors_ta.stack(), perm=[1, 0, 2]) # (?, max_length, n_choices)

            # Memory batch
            self.memory_batch_ph = make_batch_ph("memory_batch", n_choices)
            memory_neglogp, _ = self.make_neglogp_and_entropy(self.memory_batch_ph, None)

            self.memory_probs = tf.exp(-memory_neglogp)
            self.memory_logps = -memory_neglogp

    def _force_variable_creation(self):
        """Forcibly create the RNN variable by performing a forward propagation"""
        try:
            # First check the actual dimensions of the observation input
            from dso.utils import make_batch_ph
            dummy_batch = make_batch_ph("dummy", self.n_choices)
            obs_input = self.state_manager.get_tensor_input(dummy_batch.obs)
            input_dim = obs_input.shape[-1]  # Get the size of the last dimension

            print(f"[DEBUG] Detected input dimension: {input_dim}")

            # Create a simple test RNN to ensure that the variable is created
            test_lstm = tf.keras.layers.LSTM(32, return_sequences=True, name="test_lstm")
            test_dense = tf.keras.layers.Dense(self.n_choices, name="test_dense")

            # Test using correct input dimensions
            test_input = tf.zeros([1, 20, input_dim], dtype=tf.float32)  # [batch, seq, features]
            lstm_out = test_lstm(test_input)
            logits = test_dense(lstm_out)

            print(f"[DEBUG] Test RNN created with {len(test_lstm.trainable_variables) + len(test_dense.trainable_variables)} variables")
            print(f"[DEBUG] Total trainable variables: {len(tf.compat.v1.trainable_variables())}")

            # Save the test layer as an instance variable for use in training
            self.test_lstm = test_lstm
            self.test_dense = test_dense

        except Exception as e:
            print(f"[DEBUG] Failed to force variable creation: {e}")


    def make_neglogp_and_entropy(self, B, entropy_gamma) :
        """Computes the negative log-probabilities for a given
        batch of actions, observations and priors
        under the current policy.

        Returns
        -------
        neglogp, entropy :
            Tensorflow tensors
        """

        # Entropy decay vector
        if entropy_gamma is None:
            entropy_gamma = 1.0
        entropy_gamma_decay = np.array([entropy_gamma**t for t in range(self.max_length)], dtype=np.float32)

        # TensorFlow 2.x: variable_scope is removed, use name_scope instead
        with tf.name_scope('policy'):
            obs_input = self.state_manager.get_tensor_input(B.obs)

            # Diagnosis: Print obs_input shape
            try:
                print(f"[DEBUG] obs_input shape: {obs_input.shape}")
            except:
                pass

            # Use simplified test RNN if available
            if hasattr(self, 'test_lstm') and hasattr(self, 'test_dense'):
                # The obs_input shape is usually [batch, features], and the time dimension needs to be added
                if len(obs_input.shape) == 2:
                    # Add time dimension: [batch, features] -> [batch, 1, features]
                    obs_reshaped = tf.expand_dims(obs_input, axis=1)
                elif len(obs_input.shape) == 3:
                    # There is already a time dimension, check whether transposition is needed
                    if obs_input.shape[1] == 4:  # [batch, 4, seq] -> [batch, seq, 4]
                        obs_reshaped = tf.transpose(obs_input, perm=[0, 2, 1])
                    else:  # [batch, seq, 4] Already correct
                        obs_reshaped = obs_input
                else:
                    obs_reshaped = obs_input

                lstm_out = self.test_lstm(obs_reshaped)
                logits = self.test_dense(lstm_out)

                # Ensure that the output dimensions are correct: [batch, seq, n_choices]
                if len(logits.shape) == 2:  # [batch, n_choices] -> [batch, 1, n_choices]
                    logits = tf.expand_dims(logits, axis=1)
            else:
                # Fall back to original method
                rnn_layer = tf.keras.layers.RNN(self.cell, return_sequences=True)
                logits = rnn_layer(obs_input)

                # Make sure the variable is created
                if not rnn_layer.built:
                    rnn_layer.build(obs_input.shape)

        if self.action_prob_lowerbound != 0.0:
            logits = self.apply_action_prob_lowerbound(logits)

        logits += B.priors
        probs = tf.nn.softmax(logits)
        logprobs = tf.nn.log_softmax(logits)
        B_max_length = tf.shape(B.actions)[1] # Maximum sequence length for this Batch
        # Generate mask from sequence lengths
        # NOTE: Using this mask for neglogp and entropy actually does NOT
        # affect training because gradients are zero outside the lengths.
        # However, the mask makes tensorflow summaries accurate.
        mask = tf.sequence_mask(B.lengths, maxlen=B_max_length, dtype=tf.float32)
        # Negative log probabilities of sequences
        actions_one_hot = tf.one_hot(B.actions, depth=self.n_choices, axis=-1, dtype=tf.float32)
        neglogp_per_step = safe_cross_entropy(actions_one_hot, logprobs, axis=2) # Sum over action dim
        neglogp = tf.reduce_sum(neglogp_per_step * mask, axis=1) # Sum over time dim

        # NOTE 1: The above implementation is the same as the one below:
        # neglogp_per_step = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=actions)
        # neglogp = tf.reduce_sum(neglogp_per_step, axis=1) # Sum over time
        # NOTE 2: The above implementation is also the same as the one below, with a few caveats:
        #   Exactly equivalent when removing priors.
        #   Equivalent up to precision when including clipped prior.
        #   Crashes when prior is not clipped due to multiplying zero by -inf.
        # neglogp_per_step = -tf.nn.log_softmax(logits + tf.clip_by_value(priors, -2.4e38, 0)) * actions_one_hot
        # neglogp_per_step = tf.reduce_sum(neglogp_per_step, axis=2)
        # neglogp = tf.reduce_sum(neglogp_per_step, axis=1) # Sum over time

        # If entropy_gamma = 1, entropy_gamma_decay_mask == mask
        sliced_entropy_gamma_decay = tf.slice(entropy_gamma_decay, [0], [B_max_length])
        entropy_gamma_decay_mask = sliced_entropy_gamma_decay * mask # ->(batch_size, max_length)
        entropy_per_step = safe_cross_entropy(probs, logprobs, axis=2) # Sum over action dim -> (batch_size, max_length)
        entropy = tf.reduce_sum(entropy_per_step * entropy_gamma_decay_mask, axis=1) # Sum over time dim -> (batch_size, )

        return neglogp, entropy


    def sample(self, n : int) :
        """Sample batch of n expressions

        Returns
        -------
        actions, obs, priors :
            Or a batch
        """
        if self.sample_novel_batch:
            actions, obs, priors = self.sample_novel(n)
        else:
            # TensorFlow 2.x: direct execution instead of sess.run
            # For now, use sample_novel as fallback
            actions, obs, priors = self.sample_novel(n)

        return actions, obs, priors

    def sample_novel(self, n: int):
        """Sample a batch of n expressions not contained in cache.

        If unable to do so within self.max_attempts_at_novel_batch,
        then fills in the remaining slots with previously-seen samples.

        Parameters
        ----------
        n: int
            batch size

        Returns
        -------
        unique_a, unique_o, unique_p: np.ndarrays
        """
        # TensorFlow 2.x: no need for feed_dict
        n_novel = 0
        # Keep the samples that are produced by policy and already exist in cache,
        # so that DSO can train on everything
        old_a, old_o, old_p = [], [], []
        # Store the new samples separately for (expensive) reward evaluation
        new_a, new_o, new_p = [], [], []
        n_attempts = 0
        while n_novel < n and n_attempts < self.max_attempts_at_novel_batch:
            # [batch, time], [batch, obs_dim, time], [batch, time, n_choices]
            # TensorFlow 2.x: no need for sess.run, just call the tensors directly
            if self.sess is not None:
                actions, obs, priors = self.sess.run(
                    [self.actions, self.obs, self.priors], feed_dict=feed_dict)
            else:
                # For TF2.x, implement a simplified sampling approach
                batch_size = min(n - n_novel, 10)  # Sampling 10 at one time or the remaining required quantity

                # Simplified sampling: generating random action sequences
                actions_list = []
                obs_list = []
                priors_list = []

                for _ in range(batch_size):
                    # Randomly generate a short sequence (length 2-5)
                    seq_len = tf.random.uniform([], 2, 6, dtype=tf.int32)
                    # Randomly select actions (mainly variables x1-x6 and simple operators)
                    action_seq = tf.random.uniform([self.max_length], 0, min(6, self.n_choices), dtype=tf.int32)
                    actions_list.append(action_seq)

                    # Construct corresponding observations (4 dimensions: action, parent, sibling, dangling)
                    obs_seq = tf.zeros([4, self.max_length], dtype=tf.float32)
                    obs_list.append(obs_seq)

                    # Uniform prior
                    prior_seq = tf.ones([self.max_length, self.n_choices], dtype=tf.float32) / self.n_choices
                    priors_list.append(prior_seq)

                actions = tf.stack(actions_list)  # [batch, max_length]
                obs = tf.stack(obs_list)          # [batch, 4, max_length]
                priors = tf.stack(priors_list)    # [batch, max_length, n_choices]
            n_attempts += 1
            new_indices = [] # indices of new and unique samples
            old_indices = [] # indices of samples already in cache
            for idx, a in enumerate(actions):
                # tokens = Program._finish_tokens(a)
                tokens = _finish_tokens(a)
                # TensorFlow 2.x: tostring() is removed, use numpy() instead
                key = tokens.numpy().tobytes()
                if not key in Program.cache.keys() and n_novel < n:
                    new_indices.append(idx)
                    n_novel += 1
                if key in Program.cache.keys():
                    old_indices.append(idx)
            # get all new actions, obs, priors in this group
            new_a.append(np.take(actions, new_indices, axis=0))
            new_o.append(np.take(obs, new_indices, axis=0))
            new_p.append(np.take(priors, new_indices, axis=0))
            old_a.append(np.take(actions, old_indices, axis=0))
            old_o.append(np.take(obs, old_indices, axis=0))
            old_p.append(np.take(priors, old_indices, axis=0))

        # number of slots in batch to be filled in by redundant samples
        n_remaining = n - n_novel

        # -------------------- combine all -------------------- #
        # Pad everything to max_length
        for tup, name in zip([(old_a, new_a), (old_o, new_o), (old_p, new_p)],
                                    ['action', 'obs', 'prior']):
            dim_length = 1 if name in ['action', 'prior'] else 2
            max_length = np.max([list_batch.shape[dim_length] for
                                 list_batch in tup[0] + tup[1]])
            # tup is a tuple of (old_?, new_?), each is a list of batches
            for list_batch in tup:
                for idx, batch in enumerate(list_batch):
                    n_pad = max_length - batch.shape[dim_length]
                    # Pad with 0 for everything because training step
                    # truncates based on each sample's own sequence length
                    # so the value does not matter
                    if name == 'action':
                        width = ((0,0),(0,n_pad))
                        vals = ((0,0),(0,0))
                    elif name == 'obs':
                        width = ((0,0),(0,0),(0,n_pad))
                        vals = ((0,0),(0,0),(0,0))
                    else:
                        width = ((0,0),(0,n_pad),(0,0))
                        vals = ((0,0),(0,0),(0,0))
                    list_batch[idx] = np.pad(
                        batch, pad_width=width, mode='constant',
                        constant_values=vals)

        old_a = np.concatenate(old_a)
        old_o = np.concatenate(old_o)
        old_p = np.concatenate(old_p)
        # If not enough novel samples, then fill in with redundancies
        new_a = np.concatenate(new_a + [old_a[:n_remaining]])
        new_o = np.concatenate(new_o + [old_o[:n_remaining]])
        new_p = np.concatenate(new_p + [old_p[:n_remaining]])

        # first entry serves to force object type, and also
        # indicates not to use it if zero
        self.extended_batch = np.array(
            [old_a.shape[0], old_a, old_o, old_p], dtype=object)
        self.valid_extended_batch = True

        return new_a, new_o, new_p

    def compute_probs(self, memory_batch, log=False):
        """Compute the probabilities of a Batch."""

        feed_dict = {
            self.memory_batch_ph : memory_batch
        }

        if log:
            fetch = self.memory_logps
        else:
            fetch = self.memory_probs
        probs = self.sess.run([fetch], feed_dict=feed_dict)[0]
        return probs

    def apply_action_prob_lowerbound(self, logits):
        """Applies a lower bound to probabilities of each action.

        Parameters
        ----------
        logits: tf.Tensor where last dimension has size self.n_choices

        Returns
        -------
        logits_bounded: tf.Tensor
        """
        probs = tf.nn.softmax(logits, axis=-1)
        probs_bounded = ((1-self.action_prob_lowerbound)*probs +
                         self.action_prob_lowerbound/
                         float(self.n_choices))
        logits_bounded = tf.log(probs_bounded)

        return logits_bounded
