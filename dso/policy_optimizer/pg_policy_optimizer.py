
import tensorflow as tf
from dso.policy_optimizer import PolicyOptimizer
from dso.policy import Policy

class PGPolicyOptimizer(PolicyOptimizer):
    """Vanilla policy gradient policy optimizer.

    Parameters
    ----------
    cell : str
        Recurrent cell to use. Supports 'lstm' and 'gru'.

    num_layers : int
        Number of RNN layers.

    num_units : int or list of ints
        Number of RNN cell units in each of the RNN's layers. If int, the value
        is repeated for each layer.

    initiailizer : str
        Initializer for the recurrent cell. Supports 'zeros' and 'var_scale'.

    """
    def __init__(self,
            sess,  # TensorFlow 2.x doesn't use Session
            policy : Policy,
            debug : int = 0,
            summary : bool = False,
            # Optimizer hyperparameters
            optimizer : str = 'adam',
            learning_rate : float = 0.001,
            # Loss hyperparameters
            entropy_weight : float = 0.005,
            entropy_gamma : float = 1.0) -> None:
        self._debug_info_printed = False  # Mark whether debugging information has been printed
        super()._setup_policy_optimizer(sess, policy, debug, summary, optimizer, learning_rate, entropy_weight, entropy_gamma)


    def _set_loss(self):
        with tf.name_scope("losses"):
            # Retrieve rewards from batch
            r = self.sampled_batch_ph.rewards
            # TensorFlow 2.x: baseline will be passed as parameter, use 0 as default
            baseline = tf.constant(0.0, dtype=tf.float32)  # Default baseline
            self.pg_loss = tf.reduce_mean((r - baseline) * self.neglogp, name="pg_loss")
            # Loss already is set to entropy loss
            self.loss += self.pg_loss


    def _preppend_to_summary(self):
        with tf.name_scope("summary"):
            tf.summary.scalar("pg_loss", self.pg_loss)


    def train_step(self, baseline, sampled_batch):
        """Computes loss with fresh forward pass, applies gradients, and prints diagnostics."""
        B = sampled_batch
        # Diagnosis: Print obs shape
        try:
            obs_shape = B.obs.shape if hasattr(B.obs, 'shape') else tf.shape(B.obs)
            print(f"[DEBUG] B.obs shape: {obs_shape}")
        except Exception as e:
            print(f"[DEBUG] Cannot get B.obs shape: {e}")

        optimizer = getattr(self, 'optimizer_obj', None)
        assert optimizer is not None, "Optimizer not initialized"
        with tf.GradientTape() as tape:
            neglogp, entropy = self.policy.make_neglogp_and_entropy(B, self.entropy_gamma)
            entropy_loss = -self.entropy_weight * tf.reduce_mean(entropy)
            r = tf.cast(B.rewards, tf.float32)  # Ensure that rewards is float32
            base = tf.constant(0.0, dtype=tf.float32)
            neglogp = tf.cast(neglogp, tf.float32)  # Ensure that neglogp is float32
            pg_loss = tf.reduce_mean((r - base) * neglogp, name="pg_loss")
            total_loss = entropy_loss + pg_loss

            # Diagnosis: Printing loss component
            try:
                print(f"[DEBUG] entropy_loss={float(entropy_loss.numpy()):.6f}, pg_loss={float(pg_loss.numpy()):.6f}")
                print(f"[DEBUG] rewards mean={float(tf.reduce_mean(r).numpy()):.6f}, neglogp mean={float(tf.reduce_mean(neglogp).numpy()):.6f}")
            except Exception as e:
                print(f"[DEBUG] Loss component error: {e}")

        # TF2.x: Get the Keras layer variable in the policy
        if hasattr(self.policy, 'test_lstm') and hasattr(self.policy, 'test_dense'):
            vars_ = self.policy.test_lstm.trainable_variables + self.policy.test_dense.trainable_variables
        else:
            vars_ = tf.compat.v1.trainable_variables()
        grads = tape.gradient(total_loss, vars_)

        # Diagnosis: Check trainable variables and gradients (first print only)
        if not self._debug_info_printed:
            print(f"[DEBUG] Trainable vars count: {len(vars_)}")
            if len(vars_) > 0:
                print(f"[DEBUG] First var shape: {vars_[0].shape}, name: {vars_[0].name}")
            self._debug_info_printed = True

        none_grads = sum(g is None for g in grads)
        grads_ok = [g for g in grads if g is not None]
        if grads_ok:
            grad_values = [float(tf.reduce_mean(tf.abs(g)).numpy()) for g in grads_ok[:3]]  # The absolute value mean of the first 3 gradients
            print(f"[DEBUG] Sample grad abs means: {grad_values}")

        global_norm = tf.linalg.global_norm(grads_ok) if grads_ok else tf.constant(0.0)

        optimizer.apply_gradients(zip(grads, vars_))

        try:
            print(f"[INFO] loss={float(total_loss.numpy())}, none_grads={int(none_grads)}, global_grad_norm={float(global_norm.numpy()):.4f}")
        except Exception:
            pass
        return None
