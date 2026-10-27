"""NumPy Double-DQN for 2048 with prioritized replay and n-step returns.

No positional bonuses or hand-coded board tactics are used. State encoding is
configurable so compact logarithmic inputs can be compared with tile-category planes.
"""
from collections import deque
import json
import numpy as np


def encode_state(state, mode="log"):
    """Encode the board. Both encodings treat all positions identically.

    log: 16 values, empty=0 and tile=log2(tile)/16.
    tile_planes: 17 one-hot categories per cell (empty plus tile exponents 1..16),
    flattened to 272 values. Exponents above 16 share the highest category.
    """
    board = np.asarray(state, dtype=np.float32).reshape(16)
    if mode == "log":
        out = np.zeros_like(board)
        mask = board > 0
        out[mask] = np.log2(board[mask]) / 16.0
        return out
    if mode == "tile_planes":
        result = np.zeros((16, 17), dtype=np.float32)
        for i, value in enumerate(board):
            if value <= 0:
                category = 0
            else:
                category = min(16, max(1, int(round(np.log2(value)))))
            result[i, category] = 1.0
        return result.reshape(-1)
    raise ValueError(f"Unknown state encoding: {mode}")


def valid_action_mask(board):
    """Return legal actions in LEFT, DOWN, RIGHT, UP order."""
    board = np.asarray(board).reshape(4, 4)
    masks = np.zeros(4, dtype=bool)
    for r in range(4):
        row = board[r]
        for c in range(4):
            if row[c] == 0:
                continue
            if np.any(row[:c] == 0):
                masks[0] = True
            if np.any(row[c+1:] == 0):
                masks[2] = True
            if c < 3 and row[c] == row[c+1]:
                masks[0] = masks[2] = True
    for c in range(4):
        col = board[:, c]
        for r in range(4):
            if col[r] == 0:
                continue
            if np.any(col[:r] == 0):
                masks[3] = True
            if np.any(col[r+1:] == 0):
                masks[1] = True
            if r < 3 and col[r] == col[r+1]:
                masks[1] = masks[3] = True
    return masks


class PrioritizedReplayBuffer:
    """Fixed-size NumPy replay storage with proportional prioritized sampling."""
    def __init__(self, capacity, state_dim, alpha=0.6, beta_start=0.4, beta_steps=250_000, epsilon=1e-5):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_steps = int(beta_steps)
        self.epsilon = float(epsilon)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.next_masks = np.zeros((capacity, 4), dtype=np.bool_)
        self.n_steps = np.ones(capacity, dtype=np.int8)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0
        self.sample_count = 0

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, done, next_mask, n_steps=1):
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.next_masks[i] = next_mask
        self.n_steps[i] = n_steps
        self.priorities[i] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("Cannot sample an empty replay buffer")
        n = min(int(batch_size), self.size)
        scaled = np.power(self.priorities[:self.size] + self.epsilon, self.alpha)
        total = float(scaled.sum())
        probs = scaled / total if total > 0 else np.full(self.size, 1 / self.size)
        indices = self.rng.choice(self.size, size=n, replace=False, p=probs) if hasattr(self, "rng") else np.random.choice(self.size, size=n, replace=False, p=probs)
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.sample_count / self.beta_steps)
        self.sample_count += 1
        weights = np.power(self.size * probs[indices], -beta)
        weights /= max(float(weights.max()), 1e-8)
        return indices, weights.astype(np.float32)

    def update_priorities(self, indices, errors):
        vals = np.abs(np.asarray(errors, dtype=np.float32)) + self.epsilon
        self.priorities[indices] = vals
        if len(vals):
            self.max_priority = max(self.max_priority, float(vals.max()))


class NeuralNetwork:
    """Fully connected Double-DQN agent. Class name retained for UI compatibility."""
    def __init__(self, layer_sizes=None, mutation_rate=0.3, mutation_strength=0.7,
                 learning_rate=0.0007, gamma=0.99, replay_capacity=30000,
                 seed=None, state_encoding="log", n_step=3):
        if state_encoding not in ("log", "tile_planes"):
            raise ValueError(
                f"Unknown state encoding '{state_encoding}'. "
                "Choose 'log' or 'tile_planes'."
            )
        self.state_encoding = state_encoding
        self.state_dim = 16 if state_encoding == "log" else 272
        default_sizes = [self.state_dim, 48, 32, 4] if state_encoding == "log" else [self.state_dim, 96, 64, 4]
        self.layer_sizes = list(layer_sizes or default_sizes)
        if self.layer_sizes[0] != self.state_dim or self.layer_sizes[-1] != 4:
            raise ValueError(f"Architecture must have {self.state_dim} inputs and 4 outputs for {state_encoding} encoding")
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.n_step = max(1, int(n_step))
        self.rng = np.random.default_rng(seed)
        self.train_steps = 0
        self.environment_steps = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.04
        # Keep exploring for longer: the previous run reached its 4% floor
        # before its policy had learned to reliably build large tiles.
        self.epsilon_decay_steps = 400_000
        self._init_weights()
        self.target_weights = [w.copy() for w in self.weights]
        self.target_biases = [b.copy() for b in self.biases]
        self._init_adam()
        self.memory_buffer = PrioritizedReplayBuffer(replay_capacity, self.state_dim)
        self.memory_buffer.rng = self.rng
        self.nstep_queue = deque()

    def encode(self, state):
        return encode_state(state, self.state_encoding)

    def _init_weights(self):
        self.weights, self.biases = [], []
        for din, dout in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            scale = np.sqrt(2.0 / din) if dout != 4 else np.sqrt(1.0 / din)
            self.weights.append(self.rng.normal(0, scale, (dout, din)).astype(np.float32))
            self.biases.append(np.zeros((dout, 1), dtype=np.float32))
        self.last_activations = []

    def _init_adam(self):
        self.mw = [np.zeros_like(w) for w in self.weights]
        self.vw = [np.zeros_like(w) for w in self.weights]
        self.mb = [np.zeros_like(b) for b in self.biases]
        self.vb = [np.zeros_like(b) for b in self.biases]
        self.adam_t = 0

    @staticmethod
    def _forward_with(state, weights, biases, keep_activations=False):
        a = np.asarray(state, dtype=np.float32).reshape(weights[0].shape[1], -1)
        acts = [a]
        for w, b in zip(weights[:-1], biases[:-1]):
            a = np.maximum(0.0, w @ a + b)
            acts.append(a)
        out = weights[-1] @ a + biases[-1]
        acts.append(out)
        return (out, acts) if keep_activations else out

    def forward(self, state):
        out, self.last_activations = self._forward_with(self.encode(state).reshape(-1, 1), self.weights, self.biases, True)
        return out[:, 0].copy()

    def q_values(self, state, target=False):
        w = self.target_weights if target else self.weights
        b = self.target_biases if target else self.biases
        return self._forward_with(self.encode(state).reshape(-1, 1), w, b)[:, 0]

    def select_action(self, state, valid_actions, explore=True):
        valid = list(valid_actions)
        if not valid:
            return 0
        if explore:
            self.environment_steps += 1
            self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * max(0.0, 1.0 - self.environment_steps / self.epsilon_decay_steps)
            if self.rng.random() < self.epsilon:
                return int(self.rng.choice(valid))
        q = self.q_values(state)
        return int(max(valid, key=lambda a: q[a]))

    def _push_nstep(self, force=False):
        if not self.nstep_queue:
            return
        count = min(self.n_step, len(self.nstep_queue))
        if not force and count < self.n_step:
            return
        first = self.nstep_queue[0]
        total_reward = 0.0
        last = first
        used = 0
        for transition in list(self.nstep_queue)[:count]:
            total_reward += (self.gamma ** used) * transition[2]
            last = transition
            used += 1
            if transition[4]:
                break
        self.memory_buffer.add(first[0], first[1], total_reward, last[3], last[4], last[5], used)
        self.nstep_queue.popleft()

    def store_experience(self, state, action, reward, next_state, done, next_valid_actions=None):
        if next_valid_actions is None:
            next_mask = valid_action_mask(next_state)
        else:
            next_mask = np.zeros(4, dtype=bool)
            next_mask[list(next_valid_actions)] = True
        transition = (self.encode(state).copy(), int(action), float(reward),
                      self.encode(next_state).copy(), bool(done), next_mask)
        self.nstep_queue.append(transition)
        if done:
            while self.nstep_queue:
                self._push_nstep(force=True)
        else:
            self._push_nstep(force=False)

    @staticmethod
    def _rotate_action(action, quarter_turns):
        """Map LEFT, DOWN, RIGHT, UP through counter-clockwise board rotations."""
        return (int(action) + int(quarter_turns)) % 4

    def _augment_transitions(self, states, next_states, actions, masks):
        """Apply an independent random D4 symmetry to every transition in-place."""
        feature_count = self.state_dim
        planes = 17 if feature_count == 272 else 1
        for i in range(len(actions)):
            reflect = bool(self.rng.integers(0, 2))
            turns = int(self.rng.integers(0, 4))
            if not reflect and turns == 0:
                continue

            def transform(encoded):
                board = encoded.reshape(4, 4, planes) if planes > 1 else encoded.reshape(4, 4)
                if reflect:
                    board = np.flip(board, axis=1)
                if turns:
                    board = np.rot90(board, k=turns, axes=(0, 1))
                return np.asarray(board).reshape(feature_count).copy()

            states[i] = transform(states[i])
            next_states[i] = transform(next_states[i])

            # Horizontal reflection swaps LEFT/RIGHT; DOWN/UP stay unchanged.
            if reflect:
                actions[i] = (2, 1, 0, 3)[int(actions[i])]
                masks[i] = masks[i, [2, 1, 0, 3]]
            actions[i] = self._rotate_action(actions[i], turns)
            if turns:
                masks[i] = np.roll(masks[i], turns)

    def learn(self, batch_size=64):
        if len(self.memory_buffer) < max(256, batch_size):
            return 0.0
        buf = self.memory_buffer
        idx, importance = buf.sample(batch_size)
        # 2048 is symmetric under rotations and reflections. Randomly augment
        # each sampled transition with a board symmetry and the matching action
        # permutation. This adds useful training diversity without encoding any
        # preferred corner, ordering, or hand-written strategy.
        states_batch = buf.states[idx].copy()
        next_states_batch = buf.next_states[idx].copy()
        actions = buf.actions[idx].astype(np.int64).copy()
        masks_batch = buf.next_masks[idx].copy()
        self._augment_transitions(states_batch, next_states_batch, actions, masks_batch)

        states = states_batch.T
        rewards = buf.rewards[idx]
        next_states = next_states_batch.T
        dones = buf.dones[idx].astype(np.float32)
        masks = masks_batch.T
        steps = buf.n_steps[idx].astype(np.float32)

        q, acts = self._forward_with(states, self.weights, self.biases, True)
        next_online = self._forward_with(next_states, self.weights, self.biases)
        next_target = self._forward_with(next_states, self.target_weights, self.target_biases)
        next_online = np.where(masks, next_online, -1e9)
        next_actions = np.argmax(next_online, axis=0)
        next_q = next_target[next_actions, np.arange(len(idx))]
        targets = rewards + np.power(self.gamma, steps) * (1.0 - dones) * next_q
        chosen = q[actions, np.arange(len(idx))]
        td = chosen - targets
        delta = np.zeros_like(q)
        delta[actions, np.arange(len(idx))] = np.clip(td, -1.0, 1.0) * importance / len(idx)

        grad_w, grad_b = [None] * len(self.weights), [None] * len(self.biases)
        d = delta
        for layer in reversed(range(len(self.weights))):
            grad_w[layer] = d @ acts[layer].T
            grad_b[layer] = np.sum(d, axis=1, keepdims=True)
            if layer > 0:
                d = self.weights[layer].T @ d
                d *= (acts[layer] > 0).astype(np.float32)
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in grad_w + grad_b))
        if norm > 10.0:
            factor = 10.0 / (norm + 1e-8)
            grad_w = [g * factor for g in grad_w]
            grad_b = [g * factor for g in grad_b]
        self.adam_t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.weights)):
            self.mw[i] = b1*self.mw[i] + (1-b1)*grad_w[i]
            self.vw[i] = b2*self.vw[i] + (1-b2)*(grad_w[i]**2)
            self.mb[i] = b1*self.mb[i] + (1-b1)*grad_b[i]
            self.vb[i] = b2*self.vb[i] + (1-b2)*(grad_b[i]**2)
            self.weights[i] -= self.learning_rate*(self.mw[i]/(1-b1**self.adam_t))/(np.sqrt(self.vw[i]/(1-b2**self.adam_t))+eps)
            self.biases[i] -= self.learning_rate*(self.mb[i]/(1-b1**self.adam_t))/(np.sqrt(self.vb[i]/(1-b2**self.adam_t))+eps)
        buf.update_priorities(idx, np.abs(td))
        self.train_steps += 1
        if self.train_steps % 500 == 0:
            self.sync_target()
        return float(np.mean(np.abs(td)))

    def replay_learn(self, lr=None, batch_size=64):
        return self.learn(batch_size=batch_size)

    def sync_target(self):
        self.target_weights = [w.copy() for w in self.weights]
        self.target_biases = [b.copy() for b in self.biases]

    def get_weights(self):
        result = {'layer_sizes': self.layer_sizes}
        for i, (w, b) in enumerate(zip(self.weights, self.biases), 1):
            result[f'W{i}'], result[f'b{i}'] = w.copy(), b.copy()
        return result

    def set_weights(self, layer_sizes, weights, biases):
        self.layer_sizes = list(layer_sizes)
        self.state_dim = self.layer_sizes[0]
        self.weights = [np.asarray(w, dtype=np.float32).copy() for w in weights]
        self.biases = [np.asarray(b, dtype=np.float32).copy() for b in biases]
        self.target_weights = [w.copy() for w in self.weights]
        self.target_biases = [b.copy() for b in self.biases]
        self._init_adam()

    def save(self, filepath, include_training_state=True):
        data = {'layer_sizes': np.asarray(self.layer_sizes, dtype=np.int32),
                'state_encoding': np.asarray(self.state_encoding), 'state_dim': self.state_dim,
                'learning_rate': self.learning_rate, 'gamma': self.gamma, 'n_step': self.n_step,
                'train_steps': self.train_steps, 'environment_steps': self.environment_steps,
                'epsilon': self.epsilon, 'epsilon_min': self.epsilon_min,
                'epsilon_decay_steps': self.epsilon_decay_steps, 'adam_t': self.adam_t,
                'rng_state': np.asarray(json.dumps(self.rng.bit_generator.state))}
        for i, (w, b) in enumerate(zip(self.weights, self.biases), 1):
            data[f'W{i}'], data[f'b{i}'] = w, b
            data[f'TW{i}'], data[f'Tb{i}'] = self.target_weights[i-1], self.target_biases[i-1]
            if include_training_state:
                for prefix, values in [('mw', self.mw), ('vw', self.vw), ('mb', self.mb), ('vb', self.vb)]:
                    data[f'{prefix}{i}'] = values[i-1]
        if include_training_state:
            data.update({'replay_pos': self.memory_buffer.pos, 'replay_size': self.memory_buffer.size,
                         'replay_max_priority': self.memory_buffer.max_priority,
                         'replay_sample_count': self.memory_buffer.sample_count})
            n = self.memory_buffer.size
            for key in ('states','actions','rewards','next_states','dones','next_masks','n_steps','priorities'):
                data['replay_'+key] = getattr(self.memory_buffer, key)[:n].copy()
            if self.nstep_queue:
                data['pending_states'] = np.stack([e[0] for e in self.nstep_queue])
                data['pending_actions'] = np.asarray([e[1] for e in self.nstep_queue], dtype=np.int8)
                data['pending_rewards'] = np.asarray([e[2] for e in self.nstep_queue], dtype=np.float32)
                data['pending_next_states'] = np.stack([e[3] for e in self.nstep_queue])
                data['pending_dones'] = np.asarray([e[4] for e in self.nstep_queue], dtype=np.bool_)
                data['pending_masks'] = np.stack([e[5] for e in self.nstep_queue])
        np.savez_compressed(filepath, **data)

    def load(self, filepath, load_training_state=True):
        with np.load(filepath, allow_pickle=False) as d:
            self.state_encoding = str(d['state_encoding'].item()) if 'state_encoding' in d else 'log'
            if self.state_encoding not in ('log', 'tile_planes'):
                raise ValueError(f"Unsupported checkpoint state encoding: {self.state_encoding!r}")
            sizes = list(d['layer_sizes'].astype(int))
            expected_dim = 16 if self.state_encoding == 'log' else 272
            if len(sizes) < 2 or sizes[0] != expected_dim or sizes[-1] != 4:
                raise ValueError('Checkpoint encoding or network architecture is incompatible')
            self.state_dim = sizes[0]
            self.layer_sizes = sizes
            self.weights = [d[f'W{i}'].astype(np.float32).copy() for i in range(1, len(sizes))]
            self.biases = [d[f'b{i}'].astype(np.float32).copy() for i in range(1, len(sizes))]
            self.target_weights = [d[f'TW{i}'].astype(np.float32).copy() if f'TW{i}' in d else self.weights[i-1].copy() for i in range(1, len(sizes))]
            self.target_biases = [d[f'Tb{i}'].astype(np.float32).copy() if f'Tb{i}' in d else self.biases[i-1].copy() for i in range(1, len(sizes))]
            for key in ('learning_rate','gamma','n_step','train_steps','environment_steps','epsilon','epsilon_min','epsilon_decay_steps','adam_t'):
                if key in d:
                    setattr(self, key, type(getattr(self, key))(d[key].item()))
            self._init_adam()
            if 'rng_state' in d:
                self.rng.bit_generator.state = json.loads(str(d['rng_state'].item()))
            cap = self.memory_buffer.capacity
            self.memory_buffer = PrioritizedReplayBuffer(cap, self.state_dim)
            self.memory_buffer.rng = self.rng
            self.nstep_queue = deque()
            if load_training_state:
                for i in range(1, len(sizes)):
                    for prefix, attr in [('mw','mw'),('vw','vw'),('mb','mb'),('vb','vb')]:
                        if f'{prefix}{i}' in d:
                            getattr(self, attr)[i-1] = d[f'{prefix}{i}'].astype(np.float32).copy()
                if 'replay_states' in d:
                    n = min(int(d['replay_size'].item()) if 'replay_size' in d else len(d['replay_states']), cap)
                    for key in ('states','actions','rewards','next_states','dones','next_masks','n_steps','priorities'):
                        if 'replay_'+key in d:
                            getattr(self.memory_buffer, key)[:n] = d['replay_'+key][:n]
                    self.memory_buffer.size = n
                    self.memory_buffer.pos = int(d['replay_pos'].item()) % cap if 'replay_pos' in d else n % cap
                    self.memory_buffer.max_priority = float(d['replay_max_priority'].item()) if 'replay_max_priority' in d else 1.0
                    self.memory_buffer.sample_count = int(d['replay_sample_count'].item()) if 'replay_sample_count' in d else 0
                if 'pending_states' in d:
                    for s, a, r, ns, done, mask in zip(d['pending_states'], d['pending_actions'], d['pending_rewards'], d['pending_next_states'], d['pending_dones'], d['pending_masks']):
                        self.nstep_queue.append((s.copy(), int(a), float(r), ns.copy(), bool(done), mask.copy()))
