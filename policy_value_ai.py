from __future__ import annotations

import numpy as np

import selfplay_ai as game


def encode(board):
    values = np.asarray(board, dtype=np.float32).reshape(-1)
    return np.log2(np.maximum(values, 1.0)).astype(np.float32) / 16.0


def symmetries(board):
    matrix = np.asarray(board).reshape(4, 4)
    for k in range(4):
        rotated = np.rot90(matrix, k)
        yield rotated.reshape(16).copy()
        yield np.fliplr(rotated).reshape(16).copy()


class PolicyValueNet:
    def __init__(self, seed=17, hidden=96):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2 / 16), (16, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.wp = rng.normal(0, 0.02, (hidden, 4)).astype(np.float32)
        self.bp = np.zeros(4, dtype=np.float32)
        self.wv = rng.normal(0, 0.02, (hidden, 1)).astype(np.float32)
        self.bv = np.zeros(1, dtype=np.float32)

    def predict_batch(self, boards):
        x = np.asarray([encode(board) for board in boards], dtype=np.float32)
        h = np.tanh(x @ self.w1 + self.b1)
        logits = h @ self.wp + self.bp
        values = (h @ self.wv + self.bv).reshape(-1)
        return logits, values

    def predict(self, board):
        logits, values = self.predict_batch([board])
        return logits[0], float(values[0])

    def save(self, path):
        np.savez_compressed(path, w1=self.w1, b1=self.b1, wp=self.wp, bp=self.bp, wv=self.wv, bv=self.bv)

    @classmethod
    def load(cls, path):
        model = cls()
        with np.load(path) as data:
            for name in ("w1", "b1", "wp", "bp", "wv", "bv"):
                getattr(model, name)[:] = data[name]
        return model

    def train(self, boards, actions, targets, epochs=35, batch_size=256, seed=23):
        rng = np.random.default_rng(seed)
        x = np.asarray([encode(board) for board in boards], dtype=np.float32)
        y = np.asarray(actions, dtype=np.int64)
        v = np.asarray(targets, dtype=np.float32)
        params = [self.w1, self.b1, self.wp, self.bp, self.wv, self.bv]
        first = [np.zeros_like(p) for p in params]
        second = [np.zeros_like(p) for p in params]
        step = 0
        for epoch in range(epochs):
            order = rng.permutation(len(x))
            policy_loss = 0.0
            value_loss = 0.0
            batches = 0
            for start in range(0, len(order), batch_size):
                ids = order[start:start + batch_size]
                xb, yb, vb = x[ids], y[ids], v[ids]
                z = xb @ self.w1 + self.b1
                h = np.tanh(z)
                logits = h @ self.wp + self.bp
                logits -= logits.max(axis=1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=1, keepdims=True)
                pred = (h @ self.wv + self.bv).reshape(-1)
                rows = np.arange(len(ids))
                policy_loss += float(-np.log(np.maximum(probs[rows, yb], 1e-8)).mean())
                value_loss += float(np.mean((pred - vb) ** 2))
                batches += 1
                dlogits = probs
                dlogits[rows, yb] -= 1.0
                dlogits /= len(ids)
                dv = (2.0 * (pred - vb) / len(ids)).reshape(-1, 1)
                dwp = h.T @ dlogits
                dbp = dlogits.sum(axis=0)
                dwv = h.T @ dv
                dbv = dv.sum(axis=0)
                dh = dlogits @ self.wp.T + dv @ self.wv.T
                dz = dh * (1.0 - h * h)
                dw1 = xb.T @ dz
                db1 = dz.sum(axis=0)
                grads = [dw1, db1, dwp, dbp, dwv, dbv]
                step += 1
                for i, (param, grad) in enumerate(zip(params, grads)):
                    first[i] = 0.9 * first[i] + 0.1 * grad
                    second[i] = 0.999 * second[i] + 0.001 * grad * grad
                    m = first[i] / (1.0 - 0.9 ** step)
                    s = second[i] / (1.0 - 0.999 ** step)
                    param -= 0.001 * m / (np.sqrt(s) + 1e-8)
            print(
                f"epoch={epoch + 1}/{epochs} policy_loss={policy_loss / batches:.4f} "
                f"value_loss={value_loss / batches:.4f}",
                flush=True,
            )


class FastSelectiveAgent:
    def __init__(self, model, top_k=2):
        self.model = model
        self.top_k = top_k
        self.nodes = 0
        self.cache = {}

    def value(self, board):
        key = np.asarray(board, dtype=np.uint32).tobytes()
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        value = self.model.predict(board)[1]
        self.cache[key] = value
        if len(self.cache) > 20000:
            self.cache.clear()
        return value

    def choose_action(self, board):
        legal = game.legal_actions(board)
        if not legal:
            raise ValueError("No legal actions available")
        self.nodes = 0
        logits, _ = self.model.predict(board)
        ranked = sorted(legal, key=lambda action: logits[action], reverse=True)[:self.top_k]
        best_action = ranked[0]
        best_value = -float("inf")
        flat = np.asarray(board, dtype=np.uint32).reshape(-1)
        for action in ranked:
            moved, reward, changed = game._apply_action(board, action)
            if not changed:
                continue
            empty = np.flatnonzero(moved.reshape(-1) == 0)
            if len(empty) == 0:
                score = reward / 128.0 + self.value(moved)
                self.nodes += 1
            else:
                score = reward / 128.0
                child = moved.reshape(-1)
                for cell in empty:
                    cell = int(cell)
                    child[cell] = 2
                    v2 = self.value(moved)
                    self.nodes += 1
                    child[cell] = 4
                    v4 = self.value(moved)
                    self.nodes += 1
                    child[cell] = 0
                    score += (0.9 * v2 + 0.1 * v4) / len(empty)
            if score > best_value:
                best_value = score
                best_action = action
        return best_action
