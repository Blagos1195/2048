from __future__ import annotations

import numpy as np


def encode(board):
    values = np.asarray(board, dtype=np.float32).reshape(-1)
    return np.log2(np.maximum(values, 1.0)) / 16.0


class PolicyNet:
    def __init__(self, seed=41, hidden=128):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2 / 16), (16, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, 4)).astype(np.float32)
        self.b2 = np.zeros(4, dtype=np.float32)

    def predict_batch(self, boards):
        x = np.asarray([encode(board) for board in boards], dtype=np.float32)
        h = np.tanh(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def predict(self, board):
        return self.predict_batch([board])[0]

    def save(self, path):
        np.savez_compressed(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    def train(self, boards, scores, legal, epochs=40, batch_size=512, seed=43):
        rng = np.random.default_rng(seed)
        x = np.asarray([encode(board) for board in boards], dtype=np.float32)
        values = np.asarray(scores, dtype=np.float32)
        mask = np.asarray(legal, dtype=bool)
        centered = np.where(mask, values, 0.0)
        counts = mask.sum(axis=1, keepdims=True)
        means = centered.sum(axis=1, keepdims=True) / np.maximum(counts, 1)
        centered = np.where(mask, values - means, -1e9)
        variance = np.where(mask, centered * centered, 0.0).sum(axis=1, keepdims=True)
        scale = np.sqrt(variance / np.maximum(counts, 1))
        logits = centered / np.maximum(scale, 0.25) / 0.5
        logits = np.where(mask, logits, -1e9)
        logits -= logits.max(axis=1, keepdims=True)
        targets = np.exp(logits)
        targets *= mask
        targets /= np.maximum(targets.sum(axis=1, keepdims=True), 1e-12)
        params = [self.w1, self.b1, self.w2, self.b2]
        first = [np.zeros_like(p) for p in params]
        second = [np.zeros_like(p) for p in params]
        step = 0
        for epoch in range(epochs):
            order = rng.permutation(len(x))
            loss_total = 0.0
            batches = 0
            for start in range(0, len(order), batch_size):
                ids = order[start:start + batch_size]
                xb = x[ids]
                tb = targets[ids]
                z = xb @ self.w1 + self.b1
                h = np.tanh(z)
                raw = h @ self.w2 + self.b2
                raw = np.where(mask[ids], raw, -1e9)
                raw -= raw.max(axis=1, keepdims=True)
                probs = np.exp(raw)
                probs *= mask[ids]
                probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
                loss_total += float(-np.sum(tb * np.log(np.maximum(probs, 1e-9))) / len(ids))
                batches += 1
                dlogits = (probs - tb) / len(ids)
                dw2 = h.T @ dlogits
                db2 = dlogits.sum(axis=0)
                dh = dlogits @ self.w2.T
                dz = dh * (1.0 - h * h)
                dw1 = xb.T @ dz
                db1 = dz.sum(axis=0)
                grads = [dw1, db1, dw2, db2]
                step += 1
                for i, (param, grad) in enumerate(zip(params, grads)):
                    first[i] = 0.9 * first[i] + 0.1 * grad
                    second[i] = 0.999 * second[i] + 0.001 * grad * grad
                    m = first[i] / (1.0 - 0.9 ** step)
                    v = second[i] / (1.0 - 0.999 ** step)
                    param -= 0.001 * m / (np.sqrt(v) + 1e-8)
            print(f"epoch={epoch + 1}/{epochs} policy_loss={loss_total / batches:.4f}", flush=True)
