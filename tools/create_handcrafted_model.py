import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfplay_ai import NTupleValue, PATTERNS, PATTERN_COUNT, PATTERN_TABLE_SIZE

BASE = 16
PRIORITY = np.array([
    [16, 15, 14, 13],
    [9, 10, 11, 12],
    [8, 7, 6, 5],
    [1, 2, 3, 4],
], dtype=np.float32)
EXPONENTS = np.arange(PATTERN_TABLE_SIZE, dtype=np.int32)
DIGITS = np.stack([
    (EXPONENTS // (BASE ** 3)) % BASE,
    (EXPONENTS // (BASE ** 2)) % BASE,
    (EXPONENTS // BASE) % BASE,
    EXPONENTS % BASE,
], axis=1).astype(np.float32)

weights = np.zeros((PATTERN_COUNT, PATTERN_TABLE_SIZE), dtype=np.float32)

for pattern_index, cells in enumerate(PATTERNS):
    coords = [(cell // 4, cell % 4) for cell in cells]
    priorities = np.array([PRIORITY[r, c] for r, c in coords], dtype=np.float32)
    values = DIGITS

    score = (values * priorities[None, :]).sum(axis=1) * 0.0035
    score += (values == 0).sum(axis=1) * 0.08

    for i in range(4):
        for j in range(i + 1, 4):
            r1, c1 = coords[i]
            r2, c2 = coords[j]
            if abs(r1 - r2) + abs(c1 - c2) == 1:
                score -= np.abs(values[:, i] - values[:, j]) * 0.025

    if pattern_index < 8:
        differences = np.diff(values, axis=1)
        rising = np.maximum(differences, 0).sum(axis=1)
        falling = np.maximum(-differences, 0).sum(axis=1)
        score -= np.minimum(rising, falling) * 0.03

    weights[pattern_index] = score

model = NTupleValue()
model.weights[:] = weights
model.save("benchmarks/handcrafted.npz", {
    "type": "handcrafted",
    "strategy": "snake positional weighting, empty-cell reward, local smoothness, line monotonicity",
})
print(f"Created handcrafted model: shape={weights.shape}, min={weights.min():.4f}, max={weights.max():.4f}, std={weights.std():.4f}")
