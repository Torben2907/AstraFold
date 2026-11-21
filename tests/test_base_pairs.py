from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from astrafold.data import base_constants as rc
from astrafold.data import utils as du
from astrafold.data.base_pairs import (
    FORBIDDEN_LOGIT,
    MASK,
    NODE_MASKED,
    NODE_PAIRED,
    NODE_UNPAIRED,
    UNPAIRED,
    allowed_partners,
    annotate_base_pairs,
    node_states,
    pair_features,
    reveal_tokens,
    token_targets,
)
from astrafold.data.sequence import sequence_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = "GGACUUCGGUCC"
N = len(SEQUENCE)


def test_allowed_partners_are_canonical_and_apart():
    onehot = sequence_features(SEQUENCE)["onehot"][None]
    allowed = allowed_partners(onehot, torch.ones(1, N))[0]
    assert torch.equal(allowed, allowed.T)
    for i in range(N):
        for j in range(N):
            pair = SEQUENCE[i] + SEQUENCE[j]
            expected = pair in ("AU", "UA", "GC", "CG", "GU", "UG") and abs(i - j) >= 4
            assert allowed[i, j].item() == expected, (i, j)

    res_mask = torch.ones(1, N)
    res_mask[0, 0] = 0
    assert not allowed_partners(onehot, res_mask)[0, 0].any()


def test_node_states_pair_features_and_targets():
    #   0 pairs with 5 (both revealed), 1 is unpaired, 2 points to 6 but 6 is
    #   masked, 3 and 4 are masked
    tokens = torch.tensor([[5, UNPAIRED, 6, MASK, MASK, 0, MASK]])
    assert node_states(tokens).tolist() == [
        [NODE_PAIRED, NODE_UNPAIRED, NODE_PAIRED]
        + [NODE_MASKED] * 2
        + [NODE_PAIRED, NODE_MASKED]
    ]
    pairs = pair_features(tokens)[0]
    assert torch.nonzero(pairs).tolist() == [[0, 5], [5, 0]]
    assert token_targets(tokens).tolist() == [[5, 7, 6, 0, 0, 0, 0]]


def _probs(rows: dict[int, dict[int, float]], n: int) -> torch.Tensor:
    """[1, n, n + 1] probabilities; rows {i: {value: p}} (value n = unpaired)."""
    probs = torch.zeros(1, n, n + 1)
    probs[0, :, n] = 1.0
    for i, values in rows.items():
        probs[0, i] = 0.0
        for value, p in values.items():
            probs[0, i, value] = p
    return probs


def test_reveal_tokens_keeps_pairs_symmetric_and_respects_priority():
    n = 6
    tokens = torch.full((1, n), MASK)
    # 0 and 2 both want partner 4; 0 goes first, so 2 takes its second choice.
    probs = _probs({0: {4: 0.9, n: 0.1}, 2: {4: 0.8, 5: 0.15, n: 0.05}}, n)
    candidates = torch.tensor([[True, False, True, False, False, False]])
    priority = torch.tensor([[0.9, 0.0, 0.8, 0.0, 0.0, 0.0]])
    new = reveal_tokens(tokens, probs, candidates, priority)
    assert new.tolist() == [[4, MASK, 5, MASK, 0, 2]]

    # The other way round, 2 gets 4 and 0 falls back to unpaired.
    priority_2_first = torch.tensor([[0.1, 0.0, 0.9, 0.0, 0.0, 0.0]])
    new = reveal_tokens(tokens, probs, candidates, priority_2_first)
    assert new.tolist() == [[UNPAIRED, MASK, 4, MASK, 2, MASK]]


def test_reveal_tokens_only_reveals_masked_candidates():
    n = 4
    tokens = torch.tensor([[UNPAIRED, MASK, MASK, MASK]])
    probs = _probs({}, n)  # everything unpaired
    candidates = torch.tensor([[True, False, True, False]])
    new = reveal_tokens(tokens, probs, candidates, torch.zeros(1, n))
    assert new.tolist() == [[UNPAIRED, MASK, UNPAIRED, MASK]]
    assert (
        reveal_tokens(tokens, probs, torch.zeros(1, n, dtype=bool), probs[..., 0])
        is tokens
    )


def test_reveal_tokens_samples_with_the_given_noise():
    n = 6
    tokens = torch.full((1, n), MASK)
    probs = _probs({0: {4: 0.5, n: 0.5}}, n)
    candidates = torch.zeros(1, n, dtype=bool)
    candidates[0, 0] = True
    low, high = torch.full((1, n), 0.2), torch.full((1, n), 0.8)
    assert reveal_tokens(tokens, probs, candidates, torch.ones(1, n), low)[0, 0] == 4
    assert reveal_tokens(tokens, probs, candidates, torch.ones(1, n), high)[0, 0] == (
        UNPAIRED
    )


def test_reveal_tokens_sampling_matches_the_distribution():
    n, b = 6, 4000
    tokens = torch.full((b, n), MASK)
    probs = _probs({0: {4: 0.3, n: 0.7}}, n).expand(b, -1, -1)
    candidates = torch.zeros(b, n, dtype=bool)
    candidates[:, 0] = True
    torch.manual_seed(0)
    new = reveal_tokens(tokens, probs, candidates, torch.ones(b, n), torch.rand(b, n))
    assert abs((new[:, 0] == 4).float().mean().item() - 0.3) < 0.03
    assert torch.all(new[new[:, 0] == 4, 4] == 0)


def test_forbidden_logit_is_finite_in_float32():
    assert np.isfinite(np.float32(FORBIDDEN_LOGIT))


@pytest.fixture(scope="module")
def training_pickles() -> list[dict]:
    csv = PROJECT_ROOT / "metadata" / "RNA3DB_train.csv"
    if not csv.exists():
        pytest.skip("training metadata missing")
    rows = pd.read_csv(csv).sample(10, random_state=0)
    feats = []
    for path in rows.processed_path:
        name = path.split("/")[-2]
        pkl = PROJECT_ROOT / "pkls" / f"{name}_base" / f"{name}_feat.pkl"
        if not pkl.exists():
            pytest.skip("training pickles missing")
        feats.append(du.read_pkl(str(pkl)))
    return feats


def test_annotated_pairs_have_base_pair_geometry(training_pickles):
    c1 = rc.atom_order["C1'"]
    num_pairs = 0
    for feats in training_pickles:
        sequence = "".join(rc.restypes[a] for a in np.asarray(feats["aatype"]))
        positions = np.asarray(feats["all_atom_positions"], dtype=np.float32)
        tokens = annotate_base_pairs(
            sequence, positions, np.asarray(feats["all_atom_mask"]).astype(bool)
        )
        paired = np.flatnonzero(tokens >= 0)
        assert np.array_equal(tokens[tokens[paired]], paired)
        for i in paired:
            j = tokens[i]
            assert sequence[i] + sequence[j] in ("AU", "UA", "GC", "CG", "GU", "UG")
            assert abs(i - j) >= 4
            assert 9.0 < np.linalg.norm(positions[i, c1] - positions[j, c1]) < 12.0
        num_pairs += len(paired) // 2
    assert num_pairs > 50
