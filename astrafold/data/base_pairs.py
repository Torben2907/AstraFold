"""
Base-pair tokens: the discrete part of the model's input and output.

Every nucleotide i has one token: the index j of its base-pairing partner, UNPAIRED,
or MASK. The tokens of a structure form a secondary structure: pairs are symmetric
(token[i] = j <=> token[j] = i), every nucleotide has at most one partner, partners
are canonical (A-U, G-C, G-U) and at least MIN_PAIR_SEPARATION apart in sequence
(a hairpin loop has at least 3 nucleotides).

During training a random subset of the tokens is masked and the model predicts them
(`random_mask_tokens`); during sampling all tokens start masked and are revealed
step by step (`reveal_tokens`). The model sees the tokens through a per-nucleotide
state (masked / unpaired / paired, `node_states`) and a pair feature that marks the
revealed pairs (`pair_features`); it predicts, for every nucleotide, a distribution
over its N possible partners plus "unpaired" (logits [..., N, N + 1], class N =
unpaired, see `token_targets`).

The labels come from the 3D structure (`annotate_base_pairs`): canonical pairs whose
bases face each other like in a Watson-Crick (or G-U wobble) pair.
"""

import numpy as np
import torch
from torch import Tensor

from astrafold.data import base_constants as rc
from astrafold.data.sequence import ONEHOT_ORDER

UNPAIRED = -1
MASK = -2

# Per-nucleotide states the model embeds (see `node_states`).
NODE_MASKED = 0
NODE_UNPAIRED = 1
NODE_PAIRED = 2
NUM_NODE_STATES = 3

# Partners are at least this far apart in sequence (hairpin loops of >= 3 nt).
MIN_PAIR_SEPARATION = 4
CANONICAL_PAIRS = ("AU", "UA", "GC", "CG", "GU", "UG")

# Geometry of a base pair, independent of the dataset: the purine N1 to pyrimidine
# N3 distance (the central hydrogen bond of a Watson-Crick pair, about 2.9 A; up to
# about 5 A in a G-U wobble pair) and the C1'-C1' distance (about 10.6 A). The cutoffs
# sit above the 99th percentile of the annotated pairs in both the CASP15 and the
# RNA3DB training structures (Watson-Crick N1-N3 3.55 A, G-U 4.97 A, C1'-C1'
# 9.3-11.5 A).
MAX_WC_N1_N3_DISTANCE = 3.6
MAX_GU_N1_N3_DISTANCE = 5.0
C1_DISTANCE_RANGE = (9.0, 12.0)

# Logit of the partners a nucleotide cannot pair with.
FORBIDDEN_LOGIT = -1e9


def _canonical_table(order: str) -> np.ndarray:
    """[4, 4] True where the nucleotides order[a], order[b] form a canonical pair."""
    return np.array([[a + b in CANONICAL_PAIRS for b in order] for a in order])


_CANONICAL_ONEHOT = torch.from_numpy(_canonical_table(ONEHOT_ORDER)).float()


def annotate_base_pairs(
    sequence: str, atom_positions: np.ndarray, atom_mask: np.ndarray
) -> np.ndarray:
    """
    Base pairs of a structure, as tokens [N]: the partner index, or UNPAIRED.

    Params:
        sequence       : the N nucleotides (A, C, G, U)
        atom_positions : [N, 28, 3] coordinates in the atom order of
                         base_constants.atom_types (only C1', N1 and N3 are used)
        atom_mask      : [N, 28] True where the atom is resolved

    Candidate pairs are canonical, MIN_PAIR_SEPARATION apart, have the C1'-C1'
    distance of a base pair and the purine N1 within MAX_*_N1_N3_DISTANCE of the
    pyrimidine N3. A nucleotide pairs with at most one partner; candidates are
    taken greedily by their N1-N3 distance.
    """
    n = len(sequence)
    idx = np.arange(n)
    codes = np.array(["ACGU".index(nt) for nt in sequence])
    is_purine = np.isin(codes, [0, 2])
    key_atom = np.where(is_purine, rc.atom_order["N1"], rc.atom_order["N3"])
    c1_atom = rc.atom_order["C1'"]

    key = atom_positions[idx, key_atom]
    c1 = atom_positions[:, c1_atom]
    resolved = atom_mask[idx, key_atom].astype(bool) & atom_mask[:, c1_atom].astype(
        bool
    )

    key_dist = np.linalg.norm(key[:, None] - key[None], axis=-1)
    c1_dist = np.linalg.norm(c1[:, None] - c1[None], axis=-1)
    canonical = _canonical_table("ACGU")[codes[:, None], codes[None]]
    g, u = "ACGU".index("G"), "ACGU".index("U")
    is_gu = ((codes[:, None] == g) & (codes[None] == u)) | (
        (codes[:, None] == u) & (codes[None] == g)
    )
    max_key_dist = np.where(is_gu, MAX_GU_N1_N3_DISTANCE, MAX_WC_N1_N3_DISTANCE)
    candidate = (
        canonical
        & (np.abs(idx[:, None] - idx[None]) >= MIN_PAIR_SEPARATION)
        & resolved[:, None]
        & resolved[None]
        & (c1_dist > C1_DISTANCE_RANGE[0])
        & (c1_dist < C1_DISTANCE_RANGE[1])
        & (key_dist < max_key_dist)
    )

    tokens = np.full(n, UNPAIRED, dtype=np.int64)
    ii, jj = np.nonzero(np.triu(candidate, 1))
    for k in np.argsort(key_dist[ii, jj], kind="stable"):
        i, j = ii[k], jj[k]
        if tokens[i] == UNPAIRED and tokens[j] == UNPAIRED:
            tokens[i], tokens[j] = j, i
    return tokens


def allowed_partners(onehot: Tensor, res_mask: Tensor) -> Tensor:
    """
    [B, N, N] True where nucleotides i and j may pair: canonical, far enough apart
    in sequence and both valid. `onehot` is [B, N, 4] in ONEHOT_ORDER.
    """
    table = _CANONICAL_ONEHOT.to(onehot.device, onehot.dtype)
    canonical = torch.einsum("bia,ac,bjc->bij", onehot, table, onehot) > 0.5
    n = onehot.shape[1]
    idx = torch.arange(n, device=onehot.device)
    far_apart = (idx[:, None] - idx[None]).abs() >= MIN_PAIR_SEPARATION
    valid = res_mask.bool()
    return canonical & far_apart & valid[:, :, None] & valid[:, None, :]


def node_states(tokens: Tensor) -> Tensor:
    """[..., N] per-nucleotide state: NODE_MASKED, NODE_UNPAIRED or NODE_PAIRED."""
    return torch.where(
        tokens == MASK,
        NODE_MASKED,
        torch.where(tokens == UNPAIRED, NODE_UNPAIRED, NODE_PAIRED),
    )


def pair_features(tokens: Tensor) -> Tensor:
    """
    [B, N, N] True where i and j are a revealed pair (both tokens revealed and
    pointing at each other). A masked nucleotide whose partner is revealed is not
    marked, so the feature never gives away a masked token.
    """
    n = tokens.shape[-1]
    is_paired = tokens >= 0
    points_to = torch.nn.functional.one_hot(tokens.clamp(min=0), n).bool()
    points_to &= is_paired[..., None]
    return points_to & points_to.transpose(-1, -2)


def token_targets(tokens: Tensor) -> Tensor:
    """Class indices [B, N] of the token logits: the partner j, or N for UNPAIRED."""
    n = tokens.shape[-1]
    return torch.where(tokens == UNPAIRED, n, tokens).clamp(min=0)


def random_mask_tokens(
    tokens: Tensor, res_mask: Tensor, mask_probability: float | None = None
) -> tuple[Tensor, Tensor]:
    """
    Masks tokens of valid nucleotides.

    If `mask_probability` is None, a masking ratio r ~ U(0, 1) is drawn per example
    and ceil(r * num_valid) tokens (at least one) are masked, so training covers all
    masking levels seen during sampling, from fully masked to almost fully unmasked.
    Otherwise every valid token is masked independently with `mask_probability`.
    The two tokens of a pair are masked independently.
    """
    res_mask = res_mask.bool()

    if mask_probability is None:
        num_batch = tokens.shape[0]
        ratio = torch.rand(num_batch, device=tokens.device)
        num_valid = res_mask.sum(dim=-1)
        num_masked = torch.ceil(ratio * num_valid).clamp(min=1)

        # Mask the num_masked valid residues with the lowest random scores.
        scores = torch.rand_like(tokens.float()).masked_fill(~res_mask, float("inf"))
        ranks = scores.argsort(dim=-1).argsort(dim=-1)
        random_mask = (ranks < num_masked[:, None]) & res_mask
    else:
        random_mask = (torch.rand_like(tokens.float()) < mask_probability) & res_mask

    masked_tokens = tokens.clone()
    masked_tokens[random_mask] = MASK

    return masked_tokens, random_mask


def available_options(tokens: Tensor) -> Tensor:
    """
    [B, N, N + 1] True for the values a masked nucleotide i can still take: a
    partner j that is masked itself (and not i), or unpaired (always).
    """
    num_batch, n = tokens.shape
    free = tokens == MASK
    partner_ok = free[:, None, :].expand(num_batch, n, n).clone()
    partner_ok &= ~torch.eye(n, dtype=torch.bool, device=tokens.device)
    unpaired_ok = torch.ones(num_batch, n, 1, dtype=torch.bool, device=tokens.device)
    return torch.cat([partner_ok, unpaired_ok], dim=-1)


def reveal_tokens(
    tokens: Tensor,
    probs: Tensor,
    candidates: Tensor,
    priority: Tensor,
    value_noise: Tensor | None = None,
) -> Tensor:
    """
    Reveals the masked tokens marked in `candidates` and returns the new tokens.

    Candidates are revealed one after another, highest `priority` first. Each one
    takes a value from its predicted distribution restricted to the options still
    available (`available_options`): a partner j that is still masked, or unpaired.
    Revealing i -> j also reveals j -> i, so the tokens stay a valid secondary
    structure. A candidate already revealed as the partner of an earlier one is
    skipped.

    Params:
        tokens      : [B, N] current tokens
        probs       : [B, N, N + 1] predicted probabilities (class N = unpaired)
        candidates  : [B, N] True for the tokens to reveal (only masked ones count)
        priority    : [B, N] order of the reveals, highest first
        value_noise : [B, N] uniform samples; the value of token i is drawn by
                      inverse CDF sampling with value_noise[i]. None takes the most
                      probable available value instead.
    """
    candidates = candidates & (tokens == MASK)
    if not candidates.any():
        return tokens
    n = tokens.shape[1]
    new = tokens.cpu().numpy().copy()
    order = priority.float().cpu().numpy()
    noise = None if value_noise is None else value_noise.float().cpu().numpy()
    for b in candidates.any(dim=-1).nonzero().squeeze(-1).tolist():
        rows = candidates[b].nonzero().squeeze(-1)
        row_probs = probs[b, rows].float().cpu().numpy()  # [R, N + 1]
        for r in np.argsort(-order[b, rows.cpu().numpy()], kind="stable"):
            i = rows[r].item()
            if new[b, i] != MASK:
                continue
            weights = row_probs[r].copy()
            available = new[b] == MASK
            available[i] = False
            weights[:n][~available] = 0.0
            total = weights.sum()
            if total <= 0:
                value = n
            elif noise is None:
                value = int(weights.argmax())
            else:
                cdf = np.cumsum(weights) / total
                value = int(min(np.searchsorted(cdf, noise[b, i], side="right"), n))
                while weights[value] == 0:  # never land on an unavailable option
                    value -= 1
            if value == n:
                new[b, i] = UNPAIRED
            else:
                new[b, i], new[b, value] = value, i
    return torch.from_numpy(new).to(tokens.device)
