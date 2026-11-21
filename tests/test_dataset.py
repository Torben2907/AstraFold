import torch
from test_utils import load_example_batch

from astrafold.data.base_pairs import UNPAIRED, allowed_partners


def test_data(config):
    batch = load_example_batch(config)
    tokens = batch["pair_tokens"]
    assert tokens.shape == batch["res_mask"].shape
    assert tokens.dtype == torch.long
    assert torch.all(
        (tokens == UNPAIRED) | ((tokens >= 0) & (tokens < tokens.shape[1]))
    )

    # The pairs are symmetric and allowed (canonical, far enough apart, valid).
    allowed = allowed_partners(batch["onehot"].float(), batch["res_mask"])
    for b in range(tokens.shape[0]):
        paired = torch.nonzero(tokens[b] >= 0).squeeze(-1)
        partners = tokens[b, paired]
        assert torch.equal(tokens[b, partners], paired)
        assert allowed[b, paired, partners].all()
    assert (tokens >= 0).any()
