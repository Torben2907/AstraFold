"""
Neural network architecture for the flow model.

Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/models/flow_model.py
"""

import torch
from torch import Tensor, nn

from astrafold.config.hydra_schema import ModelConfig
from astrafold.data import utils as du
from astrafold.data.base_pairs import (
    FORBIDDEN_LOGIT,
    allowed_partners,
    pair_features,
)
from astrafold.data.rigid_utils import Rigid
from astrafold.models import ipa_pytorch, torsion_net
from astrafold.models.edge_embedder import EdgeEmbedder
from astrafold.models.node_embedder import NodeEmbedder


class FlowModel(nn.Module):
    def __init__(self, model_conf: ModelConfig):
        super().__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.dropout_rate = self._ipa_conf.dropout

        self.node_embedder = NodeEmbedder(model_conf.node_features)
        self.edge_embedder = EdgeEmbedder(model_conf.edge_features)

        # Base-pair token heads: a logit for every possible partner j of nucleotide
        # i, from the pair representation, and one for "unpaired", from the node.
        c_s, c_z = self._ipa_conf.c_s, self._model_conf.edge_embed_size
        self.pair_node_proj = nn.Linear(c_s, c_z)
        self.pair_head = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, 1),
        )
        self.unpaired_head = nn.Sequential(
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, 1),
        )

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f"ipa_{b}"] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=self.dropout_rate,
                norm_first=False,
            )

            self.trunk[f"seq_tfmr_{b}"] = torch.nn.TransformerEncoder(
                tfmr_layer,
                self._ipa_conf.seq_tfmr_num_layers,
                enable_nested_tensor=False,
            )
            self.trunk[f"post_tfmr_{b}"] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final"
            )
            self.trunk[f"node_transition_{b}"] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s
            )
            self.trunk[f"bb_update_{b}"] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True
            )

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f"edge_transition_{b}"] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

        # hparams taken from OpenFold's config.py
        self.angle_pred_net = torsion_net.TorsionAngleHead(
            c_in=self._ipa_conf.c_s,
            c_hidden=128,
            no_blocks=2,
            no_angles=9,
            epsilon=1e-12,
        )
        self.ipa_dropout = nn.Dropout(self.dropout_rate)

    def rigids_ang_to_nm(self, x: Rigid) -> Rigid:
        return x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)

    def rigids_nm_to_ang(self, x: Rigid) -> Rigid:
        return x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)

    def forward(self, input_feats: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Predicts the clean structure x1 from the noisy frames at flow time t, the
        sequence and the (partially masked) base-pair tokens, and the base-pair
        tokens: `token_logits` [B, N, N + 1] scores every partner j of nucleotide i
        (columns 0..N-1; pairs that are not canonical or too close in sequence get
        FORBIDDEN_LOGIT) and "unpaired" (column N).
        """
        node_mask = input_feats["res_mask"]
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        continuous_t = input_feats["t"]
        trans_t = input_feats["trans_t"]
        rotmats_t = input_feats["rotmats_t"]
        pair_tokens = input_feats["pair_tokens"]
        onehot = input_feats["onehot"]

        # Initialize node and edge embeddings
        init_node_embed = self.node_embedder(
            continuous_t, node_mask, onehot, pair_tokens
        )
        if "trans_sc" not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats["trans_sc"]

        init_edge_embed = self.edge_embedder(
            init_node_embed, trans_t, trans_sc, edge_mask, pair_features(pair_tokens)
        )

        # Initial rigids
        curr_rigids = du.create_rigid(
            rotmats_t,
            trans_t,
        )

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]

        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f"ipa_{b}"](
                node_embed, edge_embed, curr_rigids, node_mask
            )
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                node_embed, src_key_padding_mask=(1 - node_mask).bool()
            )
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)
            node_embed = self.ipa_dropout(node_embed)
            node_embed = self.trunk[f"node_transition_{b}"](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f"bb_update_{b}"](
                node_embed * node_mask[..., None]
            )
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None]
            )

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed = self.ipa_dropout(edge_embed)
                edge_embed *= edge_mask[..., None]

        # Predict the 9 torsion angles as (sin, cos) pairs
        _, pred_torsions = self.angle_pred_net(node_embed, init_node_embed)

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()

        token_logits = self._token_logits(node_embed, edge_embed, onehot, node_mask)

        return {
            "pred_torsions": pred_torsions,
            "pred_trans": pred_trans,
            "pred_rotmats": pred_rotmats,
            "token_logits": token_logits,
            "bb_frame": curr_rigids,
        }

    def _token_logits(
        self, node_embed: Tensor, edge_embed: Tensor, onehot: Tensor, node_mask: Tensor
    ) -> Tensor:
        """[B, N, N + 1] base-pair token logits (see `forward`)."""
        node_proj = self.pair_node_proj(node_embed)
        pair_repr = edge_embed + node_proj[:, :, None] + node_proj[:, None, :]
        partner_logits = self.pair_head(pair_repr).squeeze(-1).float()
        # A pair is one decision: score (i, j) and (j, i) alike.
        partner_logits = (partner_logits + partner_logits.transpose(1, 2)) / 2
        allowed = allowed_partners(onehot.float(), node_mask)
        partner_logits = partner_logits.masked_fill(~allowed, FORBIDDEN_LOGIT)
        unpaired_logits = self.unpaired_head(node_embed).float()
        return torch.cat([partner_logits, unpaired_logits], dim=-1)
