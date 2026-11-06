"""Optional L-GATr encoders using physical, unscaled (E, px, py, pz)."""

import torch
from torch import nn

from hepsetreg.backbones.backbone import _mlp_embedder


def _fourmomenta(raw, names, prefix):
    """(E, px, py, pz) from unscaled features: e/px/py/pz, or e/pt/eta/phi."""
    column = lambda name: raw[..., names.index(prefix + name)]
    if all(prefix + n in names for n in ("px", "py", "pz")):
        px, py, pz = column("px"), column("py"), column("pz")
    else:
        pt, eta, phi = column("pt"), column("eta"), column("phi")
        px, py, pz = pt * torch.cos(phi), pt * torch.sin(phi), pt * torch.sinh(eta)
    return torch.stack((column("e"), px, py, pz), dim=-1)


class LGATrBackbone(nn.Module):
    """Same return contract and token order as TransformerBackbone.

    Jet and lepton four-momenta are built from the batch's unscaled
    features (EventDataset(keep_unscaled_data=True)): `features` must
    contain jet_e and either jet_px/jet_py/jet_pz or jet_pt/jet_eta/jet_phi,
    and the same lepton names without the prefix. They enter the geometric
    stream in units of `momentum_scale` (GeV); standardised features enter
    only the scalar stream. The time and beam spurions retain the detector's
    preferred frame and beam direction. Full L-GATr uses an xy-plane
    bivector; Slim uses a directed +z vector instead.
    """

    def __init__(self, features, d_model=128, nhead=8, num_layers=4, dropout=0.1,
                 slim=False, momentum_scale=100.0):
        super().__init__()
        try:
            import lgatr
        except ImportError as exc:
            raise ImportError("Install hepsetreg[lgatr] for this encoder") from exc
        self.features = features
        self.slim = slim
        self.momentum_scale = momentum_scale
        self.d_model = d_model
        self.global_embed = _mlp_embedder(len(features["global"]), d_model)
        self.jet_embed = _mlp_embedder(len(features["jet"]), d_model)
        self.lepton_embed = _mlp_embedder(len(features["lepton"]), d_model)
        self.type_embedding = nn.Embedding(3, d_model)
        common = dict(num_blocks=num_layers, in_s_channels=d_model,
                      out_s_channels=d_model, hidden_s_channels=d_model,
                      dropout_prob=dropout, checkpoint_blocks=False,
                      naive_amp=False, compile=False)
        if slim:
            self.register_buffer("spurions", torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 1.]]))
            self.network = lgatr.LGATrSlim(
                in_v_channels=3, out_v_channels=1, hidden_v_channels=16,
                num_heads=nhead, **common)
        else:
            self.register_buffer("spurions", lgatr.get_spurions("xyplane", True))
            n = len(self.spurions)
            self.network = lgatr.LGATr(
                in_mv_channels=1 + n, out_mv_channels=1, hidden_mv_channels=16,
                attention={"num_heads": nhead}, mlp={},
                reinsert_mv_channels=tuple(range(1, 1 + n)), **common)

    def forward(self, batch, return_tokens=False):
        import lgatr

        jets, leptons = batch["jet"], batch["lepton"]
        # The event token exists even for an empty jet collection.
        valid = torch.cat((torch.ones(jets.shape[0], 1, device=jets.device, dtype=torch.bool),
                           batch["jet_mask"], batch["lepton_mask"]), dim=1)
        scalars = torch.cat((self.global_embed(batch["global"]).unsqueeze(1) + self.type_embedding.weight[0],
                             self.jet_embed(jets) + self.type_embedding.weight[1],
                             self.lepton_embed(leptons) + self.type_embedding.weight[2]), dim=1)
        momenta = []
        for name, prefix in (("jet", "jet_"), ("lepton", "")):
            if f"{name}_unscaled" not in batch:
                raise KeyError("L-GATr needs physical momenta: use EventDataset(keep_unscaled_data=True)")
            p4 = _fourmomenta(batch[f"{name}_unscaled"], self.features[name], prefix)
            mask = batch[f"{name}_mask"]
            if not torch.isfinite(p4[mask]).all():
                raise ValueError(f"non-finite {name} four-momenta")
            momenta.append(p4.masked_fill(~mask.unsqueeze(-1), 0.) / self.momentum_scale)
        p4 = torch.cat(momenta, dim=1)
        p4 = torch.cat((p4.new_zeros(p4.shape[0], 1, 4), p4), dim=1)
        vectors = p4 if self.slim else lgatr.embed_vector(p4)
        spurions = self.spurions.to(vectors).view(1, 1, -1, vectors.shape[-1])
        vectors = torch.cat((vectors.unsqueeze(-2), spurions.expand(*vectors.shape[:2], -1, -1)), dim=-2)
        scalars = scalars.masked_fill(~valid.unsqueeze(-1), 0.)
        _, tokens = self.network(vectors, scalars, attn_mask=valid[:, None, None, :])
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0.)
        return (tokens[:, 0], tokens, valid) if return_tokens else tokens[:, 0]
