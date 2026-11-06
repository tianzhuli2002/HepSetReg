"""Select an encoder without importing optional geometric dependencies."""

from hepsetreg.backbones.backbone import TransformerBackbone


def build_encoder(features, *, encoder="transformer", **kwargs):
    dimensions = dict(n_global_features=len(features["global"]),
                      n_jet_features=len(features["jet"]),
                      n_lepton_features=len(features["lepton"]))
    if encoder == "transformer":
        return TransformerBackbone(**dimensions, **kwargs)
    if encoder in ("lgatr", "lgatr_slim"):
        from hepsetreg.backbones.backbone_lgatr import LGATrBackbone
        return LGATrBackbone(features, slim=encoder == "lgatr_slim", **kwargs)
    raise ValueError(f"Unknown encoder {encoder!r}; use transformer, lgatr or lgatr_slim")
