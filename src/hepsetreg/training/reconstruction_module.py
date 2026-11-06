"""LightningModule training reconstruction heads on one shared encoder pass."""

import pytorch_lightning as pl
import torch
from torch import nn

from hepsetreg.backbones.flow_matching import flow_matching_loss, sample_flow


class ReconstructionModule(pl.LightningModule):
    """An optional AssignmentHead and velocity fields for named continuous
    targets (batch[name], standardised, conditioned on the pooled event)
    share the backbone. Each flow is trained on its supported events
    (batch[f"{name}_valid"]); the loss is the weighted sum over heads."""

    def __init__(self, backbone, assignment=None, flows=None, loss_weights=None, learning_rate=1e-3,
                 predict_n_steps=50, predict_n_samples=8):
        super().__init__()
        self.backbone = backbone
        self.assignment = assignment
        self.flows = nn.ModuleDict(flows or {})
        names = list(self.flows) + (["assignment"] if assignment is not None else [])
        self.loss_weights = {name: 1.0 for name in names} | dict(loss_weights or {})
        if not names or len(set(names)) != len(names) or set(self.loss_weights) != set(names):
            raise ValueError(f"heads {names} must be distinct and non-empty; loss_weights must name only them")
        self.learning_rate = learning_rate
        self.predict_n_steps = predict_n_steps
        self.predict_n_samples = predict_n_samples

    def losses(self, batch):
        pooled, tokens, _ = self.backbone(batch, return_tokens=True)
        losses = {}
        if self.assignment is not None:
            losses["assignment"] = self.assignment.loss(tokens, batch)
        for name, field in self.flows.items():
            valid = batch[f"{name}_valid"]
            losses[name] = (flow_matching_loss(field, batch[name][valid], pooled[valid])
                            if valid.any() else pooled.sum() * 0.)
        return losses

    def _step(self, batch, stage):
        losses = self.losses(batch)
        total = sum(self.loss_weights[name] * loss for name, loss in losses.items())
        batch_size = len(batch["global"])
        self.log_dict({f"{stage}_{k}_loss": v for k, v in losses.items()}, batch_size=batch_size)
        self.log(f"{stage}_loss", total, prog_bar=True, batch_size=batch_size)
        return total

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def predict_step(self, batch, batch_idx):
        """CPU dict: assignment indices/probabilities and, per flow, still
        standardised samples [B, predict_n_samples, n_columns]."""
        pooled, tokens, _ = self.backbone(batch, return_tokens=True)
        out = {name: sample_flow(field, pooled, field.output_dim, n_steps=self.predict_n_steps,
                                 n_samples=self.predict_n_samples).cpu()
               for name, field in self.flows.items()}
        if self.assignment is not None:
            index, prob = self.assignment.predict(tokens, batch)
            out["assignment"], out["assignment_prob"] = index.cpu(), prob.cpu()
        return out

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
