"""EventDataset plus optional truth for the reconstruction heads."""

import awkward as ak
import numpy as np
import torch

from hepsetreg.datasets.EventDatasetBase import EventDataset, load_arrays, zscore, _require_fields

# Integer truth labels aligned with each measured collection, concatenated
# like the lepton features (electrons, then muons).
LABEL_FIELDS = {"jet": ["jet_labels"], "lepton": ["el_labels", "mu_labels"]}


class ReconstructionDataset(EventDataset):
    """Same inputs, scaling and padding as EventDataset.

    With `assignment` (an AssignmentTargets), batches also contain
    "assignment" [B, n_roles]: see AssignmentTargets.truth. Labels are read
    from targets.jet_labels, targets.el_labels and targets.mu_labels, with
    0 for other objects and a negative value where truth is unknown.
    `flow_targets` maps names to ordered targets.<column> lists; each needs
    a boolean targets.<name>_valid, and batches contain the standardised
    [B, n_columns] values (zero where unsupported) and that mask. Their
    scales fit supported rows only: pass the training `target_stats` (and
    `stats`) to validation. Without truth arguments, only measured inputs
    are read, e.g. for prediction.
    """

    def __init__(self, path, *, features, assignment=None, flow_targets=None, stats=None, target_stats=None,
                 max_jets=None, keep_unscaled_data=False):
        features = {**features, "target": []}
        arrays = load_arrays(path, features)
        if not ak.all(ak.num(arrays[2], axis=1) == 2):
            raise ValueError(f"{path}: EventDataset requires exactly two leptons per event")
        super().__init__(arrays, features=features, stats=stats, max_jets=max_jets,
                         keep_unscaled_data=keep_unscaled_data)
        self.truth, self.target_stats = {}, dict(target_stats or {})
        flow_targets = flow_targets or {}
        fields = [c for name, columns in flow_targets.items() for c in [*columns, f"{name}_valid"]]
        required_sources = {role[0] for role in assignment.roles} if assignment is not None else set()
        fields += [field for source, names in LABEL_FIELDS.items()
                   if source in required_sources for field in names]
        if not fields:
            return
        targets = ak.from_parquet(path, columns=[f"targets.{f}" for f in fields])["targets"]
        _require_fields(targets, fields, f"{path} targets")
        for name, columns in flow_targets.items():
            valid = ak.to_numpy(targets[f"{name}_valid"])
            values = np.stack([ak.to_numpy(ak.fill_none(targets[c], np.nan)) for c in columns], -1).astype(np.float32)
            if valid.dtype != np.bool_ or not np.isfinite(values[valid]).all():
                raise ValueError(f"{path}: {name}_valid must be boolean and every supported {name} value finite")
            if name not in self.target_stats:
                self.target_stats[name] = (values[valid].mean(0), values[valid].std(0))
            mean, std = self.target_stats[name]
            if not (std > 0).all():
                raise ValueError(f"{path}: {name} needs supported training rows with nonzero spread")
            scaled = np.where(valid[:, None], zscore(values, mean, std), 0.).astype(np.float32)
            self.truth[name], self.truth[f"{name}_valid"] = torch.from_numpy(scaled), torch.from_numpy(valid)
        if assignment is None:
            return
        labels = {}
        for source, objects, mask in (("jet", arrays[1], self.jet_mask), ("lepton", arrays[2], self.lepton_mask)):
            if source not in required_sources:
                continue
            values = ak.concatenate([targets[f] for f in LABEL_FIELDS[source]], axis=1)
            if not ak.all(ak.num(values, axis=1) == ak.num(objects, axis=1)):
                raise ValueError(f"{path}: {LABEL_FIELDS[source]} are not aligned with the measured {source}s")
            if ak.to_numpy(ak.flatten(values)).dtype.kind not in "iu":
                raise ValueError(f"{path}: {LABEL_FIELDS[source]} must be integers")
            dense = ak.fill_none(ak.pad_none(values, mask.shape[1], clip=True), 0)
            labels[source] = (ak.to_numpy(dense), mask.numpy())
        self.truth["assignment"] = torch.from_numpy(assignment.truth(labels))

    def to(self, device):
        super().to(device)
        self.truth = {k: v.to(device) for k, v in self.truth.items()}
        return self

    def __getitem__(self, idx):
        return {**super().__getitem__(idx), **{k: v[idx] for k, v in self.truth.items()}}

    def __getitems__(self, indices):
        idx = torch.as_tensor(indices, device=self.global_tensor.device)
        return {**super().__getitems__(indices), **{k: v[idx] for k, v in self.truth.items()}}
