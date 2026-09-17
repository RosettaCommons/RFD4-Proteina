"""Pure-PyTorch port of the GearNet CATH fold classifier from NVIDIA-BioNeMo/proteina.

Reimplements the architecture from
`proteinfoundation/metrics/gearnet_utils.py
<https://github.com/NVIDIA-BioNeMo/proteina/blob/main/proteinfoundation/metrics/gearnet_utils.py>`_
(originally: `Protein Representation Learning by Geometric Structure Pretraining
<https://arxiv.org/abs/2203.06125>`_, Zhang et al.) so that its public CA-only checkpoint
(``gearnet_ca.pth``) loads directly via ``load_state_dict``, without depending on
``torch_geometric`` / ``torch_scatter`` (neither is a project dependency). NVIDIA's public
release only includes the CA-only checkpoint (not the full-backbone ``gearnet.pth``), so
this port only supports the CA-only architecture.

The graph construction (radius graph) is reimplemented below in plain PyTorch;
message aggregation reuses :py:meth:`torch.Tensor.scatter_reduce_`. Parameter names
match the published weights.

The original code is licensed by NVIDIA under a non-commercial research license
(``LicenseRef-NvidiaProprietary``)
"""

import math

import torch
from torch import Tensor, nn


def _scatter_reduce(src: Tensor, index: Tensor, reduce: str, dim_size: int | None = None) -> Tensor:
    """Reduce ``src`` rows into ``dim_size`` buckets given by ``index`` (dim=0 scatter)."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    shape = list(src.shape)
    shape[0] = dim_size
    index_expanded = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return src.new_zeros(shape).scatter_reduce_(0, index_expanded, src, reduce=reduce, include_self=False)


def _radius_graph(x: Tensor, r: float, batch: Tensor, max_num_neighbors: int | None = None) -> Tensor:
    """Undirected radius graph, batched, pure-torch replacement for ``torch_geometric.nn.radius_graph``.

    For every pair of points ``(i, j)`` in the same batch element with ``i != j`` and
    ``dist(i, j) <= r``, emits both directed edges ``(i, j)`` and ``(j, i)``. When
    ``max_num_neighbors`` is set, caps each point's neighbor count to the nearest
    ``max_num_neighbors`` (ties broken arbitrarily), matching the intent (not necessarily
    the exact tie-breaking) of the upstream implementation.

    Returns:
      Edge index of shape ``[2, num_edges]``, rows ``(source, target)``.
    """
    x2d = x.reshape(x.shape[0], -1).float()
    sources, targets = [], []
    for b in torch.unique(batch):
        idx = torch.nonzero(batch == b, as_tuple=True)[0]
        sub = x2d[idx]
        dist = torch.cdist(sub, sub)
        dist.fill_diagonal_(float("inf"))
        within = dist <= r
        if max_num_neighbors is not None and within.shape[0] > 0:
            order = torch.argsort(dist, dim=1)
            keep = torch.zeros_like(within)
            keep.scatter_(1, order[:, :max_num_neighbors], True)
            within = within & keep
        target_local, source_local = torch.nonzero(within, as_tuple=True)
        sources.append(idx[source_local])
        targets.append(idx[target_local])
    if not sources:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    return torch.stack([torch.cat(sources), torch.cat(targets)], dim=0)


def _rbf(d: Tensor, rbf_dim: int, d_min: float = 0.0, d_max: float = 20.0) -> Tensor:
    d_mu = torch.linspace(d_min, d_max, rbf_dim, device=d.device).view(1, -1)
    d_sigma = (d_max - d_min) / rbf_dim
    return torch.exp(-(((d.unsqueeze(-1) - d_mu) / d_sigma) ** 2))


def _clockwise_angle(p1: Tensor, p2: Tensor) -> Tensor:
    x = (p1 * p2).sum(dim=-1)
    y = torch.cross(p1, p2, dim=-1)
    return torch.atan2(y.norm(dim=-1), x) * torch.sign(y[..., 2])


def _get_index_embedding(indices: Tensor, embed_size: int, max_len: int = 2056) -> Tensor:
    k = torch.arange(embed_size // 2, device=indices.device)
    angle = indices[..., None] * math.pi / (max_len ** (2 * k[None] / embed_size))
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


class _Linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.module = nn.Sequential(
            nn.BatchNorm1d(in_channels, momentum=0.2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(in_channels, out_channels, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)


class _MLP(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.module = nn.Sequential(
            nn.BatchNorm1d(in_channels, momentum=0.2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512, momentum=0.2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)


class _GeometricRelationalGraphConv(nn.Module):
    """512-channel graph convolution with six relations and 256-channel edge features."""

    def __init__(self):
        super().__init__()
        self.input = _MLP(512)
        self.edge_input = _MLP(256)
        self.linear = _Linear(6 * 512, 512)
        self.output = _Linear(512, 512)

    def forward(self, h_v: Tensor, edge_index: Tensor, h_e: Tensor) -> Tensor:
        node_in, node_out, relation_type = edge_index
        message = self.input(h_v)[node_in] + self.edge_input(h_e)
        update = _scatter_reduce(message, node_out * 6 + relation_type, reduce="sum", dim_size=h_v.shape[0] * 6)
        return self.output(self.linear(update.view(h_v.shape[0], 6 * 512))) + h_v


class GearNet(nn.Module):
    """The released CA-only CATH classifier, with checkpoint-compatible parameter names."""

    def __init__(self) -> None:
        super().__init__()
        self.atom_embedding = nn.Embedding(1, 256)
        self.layers = nn.ModuleList([_GeometricRelationalGraphConv() for _ in range(8)])
        self.mlp = _MLP(512)
        self.pred_head_T = nn.Linear(512, 1336)
        self.pred_head_A = nn.Linear(512, 43)
        self.pred_head_C = nn.Linear(512, 5)

    def construct_graph(self, atom_seq_pos: Tensor, coords: Tensor, atom2batch: Tensor) -> Tensor:
        max_distance = 2
        node_in, node_out = _radius_graph(atom_seq_pos.float(), max_distance + 0.1, batch=atom2batch)
        relation = (atom_seq_pos[node_out] - atom_seq_pos[node_in] + max_distance).clamp(0, 4)
        seq_edge_list = torch.stack([node_in, node_out, relation], dim=0)

        node_in, node_out = _radius_graph(coords, 10.0, batch=atom2batch, max_num_neighbors=64)
        radius_edge_list = torch.stack([node_in, node_out, torch.full_like(node_in, 5)], dim=0)

        return torch.cat([seq_edge_list, radius_edge_list], dim=1)

    def node_feature(self, atom_seq_pos: Tensor) -> Tensor:
        atom_type_embedding = self.atom_embedding(torch.zeros_like(atom_seq_pos))
        position_embedding = _get_index_embedding(atom_seq_pos, 256, max_len=3000)
        return torch.cat([atom_type_embedding, position_embedding], dim=-1)

    def edge_feature(self, edge_list: Tensor, atom_seq_pos: Tensor, coords: Tensor, atom2batch: Tensor) -> Tensor:
        node_in, node_out, _ = edge_list
        pos_in, pos_out = coords[node_in], coords[node_out]
        rbf_feat = _rbf((pos_out - pos_in).norm(dim=-1), 128)

        rel_pos = atom_seq_pos[node_out] - atom_seq_pos[node_in]
        rel_pos_embedding = _get_index_embedding(rel_pos, 126, max_len=3000)

        center_coord = _scatter_reduce(coords, atom2batch, reduce="mean", dim_size=int(atom2batch.max().item()) + 1)
        diff_node_in = pos_in - center_coord[atom2batch[node_in]]
        diff_node_out = pos_out - center_coord[atom2batch[node_out]]
        angle = _clockwise_angle(diff_node_in, diff_node_out)
        angle_feat = torch.stack([angle.sin(), angle.cos()], dim=-1)

        return torch.cat([rbf_feat, rel_pos_embedding, angle_feat], dim=-1)

    def atom_info(self, coords: Tensor, coord_mask: Tensor, batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        flat_coords = coords[coord_mask]
        atom2batch = batch[coord_mask]

        num_residues = _scatter_reduce(torch.ones_like(batch), batch, reduce="sum")
        num_cum_residues = num_residues.cumsum(dim=0)
        residue_id = torch.arange(batch.shape[0], device=batch.device) - (num_cum_residues - num_residues)[batch]
        atom_seq_pos = residue_id[coord_mask]

        return flat_coords, atom_seq_pos, atom2batch

    def forward(self, coords: Tensor, coord_mask: Tensor, batch: Tensor) -> dict[str, Tensor]:
        """Predict per-protein CATH class logits from per-residue CA coordinates.

        Args:
          coords: Per-residue CA coordinates. Shape ``[n_res, 3]``.
          coord_mask: Which residues have a resolved CA. Shape ``[n_res]``.
          batch: Which protein each residue belongs to (supports batching multiple proteins
            in one forward pass). Shape ``[n_res]``.

        Returns:
          Dict with ``protein_feature`` (pooled embedding) and ``pred_C``, ``pred_A``,
          and ``pred_T`` classification logits.
        """
        coord_mask = coord_mask.bool()
        flat_coords, atom_seq_pos, atom2batch = self.atom_info(coords, coord_mask, batch)
        h_v = self.node_feature(atom_seq_pos)

        edge_list = self.construct_graph(atom_seq_pos, flat_coords, atom2batch)
        h_e = self.edge_feature(edge_list, atom_seq_pos, flat_coords, atom2batch)

        for layer in self.layers:
            h_v = layer(h_v, edge_list, h_e)

        protein_feature = self.mlp(_scatter_reduce(h_v, atom2batch, reduce="sum", dim_size=int(batch.max().item()) + 1))

        return {
            "protein_feature": protein_feature,
            "pred_C": self.pred_head_C(protein_feature),
            "pred_A": self.pred_head_A(protein_feature),
            "pred_T": self.pred_head_T(protein_feature),
        }
