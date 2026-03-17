"""Transform for computing Laplacian positional encodings for molecular graphs."""

from typing import Any, ClassVar

import networkx as nx
import numpy as np
import scipy.sparse
import torch
from atomworks.constants import BIOTITE_BOND_TYPE_TO_BOND_ORDER
from atomworks.io.utils.bonds import _atom_array_to_networkx_graph
from atomworks.ml.transforms._checks import check_atom_array_annotation
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray
from biotite.structure.bonds import BondType
from jaxtyping import Float, Int8
from scipy.linalg import sqrtm
from scipy.sparse.linalg import eigs

# Module-level lookup table for bond type -> bond order (computed once at import)
# Size based on all biotite BondType enum values (more robust than BIOTITE_BOND_TYPE_TO_BOND_ORDER)
_BOND_TYPE_LOOKUP = np.zeros(
    max(bt.value for bt in BondType) + 1,
    dtype=np.float32,
)

# Map standard covalent bonds from atomworks constants
for _bt, _order in BIOTITE_BOND_TYPE_TO_BOND_ORDER.items():
    _BOND_TYPE_LOOKUP[_bt.value] = _order

# Map missing bond types not in BIOTITE_BOND_TYPE_TO_BOND_ORDER
# COORDINATION: Metal-ligand bonds (common in OMOL), treat as weak single bonds
_BOND_TYPE_LOOKUP[BondType.COORDINATION.value] = 0.5
# AROMATIC: Generic aromatic bonds (delocalized electrons), treat as single bond
_BOND_TYPE_LOOKUP[BondType.AROMATIC.value] = 1.0


def _pad_or_truncate_features(
    features: Float[np.ndarray, "n_nodes n_features"],
    target_dim: int,
) -> Float[np.ndarray, "n_nodes target_dim"]:
    """Pad with zeros or truncate feature matrix to match target dimensionality.

    Args:
        features: Feature matrix of shape (n_nodes, n_features).
        target_dim: Desired number of features.

    Returns:
        Feature matrix of shape (n_nodes, target_dim).
    """
    n_nodes, n_features = features.shape

    if n_features < target_dim:
        padding = np.zeros((n_nodes, target_dim - n_features), dtype=np.float32)
        return np.concatenate([features, padding], axis=1)
    elif n_features > target_dim:
        return features[:, :target_dim]
    else:
        return features


def compute_graph_laplacian_eigenvectors(
    adjacency_matrix: Float[np.ndarray, "n_atoms n_atoms"],
    n_eigenvectors: int = 32,
    sparse_threshold: int = 100,
) -> Float[torch.Tensor, "n_atoms n_eigenvectors"]:
    """Compute graph Laplacian positional encodings from (maybe weighted) adjacency matrix.

    Uses the symmetric normalized Laplacian and extracts its eigenvectors to create
    positional encodings that capture the graph's topological structure.

    Args:
        adjacency_matrix: Square adjacency matrix (can be weighted for bond orders).
        n_eigenvectors: Number of eigenvectors to use for positional encoding.
        sparse_threshold: Use sparse eigendecomposition for graphs larger than this.

    Returns:
        Positional encoding tensor of shape (n_atoms, n_eigenvectors).
        Returns zeros for graphs with 3 or fewer nodes.
    """
    n_atoms = adjacency_matrix.shape[0]

    # Edge case: very small molecules lack meaningful graph structure
    if n_atoms <= 3:
        return torch.zeros(n_atoms, n_eigenvectors, dtype=torch.float32)

    # For small molecules, we can only compute (n_atoms - 1) eigenvectors
    n_laplacian_eigenvectors = min(n_atoms - 1, n_eigenvectors)

    # Build degree matrix from adjacency
    adjacency = adjacency_matrix.astype(np.float32)
    degree_values = np.sum(adjacency, axis=1)
    degree_matrix = np.diag(degree_values) + 1e-8 * np.eye(n_atoms)  # Add epsilon for stability

    # Compute symmetric normalized Laplacian: L = I - D^(-1/2) @ A @ D^(-1/2)
    degree_inv_sqrt = np.linalg.pinv(sqrtm(degree_matrix))
    laplacian = np.eye(n_atoms) - degree_inv_sqrt @ adjacency @ degree_inv_sqrt
    laplacian = np.asarray(laplacian, dtype=np.float32)

    # Perform eigendecomposition (dense or sparse based on graph size)
    if n_atoms < sparse_threshold:
        # Dense eigendecomposition for small graphs
        laplacian_dense = laplacian
        eigenvalues, eigenvectors = np.linalg.eig(laplacian_dense)
    else:
        # Sparse eigendecomposition for large graphs
        laplacian_sparse = scipy.sparse.coo_matrix(laplacian)
        eigenvalues, eigenvectors = eigs(
            laplacian_sparse,
            k=n_laplacian_eigenvectors + 1,
            which="SR",  # Smallest real eigenvalues
            return_eigenvectors=True,
        )

    # Sort eigenvectors by eigenvalue magnitude
    sorted_indices = eigenvalues.argsort()
    eigenvectors_sorted = np.real(eigenvectors[:, sorted_indices])

    # Skip the first (trivial) eigenvector and take the next k
    positional_encoding = eigenvectors_sorted[:, 1 : n_laplacian_eigenvectors + 1]
    positional_encoding = np.asarray(positional_encoding, dtype=np.float32)

    # Apply random sign flip to handle eigenvector sign ambiguity
    # This is standard practice in GNN positional encodings
    random_signs = -1 + 2 * np.random.randint(0, 2, (n_laplacian_eigenvectors,))
    positional_encoding *= random_signs

    # Ensure output has exactly n_eigenvectors dimensions
    positional_encoding = _pad_or_truncate_features(positional_encoding, n_eigenvectors)

    return torch.from_numpy(positional_encoding).float()


def _bond_type_matrix_to_bond_order(
    bond_type_matrix: Int8[np.ndarray, "n_atoms n_atoms"],
) -> Float[np.ndarray, "n_atoms n_atoms"]:
    """Convert biotite bond type matrix to bond order matrix (vectorized).

    Args:
        bond_type_matrix: Matrix from bonds.bond_type_matrix() with BondType int values.
            -1 indicates no bond.

    Returns:
        Bond order matrix where no bonds are 0.
    """
    valid_mask = bond_type_matrix >= 0
    result = np.zeros_like(bond_type_matrix, dtype=np.float32)
    result[valid_mask] = _BOND_TYPE_LOOKUP[bond_type_matrix[valid_mask]]
    return result


def _get_weighted_adjacency_matrix(
    atom_array: AtomArray,
) -> Float[np.ndarray, "n_atoms n_atoms"]:
    """Extract adjacency matrix with bond order weights.

    Args:
        atom_array: AtomArray with bonds.

    Returns:
        Weighted adjacency matrix with bond orders as weights.
    """
    bond_type_matrix = atom_array.bonds.bond_type_matrix()
    return _bond_type_matrix_to_bond_order(bond_type_matrix)


def featurize_bonds_as_laplacian_positional_encoding(
    atom_array: AtomArray,
    encoding_dim: int = 32,
    use_bond_order: bool = True,
) -> Float[torch.Tensor, "n_tokens encoding_dim"]:
    """Compute graph Laplacian positional encodings for atomized molecules at token level.

    For each atomized molecule, computes graph Laplacian eigenvectors and assigns
    them to the corresponding token positions. Non-atomized tokens receive zeros.
    For atomized molecules, each atom is its own token.

    Args:
        atom_array: AtomArray with atomize, molecule_iid, and token_id annotations.
        encoding_dim: Dimensionality of the positional encoding.
        use_bond_order: If True, weight edges by bond order. If False, use binary adjacency.

    Returns:
        Token-level positional encoding tensor of shape (n_tokens, encoding_dim).
    """
    token_starts = get_token_starts(atom_array, add_exclusive_stop=False)
    n_tokens = len(token_starts)  

    # Initialize zero encodings for all tokens
    graph_laplacian_pe = torch.zeros(n_tokens, encoding_dim, dtype=torch.float32)

    # Build mapping from atom index to token index
    token_id = atom_array.token_id
    unique_token_ids = np.sort(np.unique(token_id))
    atom_to_token_idx = np.searchsorted(unique_token_ids, token_id)  # [n_atoms]

    # Subset to only atomized atoms. Biotite re-indexes the BondList to the subset,
    # dropping any bonds that cross into non-atomized atoms.
    atomized_mask = atom_array.atomize  # [n_atoms] bool
    if not atomized_mask.any():
        return graph_laplacian_pe

    atomized_aa = atom_array[atomized_mask]
    atomized_indices = np.where(atomized_mask)[0]  # positions in original atom_array

    # Find connected components in the bond graph of atomized atoms.
    # We use an unweighted graph here — connectivity only, not bond orders.
    nx_graph = _atom_array_to_networkx_graph(atomized_aa, bond_order=False)
    components = list(nx.connected_components(nx_graph))

    for component_node_set in components:
        component_local_idxs = np.array(sorted(component_node_set))

        component_aa = atomized_aa[component_local_idxs]

        # Extract adjacency matrix (weighted by bond order or binary)
        if use_bond_order:
            adjacency_matrix = _get_weighted_adjacency_matrix(component_aa)
        else:
            bond_type_matrix = component_aa.bonds.bond_type_matrix()
            adjacency_matrix = (bond_type_matrix >= 0).astype(np.float32)

        # Compute graph Laplacian positional encodings for this connected component
        component_pe = compute_graph_laplacian_eigenvectors(
            adjacency_matrix,
            n_eigenvectors=encoding_dim,
        )

        # Map component atom positions back to token indices in the original array
        original_atom_indices = atomized_indices[component_local_idxs]
        token_indices = atom_to_token_idx[original_atom_indices]
        graph_laplacian_pe[token_indices] = component_pe

    return graph_laplacian_pe


class FeaturizeBondsAsLaplacianPositionalEncoding(Transform):
    """Compute graph Laplacian positional encodings for atomized molecular components.

    Computes positional encodings based on the eigenvectors of the symmetric normalized
    graph Laplacian. These atom-level encodings capture the topological structure of
    molecular graphs (ligands, ions, small molecules). Non-atomized atoms (e.g., polymer
    residues) receive zero vectors.

    The Laplacian PE provides:
    - Global graph structure information
    - Spectral graph properties
    - Bond connectivity patterns, optionally weighted by bond order

    Args:
        encoding_dim: Dimensionality of the positional encoding (default: 32).
            Determines the number of eigenvectors used.
        use_bond_order: If True, weight edges by bond order. If False, use binary adjacency.
    """

    requires_previous_transforms: ClassVar[list[str]] = ["AtomizeByCCDName"]

    def __init__(self, encoding_dim: int = 32, use_bond_order: bool = True):
        """Initialize Laplacian positional encoding transform."""
        super().__init__()
        self.encoding_dim = encoding_dim
        self.use_bond_order = use_bond_order

    def check_input(self, data: dict[str, Any]) -> None:
        """Validate that input data contains required fields."""
        check_atom_array_annotation(data, required=["atomize", "molecule_iid", "token_id"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Compute Laplacian positional encodings for each atomized molecule at token level."""
        if "feats" not in data:
            data["feats"] = {}

        data["feats"]["graph_laplacian_pe_token_level"] = featurize_bonds_as_laplacian_positional_encoding(
            data["atom_array"],
            encoding_dim=self.encoding_dim,
            use_bond_order=self.use_bond_order,
        )

        return data
