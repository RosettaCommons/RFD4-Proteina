"""Tensor utility functions for search algorithms."""

import torch


def concat_dict_tensors(dict_list: list[dict[str, torch.Tensor]], dim: int = 0) -> dict[str, torch.Tensor]:
    """Concatenate a list of dictionaries of tensors along specified dimension.

    Args:
        dict_list: List of dictionaries with the same keys and tensor values
        dim: Dimension along which to concatenate (default: 0)

    Returns:
        Dictionary with concatenated tensors for each key

    Example:
        >>> dict1 = {"a": torch.tensor([1, 2]), "b": torch.tensor([3, 4])}
        >>> dict2 = {"a": torch.tensor([5, 6]), "b": torch.tensor([7, 8])}
        >>> result = concat_dict_tensors([dict1, dict2], dim=0)
        >>> result["a"]
        tensor([1, 2, 5, 6])
    """
    if not dict_list:
        return {}

    if len(dict_list) == 1:
        return dict_list[0]

    # Get keys from first dict
    keys = dict_list[0].keys()

    # Concatenate tensors for each key
    result = {}
    for key in keys:
        tensors = [d[key] for d in dict_list if key in d]
        if tensors:
            result[key] = torch.cat(tensors, dim=dim)

    return result
