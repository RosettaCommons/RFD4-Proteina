"""Hydra plumbing: custom OmegaConf resolvers and config instantiation helpers."""

import importlib
from collections.abc import Mapping, Sequence

import hydra
from atomworks.enums import ChainType, ChainTypeInfo
from beartype.typing import Any
from lightning.fabric.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from rfproteina.callbacks.callback import BaseCallback
from rfproteina.utils.common import run_once


def non_null_values(mapping: Mapping[str, Any | None]) -> list[Any]:
    """Return non-null mapping values in insertion order.

    Examples:
        >>> non_null_values({"date": "year < 2024", "resolution": None})
        ['year < 2024']
    """
    return [value for value in mapping.values() if value is not None]


def flatten_unique_values(mapping: Mapping[str, Sequence[Any] | None]) -> list[Any]:
    """Flatten Hydra-mergeable named groups into one de-duplicated list.

    Plain YAML lists are replaced when Hydra composes configs. A mapping is merged
    by key instead, so a base config can define a common group and a child config
    can add another group without repeating the common values. The mapping keys
    are only group names; they are discarded when this function produces the
    argument passed to the instantiated object.

    For example, suppose the base config contains::

        columns_to_load:
          _target_: rfproteina.utils.hydra.flatten_unique_values
          mapping:
            common: [example_id, path, method]

    A child config can add columns with::

        columns_to_load:
          mapping:
            interface: [pn_unit_1_iid, pn_unit_2_iid, method]

    Hydra merges the two ``mapping`` dictionaries. Instantiation then returns::

        [example_id, path, method, pn_unit_1_iid, pn_unit_2_iid]

    Repeated values such as ``method`` appear only once. To remove an entire
    inherited group, override that key with ``null``::

        columns_to_load:
          mapping:
            common: null

    Individual values cannot be subtracted from an inherited group; replace that
    group's list if different membership is required.

    Examples:
        >>> flatten_unique_values({"common": ["id", "path"], "extra": ["path", "label"]})
        ['id', 'path', 'label']
        >>> flatten_unique_values({"common": None, "extra": ["label"]})
        ['label']
    """
    flattened = []
    for values in mapping.values():
        if values is None:
            continue
        for value in values:
            if value not in flattened:
                flattened.append(value)
    return flattened


#  (Custom resolvers)
@run_once
def register_resolvers() -> None:
    resolvers = {
        "resolve_import": resolve_import,
        "chain_type_info_to_regex": chain_type_info_to_regex,
        "eval": eval,  # evaluate a string as a Python expression
    }

    for name, resolver in resolvers.items():
        OmegaConf.register_new_resolver(name, resolver)


def resolve_import(module_path: str, attribute_path: str | None = None) -> Any:
    """
    Import a module and access a specific attribute from it.

    Args:
        module_path (str): The path to the module.
        attribute_path (str): The path to the attribute within the module.

    Returns:
        The imported attribute.
    """
    module = importlib.import_module(module_path)
    if attribute_path is not None:
        # Split the attribute path to navigate through nested attributes
        attributes = attribute_path.split(".")
        attr = module
        for attr_name in attributes:
            attr = getattr(attr, attr_name)
        return attr
    else:
        return module


def chain_type_info_to_regex(*args) -> Any:
    """Convert a combination of ChainType or ChainTypeInfo attributes to a regex string.

    Primarily used for filtering a dataset by chain type prior to training/validation.

    Example filter:
    - "pn_unit_1_type.astype('str').str.match('${chain_type_info_to_regex:PROTEINS}')"

    """
    regex_str = ""

    for arg in args:
        if hasattr(ChainType, arg):
            regex_str += f"{getattr(ChainType, arg).value}|"
        elif hasattr(ChainTypeInfo, arg):
            chain_types_list = getattr(ChainTypeInfo, arg)
            for ct in chain_types_list:
                regex_str += f"{ct.value}|"
        else:
            raise ValueError(f"Attribute not found for ChainType or ChainTypeInfo: {arg}.")

    # Remove the trailing '|'
    regex_str = regex_str[:-1]

    return regex_str


def _can_be_instantiated(cfg: DictConfig) -> bool:
    """Checks if a config can be instantiated."""
    return isinstance(cfg, DictConfig) and "_target_" in cfg


class InstantiationError(ValueError):
    """Raised when a config cannot be instantiated."""

    pass


def instantiate_callbacks(callbacks_cfg: DictConfig | None) -> list[BaseCallback]:
    """Instantiates callbacks from config.

    Args:
        callbacks_cfg: A DictConfig object containing callback configurations.

    Returns:
        A list of instantiated callbacks.

    Reference:
        - Lightning Hydra Template (https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/instantiators.py#L36)
    """
    callbacks: list[BaseCallback] = []

    if not callbacks_cfg:
        return callbacks

    for _, cb_conf in callbacks_cfg.items():
        if _can_be_instantiated(cb_conf):
            callbacks.append(hydra.utils.instantiate(cb_conf))
        else:
            raise InstantiationError(
                f"Skipping callback <{cb_conf}> - Not a DictConfig with `_target_` key! "
                f"Please provide a valid `_target_` for instantiation."
            )

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig | None) -> list[Logger]:
    """Instantiates loggers from config.

    Args:
        logger_cfg: A DictConfig object containing logger configurations.

    Return:
        A list of instantiated loggers.

    Reference:
        - Lightning Hydra Template (https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/instantiators.py#L36)
    """
    loggers: list[Logger] = []

    if not logger_cfg:
        return loggers

    for _, lg_conf in logger_cfg.items():
        if _can_be_instantiated(lg_conf):
            loggers.append(hydra.utils.instantiate(lg_conf))
        else:
            raise InstantiationError(
                f"Skipping logger <{lg_conf}> - Not a DictConfig with `_target_` key! "
                f"Please provide a valid `_target_` for instantiation."
            )

    return loggers
