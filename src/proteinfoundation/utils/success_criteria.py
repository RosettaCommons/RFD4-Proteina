"""Utility functions for evaluating success criteria on generated structures."""


def evaluate_success_criteria(metrics: dict, criteria_config: dict) -> bool:
    """Check if structure passes configured success criteria.

    Success criteria are defined as a list of filters, each specifying:
    - metric: Metric key to evaluate (e.g., "interface_quality.interface_pae")
    - operator: Comparison operator ("less_than", "greater_than", etc.)
    - threshold: Threshold value to compare against

    Filters can be aggregated with "all" (AND) or "any" (OR) logic.

    Args:
        metrics: Dictionary of computed metric values
        criteria_config: Success criteria configuration with "filters" and "aggregation"

    Returns:
        True if structure passes success criteria, False otherwise

    Example:
        >>> metrics = {
        ...     "interface_quality.interface_pae": 8.5,
        ...     "af2_confidence.plddt": 75.0,
        ...     "hbonds.num_hbonds": 5,
        ... }
        >>> criteria = {
        ...     "filters": [
        ...         {
        ...             "metric": "interface_quality.interface_pae",
        ...             "operator": "less_than",
        ...             "threshold": 10.0,
        ...         },
        ...         {
        ...             "metric": "af2_confidence.plddt",
        ...             "operator": "greater_than",
        ...             "threshold": 70.0,
        ...         },
        ...     ],
        ...     "aggregation": "all",
        ... }
        >>> evaluate_success_criteria(metrics, criteria)
        True
    """
    if criteria_config is None or "filters" not in criteria_config:
        # No criteria defined - all structures pass
        return True

    filters = criteria_config["filters"]
    aggregation = criteria_config.get("aggregation", "all")

    # Operator functions
    operators = {
        "less_than": lambda v, t: v < t,
        "greater_than": lambda v, t: v > t,
        "less_equal": lambda v, t: v <= t,
        "greater_equal": lambda v, t: v >= t,
        "equal": lambda v, t: v == t,
        "not_equal": lambda v, t: v != t,
    }

    # Evaluate each filter
    results = []
    for filter_spec in filters:
        metric_key = filter_spec["metric"]
        operator = filter_spec["operator"]
        threshold = filter_spec["threshold"]

        # Get metric value
        value = metrics.get(metric_key)

        if value is None:
            # Missing metric - filter fails
            results.append(False)
            continue

        # Apply operator
        if operator not in operators:
            raise ValueError(f"Unknown operator: {operator}")

        passed = operators[operator](value, threshold)
        results.append(passed)

    # Aggregate results
    if aggregation == "all":
        return all(results)
    elif aggregation == "any":
        return any(results)
    else:
        raise ValueError(f"Unknown aggregation mode: {aggregation}. Use 'all' or 'any'.")


def get_success_rate(all_metrics: list[dict], criteria_config: dict) -> dict:
    """Compute success rate and statistics across multiple structures.

    Args:
        all_metrics: List of metric dictionaries, one per structure
        criteria_config: Success criteria configuration

    Returns:
        Dictionary with success statistics:
            - success_rate: Fraction of structures passing criteria
            - num_success: Number of successful structures
            - num_total: Total number of structures
            - successful_indices: Indices of successful structures

    Example:
        >>> metrics_list = [
        ...     {"plddt": 80.0, "interface_pae": 8.0},
        ...     {"plddt": 65.0, "interface_pae": 12.0},
        ...     {"plddt": 85.0, "interface_pae": 7.0},
        ... ]
        >>> criteria = {
        ...     "filters": [{"metric": "plddt", "operator": "greater_than", "threshold": 70.0}],
        ...     "aggregation": "all",
        ... }
        >>> stats = get_success_rate(metrics_list, criteria)
        >>> stats["success_rate"]
        0.6666666666666666
    """
    if not all_metrics:
        return {"success_rate": 0.0, "num_success": 0, "num_total": 0, "successful_indices": []}

    successful_indices = []
    for i, metrics in enumerate(all_metrics):
        if evaluate_success_criteria(metrics, criteria_config):
            successful_indices.append(i)

    num_success = len(successful_indices)
    num_total = len(all_metrics)
    success_rate = num_success / num_total if num_total > 0 else 0.0

    return {
        "success_rate": success_rate,
        "num_success": num_success,
        "num_total": num_total,
        "successful_indices": successful_indices,
    }
