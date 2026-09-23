def should_stop(saved_cost: float, forgone_gain: float, threshold: float = 0.0) -> bool:
    return saved_cost - max(0.0, forgone_gain) > threshold
