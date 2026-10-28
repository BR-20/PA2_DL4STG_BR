"""Failure rules fixed before any experiment is read. Used by the prediction drafts and the report."""

TRANSFORM_FAIL_FACTOR = 1.5


def failure_flags(model_metrics, copy_metrics):
    """model_metrics and copy_metrics are eval dicts at seed averaged values."""
    e5 = model_metrics["E5_1x"]
    return {
        "mirror": model_metrics["EM"] > TRANSFORM_FAIL_FACTOR * e5,
        "rotation": model_metrics["ER"] > TRANSFORM_FAIL_FACTOR * e5,
        "speed2": model_metrics["E2_2x"] >= copy_metrics["E2_2x"],
        "speed3": model_metrics["E2_3x"] >= copy_metrics["E2_3x"],
    }
