"""Metric modules with legacy video helpers loaded only when requested."""

__all__ = ["compute_basic_video_metrics", "evaluate"]


def __getattr__(name: str):
    if name in __all__:
        from .basic_metrics import compute_basic_video_metrics, evaluate

        return {
            "compute_basic_video_metrics": compute_basic_video_metrics,
            "evaluate": evaluate,
        }[name]
    raise AttributeError(name)

__all__ = ["evaluate", "compute_basic_video_metrics"]
