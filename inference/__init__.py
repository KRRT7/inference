from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from inference.core.models.base import Model
    from inference.models.utils import (
        get_model as get_model_hint,
        get_roboflow_model as get_roboflow_model_hint,
    )

    get_model: Callable[[str, Optional[str]], Model] = get_model_hint
    get_roboflow_model: Callable[[str, Optional[str]], Model] = get_roboflow_model_hint

    from inference.core.interfaces.stream.stream import Stream as StreamHint
    from inference.core.interfaces.stream.inference_pipeline import (
        InferencePipeline as InferencePipelineHint,
    )

    Stream: type[StreamHint]
    InferencePipeline: type[InferencePipelineHint]

_LAZY_ATTRIBUTES: dict[str, Callable[[], Any]] = {
    "Stream": lambda: _import_from("inference.core.interfaces.stream.stream", "Stream"),
    "InferencePipeline": lambda: _import_from(
        "inference.core.interfaces.stream.inference_pipeline", "InferencePipeline"
    ),
    "get_model": lambda: _import_model_util("get_model"),
    "get_roboflow_model": lambda: _import_model_util("get_roboflow_model"),
}


def _import_from(module_path: str, attribute_name: str) -> Any:
    """Import and return an attribute from the specified module."""
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, attribute_name)


def _import_model_util(name: str) -> Any:
    from inference.models.utils import get_model, get_roboflow_model

    return locals()[name]


def __getattr__(name: str) -> Any:
    """Implement lazy loading for module attributes."""
    if name in _LAZY_ATTRIBUTES:
        return _LAZY_ATTRIBUTES[name]()
    raise AttributeError(f"module 'inference' has no attribute '{name}'")
