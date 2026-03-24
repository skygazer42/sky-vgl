from dataclasses import dataclass
from typing import Any, Callable, Protocol


class GraphTransform(Protocol):
    def __call__(self, value: Any) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class TransformPipeline:
    transforms: tuple[Callable[[Any], Any], ...] = ()

    def __call__(self, value: Any) -> Any:
        result = value
        for transform in self.transforms:
            result = transform(result)
        return result

    def append(self, *transforms: Callable[[Any], Any]) -> "TransformPipeline":
        return TransformPipeline(self.transforms + tuple(transforms))
