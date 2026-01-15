from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AttributeSpec:
    """
    For packed labels:
      batch["attrs"] has shape (B, A) and this attribute uses column = index
    For separate keys:
      batch[target_key] has shape (B,)
    """
    name: str
    num_classes: int
    target_key: Optional[str] = 'label'
    index: Optional[int] = None

