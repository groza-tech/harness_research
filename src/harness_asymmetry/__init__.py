"""Асимметрия агентного харнесса как источник рыночной власти.

Экспериментальный стенд к дизайн-документу исследования по специальности
5.2.2. Модель у обеих сторон одна и та же; различается только харнесс —
программная обвязка. Измеряется, как эта разница конвертируется в
захваченный излишек.
"""

__version__ = "0.1.0"

from harness_asymmetry.schemas import (
    COMPONENT_KEYS,
    HarnessVector,
    InfoRegime,
    Role,
    Scenario,
)

__all__ = [
    "__version__",
    "COMPONENT_KEYS",
    "HarnessVector",
    "InfoRegime",
    "Role",
    "Scenario",
]
