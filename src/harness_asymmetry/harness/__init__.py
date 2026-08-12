"""Харнесс: шесть независимо отключаемых компонентов и их сборка.

Главное требование дизайн-дока §7.1: каждый компонент должен включаться и
выключаться независимо, без изменения остального кода. Если компонент нельзя
отключить, не сломав другой, план эксперимента неисполним.
"""

from harness_asymmetry.harness.assembler import Harness, assemble_harness
from harness_asymmetry.harness.components.base import (
    GateResult,
    NegotiationState,
    SideIdentity,
)

__all__ = [
    "Harness",
    "assemble_harness",
    "GateResult",
    "NegotiationState",
    "SideIdentity",
]
