from __future__ import annotations

from typing import Protocol, runtime_checkable

from polygram.behavioural.report import ValidationReport


@runtime_checkable
class Confirmer(Protocol):
    """Structural protocol for objects that produce a ``ValidationReport``
    with a populated ``confirmed`` field.

    Any object with a matching ``run()`` signature satisfies this protocol
    — including :class:`~polygram.behavioural.BehaviouralValidator` with
    no modification required.
    """

    def run(self) -> ValidationReport: ...
