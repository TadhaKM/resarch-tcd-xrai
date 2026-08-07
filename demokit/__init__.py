"""The demo framework: everything a demo needs, and nothing a demo author edits.

Adding a demonstration to this robot is: create one file in `demos/`, subclass
`Demo`, implement a hook or two. It then appears in the dashboard and can be
switched to live. No registration call, no list to append to, no existing file
to touch. If you find yourself editing anything in `demokit/` to add a demo,
the framework has a gap -- say so rather than working around it.

Start with `demos/_template.py`, and see docs/ADDING_A_DEMO.md.
"""

from .base import Demo, DemoContext, IdleResult
from .registry import REGISTRY


__all__ = ["Demo", "DemoContext", "IdleResult", "REGISTRY"]
