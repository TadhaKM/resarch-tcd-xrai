"""Finding the demos: every module in `demos/` that doesn't start with `_`.

There is deliberately no list of demos anywhere. A list is a second place to
edit, and the whole promise of this framework is that adding a demonstration
means adding one file. Discovery is therefore purely "what is in the folder",
and a demo that doesn't appear is either named with a leading underscore or
failed to import -- both of which are logged loudly at startup.

Like base.py this is a stdlib-only leaf; see that module's docstring for why.
"""

import importlib
import logging
import pkgutil
import threading
from typing import Optional

from .base import Demo


logger = logging.getLogger(__name__)

#: The package scanned for demos. A source checkout, which is how install.ps1
#: and start_reachy.ps1 run this. If the app is ever frozen or shipped as a
#: wheel, pkgutil.iter_modules stops working and this needs importlib.resources.
DEMOS_PACKAGE = "demos"

#: Used when the selected demo is missing or disabled. Deliberately not the id
#: of any real demo, so "what is running" never lies on the dashboard.
FALLBACK_ID = "_fallback"

#: Consecutive failures before a demo is set aside. Consecutive, not total: a
#: demo that fails once an hour on an open day should not quietly disappear
#: after three hours of otherwise faultless operation.
MAX_CONSECUTIVE_FAILURES = 3


class Registry:
    """The demos found on disk, and which of them are still trusted."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._demos: dict[str, Demo] = {}
        self._order: list[str] = []
        self._failures: dict[str, int] = {}
        self._disabled: dict[str, str] = {}
        #: Ids that arrived after startup rather than from a file on disk --
        #: features Hub staff wrote from the dashboard. Tracked so register()
        #: and unregister() can refuse to touch anything that came from code.
        self._runtime: set[str] = set()
        self._discovered = False

    # --- discovery -------------------------------------------------------

    def discover(self, package_name: str = DEMOS_PACKAGE) -> None:
        """Import every demo module and register what it defines.

        Built into locals and published under the lock at the end, rather than
        holding the lock across imports: importing a module takes CPython's own
        per-module import lock, and holding two locks in an order no other
        thread agrees on is how deadlocks happen. The dashboard thread can call
        into here through snapshot().
        """
        found: dict[str, Demo] = {}
        try:
            package = importlib.import_module(package_name)
        except Exception:
            logger.exception("No demos package (%s); the robot will answer but do nothing else", package_name)
            self._publish(found)
            return

        for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
            if info.name.startswith("_"):
                continue
            module_name = f"{package_name}.{info.name}"
            try:
                module = importlib.import_module(module_name)
            except Exception:
                # One broken demo must not cost every other one. This is the
                # single most likely thing to go wrong when a student adds a
                # file, so it is reported by name.
                logger.exception("Demo %s failed to import and was skipped", module_name)
                continue
            for demo in self._demos_in(module, info.name):
                if demo.id in found:
                    logger.error(
                        "Two demos claim the id %r (%s and %s); keeping the first",
                        demo.id,
                        type(found[demo.id]).__module__,
                        type(demo).__module__,
                    )
                    continue
                found[demo.id] = demo

        self._publish(found)
        logger.info("Loaded %d demo(s): %s", len(found), ", ".join(sorted(found)) or "none")

    def _demos_in(self, module: object, module_stem: str) -> list[Demo]:
        """Instantiate every concrete Demo subclass defined in this module."""
        out: list[Demo] = []
        for attr in vars(module).values():
            if not isinstance(attr, type) or not issubclass(attr, Demo) or attr is Demo:
                continue
            # Only classes defined here, so importing a shared base class into
            # a demo file does not register it a second time.
            if attr.__module__ != getattr(module, "__name__", None):
                continue
            if getattr(attr, "__abstractmethods__", None):
                continue
            try:
                demo = attr()
            except Exception:
                logger.exception("Demo class %s could not be constructed; skipped", attr.__name__)
                continue
            # Read from __dict__, never the inherited attribute: a subclassed
            # demo would otherwise inherit its parent's id, collide, and be
            # dropped -- silently costing exactly the "just add a file" promise
            # this registry exists to keep.
            demo.id = attr.__dict__.get("id") or module_stem
            demo.label = attr.__dict__.get("label") or demo.id.replace("_", " ").title()
            out.append(demo)
        return out

    def _publish(self, found: dict[str, Demo]) -> None:
        ordered = sorted(found.values(), key=lambda d: (d.order, d.label))
        with self._lock:
            self._demos = {d.id: d for d in ordered}
            self._order = [d.id for d in ordered]
            self._discovered = True

    def register(self, demo: Demo) -> bool:
        """Add or replace one already-built demo while the robot is running.

        This is what lets a feature written from the dashboard become a button
        without a restart. Everything downstream already copes: the loop
        republishes dashboard_entries() every cycle, the browser re-renders on
        a content change, and the runner rebuilds its trigger table on every
        utterance. The registry was the only thing that latched.

        False, and nothing changed, if the id belongs to a demo found on disk.
        A stored feature must never shadow a committed one -- the code is the
        thing that was reviewed.

        Not a way to hot-reload code: discover() imports each module once and a
        second call returns the same module object. This takes an instance
        someone has already constructed.
        """
        with self._lock:
            if demo.id in self._demos and demo.id not in self._runtime:
                logger.error(
                    "Refusing to register %r over a demo of the same name", demo.id
                )
                return False
            merged = dict(self._demos)
            merged[demo.id] = demo
            # Rebuilt into locals and swapped whole, the way _publish does it:
            # readers tolerate the dict being replaced between two calls, but
            # not being mutated while one of them is iterating it.
            ordered = sorted(merged.values(), key=lambda d: (d.order, d.label))
            self._demos = {d.id: d for d in ordered}
            self._order = [d.id for d in ordered]
            self._runtime.add(demo.id)
            # An edited feature starts clean. The version being replaced may
            # have been set aside for failing three times, and the edit is
            # usually the fix -- leaving it disabled means the operator changes
            # the wording, nothing happens, and there is no obvious cure.
            self._failures.pop(demo.id, None)
            self._disabled.pop(demo.id, None)
        return True

    def is_runtime(self, demo_id: str) -> bool:
        """Whether this id was registered at runtime rather than found on disk.

        Needed to tell "that name belongs to a demo in the code" from "that is
        this very feature, already registered" -- which re-validating a stored
        feature otherwise reports as a clash with itself.
        """
        with self._lock:
            return demo_id in self._runtime

    def unregister(self, demo_id: str) -> bool:
        """Remove a demo that register() added. False for anything else."""
        with self._lock:
            if demo_id not in self._runtime:
                return False
            remaining = {i: d for i, d in self._demos.items() if i != demo_id}
            ordered = sorted(remaining.values(), key=lambda d: (d.order, d.label))
            self._demos = {d.id: d for d in ordered}
            self._order = [d.id for d in ordered]
            self._runtime.discard(demo_id)
            self._failures.pop(demo_id, None)
            self._disabled.pop(demo_id, None)
        return True

    def ensure_discovered(self) -> None:
        with self._lock:
            done = self._discovered
        if not done:
            self.discover()

    # --- lookup ----------------------------------------------------------

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._order)

    def get(self, demo_id: str) -> Optional[Demo]:
        with self._lock:
            return self._demos.get(demo_id)

    def default_id(self) -> str:
        """The demo to start in: the first that is usable, by order."""
        with self._lock:
            for demo_id in self._order:
                if demo_id not in self._disabled:
                    return demo_id
            return self._order[0] if self._order else FALLBACK_ID

    def is_available(self, demo_id: str, capabilities: frozenset[str]) -> tuple[bool, str]:
        """Whether a demo can run now, and why not when it cannot."""
        demo = self.get(demo_id)
        if demo is None:
            return False, "not installed"
        with self._lock:
            reason = self._disabled.get(demo_id)
        if reason:
            return False, reason
        missing = [need for need in demo.requires if need not in capabilities]
        if missing:
            return False, f"needs {', '.join(missing)}"
        return True, ""

    # --- health ----------------------------------------------------------

    def record_success(self, demo_id: str) -> None:
        """Clear the failure count after a clean hook."""
        with self._lock:
            self._failures.pop(demo_id, None)

    def record_failure(self, demo_id: str) -> bool:
        """Count a failure. Returns True if the demo has now been set aside."""
        with self._lock:
            count = self._failures.get(demo_id, 0) + 1
            self._failures[demo_id] = count
            if count < MAX_CONSECUTIVE_FAILURES:
                return False
            self._disabled[demo_id] = f"failed {count} times in a row"
        logger.error(
            "Demo %r failed %d times in a row and has been set aside; "
            "fix it and re-enable it from the dashboard.",
            demo_id,
            count,
        )
        return True

    def enable(self, demo_id: str) -> bool:
        """Put a set-aside demo back in service."""
        with self._lock:
            existed = self._disabled.pop(demo_id, None) is not None
            self._failures.pop(demo_id, None)
        if existed:
            logger.info("Demo %r re-enabled.", demo_id)
        return existed

    # --- dashboard -------------------------------------------------------

    def dashboard_entries(self, capabilities: frozenset[str]) -> list[dict]:
        """The demo list the dashboard renders, in display order."""
        entries = []
        for demo_id in self.ids():
            demo = self.get(demo_id)
            if demo is None:
                continue
            available, reason = self.is_available(demo_id, capabilities)
            entries.append(
                {
                    "id": demo_id,
                    "label": demo.label,
                    "help": demo.help,
                    "available": available,
                    "note": reason,
                }
            )
        return entries


REGISTRY = Registry()
