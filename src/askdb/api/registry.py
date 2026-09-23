"""Finding and holding open the databases the service can answer against.

Loading a catalog means counting rows in every table, which on the larger
Mini-Dev databases takes seconds. Doing that per request would make the first
question of every conversation slow for no reason, so catalogs are built once
and kept.
"""

from pathlib import Path

from askdb.agent.tools import Toolbox
from askdb.data import minidev
from askdb.db import catalog


class UnknownDatabase(KeyError):
    """Asked for a database that is not available."""


class DatabaseRegistry:
    def __init__(self, databases_root: Path) -> None:
        self.root = databases_root
        self._toolboxes: dict[str, Toolbox] = {}

    def available(self) -> list[str]:
        """Database ids present on disk, in a stable order."""
        if not self.root.is_dir():
            return []
        found = []
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("__"):
                continue
            try:
                minidev.resolve_database(self.root, entry.name)
            except minidev.DatasetError:
                continue
            found.append(entry.name)
        return found

    def toolbox(self, db_id: str) -> Toolbox:
        if db_id in self._toolboxes:
            return self._toolboxes[db_id]

        try:
            path = minidev.resolve_database(self.root, db_id)
        except minidev.DatasetError as error:
            raise UnknownDatabase(db_id) from error

        box = Toolbox(path, catalog.load(path))
        self._toolboxes[db_id] = box
        return box
