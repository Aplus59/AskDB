import sqlite3
from pathlib import Path

import pytest

SCHEMA = """
CREATE TABLE artist (
    artist_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT
);

CREATE TABLE album (
    album_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    artist_id INTEGER,
    released INTEGER,
    FOREIGN KEY (artist_id) REFERENCES artist(artist_id)
);

-- Declared without naming the target column, which SQLite reports differently.
CREATE TABLE track (
    track_id INTEGER PRIMARY KEY,
    album_id INTEGER REFERENCES album,
    title TEXT
);
"""

ROWS = """
INSERT INTO artist VALUES (1, 'Miles Davis', 'US'), (2, 'Alice Coltrane', 'US'),
                          (3, 'Ali Farka Toure', 'ML');
INSERT INTO album VALUES (1, 'Kind of Blue', 1, 1959), (2, 'Journey in Satchidananda', 2, 1971);
INSERT INTO track VALUES (1, 1, 'So What'), (2, 1, 'Blue in Green'), (3, 2, 'Shiva-Loka');
"""


@pytest.fixture(scope="session")
def sample_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("db") / "music.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executescript(ROWS)
    conn.commit()
    conn.close()
    return path
