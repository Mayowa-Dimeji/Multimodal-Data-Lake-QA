import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DB_DIR = BASE / "data_lake" / "db"
DB_PATH = DB_DIR / "movies.db"
SEED_SQL = DB_DIR / "seed.sql"

def main():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()  # fresh build
    con = sqlite3.connect(DB_PATH)
    with open(SEED_SQL, "r") as f:
        con.executescript(f.read())
    con.commit()
    con.close()
    print(f"Seeded SQLite database at {DB_PATH}")

if __name__ == "__main__":
    main()
