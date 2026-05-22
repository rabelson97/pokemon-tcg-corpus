import json
import sqlite3
from pathlib import Path

def main():
    db_path = Path("/Users/rabelson/Developer/GitHub/pokemon-tcg-corpus/embeddings.db")
    cache_path = Path("/Users/rabelson/Developer/GitHub/pokemon-tcg-corpus/build/tcgdex-detail-cache.jsonl")

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    if not cache_path.exists():
        print(f"Cache not found at {cache_path}")
        return

    print("Connecting to database...")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # 1. Migrate schema if types column is missing
        columns = {str(row[1]) for row in cursor.execute("PRAGMA table_info(cards);").fetchall()}
        if "types" not in columns:
            print("Adding 'types' column to cards table...")
            cursor.execute("ALTER TABLE cards ADD COLUMN types TEXT;")
            conn.commit()
        else:
            print("'types' column already exists in schema.")

        # 2. Parse cache and build a mapping of upstream_id -> types_str
        print("Parsing TCGdex cache...")
        type_mapping = {}
        with cache_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                upstream_id = entry.get("upstream_id")
                payload = entry.get("payload", {})
                types_list = payload.get("types")
                
                if upstream_id and types_list:
                    types_str = ",".join(str(t) for t in types_list)
                    type_mapping[upstream_id] = types_str

        print(f"Found {len(type_mapping)} card type records in cache.")

        # 3. Update the cards table
        print("Updating database cards...")
        updates = []
        for upstream_id, types_str in type_mapping.items():
            updates.append((types_str, upstream_id))
            
        cursor.executemany(
            "UPDATE cards SET types = ? WHERE upstream_id = ?;",
            updates
        )
        conn.commit()
        
        # 4. Verify updates
        affected = cursor.execute("SELECT COUNT(*) FROM cards WHERE types IS NOT NULL;").fetchone()[0]
        total = cursor.execute("SELECT COUNT(*) FROM cards;").fetchone()[0]
        print(f"Successfully populated 'types' for {affected}/{total} cards!")

if __name__ == "__main__":
    main()
