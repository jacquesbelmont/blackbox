#!/usr/bin/env python3

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from feith_migrator import Config, FeithRecord


def iter_json(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            yield item
    elif isinstance(data, dict):
        yield data
    else:
        raise ValueError("JSON must be an object or array")


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_csv(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def normalize_row(row: Dict) -> Dict:
    # Accept a few common aliases to reduce friction when exporting
    aliases = {
        "id": "doc_id",
        "document_id": "doc_id",
        "feith_id": "doc_id",
        "type": "doc_type",
        "created": "create_date",
        "created_at": "create_date",
        "metadata": "metadata_xml",
        "ocr": "ocr_text",
        "path": "file_path",
        "status": "migration_status",
    }

    normalized = {}
    for k, v in row.items():
        key = aliases.get(k, k)
        normalized[key] = v

    if "migration_status" not in normalized or not normalized["migration_status"]:
        normalized["migration_status"] = "PENDING"

    required = ["doc_id"]
    for r in required:
        if not normalized.get(r):
            raise ValueError(f"Missing required field: {r}")

    # Truncate very long fields if needed (keeps DB inserts stable for demo)
    if normalized.get("doc_id"):
        normalized["doc_id"] = str(normalized["doc_id"])[:50]
    if normalized.get("doc_type"):
        normalized["doc_type"] = str(normalized["doc_type"])[:100]
    if normalized.get("create_date"):
        normalized["create_date"] = str(normalized["create_date"])[:50]
    if normalized.get("file_path"):
        normalized["file_path"] = str(normalized["file_path"])[:500]

    return normalized


def load_rows(path: str) -> List[Dict]:
    ext = os.path.splitext(path.lower())[1]

    if ext == ".json":
        rows = list(iter_json(path))
    elif ext in {".jsonl", ".ndjson"}:
        rows = list(iter_jsonl(path))
    elif ext == ".csv":
        rows = list(iter_csv(path))
    else:
        raise ValueError("Unsupported file extension. Use .json, .jsonl/.ndjson, or .csv")

    return [normalize_row(r) for r in rows]


def main():
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Load Feith-like samples into feith_documents table")
    parser.add_argument("--input", required=True, help="Path to CSV/JSON/JSONL with Feith-like records")
    parser.add_argument(
        "--db-url",
        default=Config.FEITH_DB_URL or "sqlite:///feith_mock.db",
        help="SQLAlchemy database URL for the simulated Feith DB (defaults to FEITH_DB_URL or sqlite:///feith_mock.db)",
    )
    parser.add_argument("--schema", default=Config.FEITH_DB_SCHEMA or "", help="Optional schema (leave blank for default)")
    parser.add_argument("--mode", choices=["merge", "skip"], default="merge", help="merge=upsert by doc_id; skip=ignore existing")
    args = parser.parse_args()

    if args.schema:
        # for loading we override the model schema dynamically
        FeithRecord.__table__.schema = args.schema

    engine = create_engine(args.db_url, poolclass=NullPool)
    FeithRecord.__table__.create(engine, checkfirst=True)

    Session = sessionmaker(bind=engine)
    session = Session()

    rows = load_rows(args.input)

    inserted = 0
    updated_or_skipped = 0

    try:
        for row in rows:
            doc_id = row.get("doc_id")
            existing = session.query(FeithRecord).filter(FeithRecord.doc_id == doc_id).first()

            if existing is not None and args.mode == "skip":
                updated_or_skipped += 1
                continue

            record = FeithRecord(
                doc_id=row.get("doc_id"),
                doc_type=row.get("doc_type"),
                create_date=row.get("create_date"),
                metadata_xml=row.get("metadata_xml"),
                ocr_text=row.get("ocr_text"),
                file_path=row.get("file_path"),
                migration_status=row.get("migration_status") or "PENDING",
            )

            # merge performs an upsert-like behavior keyed by PK
            session.merge(record)
            inserted += 1

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    total = len(rows)
    print(f"Loaded {inserted} record(s) into feith_documents ({updated_or_skipped} skipped/updated out of {total})")
    print(f"DB: {args.db_url}")


if __name__ == "__main__":
    main()
