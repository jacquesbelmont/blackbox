#!/usr/bin/env python3

import asyncio

from feith_migrator import MigrationPipeline, Config, TransformedRecord


async def wednesday_demo():
    print("\n" + "=" * 70)
    print("FEITH MIGRATION - PROOF OF CONCEPT DEMONSTRATION")
    print("=" * 70 + "\n")

    pipeline = MigrationPipeline()

    total = pipeline.count_pending_records()
    print(f"Pending records: {total:,}\n")

    demo_batch_size = min(20, max(1, total))
    Config.BATCH_SIZE = demo_batch_size

    batch = pipeline.extract_batch(demo_batch_size)
    print(f"Extracted batch: {len(batch)} record(s)\n")

    print("Transforming with LLM...")
    await pipeline.process_batch(batch)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Processed:      {pipeline.stats.processed}")
    print(f"Auto-approved:  {pipeline.stats.auto_approved}")
    print(f"Manual review:  {pipeline.stats.manual_review}")
    print(f"Rejected:       {pipeline.stats.rejected}")
    print(f"Errors:         {pipeline.stats.errors}")
    print("=" * 70)

    staged = pipeline.target_session.query(TransformedRecord).order_by(TransformedRecord.processed_at.desc()).limit(3).all()
    if staged:
        print("\nSample staged records:")
        for idx, record in enumerate(staged, 1):
            print(f"{idx}. {record.feith_id} | {record.document_type} | {record.status} | confidence={record.confidence_score}")

    await pipeline.cleanup()


if __name__ == "__main__":
    asyncio.run(wednesday_demo())
