import asyncio
import json
import sys

from feith_migrator import MigrationPipeline, FeithRecord


async def validate_one(feith_id: str):
    pipeline = MigrationPipeline()

    record = pipeline.feith_session.query(FeithRecord).filter(
        FeithRecord.doc_id == feith_id
    ).first()

    if not record:
        print(f"Record {feith_id} not found")
        await pipeline.cleanup()
        return

    print(f"Transforming {feith_id}...")
    transformed = await pipeline.transformer.transform_single(record)

    print("\n" + "=" * 60)
    print("TRANSFORMATION RESULT")
    print("=" * 60)
    print(json.dumps(transformed, indent=2))
    print("=" * 60)

    status = pipeline.validate_record(transformed)
    print(f"\nStatus: {status}")

    if input("Save to staging? (y/n): ").lower() == "y":
        pipeline.save_to_staging(transformed, status)
        print("Saved")

    await pipeline.cleanup()


feith_id_arg = sys.argv[1] if len(sys.argv) > 1 else "DOC-00001"
asyncio.run(validate_one(feith_id_arg))
