import asyncio

from feith_migrator import MigrationPipeline, Config


async def demo():
    Config.BATCH_SIZE = 10

    pipeline = MigrationPipeline()

    batch = pipeline.extract_batch(10)
    print(f"Extracted {len(batch)} test records")

    await pipeline.process_batch(batch)

    print("\n=== Test Results ===")
    print(f"Processed: {pipeline.stats.processed}")
    print(f"Auto-approved: {pipeline.stats.auto_approved}")
    print(f"Manual review: {pipeline.stats.manual_review}")
    print(f"Rejected: {pipeline.stats.rejected}")

    await pipeline.cleanup()


asyncio.run(demo())
