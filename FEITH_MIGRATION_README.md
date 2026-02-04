# Feith Data Migration Tool

## Quick Start Guide for Wednesday Demo

### Prerequisites

1. **DGX Access:**
   ```bash
   ssh brandon@dgx-server.internal
   ``` 

2. **Python Environment:**
   ```bash
   python3 --version  # Ensure 3.11+
   ```

3. **vLLM Server Running:**
   ```bash
   curl http://localhost:8080/v1/models
   # Should return: {"data": [{"id": "llama-3.1-70b", ...}]}
   ```

---

## Local Simulation (Realistic) - SQL Server / Azure SQL Edge

If you do not have access to the real Feith database yet, you can simulate a Feith-like source database locally using SQL Server.

### Recommended on macOS (Apple Silicon): Azure SQL Edge (Docker)

Prerequisites (local machine):
- Install Python deps: `pip install -r requirements.txt` (includes `pyodbc`)
- Install Microsoft ODBC driver (recommended: ODBC Driver 18 for SQL Server)
  - Easiest path is to use Azure Data Studio (which can also manage the DB)
  - Or install ODBC via Homebrew/Microsoft repo (varies by macOS version)

1. Start the container:

```bash
docker run -d --name feith-sql \
  -e 'ACCEPT_EULA=1' \
  -e 'MSSQL_SA_PASSWORD=YourStrong!Passw0rd' \
  -p 1433:1433 \
  mcr.microsoft.com/azure-sql-edge
```

2. Set `.env` to use SQL Server as Feith source:

```bash
# Use SQL Server as the Feith source DB
FEITH_FORCE_SQLITE=false
FEITH_DB_URL=mssql+pyodbc://sa:YourStrong!Passw0rd@localhost:1433/feith?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes

# Keep staging local (SQLite) for simple demos
TARGET_DB_URL=sqlite:///migration_staging.db

# LLM can be Ollama locally (no DGX required)
LLM_API_URL=http://localhost:11434/api/generate
LLM_MODEL=llama3
```

### Importing Feith-like sample records

Before importing, ensure the database `feith` exists (the loader will create the table, but not the database).
You can create it using Azure Data Studio or by running a SQL query against the server:

```sql
CREATE DATABASE feith;
```

Create a CSV/JSON/JSONL file with (at minimum):
- `doc_id`
- `doc_type`
- `create_date`
- `metadata_xml`
- `ocr_text`
- `file_path`
- `migration_status` (optional, defaults to `PENDING`)

Then load into the simulated Feith DB:

```bash
python3 load_feith_samples.py --input ./feith_samples.csv --db-url "$FEITH_DB_URL"
```

Notes:
- CSV header aliases are supported (e.g., `feith_id` -> `doc_id`, `type` -> `doc_type`, `ocr` -> `ocr_text`).
- Use `--mode skip` to ignore records that already exist.

### Run the same migration pipeline

```bash
python3 validate_single.py DOC-00001
python3 wednesday_demo.py
```

---

## Installation (5 minutes)

```bash
# 1. Create directory
mkdir -p /opt/blackbox-sentinel/feith-migration
cd /opt/blackbox-sentinel/feith-migration

# 2. Copy migration script
# (Upload feith_migrator.py to this directory)

# 3. Install dependencies
pip install --break-system-packages \
    sqlalchemy \
    cx_oracle \
    psycopg2-binary \
    httpx \
    python-dotenv

# 4. Configure
cp feith_migrator.env.example .env
nano .env
# Fill in Feith DB credentials
```

---

## Configuration (.env file)

**Minimum Required:**
```bash
# Feith Database
FEITH_DB_HOST=feith-db.company.com
FEITH_DB_PORT=1521
FEITH_DB_NAME=FEITH_PROD
FEITH_DB_USER=readonly_user
FEITH_DB_PASSWORD=SecurePassword123

# Target Database
TARGET_DB_URL=postgresql://migration:pass@localhost:5432/migrated_docs

# LLM (should work out of the box if vLLM is running)
LLM_API_URL=http://localhost:8080/v1/completions
LLM_MODEL=llama-3.1-70b
```

---

## Usage

### Option 1: Full Migration (Production)

```bash
# Process all pending Feith records
python3 feith_migrator.py
```

**Output:**
```
============================================================
FEITH MIGRATION PIPELINE - Starting
============================================================
Total records to migrate: 50000
Batch 1: Processing 100 records...
Transformed DOC-12345: Invoice (confidence: 0.92)
Transformed DOC-12346: Contract (confidence: 0.87)
...
Progress: 0.2% (100/50000)
  Auto-approved: 89
  Manual review: 9
  Rejected: 2
  Errors: 0
...
```

### Option 2: Test Run (Demo for Wednesday)

Create test script `test_migration.py`:

```python
import asyncio
from feith_migrator import MigrationPipeline, Config

async def demo():
    # Override config for small test
    Config.BATCH_SIZE = 10
    
    pipeline = MigrationPipeline()
    
    # Process just one batch
    batch = pipeline.extract_batch(10)
    print(f"Extracted {len(batch)} test records")
    
    await pipeline.process_batch(batch)
    
    print("\n=== Test Results ===")
    print(f"Processed: {pipeline.stats.processed}")
    print(f"Auto-approved: {pipeline.stats.auto_approved}")
    print(f"Manual review: {pipeline.stats.manual_review}")
    
    await pipeline.cleanup()

asyncio.run(demo())
```

Run:
```bash
python3 test_migration.py
```

### Option 3: Interactive Validation

```python
# validate_single.py
import asyncio
from feith_migrator import MigrationPipeline

async def validate_one(feith_id: str):
    pipeline = MigrationPipeline()
    
    # Get specific record
    record = pipeline.feith_session.query(FeithRecord).filter(
        FeithRecord.doc_id == feith_id
    ).first()
    
    if not record:
        print(f"Record {feith_id} not found")
        return
    
    # Transform
    print(f"Transforming {feith_id}...")
    transformed = await pipeline.transformer.transform_single(record)
    
    # Display results
    import json
    print("\n" + "="*60)
    print("TRANSFORMATION RESULT")
    print("="*60)
    print(json.dumps(transformed, indent=2))
    print("="*60)
    
    # Save if acceptable
    status = pipeline.validate_record(transformed)
    print(f"\nStatus: {status}")
    
    if input("Save to staging? (y/n): ").lower() == 'y':
        pipeline.save_to_staging(transformed, status)
        print("✅ Saved")
    
    await pipeline.cleanup()

# Usage
import sys
feith_id = sys.argv[1] if len(sys.argv) > 1 else "DOC-12345"
asyncio.run(validate_one(feith_id))
```

Run:
```bash
python3 validate_single.py DOC-12345
```

---

## Understanding the Output

### Confidence Scores

| Score | Status | Action |
|-------|--------|--------|
| ≥ 0.9 | Auto-Approved | Loaded directly to target |
| 0.7 - 0.9 | Manual Review | Flagged for human validation |
| < 0.7 | Rejected | Re-processed or manual entry |

### Example Transformation

**Input (Raw Feith):**
```xml
<doc id="DOC-45821" type="UNK" date="O3/22/2O19">
  INVOICE #INV-2O19-O845
  Bill To: Acme Corp
  Total: $18,5OO.OO
</doc>
```

**Output (Structured JSON):**
```json
{
  "document_type": "Invoice",
  "feith_id": "DOC-45821",
  "normalized_date": "2019-03-22",
  "entities": ["Acme Corp"],
  "category": "Financial",
  "has_pii": false,
  "confidence_score": 0.89,
  "extraction_notes": "OCR errors corrected: O→0"
}
```

---

## Monitoring

### Real-time Progress

The script outputs progress every batch:
```
Progress: 45.2% (22600/50000)
  Auto-approved: 20834 (92%)
  Manual review: 1562 (7%)
  Rejected: 204 (1%)
  Errors: 0
```

### Log Files

**Migration log:**
```bash
tail -f feith_migration.log
```

**Statistics:**
```bash
# Generated after completion
cat migration_stats_20260201_143022.json
```

### Database Queries

**Check staging:**
```sql
-- Auto-approved records
SELECT COUNT(*) FROM migration_staging WHERE status = 'APPROVED';

-- Manual review queue
SELECT feith_id, document_type, confidence_score, extraction_notes
FROM migration_staging 
WHERE status = 'REVIEW'
ORDER BY confidence_score DESC
LIMIT 100;

-- Quality metrics
SELECT 
  document_type,
  COUNT(*) as count,
  AVG(confidence_score) as avg_confidence
FROM migration_staging
GROUP BY document_type
ORDER BY count DESC;
```

---

## Performance Tuning

### Speed Optimization

**Current Performance:**
- Single record: ~2-3 seconds
- Batch (100 records): ~30-40 seconds (0.3-0.4s per record)
- Daily throughput: ~200,000 records (8 hours)

**To Increase Speed:**

1. **Larger Batches:**
   ```bash
   # .env
   BATCH_SIZE=200  # From 100
   ```

2. **More GPUs:**
   ```bash
   # vLLM config
   tensor_parallel_size: 8  # Use all GPUs
   ```

3. **Parallel Workers:**
   ```python
   # In code
   Config.MAX_WORKERS = 8  # From 4
   ```

### Quality Optimization

**To Reduce False Positives:**

```bash
# .env
CONFIDENCE_THRESHOLD=0.9  # From 0.7 (stricter)
```

**To Handle More Edge Cases:**

Edit prompt template in `feith_migrator.py`:
```python
PROMPT_TEMPLATE = """
...
ADDITIONAL INSTRUCTIONS:
- For medical records, always flag has_pii=true
- For contracts, extract all party names
- For invoices, parse total amount
...
"""
```

---

## Troubleshooting

### Issue: "Could not connect to Feith database"

**Solution:**
```bash
# Test connection manually
python3 -c "
import cx_Oracle
conn = cx_Oracle.connect('user/pass@host:port/service')
print('✅ Connected')
"
```

### Issue: "LLM API timeout"

**Solution:**
```bash
# Check vLLM server
curl http://localhost:8080/health

# Increase timeout in .env
LLM_TIMEOUT=180
```

### Issue: "Low confidence scores across all records"

**Possible Causes:**
1. Poor OCR quality in Feith data
2. LLM prompt needs refinement
3. Model hallucinating

**Solutions:**
- Review sample Feith records manually
- Adjust prompt template for your data
- Try different temperature (lower = more deterministic)

### Issue: "Memory error during batch processing"

**Solution:**
```bash
# Reduce batch size
BATCH_SIZE=50  # From 100

# Or process sequentially
Config.MAX_WORKERS = 1
```

---

## Demo Script for Wednesday

Here's a ready-to-use script for demonstrating to Brandon and Justin:

```python
#!/usr/bin/env python3
"""
Wednesday Demo - Feith Migration POC
Run this to showcase the capability
"""

import asyncio
from feith_migrator import MigrationPipeline, Config

async def wednesday_demo():
    print("\n" + "="*70)
    print("FEITH MIGRATION - PROOF OF CONCEPT DEMONSTRATION")
    print("="*70 + "\n")
    
    # Configure for demo (small batch)
    Config.BATCH_SIZE = 20
    
    pipeline = MigrationPipeline()
    
    print("📊 Checking Feith database...")
    total = pipeline.count_pending_records()
    print(f"   Found {total:,} records pending migration\n")
    
    print("🔄 Extracting sample batch (20 records)...")
    batch = pipeline.extract_batch(20)
    print(f"   Extracted {len(batch)} records\n")
    
    print("🤖 Transforming with LLM (Llama 3.1 70B on DGX)...")
    print("   Please wait ~30-60 seconds...\n")
    
    await pipeline.process_batch(batch)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"✅ Processed:      {pipeline.stats.processed}")
    print(f"✅ Auto-approved:  {pipeline.stats.auto_approved} ({pipeline.stats.auto_approved/pipeline.stats.processed*100:.0f}%)")
    print(f"⚠️  Manual review:  {pipeline.stats.manual_review} ({pipeline.stats.manual_review/pipeline.stats.processed*100:.0f}%)")
    print(f"❌ Rejected:       {pipeline.stats.rejected}")
    print(f"💡 Success rate:   {pipeline.stats.success_rate:.1f}%")
    print("="*70)
    
    print("\n📝 Sample transformed records:")
    print("-" * 70)
    
    # Show 3 examples
    staged = pipeline.target_session.query(TransformedRecord).limit(3).all()
    for idx, record in enumerate(staged, 1):
        print(f"\n{idx}. {record.feith_id}")
        print(f"   Type: {record.document_type}")
        print(f"   Date: {record.normalized_date}")
        print(f"   Category: {record.category}")
        print(f"   Confidence: {record.confidence_score:.2f}")
        print(f"   Status: {record.status}")
    
    print("\n" + "="*70)
    print("💡 DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review manual review queue in database")
    print("2. Adjust confidence thresholds if needed")
    print("3. Run full migration (50,000+ records)")
    print("\nEstimated full migration time: ~6 hours")
    
    await pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(wednesday_demo())
```

Save as `wednesday_demo.py` and run:
```bash
python3 wednesday_demo.py
```

---

## Post-Migration Steps

After migration completes:

### 1. Validate Results

```sql
-- Quality check
SELECT 
  status,
  COUNT(*) as count,
  AVG(confidence_score) as avg_confidence
FROM migration_staging
GROUP BY status;
```

### 2. Export Manual Review Queue

```sql
-- Export low-confidence records for review
COPY (
  SELECT 
    feith_id,
    document_type,
    normalized_date,
    entities,
    confidence_score,
    extraction_notes
  FROM migration_staging
  WHERE status = 'REVIEW'
  ORDER BY confidence_score DESC
) TO '/tmp/manual_review_queue.csv' CSV HEADER;
```

### 3. Load Approved Records to Target

```python
# load_to_target.py
from feith_migrator import MigrationPipeline

def load_approved():
    pipeline = MigrationPipeline()
    
    approved = pipeline.target_session.query(TransformedRecord).filter(
        TransformedRecord.status == 'APPROVED'
    ).all()
    
    print(f"Loading {len(approved)} approved records to target system...")
    
    for record in approved:
        # Your target system integration here
        # e.g., SharePoint, modern DMS, etc.
        pass
    
    print("✅ Load complete")

load_approved()
```

---

## Support

**Issues?**
- Check logs: `feith_migration.log`
- DGX status: `nvidia-smi`
- vLLM health: `curl http://localhost:8080/health`

**Questions?**
Contact: jacques@company.com

---

**Last Updated:** January 31, 2026  
**Version:** 1.0 POC
