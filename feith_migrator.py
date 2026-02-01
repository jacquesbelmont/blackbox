#!/usr/bin/env python3
"""
Feith Data Migration Script - Production Ready
Blackbox Sentinel - Module 3: Data Intelligence
 
Purpose: Migrate legacy Feith document management data to modern systems
         using LLM-powered transformation for unstructured data

Author: Jacques - AI Engineer
Date: January 31, 2026
"""

import os
import json # for parsing LLM responses
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import re

# Database & ORM
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import NoSuchModuleError

# HTTP & API
import httpx

# Configuration
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feith_migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Migration configuration"""
    
    # Feith Database (Source)
    FEITH_DB_HOST = os.getenv("FEITH_DB_HOST", "feith-db.internal")
    FEITH_DB_PORT = os.getenv("FEITH_DB_PORT", "1521")
    FEITH_DB_NAME = os.getenv("FEITH_DB_NAME", "FEITH_PROD")
    FEITH_DB_USER = os.getenv("FEITH_DB_USER", "readonly_user")
    FEITH_DB_PASSWORD = os.getenv("FEITH_DB_PASSWORD", "")
    FEITH_DB_SCHEMA = os.getenv("FEITH_DB_SCHEMA", "") or None
    FEITH_WRITEBACK_ENABLED = os.getenv("FEITH_WRITEBACK_ENABLED", "false").lower() in {"1", "true", "yes"}
    
    # Target Database (Destination)
    TARGET_DB_URL = os.getenv("TARGET_DB_URL", "postgresql://user:pass@localhost/target_db")
    
    # LLM API (vLLM on DGX)
    LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8080/v1/completions")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-70b")
    
    # Processing Settings
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

    # Runtime
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
    
    # Validation
    AUTO_APPROVE_THRESHOLD = 0.9
    MANUAL_REVIEW_THRESHOLD = 0.7
    REJECT_THRESHOLD = 0.5


# ============================================================================
# DATA MODELS
# ============================================================================

Base = declarative_base()

class FeithRecord(Base):
    """Source Feith record (read-only)"""
    __tablename__ = 'feith_documents'
    __table_args__ = {'schema': Config.FEITH_DB_SCHEMA} if Config.FEITH_DB_SCHEMA else {}
    
    doc_id = Column(String(50), primary_key=True)
    doc_type = Column(String(100))
    create_date = Column(String(50))
    metadata_xml = Column(Text)
    ocr_text = Column(Text)
    file_path = Column(String(500))
    migration_status = Column(String(20), default='PENDING')


class TransformedRecord(Base):
    """Transformed record in staging"""
    __tablename__ = 'migration_staging'
    
    id = Column(Integer, primary_key=True)
    feith_id = Column(String(50), unique=True, nullable=False)
    document_type = Column(String(100))
    normalized_date = Column(DateTime)
    entities = Column(Text)  # JSON array
    category = Column(String(100))
    has_pii = Column(Boolean, default=False)
    pii_types = Column(Text)  # JSON array
    confidence_score = Column(Float)
    extraction_notes = Column(Text)
    requires_review = Column(Boolean, default=False)
    processed_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default='STAGED')  # STAGED, APPROVED, REJECTED


@dataclass
class MigrationStats:
    """Migration statistics"""
    total_records: int = 0
    processed: int = 0
    auto_approved: int = 0
    manual_review: int = 0
    rejected: int = 0
    errors: int = 0
    start_time: datetime = None
    end_time: datetime = None
    
    def to_dict(self):
        return asdict(self)
    
    @property
    def success_rate(self):
        if self.processed == 0:
            return 0.0
        return (self.auto_approved / self.processed) * 100
    
    @property
    def duration_seconds(self):
        if not self.start_time or not self.end_time:
            return 0
        return (self.end_time - self.start_time).total_seconds()


# ============================================================================
# LLM CLIENT
# ============================================================================

class LLMClient:
    """Client for vLLM inference server"""
    
    def __init__(self, api_url: str, model: str, timeout_seconds: float = 120.0):
        self.api_url = api_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=timeout_seconds)
    
    async def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        """Generate completion from LLM"""
        try:
            response = await self.client.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": ["```", "---"]
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise
    
    async def batch_generate(self, prompts: List[str], max_tokens: int = 1024) -> List[str]:
        """Generate completions in batch (faster)"""
        tasks = [self.generate(prompt, max_tokens) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def close(self):
        await self.client.aclose()


# ============================================================================
# FEITH TRANSFORMER
# ============================================================================

class FeithTransformer:
    """Transform Feith records using LLM"""
    
    PROMPT_TEMPLATE = """You are a Senior Data Engineer specializing in legacy document migration.

INPUT (Raw Feith Record):
{feith_record}

TASK: Transform into structured format

INSTRUCTIONS:
1. Identify document type: Contract, Invoice, Medical Record, Technical Drawing, Email, Correspondence, Report, Form, Other
2. Extract metadata:
   - Document ID (preserve original Feith ID)
   - Date (normalize to YYYY-MM-DD format, handle OCR errors like "O1/15/2O23" → "2023-01-15")
   - Parties/Entities (names, companies, relevant entities)
   - Category (Legal, Financial, Engineering, HR, Medical, Administrative)
3. Detect PII (flag for HIPAA compliance: names, SSN, medical info)
4. Confidence score (0.0-1.0) based on data quality

OUTPUT (strict JSON format, no markdown):
{{
  "document_type": "Invoice",
  "feith_id": "DOC-12345",
  "normalized_date": "2023-01-15",
  "entities": ["Acme Corp", "John Smith"],
  "category": "Financial",
  "has_pii": false,
  "pii_types": [],
  "confidence_score": 0.92,
  "extraction_notes": "OCR date corrected from O1/15/2O23"
}}

EDGE CASES:
- Missing date → use null
- Illegible OCR → confidence < 0.5, note in extraction_notes
- Ambiguous document type → mark as "Other", confidence < 0.7
- Multiple entities → list all significant ones (limit 10)

JSON OUTPUT:
"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def prepare_record_text(self, record: FeithRecord) -> str:
        """Format Feith record for LLM prompt"""
        record_text = f"""
Feith ID: {record.doc_id}
Type: {record.doc_type or 'Unknown'}
Date: {record.create_date or 'Not specified'}

Metadata:
{record.metadata_xml or 'N/A'}

OCR Text (excerpt):
{(record.ocr_text or 'N/A')[:1000]}...
"""
        return record_text.strip()
    
    async def transform_single(self, record: FeithRecord) -> Dict[str, Any]:
        """Transform single Feith record"""
        try:
            # Prepare prompt
            record_text = self.prepare_record_text(record)
            prompt = self.PROMPT_TEMPLATE.format(feith_record=record_text)
            
            # LLM generation
            response = await self.llm.generate(prompt, max_tokens=512)
            
            # Parse JSON response
            # Clean response (remove markdown if present)
            response = response.replace("```json", "").replace("```", "").strip()
            
            try:
                transformed = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    transformed = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON from LLM response")
            
            # Validate required fields
            required_fields = ["document_type", "feith_id", "confidence_score"]
            for field in required_fields:
                if field not in transformed:
                    transformed[field] = None
            
            # Ensure confidence score is valid
            if transformed["confidence_score"] is None:
                transformed["confidence_score"] = 0.5
            
            logger.info(f"Transformed {record.doc_id}: {transformed['document_type']} (confidence: {transformed['confidence_score']})")
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming {record.doc_id}: {e}")
            # Return low-confidence fallback
            return {
                "document_type": "Other",
                "feith_id": record.doc_id,
                "normalized_date": None,
                "entities": [],
                "category": "Unknown",
                "has_pii": False,
                "pii_types": [],
                "confidence_score": 0.3,
                "extraction_notes": f"Error during transformation: {str(e)}"
            }
    
    async def transform_batch(self, records: List[FeithRecord]) -> List[Dict[str, Any]]:
        """Transform batch of records"""
        logger.info(f"Transforming batch of {len(records)} records...")
        
        # Prepare all prompts
        prompts = []
        for record in records:
            record_text = self.prepare_record_text(record)
            prompt = self.PROMPT_TEMPLATE.format(feith_record=record_text)
            prompts.append(prompt)
        
        # Batch LLM generation
        responses = await self.llm.batch_generate(prompts, max_tokens=512)
        
        # Parse responses
        transformed_records = []
        for idx, (record, response) in enumerate(zip(records, responses)):
            if isinstance(response, Exception):
                logger.error(f"Batch error for {record.doc_id}: {response}")
                transformed_records.append({
                    "document_type": "Other",
                    "feith_id": record.doc_id,
                    "confidence_score": 0.3,
                    "extraction_notes": f"Batch processing error: {str(response)}"
                })
            else:
                try:
                    # Clean and parse JSON
                    response = response.replace("```json", "").replace("```", "").strip()
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        transformed = json.loads(json_match.group())
                        transformed_records.append(transformed)
                    else:
                        raise ValueError("No JSON found")
                except Exception as e:
                    logger.error(f"Parse error for {record.doc_id}: {e}")
                    transformed_records.append({
                        "feith_id": record.doc_id,
                        "confidence_score": 0.3,
                        "extraction_notes": f"Parse error: {str(e)}"
                    })
        
        logger.info(f"Batch transformation complete: {len(transformed_records)} records")
        return transformed_records


# ============================================================================
# MIGRATION PIPELINE
# ============================================================================

class MigrationPipeline:
    """Main migration pipeline orchestrator"""
    
    def __init__(self):
        # Database connections
        self.feith_engine = self._create_feith_engine()
        self.target_engine = create_engine(Config.TARGET_DB_URL, poolclass=NullPool)
        
        # Create tables
        TransformedRecord.__table__.create(self.target_engine, checkfirst=True)

        # SQLite demo support (creates mock Feith table and seeds data)
        if self.feith_engine.dialect.name == "sqlite":
            FeithRecord.__table__.create(self.feith_engine, checkfirst=True)

        # Sessions
        FeithSession = sessionmaker(bind=self.feith_engine)
        TargetSession = sessionmaker(bind=self.target_engine)
        self.feith_session = FeithSession()
        self.target_session = TargetSession()

        # Seed demo data after session exists
        if self.feith_engine.dialect.name == "sqlite":
            self._seed_sqlite_demo_data_if_needed()
        
        # LLM client
        self.llm = LLMClient(Config.LLM_API_URL, Config.LLM_MODEL, timeout_seconds=Config.LLM_TIMEOUT)
        self.transformer = FeithTransformer(self.llm)
        
        # Statistics
        self.stats = MigrationStats()

        # Pagination state (avoid relying on writes to Feith)
        self._last_doc_id: Optional[str] = None

    def _seed_sqlite_demo_data_if_needed(self):
        existing = self.feith_session.query(FeithRecord).count()
        if existing > 0:
            return

        demo_records = [
            FeithRecord(
                doc_id="DOC-00001",
                doc_type="UNK",
                create_date="O3/22/2O19",
                metadata_xml="<doc id='DOC-00001' type='UNK' date='O3/22/2O19' />",
                ocr_text="INVOICE #INV-2O19-O845\nBill To: Acme Corporation\nTotal: $18,5OO.OO",
                file_path="/feith/storage/DOC-00001.pdf",
                migration_status="PENDING",
            ),
            FeithRecord(
                doc_id="DOC-00002",
                doc_type="Contract",
                create_date="2020-11-05",
                metadata_xml="<contract id='DOC-00002' />",
                ocr_text="SERVICE AGREEMENT\nParties: Example LLC and John Smith\nEffective Date: 11/05/2020",
                file_path="/feith/storage/DOC-00002.pdf",
                migration_status="PENDING",
            ),
            FeithRecord(
                doc_id="DOC-00003",
                doc_type="Medical",
                create_date="2021-08-17",
                metadata_xml="<medical_record id='DOC-00003' />",
                ocr_text="PATIENT: Jane Doe\nMRN: MRN-12345\nDiagnosis: Diabetes",
                file_path="/feith/storage/DOC-00003.pdf",
                migration_status="PENDING",
            ),
        ]

        self.feith_session.add_all(demo_records)
        self.feith_session.commit()
    
    def _create_feith_engine(self):
        """Create Feith database engine (Oracle)"""
        # For Oracle: oracle+cx_oracle://user:pass@host:port/service
        # For PostgreSQL: postgresql://user:pass@host:port/db
        
        # This is a placeholder - adjust based on actual Feith DB type
        feith_url_cx = f"oracle+cx_oracle://{Config.FEITH_DB_USER}:{Config.FEITH_DB_PASSWORD}@{Config.FEITH_DB_HOST}:{Config.FEITH_DB_PORT}/{Config.FEITH_DB_NAME}"
        feith_url_oracledb = f"oracle+oracledb://{Config.FEITH_DB_USER}:{Config.FEITH_DB_PASSWORD}@{Config.FEITH_DB_HOST}:{Config.FEITH_DB_PORT}/?service_name={Config.FEITH_DB_NAME}"

        try:
            return create_engine(feith_url_cx, poolclass=NullPool)
        except NoSuchModuleError as e:
            logger.warning(f"Oracle cx_Oracle driver not available: {e}")
        except Exception as e:
            logger.warning(f"Oracle cx_Oracle connection error: {e}")

        try:
            return create_engine(feith_url_oracledb, poolclass=NullPool)
        except NoSuchModuleError as e:
            logger.warning(f"Oracle oracledb driver not available: {e}")
        except Exception as e:
            logger.warning(f"Oracle oracledb connection error: {e}")

        logger.warning("Using SQLite mock for demonstration")
        return create_engine("sqlite:///feith_mock.db")
    
    def count_pending_records(self) -> int:
        """Count records pending migration"""
        return self.feith_session.query(FeithRecord).filter(
            FeithRecord.migration_status == 'PENDING'
        ).count()
    
    def extract_batch(self, batch_size: int) -> List[FeithRecord]:
        """Extract batch of Feith records"""
        query = self.feith_session.query(FeithRecord).filter(
            FeithRecord.migration_status == 'PENDING'
        )

        if self._last_doc_id is not None:
            query = query.filter(FeithRecord.doc_id > self._last_doc_id)

        records = query.order_by(FeithRecord.doc_id.asc()).limit(batch_size).all()
        if records:
            self._last_doc_id = records[-1].doc_id

        return records
    
    def validate_record(self, transformed: Dict[str, Any]) -> str:
        """Validate transformed record and determine status"""
        confidence = transformed.get("confidence_score", 0.0)
        
        if confidence >= Config.AUTO_APPROVE_THRESHOLD:
            return "APPROVED"
        elif confidence >= Config.MANUAL_REVIEW_THRESHOLD:
            return "REVIEW"
        else:
            return "REJECTED"
    
    def save_to_staging(self, transformed: Dict[str, Any], status: str):
        """Save transformed record to staging table"""
        try:
            # Parse entities and pii_types as JSON
            entities_json = json.dumps(transformed.get("entities", []))
            pii_types_json = json.dumps(transformed.get("pii_types", []))
            
            # Parse date
            date_str = transformed.get("normalized_date")
            normalized_date = None
            if date_str:
                try:
                    normalized_date = datetime.strptime(date_str, "%Y-%m-%d")
                except:
                    pass
            
            # Create record
            feith_id = transformed.get("feith_id")
            staged_record = self.target_session.query(TransformedRecord).filter(
                TransformedRecord.feith_id == feith_id
            ).first()

            if staged_record is None:
                staged_record = TransformedRecord(feith_id=feith_id)
                self.target_session.add(staged_record)

            staged_record.document_type = transformed.get("document_type")
            staged_record.normalized_date = normalized_date
            staged_record.entities = entities_json
            staged_record.category = transformed.get("category")
            staged_record.has_pii = transformed.get("has_pii", False)
            staged_record.pii_types = pii_types_json
            staged_record.confidence_score = transformed.get("confidence_score", 0.0)
            staged_record.extraction_notes = transformed.get("extraction_notes")
            staged_record.requires_review = (status == "REVIEW")
            staged_record.status = status
            staged_record.processed_at = datetime.utcnow()

            self.target_session.commit()
            
        except Exception as e:
            logger.error(f"Error saving to staging: {e}")
            self.target_session.rollback()
            raise
    
    def update_feith_status(self, feith_id: str, status: str):
        """Update Feith record migration status"""
        if not Config.FEITH_WRITEBACK_ENABLED:
            return
        try:
            record = self.feith_session.query(FeithRecord).filter(
                FeithRecord.doc_id == feith_id
            ).first()
            
            if record:
                record.migration_status = status
                self.feith_session.commit()
        except Exception as e:
            logger.error(f"Error updating Feith status: {e}")
            self.feith_session.rollback()
    
    async def process_batch(self, batch: List[FeithRecord]):
        """Process a batch of records"""
        if not batch:
            return
        
        logger.info(f"Processing batch of {len(batch)} records...")
        
        # Transform with LLM
        transformed_records = await self.transformer.transform_batch(batch)
        
        # Validate and save
        for original, transformed in zip(batch, transformed_records):
            try:
                # Validate
                status = self.validate_record(transformed)
                
                # Save to staging
                self.save_to_staging(transformed, status)
                
                # Update Feith status
                self.update_feith_status(original.doc_id, "PROCESSED")
                
                # Update stats
                self.stats.processed += 1
                if status == "APPROVED":
                    self.stats.auto_approved += 1
                elif status == "REVIEW":
                    self.stats.manual_review += 1
                else:
                    self.stats.rejected += 1
                
            except Exception as e:
                logger.error(f"Error processing {original.doc_id}: {e}")
                self.stats.errors += 1
    
    async def run(self):
        """Run full migration pipeline"""
        logger.info("="*60)
        logger.info("FEITH MIGRATION PIPELINE - Starting")
        logger.info("="*60)
        
        self.stats.start_time = datetime.utcnow()
        
        # Count total records
        self.stats.total_records = self.count_pending_records()
        logger.info(f"Total records to migrate: {self.stats.total_records}")
        
        if self.stats.total_records == 0:
            logger.warning("No pending records found")
            return
        
        # Process in batches
        batch_num = 1
        while True:
            # Extract batch
            batch = self.extract_batch(Config.BATCH_SIZE)
            
            if not batch:
                logger.info("No more records to process")
                break
            
            logger.info(f"Batch {batch_num}: Processing {len(batch)} records...")
            
            # Process
            await self.process_batch(batch)
            
            # Progress
            progress = (self.stats.processed / self.stats.total_records) * 100
            logger.info(f"Progress: {progress:.1f}% ({self.stats.processed}/{self.stats.total_records})")
            logger.info(f"  Auto-approved: {self.stats.auto_approved}")
            logger.info(f"  Manual review: {self.stats.manual_review}")
            logger.info(f"  Rejected: {self.stats.rejected}")
            logger.info(f"  Errors: {self.stats.errors}")
            
            batch_num += 1
        
        self.stats.end_time = datetime.utcnow()
        
        # Final report
        self.print_final_report()
    
    def print_final_report(self):
        """Print final migration report"""
        logger.info("")
        logger.info("="*60)
        logger.info("MIGRATION COMPLETE - Final Report")
        logger.info("="*60)
        logger.info(f"Total Records:    {self.stats.total_records}")
        logger.info(f"Processed:        {self.stats.processed}")
        denom = self.stats.processed if self.stats.processed else 1
        logger.info(f"Auto-Approved:    {self.stats.auto_approved} ({(self.stats.auto_approved/denom*100):.1f}%)")
        logger.info(f"Manual Review:    {self.stats.manual_review} ({(self.stats.manual_review/denom*100):.1f}%)")
        logger.info(f"Rejected:         {self.stats.rejected} ({(self.stats.rejected/denom*100):.1f}%)")
        logger.info(f"Errors:           {self.stats.errors}")
        logger.info(f"Success Rate:     {self.stats.success_rate:.1f}%")
        logger.info(f"Duration:         {self.stats.duration_seconds:.1f} seconds")
        rps = (self.stats.processed / self.stats.duration_seconds) if self.stats.duration_seconds else 0.0
        logger.info(f"Records/Second:   {rps:.2f}")
        logger.info("="*60)
        
        # Save stats to file
        stats_file = f"migration_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2, default=str)
        logger.info(f"Stats saved to: {stats_file}")
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.llm.close()
        self.feith_session.close()
        self.target_session.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution"""
    pipeline = MigrationPipeline()
    
    try:
        await pipeline.run()
    except KeyboardInterrupt:
        logger.warning("Migration interrupted by user")
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
