import os
import json
import random
import uuid
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker

# --- CONFIGURATION ---
DB_PATH = "sqlite:///medical_claims_v6.db"
Base = declarative_base()

# --- DATABASE MODEL (Must match your main app) ---
class CallLog(Base):
    __tablename__ = "call_logs"
    id = Column(Integer, primary_key=True)
    call_id = Column(String, unique=True)
    filename = Column(String)
    processed_at = Column(DateTime, default=datetime.now)
    duration_seconds = Column(Float)
    extracted_data = Column(JSON)
    summary = Column(Text)
    transcription = Column(JSON)
    raw_filepath = Column(String)
    claim_status = Column(String)
    confidence_score = Column(Float)
    processing_time = Column(Float)

# --- INIT DB CONNECTION ---
engine = create_engine(DB_PATH)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# --- SAMPLE DATA ---
PATIENTS = ["John Smith", "Sarah Connor", "Michael Scott", "Dwight Schrute", "Pam Beesly", "Bruce Wayne", "Peter Parker"]
DENIAL_REASONS = ["Policy Expired", "Procedure Not Covered", "Out of Network", "Duplicate Claim", "Lack of Authorization"]
PROCEDURES = ["MRI Scan", "Blood Test", "X-Ray", "Physical Therapy", "Emergency Room Visit"]

def generate_fake_log():
    # 1. Decide Status
    outcome = random.choice(["approved", "denied", "under review"])
    
    # 2. Generate Timestamps (Spread over last 7 days)
    days_ago = random.randint(0, 7)
    processed_time = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59))
    
    # 3. Generate Data based on Status
    patient = random.choice(PATIENTS)
    claim_id = f"CLM-{random.randint(10000, 99999)}"
    amount = random.randint(100, 5000)
    
    data = {
        "Patient_Name": patient,
        "Date_of_Birth": "1985-05-20",
        "Policy_ID": f"POL-{random.randint(100000, 999999)}",
        "Claim_Number": claim_id,
        "Date_of_Service": "2025-11-01",
        "Claim_Status": outcome.title(), # Capitalize first letter
        "Amount_USD": f"${amount}.00",
        "Amount_INR": f"₹{amount * 83:,.2f}",
        "Representative_Name": "AI Agent",
        "Reference_Number": str(uuid.uuid4())[:8],
        "Follow_Up_Required": "No"
    }

    confidence = 0.95
    
    # Specific tweaks for Denied/Under Review
    if outcome == "denied":
        data["Denial_Reason"] = random.choice(DENIAL_REASONS)
        data["Follow_Up_Required"] = "Yes"
    elif outcome == "under review":
        data["Claim_Status"] = "Pending Info"
        confidence = 0.45
        data["Denial_Reason"] = "Missing documentation"

    # 4. Create Entry
    log = CallLog(
        call_id=str(uuid.uuid4()),
        filename=f"recording_{patient.replace(' ', '_').lower()}_{random.randint(1,100)}.mp3",
        processed_at=processed_time,
        duration_seconds=random.uniform(30.0, 300.0),
        extracted_data=data,
        summary=f"Automated verification for {patient}.",
        transcription=[{"speaker": "System", "text": "This is a generated log for testing purposes."}],
        raw_filepath="synthetic_data",
        claim_status=outcome,
        confidence_score=confidence,
        processing_time=random.uniform(1.5, 5.0)
    )
    return log

# --- EXECUTION ---
print("🌱 Seeding database with 50 synthetic records...")
try:
    logs = [generate_fake_log() for _ in range(50)]
    session.add_all(logs)
    session.commit()
    print("✅ Success! Added 50 records.")
    print("📊 Breakdown:")
    print(f"   - Approved: {len([l for l in logs if l.claim_status == 'approved'])}")
    print(f"   - Denied: {len([l for l in logs if l.claim_status == 'denied'])}")
    print(f"   - Under Review: {len([l for l in logs if l.claim_status == 'under review'])}")
    print("\n🚀 Run your main app now to see the Dashboard update!")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    session.close()