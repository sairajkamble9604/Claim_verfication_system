"""
claim_verification_v26.0_windows_safe.py
- *** WINDOWS FILE LOCK FIX ***
- Fix 1: Generates PDF in the local folder (not %TEMP%) to avoid Windows permission crashes.
- Fix 2: Displays ACTUAL error message in UI if PDF fails (for easier debugging).
- Fix 3: Logo loading is isolated (PDF will generate even if logo is broken).
- Core: Stable Whisper + Native AI + Pro UI.
"""
# -*- coding: utf-8 -*-
import os
import json
import re
import time
import pandas as pd
from datetime import datetime
import uuid
import logging
import urllib.request
import urllib.error
import ast
import shutil
import streamlit as st
from streamlit_tags import st_tags
import torch
import whisper 
from pydub import AudioSegment

# --- IMPORTS ---
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import func, desc

import plotly.express as px

# --- PRO PDF ENGINE ---
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# --------------------------
# 1. Configuration & Styling
# --------------------------
st.set_page_config(
    page_title="Claim Verification AI",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# --- DARK THEME CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .block-container { padding-top: 1.5rem; }
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #1e1e1e, #252525);
        border: 1px solid #333;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    h1, h2, h3, p, div, span, label { color: #FAFAFA !important; }
    section[data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #333; }
    .data-row { padding: 8px 0; border-bottom: 1px solid #333; }
    .data-label { font-weight: bold; color: #bbb !important; width: 40%; display: inline-block; }
    .data-val { color: #fff !important; font-weight: 500; }
    .trans-box {
        background-color: #1c1c1c;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #444;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

LOG_FILE = "system.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

class Config:
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.m4a', '.flac', '.aac']
    WHISPER_MODELS = ['tiny', 'base', 'small', 'medium', 'large']
    DEFAULT_MODEL = 'base'
    USD_TO_INR = 83.0 
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "llama3" 
    DB_PATH = "sqlite:///medical_claims_v6.db"

config = Config()

# --------------------------
# 2. Database
# --------------------------
Base = declarative_base()

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

@st.cache_resource
def get_db_session():
    engine = create_engine(config.DB_PATH, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

Session = get_db_session()

# --------------------------
# 3. Core Logic
# --------------------------
REQUIRED_SCHEMA = {
  "Patient_Name": "", "Date_of_Birth": "", "Policy_ID": "", "Claim_Number": "",
  "Date_of_Service": "", "Denial_Reason": "", "Claim_Status": "", "Amount_USD": "",
  "Amount_INR": "", "Representative_Name": "", "Reference_Number": "", "Follow_Up_Required": "Yes/No"
}

def check_ollama_connection():
    try:
        req = urllib.request.Request(config.OLLAMA_URL.replace("/api/generate", ""), method="HEAD")
        with urllib.request.urlopen(req, timeout=1) as r: return r.status == 200
    except: return False

def check_ffmpeg():
    return shutil.which("ffmpeg") is not None

def send_ollama_request(prompt):
    try:
        data = json.dumps({"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode('utf-8')
        req = urllib.request.Request(config.OLLAMA_URL, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8')).get('response', '')
    except: return ""

def robust_json_parse(raw_text):
    try: return json.loads(raw_text)
    except: pass
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: 
            try: return ast.literal_eval(match.group(0))
            except: pass
    return {}

def extract_claim_info_native(transcript_text):
    prompt = f"Extract to JSON. Schema: {json.dumps(REQUIRED_SCHEMA)}. Text: '{transcript_text}'"
    raw = send_ollama_request(prompt)
    parsed = robust_json_parse(raw)
    validated = {}
    for k in REQUIRED_SCHEMA:
        val = parsed.get(k, "")
        val_str = str(val[0] if isinstance(val, list) else val).strip()
        if re.match(r'^[\W_]+$', val_str) or val_str.lower() in ['none', 'null', 'n/a']: 
            val_str = ""
        validated[k] = val_str
    return validated

def refine_transcript_with_ollama(raw_text):
    prompt = f"Rewrite with 'Agent:' and 'Client:' labels. Concise.\nRaw: '{raw_text}'"
    refined = send_ollama_request(prompt)
    clean_lines = []
    for line in refined.split('\n'):
        if ":" in line and len(line.split(":")[0]) < 20: 
            clean = line.replace('*', '').strip()
            clean_lines.append(clean)
    return "\n".join(clean_lines) if clean_lines else raw_text

def normalize_status(raw_status):
    s = str(raw_status).lower().strip()
    if "approv" in s or "accept" in s or "paid" in s: return "approved"
    elif "den" in s or "reject" in s: return "denied"
    else: return "under review"

# --------------------------
# 4. Processing Engine
# --------------------------
@st.cache_resource
def load_whisper_model(size): return whisper.load_model(size)

class TranscriptionEngine:
    def transcribe(self, path, size="base"):
        start = time.time()
        try:
            if os.path.getsize(path) < 100: return {"error": "Empty file", "segments": []}
            model = load_whisper_model(size)
            result = model.transcribe(path)
            dur = result['segments'][-1]['end'] if result['segments'] else 0
            return {"segments": result.get("segments", []), "duration": dur, "processing_time": time.time()-start}
        except Exception as e: return {"error": str(e), "segments": [], "duration": 0, "processing_time": 0}

class AudioProcessor:
    @staticmethod
    def convert_to_wav(path):
        # Windows Safe: Create local file instead of system temp
        try:
            if path.endswith('.wav'): return path
            local_wav = f"temp_{uuid.uuid4().hex}.wav"
            AudioSegment.from_file(path).export(local_wav, format='wav')
            return local_wav
        except: return path

def process_single_file(uploaded_file, model_size, use_deep_ai):
    try:
        # Windows Safe: Save uploaded file locally
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp3"
        raw_path = f"temp_raw_{uuid.uuid4().hex}{suffix}"
        
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.read())
        
        wav_path = AudioProcessor.convert_to_wav(raw_path)
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0: 
            return {"error": "Corrupt audio file", "filename": uploaded_file.name}

        trans_res = TranscriptionEngine().transcribe(wav_path, model_size)
        if trans_res.get("error"): return {"error": trans_res['error'], "filename": uploaded_file.name}
        
        full_text = " ".join([s['text'].strip() for s in trans_res['segments']])
        
        refined_segments = []
        if use_deep_ai:
            refined_text = refine_transcript_with_ollama(full_text)
            for line in refined_text.split('\n'):
                if ":" in line:
                    p = line.split(":", 1)
                    refined_segments.append({"speaker": p[0].strip(), "text": p[1].strip()})
        if not refined_segments: refined_segments = [{"speaker": "Unknown", "text": full_text}]
        
        extracted = extract_claim_info_native(full_text) 
        
        if extracted.get("Amount_USD"):
            try: 
                clean = re.sub(r'[^\d\.]','', str(extracted['Amount_USD']))
                if clean: extracted["Amount_INR"] = f"₹{float(clean) * config.USD_TO_INR:,.2f}"
            except: pass
        
        clean_status = normalize_status(str(extracted.get("Claim_Status", "Under Review")))
        extracted["Claim_Status"] = clean_status.title()
        has_claim = bool(extracted.get("Claim_Number")) or bool(extracted.get("Policy_ID"))
        
        validation = {
            "is_valid": has_claim, "confidence": 1.0 if has_claim else 0.5, 
            "status": clean_status, "has_approval": "approved" == clean_status, "has_denial": "denied" == clean_status
        }
        
        save_to_db(uploaded_file.name, trans_res, extracted, validation, refined_segments, raw_path)
        
        # Safe Cleanup
        try:
            if os.path.exists(raw_path): os.remove(raw_path)
            if os.path.exists(wav_path) and wav_path != raw_path: os.remove(wav_path)
        except: pass

        return {
            "filename": uploaded_file.name, "data": extracted, "transcript": refined_segments,
            "validation": validation, "duration": trans_res['duration'], "processing_time": trans_res['processing_time']
        }
    except Exception as e: return {"error": str(e), "filename": uploaded_file.name}

def save_to_db(filename, trans, data, val, transcript, path):
    try:
        session = Session()
        name, ext = os.path.splitext(filename)
        existing = session.query(CallLog).filter(CallLog.filename.like(f"{name}%")).count()
        db_filename = f"{name} ({existing + 1}){ext}" if existing > 0 else filename
        session.add(CallLog(
            call_id=str(uuid.uuid4()), filename=db_filename, processed_at=datetime.now(),
            duration_seconds=trans['duration'], extracted_data=data, transcription=transcript,
            summary=f"Claim {data.get('Claim_Number', 'N/A')}", raw_filepath=path, 
            claim_status=val['status'], confidence_score=val['confidence'], processing_time=trans['processing_time']
        ))
        session.commit()
        session.close()
    except: pass

# --------------------------
# 5. PRO PDF GENERATOR (WINDOWS SAFE)
# --------------------------
def clean_text_for_pdf(text):
    """Removes emojis and special characters that crash ReportLab."""
    if not text: return ""
    text = text.replace('🗣️', '') 
    # Standardize text to Latin-1 compatible to avoid unicode crashes
    return text.encode('ascii', 'ignore').decode('ascii')

def generate_pdf(data):
    try:
        # Windows Safe: Use local directory instead of system temp
        pdf_filename = f"report_{uuid.uuid4().hex}.pdf"
        path = os.path.abspath(pdf_filename)
        
        doc = SimpleDocTemplate(path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Logo Failsafe
        logo_path = r"C:\Users\kambl\Desktop\Simdaa_Logo.png"
        if not os.path.exists(logo_path): logo_path = "assets/Simdaa_Logo.png"
        if os.path.exists(logo_path):
            try:
                im = Image(logo_path, width=2*inch, height=0.6*inch)
                im.hAlign = 'LEFT'
                story.append(im)
                story.append(Spacer(1, 20))
            except: pass
        
        story.append(Paragraph(f"Claim Verification Report", styles['Title']))
        story.append(Paragraph(f"File: {clean_text_for_pdf(data['filename'])}", styles['Heading3']))
        
        # Status
        status_text = data['validation']['status'].upper()
        bg_color = colors.mistyrose if "DENIED" in status_text else colors.lightgreen if "APPROVED" in status_text else colors.lightyellow
        
        t = Table([["Current Status:", status_text]], colWidths=[1.5*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (1, 0), (1, 0), bg_color),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 20))
        
        # Data Table
        story.append(Paragraph("Extracted Details", styles['Heading2']))
        table_data = [["Field", "Value"]]
        if data.get("data"):
            for k, v in data['data'].items():
                clean_val = clean_text_for_pdf(str(v) if v else "Not Mentioned")
                val_para = Paragraph(clean_val, styles['Normal'])
                table_data.append([k.replace('_', ' '), val_para])
        
        t2 = Table(table_data, colWidths=[2.5*inch, 4.5*inch])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        for i in range(1, len(table_data)):
            if i % 2 == 0: t2.setStyle(TableStyle([('BACKGROUND', (0, i), (-1, i), colors.whitesmoke)]))
        story.append(t2)
        story.append(Spacer(1, 20))
        
        # Transcript
        story.append(Paragraph("Transcript", styles['Heading2']))
        if data.get('transcript'):
            trans_data = [["Speaker", "Message"]]
            for seg in data['transcript']:
                spk = clean_text_for_pdf(seg.get('speaker', 'Unknown').replace('*', ''))
                txt = clean_text_for_pdf(seg.get('text', ''))
                
                msg = Paragraph(txt, styles['Normal'])
                trans_data.append([spk, msg])
            
            t3 = Table(trans_data, colWidths=[1.5*inch, 5.5*inch])
            t3.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('PADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(t3)
        
        doc.build(story)
        return path
    except Exception as e:
        # IMPORTANT: This returns the error so you can see it in the UI
        return f"ERROR: {str(e)}"

class EnhancedExportEngine:
    @staticmethod
    def to_json(data): return json.dumps(data, indent=2, default=str)

def prepare_comprehensive_report():
    try:
        session = Session()
        all_calls = session.query(CallLog).all()
        report = {"generated_at": datetime.now().isoformat(), "calls": []}
        for call in all_calls:
            report["calls"].append({
                "filename": call.filename, "status": call.claim_status,
                "data": call.extracted_data
            })
        session.close()
        return EnhancedExportEngine.to_json(report)
    except: return "{}"

# --------------------------
# 6. UI Components
# --------------------------
def render_dashboard():
    st.title("📊 Executive Dashboard")
    session = Session()
    now = datetime.now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    try:
        total = session.query(CallLog).count()
        today = session.query(CallLog).filter(CallLog.processed_at >= start).count()
        denied = session.query(CallLog).filter(CallLog.claim_status.ilike('%denied%')).count()
    except: total, today, denied = 0, 0, 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Processed", total, delta="+5 this week")
    c2.metric("Processed Today", today, delta="Live")
    c3.metric("Denied Claims", denied, delta_color="inverse")
    st.markdown("---")
    
    col_charts, col_recent = st.columns([2, 1])
    with col_charts:
        st.subheader("Analytics")
        raw_data = session.query(CallLog.claim_status).all()
        if raw_data:
            status_list = [normalize_status(r[0]) for r in raw_data]
            df = pd.DataFrame(status_list, columns=['Status'])
            df['Status'] = df['Status'].str.title()
            chart_df = df['Status'].value_counts().reset_index()
            chart_df.columns = ['Status', 'Count']
            
            fig = px.pie(chart_df, values='Count', names='Status', hole=0.6, color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"), showlegend=True, margin=dict(t=0,b=0,l=0,r=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No data yet.")
        
    with col_recent:
        st.subheader("Recent")
        calls = session.query(CallLog).order_by(desc(CallLog.processed_at)).limit(5).all()
        if calls:
            for c in calls:
                s = c.claim_status.lower()
                icon = "🔴" if "denied" in s else "🟢" if "approved" in s else "⚪"
                st.caption(f"{c.processed_at.strftime('%H:%M')} • {c.claim_status.upper()}")
                st.write(f"{icon} {c.filename[:20]}...")
                st.divider()
    session.close()

def render_processor():
    st.title("Claim Verification System")
    if not check_ollama_connection(): st.warning("⚠️ **Ollama offline.** Run `ollama serve`.")
    if not check_ffmpeg(): st.error("🚫 **FFmpeg missing.** Install it to process audio.")
    
    with st.container():
        c1, c2 = st.columns([2, 1])
        with c1: files = st.file_uploader("Upload Calls", type=config.SUPPORTED_FORMATS, accept_multiple_files=True)
        with c2:
            model = st.selectbox("Accuracy Model", config.WHISPER_MODELS, index=1)
            deep_ai = st.checkbox("Deep Speaker ID", value=True)
            if files: start = st.button("🚀 Process Batch", type="primary", use_container_width=True)

    if files and 'start' in locals() and start:
        bar = st.progress(0)
        for i, f in enumerate(files):
            with st.status(f"Processing **{f.name}**...", expanded=True) as status:
                res = process_single_file(f, model, deep_ai)
                if res.get("error"):
                    status.update(label="❌ Failed", state="error")
                    st.error(res['error'])
                else:
                    status.update(label="✅ Complete", state="complete", expanded=False)
                    st.toast(f"Processed: {f.name}", icon="✅")
                    
                    with st.container():
                        st.markdown(f"### {res['filename']}")
                        status_c = "red" if res['validation']['has_denial'] else "green" if res['validation']['has_approval'] else "orange"
                        st.caption(f"STATUS: :{status_c}[{res['validation']['status'].upper()}]")
                        
                        st.markdown("#### 📝 Extracted Data")
                        if res.get("data"):
                            for k, v in res['data'].items():
                                st.markdown(f"""
                                <div class="data-row">
                                    <span class="data-label">{k.replace('_',' ')}:</span>
                                    <span class="data-val">{v if v else '-'}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("#### 💬 Transcript")
                        with st.container(height=300):
                            st.markdown('<div class="trans-box">', unsafe_allow_html=True)
                            for s in res['transcript']:
                                st.markdown(f"**{s.get('speaker','').replace('*','')}**: {s.get('text','')}<br>", unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown("---")
                        c1, c2 = st.columns(2)
                        c1.download_button("📥 JSON Data", json.dumps(res, indent=2), f"{res['filename']}.json", use_container_width=True)
                        
                        # --- PDF BUTTON FIX ---
                        pdf_result = generate_pdf(res)
                        
                        if pdf_result and not pdf_result.startswith("ERROR"): 
                            c2.download_button("📄 Official PDF Report", open(pdf_result, "rb"), f"{res['filename']}.pdf", use_container_width=True)
                        else:
                            # Show the actual error message to the user
                            c2.error(f"PDF Failed: {pdf_result}")
                        # ----------------------
                        
            bar.progress((i+1)/len(files))

def render_history():
    st.title("📜 Records")
    session = Session()
    search = st.text_input("Search Files")
    query = session.query(CallLog).order_by(desc(CallLog.processed_at))
    if search: query = query.filter(CallLog.filename.ilike(f"%{search}%"))
    calls = query.all()
    if calls:
        st.write(f"Found **{len(calls)}** records.")
        for c in calls:
            with st.expander(f"{c.filename} | {c.claim_status.upper()}"):
                st.markdown("#### Extracted Data")
                if c.extracted_data:
                    for k, v in c.extracted_data.items():
                        st.markdown(f"""<div class="data-row"><span class="data-label">{k.replace('_',' ')}:</span> <span class="data-val">{v if v else '-'}</span></div>""", unsafe_allow_html=True)
                st.markdown("#### Transcript")
                if c.transcription:
                    with st.container(height=200):
                        for s in c.transcription:
                            st.text(f"{s.get('speaker','').replace('*','')}: {s.get('text','')}")
    else: st.warning("No records.")
    session.close()

def render_database():
    st.title("💾 Database")
    session = Session()
    calls = session.query(CallLog).all()
    if calls:
        data = [{"Date": c.processed_at, "File": c.filename, "Status": c.claim_status, "Duration": f"{c.duration_seconds:.1f}s" if c.duration_seconds else "0"} for c in calls]
        st.dataframe(pd.DataFrame(data), use_container_width=True)
    else: st.info("Empty.")
    session.close()

def render_settings():
    st.title("⚙️ Config")
    with st.container(border=True):
        st.success("Online")
        st.write(f"DB: {config.DB_PATH}")
        if check_ollama_connection(): st.info("Ollama: Connected")
        else: st.error("Ollama: Disconnected")

if 'processed_files' not in st.session_state: st.session_state.processed_files = []

def main():
    with st.sidebar:
        logo_path = r"C:\Users\kambl\Desktop\Simdaa_Logo.png"
        if not os.path.exists(logo_path): logo_path = "assets/Simdaa_Logo.png"
        if os.path.exists(logo_path): st.image(logo_path, width=160)
        else: st.markdown("## 🏥 **Claim Verification**")
        
        st.markdown("---")
        page = st.radio("Menu", ["Dashboard", "Process Calls", "History", "Database", "Settings"])
        st.markdown("---")
        report_json = prepare_comprehensive_report()
        st.download_button("📊 Full Report", report_json, f"report.json", "application/json", use_container_width=True)

    if page == "Dashboard": render_dashboard()
    elif page == "Process Calls": render_processor()
    elif page == "History": render_history()
    elif page == "Database": render_database()
    elif page == "Settings": render_settings()

if __name__ == "__main__":
    main()