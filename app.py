
# ============================================================
#   SMART ATTENDANCE SYSTEM — BSCS Final Year Project
#   Author   : AI-Generated (Portfolio / FYP Quality)
#   Stack    : Python · Streamlit · OpenCV · SQLite · Ngrok
#   Platform : Google Colab
# ============================================================

# ─────────────────────────────────────────────────────────────
# SECTION 0 ── GOOGLE COLAB SETUP  (run this cell first)
# ─────────────────────────────────────────────────────────────
COLAB_SETUP_CELL = """
# ── Paste the lines below into a Google Colab code cell ──

# 1. Install all required libraries
!pip install streamlit pyngrok opencv-python-headless face_recognition \
             Pillow pandas openpyxl matplotlib seaborn fpdf2 bcrypt \
             streamlit-option-menu plotly -q

# 2. Download & run the app
!wget -q https://raw.githubusercontent.com/your-repo/smart-attendance/main/smart_attendance_system.py

# 3. Launch with Ngrok (replace YOUR_AUTHTOKEN)
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")
public_url = ngrok.connect(8501)
print(f"\\n🌐 PUBLIC URL → {public_url}")

import subprocess, threading
def run_streamlit():
    subprocess.run(["streamlit", "run", "smart_attendance_system.py",
                    "--server.port=8501", "--server.headless=true",
                    "--server.enableCORS=false"])

t = threading.Thread(target=run_streamlit, daemon=True)
t.start()
"""

# ─────────────────────────────────────────────────────────────
# SECTION 1 ── LIBRARY IMPORTS
# ─────────────────────────────────────────────────────────────
import streamlit as st
import sqlite3
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import bcrypt
import base64
import io
import json
import re
import logging
import hashlib
import time
import datetime
import pickle
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF
import warnings
warnings.filterwarnings("ignore")

# Optional: face_recognition (graceful fallback if dlib missing)
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    st.warning("face_recognition library not available — using Haarcascade fallback.")

# ─────────────────────────────────────────────────────────────
# SECTION 2 ── GLOBAL CONSTANTS & LOGGER
# ─────────────────────────────────────────────────────────────
DB_PATH         = "attendance.db"
UPLOAD_DIR      = "student_images"
ENCODINGS_FILE  = "face_encodings.pkl"
REPORTS_DIR     = "reports"
LOGS_DIR        = "logs"
APP_TITLE       = "SmartAttend AI"
APP_ICON        = "🎓"
CONFIDENCE_THRESHOLD = 0.50        # face-match confidence (0–1)
VERSION         = "2.0.0"

for _d in [UPLOAD_DIR, REPORTS_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "app.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# SECTION 3 ── DATABASE LAYER
# ─────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Return a thread-safe SQLite connection with WAL mode enabled."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Create all required tables on first run."""
    conn = get_connection()
    cur  = conn.cursor()

    # ── Admins ──────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name     TEXT,
            role          TEXT DEFAULT 'admin',
            created_at    TEXT DEFAULT (datetime('now')),
            last_login    TEXT
        )
    """)

    # ── Students ─────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number   TEXT UNIQUE NOT NULL,
            full_name     TEXT NOT NULL,
            email         TEXT,
            department    TEXT,
            semester      TEXT,
            image_path    TEXT,
            face_encoded  INTEGER DEFAULT 0,
            created_at    TEXT DEFAULT (datetime('now')),
            status        TEXT DEFAULT 'active'
        )
    """)

    # ── Attendance ────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id    INTEGER NOT NULL,
            roll_number   TEXT NOT NULL,
            full_name     TEXT NOT NULL,
            date          TEXT NOT NULL,
            time          TEXT NOT NULL,
            status        TEXT DEFAULT 'present',
            method        TEXT DEFAULT 'face_recognition',
            confidence    REAL DEFAULT 1.0,
            marked_by     TEXT DEFAULT 'system',
            FOREIGN KEY (student_id) REFERENCES students(id),
            UNIQUE (roll_number, date)
        )
    """)

    # ── Activity Logs ─────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            admin      TEXT,
            action     TEXT,
            details    TEXT,
            timestamp  TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Face Encodings Metadata ───────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_data (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  INTEGER UNIQUE NOT NULL,
            roll_number TEXT UNIQUE NOT NULL,
            encoding    BLOB,
            updated_at  TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialised.")


# ─────────────────────────────────────────────────────────────
# SECTION 4 ── AUTHENTICATION MODULE
# ─────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def create_admin(username: str, email: str, password: str, full_name: str) -> dict:
    """Insert a new admin account. Returns {success, message}."""
    if len(password) < 6:
        return {"success": False, "message": "Password must be at least 6 characters."}
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return {"success": False, "message": "Invalid email address."}
    try:
        conn = get_connection()
        conn.execute(
            "INSERT INTO admins (username, email, password_hash, full_name) VALUES (?,?,?,?)",
            (username.strip(), email.strip(), hash_password(password), full_name.strip())
        )
        conn.commit()
        conn.close()
        logger.info(f"Admin created: {username}")
        return {"success": True, "message": "Account created successfully!"}
    except sqlite3.IntegrityError as e:
        return {"success": False, "message": f"Username or email already exists. ({e})"}


def login_admin(username: str, password: str) -> dict:
    """Validate credentials. Returns {success, admin_data/message}."""
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM admins WHERE username=?", (username.strip(),)
    ).fetchone()
    conn.close()
    if not row:
        return {"success": False, "message": "User not found."}
    if not verify_password(password, row["password_hash"]):
        return {"success": False, "message": "Incorrect password."}
    # update last_login
    conn = get_connection()
    conn.execute("UPDATE admins SET last_login=datetime('now') WHERE id=?", (row["id"],))
    conn.commit()
    conn.close()
    logger.info(f"Admin logged in: {username}")
    return {"success": True, "admin": dict(row)}


def log_activity(admin: str, action: str, details: str = ""):
    """Record admin activity."""
    try:
        conn = get_connection()
        conn.execute(
            "INSERT INTO activity_logs (admin, action, details) VALUES (?,?,?)",
            (admin, action, details)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"log_activity error: {e}")


# ─────────────────────────────────────────────────────────────
# SECTION 5 ── STUDENT MANAGEMENT MODULE
# ─────────────────────────────────────────────────────────────

def add_student(roll: str, name: str, email: str, dept: str, sem: str,
                image_bytes: bytes = None, image_name: str = "") -> dict:
    """Insert student record, save image, trigger encoding."""
    if not roll.strip() or not name.strip():
        return {"success": False, "message": "Roll number and name are required.", "face_encoded": False}
    img_path = ""
    if image_bytes:
        ext = Path(image_name).suffix.lower() if image_name else ".jpg"
        if ext not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"
        img_path = os.path.join(UPLOAD_DIR, f"{roll.strip()}{ext}")
        with open(img_path, "wb") as f:
            f.write(image_bytes)
    try:
        conn = get_connection()
        conn.execute(
            """INSERT INTO students
               (roll_number, full_name, email, department, semester, image_path)
               VALUES (?,?,?,?,?,?)""",
            (roll.strip(), name.strip(), email.strip(), dept.strip(), sem.strip(), img_path)
        )
        conn.commit()
        conn.close()

        # encode face if image provided — and report back whether it actually worked
        face_encoded = False
        face_message = ""
        if img_path:
            face_encoded = encode_student_face(roll.strip(), img_path)
            if not face_encoded:
                if not FACE_RECOGNITION_AVAILABLE:
                    face_message = ("⚠️ Photo saved, but the face-recognition engine is not "
                                     "installed correctly, so no face encoding was created. "
                                     "This student CANNOT be matched by the camera yet. "
                                     "Re-run the install cell, then use 'Re-encode Face' below.")
                else:
                    face_message = ("⚠️ Photo saved, but no clear face was detected in it, so "
                                     "no face encoding was created. This student CANNOT be "
                                     "matched by the camera yet. Please re-upload a clear, "
                                     "front-facing, well-lit photo with only one face in it.")

        logger.info(f"Student added: {roll} (face_encoded={face_encoded})")
        base_msg = f"Student '{name}' added successfully."
        if img_path and not face_encoded:
            return {"success": True, "message": base_msg + " " + face_message, "face_encoded": False}
        return {"success": True, "message": base_msg, "face_encoded": face_encoded}
    except sqlite3.IntegrityError:
        return {"success": False, "message": f"Roll number '{roll}' already exists.", "face_encoded": False}
    except Exception as e:
        return {"success": False, "message": str(e), "face_encoded": False}


def get_all_students() -> pd.DataFrame:
    conn = get_connection()
    df   = pd.read_sql_query("SELECT * FROM students ORDER BY id DESC", conn)
    conn.close()
    return df


def get_student_by_roll(roll: str) -> dict | None:
    conn = get_connection()
    row  = conn.execute("SELECT * FROM students WHERE roll_number=?", (roll,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_student(roll: str, name: str, email: str, dept: str, sem: str,
                   status: str) -> dict:
    try:
        conn = get_connection()
        conn.execute(
            """UPDATE students SET full_name=?, email=?, department=?,
               semester=?, status=? WHERE roll_number=?""",
            (name, email, dept, sem, status, roll)
        )
        conn.commit()
        conn.close()
        return {"success": True, "message": "Student updated."}
    except Exception as e:
        return {"success": False, "message": str(e)}


def delete_student(roll: str) -> dict:
    try:
        conn = get_connection()
        student = conn.execute(
            "SELECT * FROM students WHERE roll_number=?", (roll,)
        ).fetchone()
        if student and student["image_path"] and os.path.exists(student["image_path"]):
            os.remove(student["image_path"])
        conn.execute("DELETE FROM students    WHERE roll_number=?", (roll,))
        conn.execute("DELETE FROM attendance  WHERE roll_number=?", (roll,))
        conn.execute("DELETE FROM face_data   WHERE roll_number=?", (roll,))
        conn.commit()
        conn.close()
        logger.info(f"Student deleted: {roll}")
        return {"success": True, "message": "Student deleted."}
    except Exception as e:
        return {"success": False, "message": str(e)}


# ─────────────────────────────────────────────────────────────
# SECTION 6 ── FACE RECOGNITION MODULE
# ─────────────────────────────────────────────────────────────

def encode_student_face(roll: str, img_path: str) -> bool:
    """Compute and store face encoding for a student image."""
    if not FACE_RECOGNITION_AVAILABLE:
        logger.warning(f"Cannot encode {roll}: face_recognition library not available.")
        return False
    if not img_path or not os.path.exists(img_path):
        logger.error(f"Cannot encode {roll}: image file missing at {img_path}")
        return False
    try:
        img = face_recognition.load_image_file(img_path)

        # Downscale very large photos (common with modern phone cameras) —
        # oversized images slow down / can fail HOG detection.
        h, w = img.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        locations = face_recognition.face_locations(img, model="hog")
        if not locations:
            logger.warning(f"No face detected in image for {roll} ({img_path})")
            return False
        encs = face_recognition.face_encodings(img, known_face_locations=locations, num_jitters=1)
        if not encs:
            logger.warning(f"Face detected but encoding failed for {roll} ({img_path})")
            return False
        if len(locations) > 1:
            logger.warning(f"Multiple faces ({len(locations)}) found for {roll}; using the first one.")
        encoding_blob = pickle.dumps(encs[0])
        conn = get_connection()
        student = conn.execute(
            "SELECT id FROM students WHERE roll_number=?", (roll,)
        ).fetchone()
        if not student:
            conn.close()
            return False
        conn.execute(
            """INSERT OR REPLACE INTO face_data (student_id, roll_number, encoding, updated_at)
               VALUES (?, ?, ?, datetime('now'))""",
            (student["id"], roll, encoding_blob)
        )
        conn.execute(
            "UPDATE students SET face_encoded=1 WHERE roll_number=?", (roll,)
        )
        conn.commit()
        conn.close()
        logger.info(f"Face encoded for {roll}")
        return True
    except Exception as e:
        logger.error(f"encode_student_face error ({roll}): {e}")
        return False


def load_all_encodings() -> list[dict]:
    """Load all stored encodings from DB."""
    if not FACE_RECOGNITION_AVAILABLE:
        return []
    try:
        conn = get_connection()
        rows = conn.execute("SELECT roll_number, encoding FROM face_data").fetchall()
        conn.close()
        result = []
        for r in rows:
            enc = pickle.loads(r["encoding"])
            result.append({"roll_number": r["roll_number"], "encoding": enc})
        return result
    except Exception as e:
        logger.error(f"load_all_encodings error: {e}")
        return []


def recognize_face_in_image(img_array: np.ndarray) -> list[dict]:
    """
    Given a BGR numpy image, return list of recognised faces:
    [{roll_number, name, confidence, bbox}]
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return []
    known = load_all_encodings()
    if not known:
        return []
    rgb       = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)
    results   = []
    known_encs  = [k["encoding"]    for k in known]
    known_rolls = [k["roll_number"] for k in known]
    for enc, loc in zip(encodings, locations):
        distances  = face_recognition.face_distance(known_encs, enc)
        best_idx   = int(np.argmin(distances))
        confidence = float(1 - distances[best_idx])
        if confidence >= CONFIDENCE_THRESHOLD:
            roll    = known_rolls[best_idx]
            student = get_student_by_roll(roll)
            name    = student["full_name"] if student else "Unknown"
        else:
            roll, name = "UNKNOWN", "Unknown"
        top, right, bottom, left = loc
        results.append({
            "roll_number": roll,
            "name":        name,
            "confidence":  round(confidence * 100, 1),
            "bbox":        (left, top, right, bottom)
        })
    return results


def detect_faces_haar(img_array: np.ndarray) -> list[tuple]:
    """Fallback Haarcascade face detection returning list of (x,y,w,h)."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray    = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces   = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                        minSize=(60, 60))
    return [tuple(f) for f in faces] if len(faces) > 0 else []


def draw_face_annotations(img_array: np.ndarray, faces: list[dict]) -> np.ndarray:
    """Draw coloured bounding boxes and labels on image."""
    annotated = img_array.copy()
    for face in faces:
        l, t, r, b = face["bbox"]
        colour = (0, 200, 100) if face["roll_number"] != "UNKNOWN" else (0, 0, 220)
        cv2.rectangle(annotated, (l, t), (r, b), colour, 2)
        label = f"{face['name']} ({face['confidence']}%)"
        cv2.putText(annotated, label, (l, t - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return annotated


# ─────────────────────────────────────────────────────────────
# SECTION 7 ── ATTENDANCE MODULE
# ─────────────────────────────────────────────────────────────

def mark_attendance(roll: str, method: str = "manual",
                    confidence: float = 1.0, marked_by: str = "admin") -> dict:
    """Mark a student present today. Prevents duplicates."""
    student = get_student_by_roll(roll)
    if not student:
        return {"success": False, "message": f"Student {roll} not found."}
    today = datetime.date.today().isoformat()
    now   = datetime.datetime.now().strftime("%H:%M:%S")
    try:
        conn = get_connection()
        conn.execute(
            """INSERT INTO attendance
               (student_id, roll_number, full_name, date, time,
                status, method, confidence, marked_by)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (student["id"], roll, student["full_name"],
             today, now, "present", method, confidence, marked_by)
        )
        conn.commit()
        conn.close()
        logger.info(f"Attendance marked: {roll} on {today}")
        return {"success": True,
                "message": f"✅ Attendance marked for {student['full_name']}"}
    except sqlite3.IntegrityError:
        return {"success": False,
                "message": f"⚠️ {student['full_name']} already marked today."}
    except Exception as e:
        return {"success": False, "message": str(e)}


def mark_absent(roll: str, date: str) -> dict:
    """Explicitly mark a student absent on a given date."""
    student = get_student_by_roll(roll)
    if not student:
        return {"success": False, "message": "Student not found."}
    try:
        conn = get_connection()
        conn.execute(
            """INSERT OR IGNORE INTO attendance
               (student_id, roll_number, full_name, date, time, status, method)
               VALUES (?,?,?,?,?,?,?)""",
            (student["id"], roll, student["full_name"],
             date, "00:00:00", "absent", "manual")
        )
        conn.commit()
        conn.close()
        return {"success": True, "message": f"Absent marked for {roll} on {date}."}
    except Exception as e:
        return {"success": False, "message": str(e)}


def get_attendance_df(date: str = None, roll: str = None,
                      dept: str = None) -> pd.DataFrame:
    """Flexible attendance query with optional filters."""
    query  = "SELECT * FROM attendance WHERE 1=1"
    params = []
    if date:
        query += " AND date=?";  params.append(date)
    if roll:
        query += " AND roll_number=?";  params.append(roll)
    conn = get_connection()
    df   = pd.read_sql_query(query, conn, params=params)
    conn.close()
    if dept and not df.empty:
        students = get_all_students()
        df = df.merge(students[["roll_number", "department"]], on="roll_number", how="left")
        df = df[df["department"] == dept]
    return df.sort_values("date", ascending=False) if not df.empty else df


def get_attendance_stats() -> dict:
    """Return high-level stats dictionary."""
    conn   = get_connection()
    total  = conn.execute("SELECT COUNT(*) FROM students WHERE status='active'").fetchone()[0]
    today  = datetime.date.today().isoformat()
    today_p= conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE date=? AND status='present'", (today,)
    ).fetchone()[0]
    all_records = conn.execute("SELECT COUNT(*) FROM attendance WHERE status='present'").fetchone()[0]
    total_days  = conn.execute("SELECT COUNT(DISTINCT date) FROM attendance").fetchone()[0]
    conn.close()
    avg_pct = round(all_records / max(total * total_days, 1) * 100, 1)
    return {
        "total_students": total,
        "present_today":  today_p,
        "absent_today":   max(0, total - today_p),
        "total_days":     total_days,
        "avg_attendance": avg_pct,
        "total_records":  all_records,
    }


def get_student_attendance_percent(roll: str) -> float:
    conn      = get_connection()
    total_d   = conn.execute("SELECT COUNT(DISTINCT date) FROM attendance").fetchone()[0]
    present_d = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE roll_number=? AND status='present'", (roll,)
    ).fetchone()[0]
    conn.close()
    return round(present_d / max(total_d, 1) * 100, 1)


# ─────────────────────────────────────────────────────────────
# SECTION 8 ── REPORT GENERATION
# ─────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, filename: str = "attendance.csv") -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def export_excel(df: pd.DataFrame, filename: str = "attendance.xlsx") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Attendance")
    return buf.getvalue()


def export_pdf(df: pd.DataFrame, title: str = "Attendance Report") -> bytes:
    """Generate a styled PDF report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_fill_color(30, 58, 138)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C", fill=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    # Table header
    cols    = list(df.columns)
    col_w   = min(35, 180 // len(cols))
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(59, 130, 246)
    pdf.set_text_color(255, 255, 255)
    for c in cols:
        pdf.cell(col_w, 8, str(c)[:14], border=1, fill=True)
    pdf.ln()
    # Table rows
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(30, 30, 30)
    for i, row in df.iterrows():
        fill = i % 2 == 0
        pdf.set_fill_color(240, 246, 255) if fill else pdf.set_fill_color(255, 255, 255)
        for c in cols:
            pdf.cell(col_w, 7, str(row[c])[:14], border=1, fill=fill)
        pdf.ln()
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# SECTION 9 ── STREAMLIT CUSTOM CSS  (Professional Dark Theme)
# ─────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* ── Google Fonts ───────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root Variables ─────────────────────────────────────── */
:root {
  --bg:        #0f172a;
  --bg2:       #1e293b;
  --bg3:       #334155;
  --accent:    #3b82f6;
  --accent2:   #6366f1;
  --success:   #10b981;
  --warn:      #f59e0b;
  --danger:    #ef4444;
  --text:      #e2e8f0;
  --muted:     #94a3b8;
  --border:    rgba(255,255,255,0.08);
  --radius:    14px;
  --shadow:    0 4px 30px rgba(0,0,0,0.4);
}

/* ── Global Resets ──────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  background:  var(--bg) !important;
  color:       var(--text) !important;
}

/* ── Streamlit Main ─────────────────────────────────────── */
.stApp { background: var(--bg) !important; }
.block-container { padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1300px !important; }

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Metric Cards ───────────────────────────────────────── */
[data-testid="metric-container"] {
  background:    var(--bg2) !important;
  border:        1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding:       1.25rem 1.5rem !important;
  box-shadow:    var(--shadow) !important;
}

/* ── Buttons ────────────────────────────────────────────── */
.stButton > button {
  background:    linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  color:         white !important;
  border:        none !important;
  border-radius: 10px !important;
  padding:       0.55rem 1.4rem !important;
  font-weight:   700 !important;
  letter-spacing:0.3px !important;
  transition:    all 0.2s ease !important;
  box-shadow:    0 2px 14px rgba(59,130,246,0.35) !important;
}
.stButton > button:hover {
  transform:   translateY(-2px) !important;
  box-shadow:  0 4px 22px rgba(59,130,246,0.5) !important;
}

/* ── Inputs ─────────────────────────────────────────────── */
.stTextInput input, .stSelectbox select,
.stTextArea textarea {
  background:    var(--bg3) !important;
  color:         var(--text) !important;
  border:        1px solid var(--border) !important;
  border-radius: 10px !important;
  padding:       0.5rem 0.75rem !important;
}

/* ── Tables ─────────────────────────────────────────────── */
.stDataFrame, [data-testid="stTable"] {
  background:    var(--bg2) !important;
  border-radius: var(--radius) !important;
  border:        1px solid var(--border) !important;
}

/* ── Tabs ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background:    var(--bg2) !important;
  border-radius: 12px !important;
  gap:           6px !important;
  padding:       6px !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px !important;
  color:         var(--muted) !important;
  font-weight:   600 !important;
}
.stTabs [aria-selected="true"] {
  background: var(--accent) !important;
  color:      white !important;
}

/* ── Alerts ─────────────────────────────────────────────── */
.stSuccess { background: rgba(16,185,129,0.12) !important; border-left: 4px solid var(--success) !important; border-radius: 8px !important; }
.stError   { background: rgba(239,68,68,0.12)  !important; border-left: 4px solid var(--danger)  !important; border-radius: 8px !important; }
.stWarning { background: rgba(245,158,11,0.12) !important; border-left: 4px solid var(--warn)    !important; border-radius: 8px !important; }
.stInfo    { background: rgba(59,130,246,0.12) !important; border-left: 4px solid var(--accent)  !important; border-radius: 8px !important; }

/* ── Card Helper ─────────────────────────────────────────── */
.sa-card {
  background:    var(--bg2);
  border:        1px solid var(--border);
  border-radius: var(--radius);
  padding:       1.4rem 1.6rem;
  box-shadow:    var(--shadow);
  margin-bottom: 1rem;
}
.sa-card-title {
  font-size:   1.05rem;
  font-weight: 700;
  color:       var(--accent);
  margin-bottom: 0.5rem;
}

/* ── Header Banner ─────────────────────────────────────── */
.sa-header {
  background:    linear-gradient(135deg, #1e3a8a 0%, #1e293b 60%, #0f172a 100%);
  border-radius: var(--radius);
  padding:       2rem 2.5rem;
  margin-bottom: 1.5rem;
  border:        1px solid var(--border);
  display:       flex;
  align-items:   center;
  gap:           1rem;
}
.sa-header h1 { font-size: 2rem; font-weight: 800; margin: 0; }
.sa-header p  { color: var(--muted); margin: 0.25rem 0 0 0; }

/* ── Badge ──────────────────────────────────────────────── */
.badge-green  { background: rgba(16,185,129,0.18); color: #10b981; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; }
.badge-red    { background: rgba(239,68,68,0.18);  color: #ef4444; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; }
.badge-blue   { background: rgba(59,130,246,0.18); color: #3b82f6; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; }

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar        { width: 6px; height: 6px; }
::-webkit-scrollbar-track  { background: var(--bg2); }
::-webkit-scrollbar-thumb  { background: var(--bg3); border-radius: 3px; }
</style>
"""


# ─────────────────────────────────────────────────────────────
# SECTION 10 ── UI HELPER COMPONENTS
# ─────────────────────────────────────────────────────────────

def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", icon: str = ""):
    st.markdown(f"""
    <div class="sa-header">
      <div style="font-size:2.8rem">{icon}</div>
      <div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


def stat_row(stats: dict):
    """Render a row of Streamlit metric cards."""
    cols = st.columns(len(stats))
    for col, (label, (val, delta)) in zip(cols, stats.items()):
        col.metric(label, val, delta)


def card(title: str, content_html: str):
    st.markdown(f"""
    <div class="sa-card">
      <div class="sa-card-title">{title}</div>
      {content_html}
    </div>
    """, unsafe_allow_html=True)


def badge(text: str, colour: str = "blue") -> str:
    return f'<span class="badge-{colour}">{text}</span>'


# ─────────────────────────────────────────────────────────────
# SECTION 11 ── PAGE: LOGIN / SIGNUP
# ─────────────────────────────────────────────────────────────

def page_auth():
    inject_css()
    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem 0;'>
      <div style='font-size:3.5rem'>🎓</div>
      <h1 style='font-size:2.2rem; font-weight:800; margin:0;'>SmartAttend AI</h1>
      <p style='color:#94a3b8; font-size:1rem;'>AI-Powered Attendance Management Platform</p>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["🔑  Sign In", "📝  Create Account"])

    with tab_login:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="your-username")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
        if submitted:
            if not username or not password:
                st.error("Please fill in all fields.")
            else:
                result = login_admin(username, password)
                if result["success"]:
                    st.session_state["admin"]         = result["admin"]
                    st.session_state["authenticated"] = True
                    log_activity(username, "LOGIN", "Successful login")
                    st.success("Welcome back! Redirecting…")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error(result["message"])

    with tab_signup:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("signup_form"):
            c1, c2 = st.columns(2)
            full_name = c1.text_input("Full Name")
            username  = c2.text_input("Username")
            email     = st.text_input("Email Address")
            p1        = st.text_input("Password",        type="password")
            p2        = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
        if submitted:
            if p1 != p2:
                st.error("Passwords do not match.")
            else:
                result = create_admin(username, email, p1, full_name)
                if result["success"]:
                    st.success(result["message"] + " Please sign in.")
                else:
                    st.error(result["message"])


# ─────────────────────────────────────────────────────────────
# SECTION 12 ── PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────

def page_dashboard():
    inject_css()
    admin  = st.session_state["admin"]
    page_header("Dashboard", f"Welcome back, {admin['full_name'] or admin['username']} 👋", "📊")

    stats = get_attendance_stats()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total Students",    stats["total_students"])
    c2.metric("✅ Present Today",      stats["present_today"],
              f"+{stats['present_today']}")
    c3.metric("❌ Absent Today",       stats["absent_today"])
    c4.metric("📅 Tracking Days",      stats["total_days"])
    c5.metric("📈 Avg Attendance",     f"{stats['avg_attendance']}%")
    st.markdown("---")

    col_a, col_b = st.columns([3, 2])

    # ── Daily trend line chart ─────────────────────────────
    with col_a:
        st.subheader("📈 Attendance Trend (Last 14 Days)")
        df = get_attendance_df()
        if not df.empty:
            daily = (df[df["status"] == "present"]
                     .groupby("date")["roll_number"].count()
                     .reset_index()
                     .rename(columns={"roll_number": "present"}))
            daily = daily.sort_values("date").tail(14)
            fig = px.line(daily, x="date", y="present",
                          markers=True,
                          labels={"date": "Date", "present": "Students Present"},
                          template="plotly_dark",
                          color_discrete_sequence=["#3b82f6"])
            fig.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#1e293b",
                               font_color="#e2e8f0", margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No attendance data yet.")

    # ── Donut chart ─────────────────────────────────────────
    with col_b:
        st.subheader("🍩 Today's Ratio")
        p, a = stats["present_today"], stats["absent_today"]
        if p + a > 0:
            fig2 = go.Figure(go.Pie(
                labels=["Present", "Absent"],
                values=[p, a],
                hole=0.55,
                marker_colors=["#10b981", "#ef4444"],
            ))
            fig2.update_layout(
                plot_bgcolor="#1e293b", paper_bgcolor="#1e293b",
                font_color="#e2e8f0",
                showlegend=True,
                legend=dict(orientation="h", y=-0.1),
                margin=dict(l=0,r=0,t=10,b=0),
            )
            fig2.add_annotation(text=f"{round(p/(p+a)*100)}%", x=0.5, y=0.5,
                                 font_size=24, showarrow=False,
                                 font_color="#e2e8f0")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data for today yet.")

    # ── Department attendance bar ──────────────────────────
    st.markdown("---")
    st.subheader("🏫 Attendance by Department")
    df_all = get_attendance_df()
    if not df_all.empty:
        students = get_all_students()
        merged   = df_all.merge(students[["roll_number","department"]], on="roll_number", how="left")
        dept_grp = (merged[merged["status"]=="present"]
                    .groupby("department")["roll_number"].count()
                    .reset_index().rename(columns={"roll_number":"count"}))
        if not dept_grp.empty:
            fig3 = px.bar(dept_grp, x="department", y="count",
                          color="count", color_continuous_scale="Blues",
                          template="plotly_dark",
                          labels={"department":"Department","count":"Present Count"})
            fig3.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#1e293b",
                                font_color="#e2e8f0", margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No data to visualize yet.")

    # ── Recent attendance table ────────────────────────────
    st.markdown("---")
    st.subheader("🕐 Recent Attendance Records")
    recent = get_attendance_df().head(10)
    if not recent.empty:
        disp = recent[["roll_number","full_name","date","time","status","method","confidence"]].copy()
        disp.columns = ["Roll","Name","Date","Time","Status","Method","Confidence"]
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info("No records found.")


# ─────────────────────────────────────────────────────────────
# SECTION 13 ── PAGE: FACE ATTENDANCE
# ─────────────────────────────────────────────────────────────

def page_face_attendance():
    inject_css()
    page_header("Face Attendance", "Upload a photo or use webcam snapshot to mark attendance", "📷")

    tab_upload, tab_webcam, tab_manual = st.tabs([
        "📁 Upload Photo", "🎥 Webcam Snapshot", "✏️ Manual Entry"
    ])

    # ── Upload Photo ─────────────────────────────────────────
    with tab_upload:
        st.markdown("Upload a classroom or individual photo to auto-detect and mark attendance.")
        uploaded = st.file_uploader("Choose image", type=["jpg","jpeg","png"])
        if uploaded:
            img_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img       = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Could not read image.")
                return
            col_orig, col_ann = st.columns(2)
            col_orig.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                            caption="Original", use_column_width=True)
            if FACE_RECOGNITION_AVAILABLE:
                with st.spinner("Analysing faces…"):
                    faces = recognize_face_in_image(img)
                if faces:
                    annotated = draw_face_annotations(img, faces)
                    col_ann.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                   caption="Detected Faces", use_column_width=True)
                    st.success(f"Detected {len(faces)} face(s).")
                    for face in faces:
                        if face["roll_number"] != "UNKNOWN":
                            res = mark_attendance(face["roll_number"],
                                                  method="face_recognition",
                                                  confidence=face["confidence"]/100,
                                                  marked_by=st.session_state["admin"]["username"])
                            if res["success"]:
                                st.success(res["message"])
                            else:
                                st.warning(res["message"])
                        else:
                            st.warning(f"Unknown face detected — could not mark attendance.")
                else:
                    st.warning("No known faces detected.")
            else:
                # Haarcascade fallback
                faces_haar = detect_faces_haar(img)
                ann = img.copy()
                for (x,y,w,h) in faces_haar:
                    cv2.rectangle(ann, (x,y),(x+w,y+h),(0,200,100),2)
                col_ann.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                               caption=f"Detected {len(faces_haar)} face(s) (Haarcascade)",
                               use_column_width=True)
                st.info("face_recognition not available — faces detected but not identified. Mark manually.")

    # ── Webcam Snapshot ──────────────────────────────────────
    with tab_webcam:
        st.info("📸 Use your device camera to take a live snapshot.")
        webcam_img = st.camera_input("Take a photo")
        if webcam_img:
            img_bytes = np.frombuffer(webcam_img.read(), np.uint8)
            img       = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if FACE_RECOGNITION_AVAILABLE:
                with st.spinner("Recognising faces…"):
                    faces = recognize_face_in_image(img)
                for face in faces:
                    if face["roll_number"] != "UNKNOWN":
                        res = mark_attendance(face["roll_number"],
                                              method="webcam",
                                              confidence=face["confidence"]/100,
                                              marked_by=st.session_state["admin"]["username"])
                        st.success(res["message"]) if res["success"] else st.warning(res["message"])
                    else:
                        st.warning("Unrecognised face — please mark manually.")
            else:
                st.warning("face_recognition unavailable. Use Manual Entry.")

    # ── Manual Entry ─────────────────────────────────────────
    with tab_manual:
        students_df = get_all_students()
        if students_df.empty:
            st.info("No students registered yet.")
        else:
            options = {f"{r['roll_number']} — {r['full_name']}": r['roll_number']
                       for _, r in students_df.iterrows()}
            selected = st.selectbox("Select Student", list(options.keys()))
            col_btn1, col_btn2 = st.columns(2)
            if col_btn1.button("✅ Mark Present", use_container_width=True):
                roll = options[selected]
                res  = mark_attendance(roll, method="manual",
                                       marked_by=st.session_state["admin"]["username"])
                st.success(res["message"]) if res["success"] else st.warning(res["message"])
            if col_btn2.button("❌ Mark Absent", use_container_width=True):
                roll  = options[selected]
                today = datetime.date.today().isoformat()
                res   = mark_absent(roll, today)
                st.success(res["message"]) if res["success"] else st.error(res["message"])


# ─────────────────────────────────────────────────────────────
# SECTION 14 ── PAGE: STUDENT MANAGEMENT
# ─────────────────────────────────────────────────────────────

DEPARTMENTS = ["Computer Science", "Software Engineering", "Electrical Engineering",
               "Business Administration", "Mathematics", "Physics", "Other"]
SEMESTERS   = [f"Semester {i}" for i in range(1, 9)]


def page_students():
    inject_css()
    page_header("Student Management", "Register, edit and manage student profiles", "👨‍🎓")

    tab_list, tab_add, tab_edit, tab_delete = st.tabs([
        "📋 All Students", "➕ Add Student", "✏️ Edit Student", "🗑️ Delete Student"
    ])

    # ── All Students ─────────────────────────────────────────
    with tab_list:
        df = get_all_students()
        if df.empty:
            st.info("No students registered yet.")
        else:
            search = st.text_input("🔍 Search (name / roll / department)", "")
            if search:
                mask = (df["full_name"].str.contains(search, case=False, na=False) |
                        df["roll_number"].str.contains(search, case=False, na=False) |
                        df["department"].str.contains(search, case=False, na=False))
                df = df[mask]
            st.dataframe(df[["roll_number","full_name","department","semester",
                              "email","status","face_encoded","created_at"]],
                          use_container_width=True, hide_index=True)
            st.caption(f"Total: **{len(df)}** students")

            unencoded = df[(df["face_encoded"] == 0) & (df["image_path"].fillna("") != "")]
            if not unencoded.empty:
                st.warning(f"⚠️ {len(unencoded)} student(s) have a photo on file but NO face "
                            "encoding — they cannot be matched by the camera yet.")
                if st.button("🔁 Re-encode All Unmatched Photos", use_container_width=True):
                    if not FACE_RECOGNITION_AVAILABLE:
                        st.error("Face-recognition engine is not available in this session. "
                                  "Re-run the install cell first.")
                    else:
                        fixed, failed = 0, []
                        for _, row in unencoded.iterrows():
                            if row["image_path"] and os.path.exists(row["image_path"]):
                                if encode_student_face(row["roll_number"], row["image_path"]):
                                    fixed += 1
                                else:
                                    failed.append(row["roll_number"])
                            else:
                                failed.append(row["roll_number"])
                        st.success(f"✅ Fixed {fixed} student(s).")
                        if failed:
                            st.warning(f"Still unresolved (need a new photo): {', '.join(failed)}")
                        st.rerun()

    # ── Add Student ──────────────────────────────────────────
    with tab_add:
        with st.form("add_student_form"):
            c1, c2 = st.columns(2)
            roll     = c1.text_input("Roll Number *", placeholder="CS-2024-001")
            name     = c2.text_input("Full Name *",   placeholder="John Doe")
            email    = st.text_input("Email Address",  placeholder="john@example.com")
            dept     = st.selectbox("Department", DEPARTMENTS)
            sem      = st.selectbox("Semester",    SEMESTERS)
            img_file = st.file_uploader("Profile Photo (for face recognition)", type=["jpg","jpeg","png"])
            submitted = st.form_submit_button("➕ Add Student", use_container_width=True)

        if submitted:
            img_bytes, img_name = (None, "")
            if img_file:
                img_bytes = img_file.read()
                img_name  = img_file.name
            res = add_student(roll, name, email, dept, sem, img_bytes, img_name)
            if res["success"]:
                if img_bytes and not res.get("face_encoded"):
                    st.warning(res["message"])
                else:
                    st.success(res["message"])
                log_activity(st.session_state["admin"]["username"], "ADD_STUDENT", f"Roll: {roll}")
            else:
                st.error(res["message"])

    # ── Edit Student ─────────────────────────────────────────
    with tab_edit:
        rolls = [r for r in get_all_students()["roll_number"]]
        if not rolls:
            st.info("No students to edit.")
        else:
            sel_roll = st.selectbox("Select Student (Roll Number)", rolls, key="edit_sel")
            stu      = get_student_by_roll(sel_roll)
            if stu:
                with st.form("edit_form"):
                    c1, c2 = st.columns(2)
                    e_name  = c1.text_input("Full Name", value=stu["full_name"])
                    e_email = c2.text_input("Email",     value=stu["email"] or "")
                    e_dept  = st.selectbox("Department", DEPARTMENTS,
                                           index=DEPARTMENTS.index(stu["department"])
                                                 if stu["department"] in DEPARTMENTS else 0)
                    e_sem   = st.selectbox("Semester", SEMESTERS,
                                           index=SEMESTERS.index(stu["semester"])
                                                 if stu["semester"] in SEMESTERS else 0)
                    e_stat  = st.selectbox("Status", ["active","inactive"],
                                           index=0 if stu["status"]=="active" else 1)
                    submitted = st.form_submit_button("💾 Save Changes", use_container_width=True)
                if submitted:
                    res = update_student(sel_roll, e_name, e_email, e_dept, e_sem, e_stat)
                    st.success(res["message"]) if res["success"] else st.error(res["message"])

                # ── Face Photo Management (this is what fixes recognition) ──
                st.markdown("---")
                st.markdown("#### 📷 Face Photo")
                col_prev, col_status = st.columns([1, 2])
                with col_prev:
                    if stu["image_path"] and os.path.exists(stu["image_path"]):
                        st.image(stu["image_path"], width=160, caption="Current photo")
                    else:
                        st.info("No photo on file.")
                with col_status:
                    if stu["face_encoded"]:
                        st.success("✅ Face encoding is stored — this student CAN be matched by the camera.")
                    else:
                        st.warning("⚠️ No face encoding stored — this student CANNOT be matched by "
                                    "the camera yet. Upload a clear photo below and encode it.")

                new_photo = st.file_uploader(
                    "Upload / replace photo (clear, front-facing, one face only)",
                    type=["jpg", "jpeg", "png"], key=f"edit_photo_{sel_roll}"
                )
                col_a, col_b = st.columns(2)
                if col_a.button("💾 Save Photo & Encode Face", use_container_width=True, key=f"save_photo_{sel_roll}"):
                    if not new_photo:
                        st.error("Please choose an image file first.")
                    else:
                        ext = Path(new_photo.name).suffix.lower()
                        if ext not in (".jpg", ".jpeg", ".png"):
                            ext = ".jpg"
                        new_path = os.path.join(UPLOAD_DIR, f"{sel_roll}{ext}")
                        with open(new_path, "wb") as f:
                            f.write(new_photo.read())
                        conn = get_connection()
                        conn.execute("UPDATE students SET image_path=? WHERE roll_number=?",
                                     (new_path, sel_roll))
                        conn.commit()
                        conn.close()
                        ok = encode_student_face(sel_roll, new_path)
                        if ok:
                            st.success("✅ Photo saved and face encoded successfully. "
                                       "This student can now be recognised by the camera.")
                            st.rerun()
                        elif not FACE_RECOGNITION_AVAILABLE:
                            st.error("Photo saved, but the face-recognition engine is not available "
                                      "in this session. Re-run the install cell in the notebook, "
                                      "then click this button again.")
                        else:
                            st.error("Photo saved, but no clear face was detected in it. "
                                      "Try a sharper, front-facing, well-lit photo with only one face.")
                if col_b.button("🔁 Re-encode Current Photo", use_container_width=True, key=f"reencode_{sel_roll}"):
                    if stu["image_path"] and os.path.exists(stu["image_path"]):
                        ok = encode_student_face(sel_roll, stu["image_path"])
                        if ok:
                            st.success("✅ Re-encoded successfully.")
                            st.rerun()
                        else:
                            st.error("Could not encode a face from the existing photo. Please upload a new one.")
                    else:
                        st.error("No existing photo to re-encode. Please upload one.")

    # ── Delete Student ───────────────────────────────────────
    with tab_delete:
        rolls_del = [r for r in get_all_students()["roll_number"]]
        if not rolls_del:
            st.info("No students to delete.")
        else:
            del_roll = st.selectbox("Select Student to Delete", rolls_del, key="del_sel")
            stu_del  = get_student_by_roll(del_roll)
            if stu_del:
                st.warning(f"⚠️ You are about to delete **{stu_del['full_name']}** and ALL their records.")
                confirm = st.text_input("Type the roll number to confirm deletion")
                if st.button("🗑️ Confirm Delete", type="primary"):
                    if confirm.strip() == del_roll:
                        res = delete_student(del_roll)
                        st.success(res["message"]) if res["success"] else st.error(res["message"])
                        log_activity(st.session_state["admin"]["username"],
                                     "DELETE_STUDENT", f"Roll: {del_roll}")
                    else:
                        st.error("Roll number does not match. Deletion cancelled.")


# ─────────────────────────────────────────────────────────────
# SECTION 15 ── PAGE: ATTENDANCE RECORDS
# ─────────────────────────────────────────────────────────────

def page_attendance():
    inject_css()
    page_header("Attendance Records", "View, filter and export attendance data", "📋")

    # Filters
    c1, c2, c3, c4 = st.columns(4)
    sel_date = c1.date_input("Filter by Date", value=None)
    sel_roll = c2.text_input("Filter by Roll", "")
    students_df = get_all_students()
    depts       = ["All"] + sorted(students_df["department"].dropna().unique().tolist())
    sel_dept    = c3.selectbox("Department", depts)
    sel_status  = c4.selectbox("Status", ["All","present","absent"])

    date_str = sel_date.isoformat() if sel_date else None
    roll_str = sel_roll.strip() or None
    dept_str = None if sel_dept == "All" else sel_dept

    df = get_attendance_df(date=date_str, roll=roll_str, dept=dept_str)
    if sel_status != "All":
        df = df[df["status"] == sel_status]

    st.markdown(f"**{len(df)} record(s) found.**")
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("#### 📥 Export")
        col_csv, col_excel, col_pdf = st.columns(3)
        col_csv.download_button("⬇️ CSV", export_csv(df),
                                 file_name="attendance.csv", mime="text/csv")
        col_excel.download_button("⬇️ Excel", export_excel(df),
                                   file_name="attendance.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        col_pdf.download_button("⬇️ PDF Report", export_pdf(df, "Attendance Report"),
                                  file_name="attendance_report.pdf", mime="application/pdf")
    else:
        st.info("No records match the selected filters.")


# ─────────────────────────────────────────────────────────────
# SECTION 16 ── PAGE: ANALYTICS
# ─────────────────────────────────────────────────────────────

def page_analytics():
    inject_css()
    page_header("Analytics", "Deep insights into attendance patterns", "📊")

    df = get_attendance_df()
    if df.empty:
        st.info("No attendance data available yet. Start marking attendance to see analytics.")
        return

    students = get_all_students()
    df = df.merge(students[["roll_number","department","semester"]], on="roll_number", how="left")

    # Row 1: Heatmap + Monthly bar
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🗓️ Monthly Attendance Heatmap")
        df["month"]   = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
        monthly_present = (df[df["status"]=="present"]
                           .groupby("month")["roll_number"].count()
                           .reset_index().rename(columns={"roll_number":"count"}))
        fig = px.bar(monthly_present, x="month", y="count",
                     color="count", color_continuous_scale="Blues",
                     template="plotly_dark",
                     labels={"month":"Month","count":"Present Count"})
        fig.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#1e293b",
                           font_color="#e2e8f0", margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("🏫 Department-wise Breakdown")
        dept_data = (df[df["status"]=="present"]
                     .groupby("department")["roll_number"].count()
                     .reset_index().rename(columns={"roll_number":"count"}))
        fig2 = px.pie(dept_data, names="department", values="count",
                       hole=0.4, template="plotly_dark",
                       color_discrete_sequence=px.colors.sequential.Blues_r)
        fig2.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#1e293b",
                            font_color="#e2e8f0", margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Top students by attendance
    st.markdown("---")
    st.subheader("🏆 Student Attendance Percentage")
    total_days = df["date"].nunique()
    if total_days > 0:
        stu_pct = (df[df["status"]=="present"]
                   .groupby(["roll_number","full_name"])["date"].count()
                   .reset_index().rename(columns={"date":"present_days"}))
        stu_pct["percentage"] = (stu_pct["present_days"] / total_days * 100).round(1)
        stu_pct = stu_pct.sort_values("percentage", ascending=False)

        fig3 = px.bar(stu_pct.head(20), x="full_name", y="percentage",
                       color="percentage",
                       color_continuous_scale=["#ef4444","#f59e0b","#10b981"],
                       range_color=[0,100], template="plotly_dark",
                       labels={"full_name":"Student","percentage":"Attendance %"})
        fig3.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#1e293b",
                            font_color="#e2e8f0", margin=dict(l=0,r=0,t=10,b=0),
                            xaxis_tickangle=-30)
        fig3.add_hline(y=75, line_dash="dash", line_color="#f59e0b",
                        annotation_text="75% threshold")
        st.plotly_chart(fig3, use_container_width=True)

        # Students below threshold
        below = stu_pct[stu_pct["percentage"] < 75]
        if not below.empty:
            st.warning(f"⚠️ **{len(below)} students** have attendance below 75%:")
            st.dataframe(below[["roll_number","full_name","present_days","percentage"]],
                          use_container_width=True, hide_index=True)

    # Row 3: Recognition method breakdown
    st.markdown("---")
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("🤖 Recognition Methods")
        method_data = df[df["status"]=="present"]["method"].value_counts().reset_index()
        method_data.columns = ["method","count"]
        fig4 = px.pie(method_data, names="method", values="count",
                       template="plotly_dark",
                       color_discrete_sequence=["#3b82f6","#10b981","#f59e0b"])
        fig4.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#1e293b",
                            font_color="#e2e8f0", margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig4, use_container_width=True)

    with col_d:
        st.subheader("📅 Day-of-Week Pattern")
        df["dow"] = pd.to_datetime(df["date"]).dt.day_name()
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_data  = (df[df["status"]=="present"]
                     .groupby("dow")["roll_number"].count()
                     .reindex(dow_order, fill_value=0)
                     .reset_index().rename(columns={"roll_number":"count"}))
        fig5 = px.bar(dow_data, x="dow", y="count",
                       color="count", color_continuous_scale="Blues",
                       template="plotly_dark",
                       labels={"dow":"Day","count":"Present Count"})
        fig5.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#1e293b",
                            font_color="#e2e8f0", margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig5, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# SECTION 17 ── PAGE: ACTIVITY LOGS
# ─────────────────────────────────────────────────────────────

def page_logs():
    inject_css()
    page_header("Activity Logs", "Complete admin activity audit trail", "🗒️")
    conn = get_connection()
    df   = pd.read_sql_query(
        "SELECT * FROM activity_logs ORDER BY id DESC LIMIT 200", conn
    )
    conn.close()
    if df.empty:
        st.info("No activity logs yet.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export Logs (CSV)", export_csv(df),
                            file_name="activity_logs.csv", mime="text/csv")


# ─────────────────────────────────────────────────────────────
# SECTION 18 ── PAGE: SETTINGS
# ─────────────────────────────────────────────────────────────

def page_settings():
    inject_css()
    page_header("Settings", "Manage account and application configuration", "⚙️")
    admin = st.session_state["admin"]

    tab_profile, tab_password, tab_about = st.tabs([
        "👤 Profile", "🔒 Change Password", "ℹ️ About"
    ])

    with tab_profile:
        with st.form("profile_form"):
            full_name = st.text_input("Full Name", value=admin.get("full_name",""))
            email     = st.text_input("Email",     value=admin.get("email",""))
            submitted = st.form_submit_button("Save Profile")
        if submitted:
            try:
                conn = get_connection()
                conn.execute(
                    "UPDATE admins SET full_name=?, email=? WHERE id=?",
                    (full_name, email, admin["id"])
                )
                conn.commit()
                conn.close()
                st.session_state["admin"]["full_name"] = full_name
                st.session_state["admin"]["email"]     = email
                st.success("Profile updated!")
            except Exception as e:
                st.error(str(e))

    with tab_password:
        with st.form("pw_form"):
            cur_pw  = st.text_input("Current Password", type="password")
            new_pw  = st.text_input("New Password",     type="password")
            conf_pw = st.text_input("Confirm New",      type="password")
            submitted = st.form_submit_button("Change Password")
        if submitted:
            if not verify_password(cur_pw, admin["password_hash"]):
                st.error("Current password is incorrect.")
            elif new_pw != conf_pw:
                st.error("New passwords do not match.")
            elif len(new_pw) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                conn = get_connection()
                conn.execute(
                    "UPDATE admins SET password_hash=? WHERE id=?",
                    (hash_password(new_pw), admin["id"])
                )
                conn.commit()
                conn.close()
                st.success("Password changed successfully!")
                log_activity(admin["username"], "CHANGE_PASSWORD")

    with tab_about:
        st.markdown(f"""
        <div class="sa-card">
          <div class="sa-card-title">SmartAttend AI — v{VERSION}</div>
          <p>AI-powered attendance management system built as a BSCS Final Year Project.</p>
          <br>
          <b>Technology Stack</b>
          <ul>
            <li>Python 3.10+</li>
            <li>Streamlit</li>
            <li>OpenCV + face_recognition</li>
            <li>SQLite</li>
            <li>Plotly / Matplotlib</li>
            <li>FPDF2 / OpenPyXL</li>
          </ul>
          <br>
          <b>Features:</b> Face Recognition · Webcam Attendance · Analytics · PDF Reports · CSV/Excel Export
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SECTION 19 ── SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────

NAV_PAGES = {
    "📊 Dashboard":       page_dashboard,
    "📷 Face Attendance": page_face_attendance,
    "👨‍🎓 Students":       page_students,
    "📋 Attendance":      page_attendance,
    "📈 Analytics":       page_analytics,
    "🗒️ Activity Logs":   page_logs,
    "⚙️ Settings":        page_settings,
}


def sidebar():
    admin = st.session_state.get("admin", {})
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:1.5rem 0 1rem 0;">
          <div style="font-size:2.8rem">🎓</div>
          <div style="font-weight:800; font-size:1.15rem; color:#e2e8f0;">SmartAttend AI</div>
          <div style="color:#94a3b8; font-size:0.8rem; margin-top:0.25rem;">v{VERSION}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#334155; border-radius:12px; padding:0.75rem 1rem;
                    margin-bottom:1rem; font-size:0.88rem;">
          <b>👤 {admin.get('full_name') or admin.get('username','Admin')}</b><br>
          <span style='color:#94a3b8'>{admin.get('email','')}</span>
        </div>
        """, unsafe_allow_html=True)

        for page_name in NAV_PAGES:
            if st.button(page_name, use_container_width=True, key=f"nav_{page_name}"):
                st.session_state["page"] = page_name

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            log_activity(admin.get("username",""), "LOGOUT")
            for k in ["authenticated","admin","page"]:
                st.session_state.pop(k, None)
            st.rerun()

        # Quick stats in sidebar
        try:
            stats = get_attendance_stats()
            st.markdown(f"""
            <div style="background:#1e293b; border-radius:10px; padding:0.75rem;
                        font-size:0.82rem; color:#94a3b8; margin-top:0.5rem;">
              👥 Students: <b style='color:#e2e8f0'>{stats['total_students']}</b><br>
              ✅ Present Today: <b style='color:#10b981'>{stats['present_today']}</b><br>
              📈 Avg: <b style='color:#3b82f6'>{stats['avg_attendance']}%</b>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# SECTION 20 ── MAIN APPLICATION ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    # ── Streamlit page config ──────────────────────────────
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help":    "https://github.com/",
            "About":       f"# {APP_TITLE} v{VERSION}\nBSCS Final Year Project",
            "Report a bug": None,
        }
    )

    # ── One-time DB init ───────────────────────────────────
    init_database()

    # ── Session defaults ───────────────────────────────────
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "page" not in st.session_state:
        st.session_state["page"] = "📊 Dashboard"

    # ── Auth gate ──────────────────────────────────────────
    if not st.session_state["authenticated"]:
        page_auth()
        return

    # ── Authenticated layout ───────────────────────────────
    sidebar()
    page_fn = NAV_PAGES.get(st.session_state["page"], page_dashboard)
    page_fn()


if __name__ == "__main__":
    main()