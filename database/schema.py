"""
database/schema.py
PostgreSQL Database Schema + Setup for Voter ID Survey System
"""

# pip install psycopg2-binary sqlalchemy

import sqlite3  # Using SQLite for demo; swap with PostgreSQL in production
import json
from datetime import datetime

# ─── Database Connection ──────────────────────────────────────────────────────
# For PostgreSQL (production):
# from sqlalchemy import create_engine
# engine = create_engine("postgresql://user:password@localhost:5432/voter_survey_db")

# For SQLite (demo):
DB_PATH = "database/voter_survey.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

# ─── Create Tables ────────────────────────────────────────────────────────────
CREATE_TABLES_SQL = """

-- Districts table
CREATE TABLE IF NOT EXISTS districts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    state       TEXT NOT NULL DEFAULT 'Uttar Pradesh',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Families table
CREATE TABLE IF NOT EXISTS families (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id       TEXT NOT NULL UNIQUE,
    head_name       TEXT NOT NULL,
    village         TEXT,
    district_id     INTEGER REFERENCES districts(id),
    address         TEXT,
    surveyor_name   TEXT,
    survey_date     DATE DEFAULT CURRENT_DATE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Members table (one row per person)
CREATE TABLE IF NOT EXISTS members (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id       TEXT REFERENCES families(family_id),
    full_name       TEXT NOT NULL,
    date_of_birth   DATE,
    age             INTEGER NOT NULL,
    gender          TEXT CHECK(gender IN ('Male','Female','Other')) NOT NULL,
    has_voter_id    BOOLEAN NOT NULL DEFAULT FALSE,
    voter_id_number TEXT,               -- if they have one
    is_eligible     BOOLEAN GENERATED ALWAYS AS (age >= 18) STORED,
    aadhar_linked   BOOLEAN DEFAULT FALSE,
    document_type   TEXT,               -- what doc was used to verify
    agent_verified  BOOLEAN DEFAULT FALSE,  -- verified by AI agent?
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Survey logs (audit trail)
CREATE TABLE IF NOT EXISTS survey_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id   TEXT,
    action      TEXT,               -- 'INSERT', 'UPDATE', 'AGENT_VERIFY'
    agent_name  TEXT,               -- which AI agent performed this
    details     TEXT,               -- JSON details
    timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Anomaly flags
CREATE TABLE IF NOT EXISTS anomaly_flags (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    member_id   INTEGER REFERENCES members(id),
    flag_type   TEXT,               -- 'DUPLICATE', 'AGE_ERROR', 'SUSPICIOUS'
    description TEXT,
    resolved    BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def setup_database():
    """Initialize the database with all tables"""
    conn = get_connection()
    conn.executescript(CREATE_TABLES_SQL)
    conn.commit()
    conn.close()
    print("[DB] Database setup complete!")

# ─── CRUD Operations ─────────────────────────────────────────────────────────

def insert_family(family_data: dict) -> str:
    """Insert a new family survey record"""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO families (family_id, head_name, village, address, surveyor_name)
            VALUES (?, ?, ?, ?, ?)
        """, (
            family_data["family_id"],
            family_data["head"],
            family_data.get("village", ""),
            family_data.get("address", ""),
            family_data.get("surveyor", "AI Agent")
        ))
        for m in family_data.get("members", []):
            cur.execute("""
                INSERT INTO members (family_id, full_name, age, gender, has_voter_id, 
                                     voter_id_number, document_type, agent_verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                family_data["family_id"],
                m["name"], m["age"], m["gender"],
                m.get("has_voter_id", False),
                m.get("voter_id_number", None),
                m.get("document_type", "Manual"),
                m.get("agent_verified", False)
            ))
        # Log the action
        cur.execute("""
            INSERT INTO survey_logs (family_id, action, agent_name, details)
            VALUES (?, 'INSERT', 'Survey Agent', ?)
        """, (family_data["family_id"], json.dumps({"members": len(family_data.get("members",[]))})))
        conn.commit()
        return f"✅ Family {family_data['family_id']} inserted successfully"
    except Exception as e:
        conn.rollback()
        return f"❌ Error: {str(e)}"
    finally:
        conn.close()

def get_all_families() -> list:
    """Retrieve all family records with members"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM families")
    families = cur.fetchall()
    result = []
    for fam in families:
        cur.execute("SELECT * FROM members WHERE family_id = ?", (fam[1],))
        members = cur.fetchall()
        result.append({"family": fam, "members": members})
    conn.close()
    return result

def get_district_stats(district: str) -> dict:
    """Get voter ID statistics for a specific district"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            COUNT(*) as total_members,
            SUM(CASE WHEN age >= 18 THEN 1 ELSE 0 END) as eligible,
            SUM(CASE WHEN has_voter_id = 1 THEN 1 ELSE 0 END) as have_id,
            SUM(CASE WHEN gender='Female' AND has_voter_id=1 THEN 1 ELSE 0 END) as female_id,
            SUM(CASE WHEN gender='Male' AND has_voter_id=1 THEN 1 ELSE 0 END) as male_id
        FROM members m
        JOIN families f ON m.family_id = f.family_id
        WHERE f.district_id IN (SELECT id FROM districts WHERE name = ?)
    """, (district,))
    row = cur.fetchone()
    conn.close()
    return {
        "district": district,
        "total": row[0], "eligible": row[1], "have_id": row[2],
        "female_id": row[3], "male_id": row[4],
        "coverage_pct": round((row[2]/row[1])*100, 1) if row[1] else 0
    }

def flag_anomaly(member_id: int, flag_type: str, description: str):
    """Flag a suspicious or anomalous record"""
    conn = get_connection()
    conn.execute("""
        INSERT INTO anomaly_flags (member_id, flag_type, description)
        VALUES (?, ?, ?)
    """, (member_id, flag_type, description))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    setup_database()
    print("[DB] Schema:")
    print("  - districts")
    print("  - families")
    print("  - members (with eligibility auto-computed)")
    print("  - survey_logs (audit trail)")
    print("  - anomaly_flags")
