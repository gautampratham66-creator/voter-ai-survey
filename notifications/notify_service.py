"""
notifications/notify_service.py

Sends automatic outreach notifications (SMS and/or Email) to every citizen
who is:
    - eligible for a Voter ID  (age >= 18), AND
    - does not currently have one (has_voter_id == "No")

Pluggable design: SMSChannel and EmailChannel each implement `.send(record)`.
Add/remove channels freely; NotificationManager fans out to whichever
channels are configured.

Credentials are read from environment variables (see .env.example). If
credentials for a channel are missing, that channel runs in DRY-RUN mode:
it logs what it *would* send instead of actually sending, so you can wire
this up safely before you have real API keys.

    pip install twilio python-dotenv   (email uses stdlib smtplib, no extra dep)
"""

import os
import csv
import smtplib
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; env vars can be set another way


# ─── Data model ──────────────────────────────────────────────────────────────
@dataclass
class VoterRecord:
    """One person who needs an outreach notification."""
    name: str
    age: int
    gender: str
    district: str
    phone_number: Optional[str] = None
    email: Optional[str] = None
    nearest_seva_camp: Optional[str] = None
    risk_level: Optional[str] = None
    family_id: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def message_text(self) -> str:
        camp = f" Nearest Seva Kendra: {self.nearest_seva_camp}." if self.nearest_seva_camp else ""
        return (
            f"Dear {self.name}, our records show you do not yet have a Voter ID. "
            f"You are eligible to register.{camp} "
            f"Please visit with valid ID proof (Aadhaar/Birth Certificate) to complete enrollment. "
            f"— District Election Office, {self.district}"
        )


@dataclass
class NotificationResult:
    record: VoterRecord
    channel: str
    status: str          # "sent" | "dry_run" | "skipped" | "failed"
    detail: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ─── Channels ────────────────────────────────────────────────────────────────
class NotificationChannel(ABC):
    name: str = "base"

    @abstractmethod
    def is_configured(self) -> bool:
        """True if real credentials are present (send for real). False = dry-run."""
        ...

    @abstractmethod
    def send(self, record: VoterRecord) -> NotificationResult:
        ...


class SMSChannel(NotificationChannel):
    """SMS via Twilio. Dry-runs (logs only) if credentials aren't set."""
    name = "sms"

    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER")
        self._client = None

    def is_configured(self) -> bool:
        return bool(self.account_sid and self.auth_token and self.from_number)

    def _client_or_none(self):
        if self._client is None and self.is_configured():
            from twilio.rest import Client  # imported lazily; only needed for real sends
            self._client = Client(self.account_sid, self.auth_token)
        return self._client

    def send(self, record: VoterRecord) -> NotificationResult:
        if not record.phone_number:
            return NotificationResult(record, self.name, "skipped", "no phone_number on record")

        body = record.message_text()

        if not self.is_configured():
            print(f"[SMS][DRY-RUN] -> {record.phone_number}: {body}")
            return NotificationResult(record, self.name, "dry_run", "TWILIO_* env vars not set")

        try:
            client = self._client_or_none()
            msg = client.messages.create(
                body=body,
                from_=self.from_number,
                to=self._e164(record.phone_number),
            )
            return NotificationResult(record, self.name, "sent", f"sid={msg.sid}")
        except Exception as e:
            return NotificationResult(record, self.name, "failed", str(e))

    @staticmethod
    def _e164(phone: str) -> str:
        phone = str(phone).strip()
        if not phone.startswith("+"):
            # Default to India country code; adjust for your deployment
            phone = "+91" + phone.lstrip("0")
        return phone


class EmailChannel(NotificationChannel):
    """Email via SMTP (Gmail, SendGrid SMTP relay, office SMTP, etc). Dry-runs if unset."""
    name = "email"

    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.from_address = os.getenv("SMTP_FROM_ADDRESS", self.smtp_user)

    def is_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_user and self.smtp_password)

    def send(self, record: VoterRecord) -> NotificationResult:
        if not record.email:
            return NotificationResult(record, self.name, "skipped", "no email on record")

        body = record.message_text()

        if not self.is_configured():
            print(f"[EMAIL][DRY-RUN] -> {record.email}: {body}")
            return NotificationResult(record, self.name, "dry_run", "SMTP_* env vars not set")

        try:
            msg = EmailMessage()
            msg["Subject"] = "Action needed: Complete your Voter ID registration"
            msg["From"] = self.from_address
            msg["To"] = record.email
            msg.set_content(body)

            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            return NotificationResult(record, self.name, "sent")
        except Exception as e:
            return NotificationResult(record, self.name, "failed", str(e))


# ─── Manager ─────────────────────────────────────────────────────────────────
class NotificationManager:
    def __init__(self, channels=None):
        self.channels = channels if channels is not None else [SMSChannel(), EmailChannel()]

    def notify(self, records: list) -> list:
        """Send (or dry-run) notifications for every record, on every channel."""
        results = []
        for record in records:
            for channel in self.channels:
                results.append(channel.send(record))
        return results

    def notify_and_log(self, records: list, log_path: str = "notifications/outreach_log.csv") -> list:
        results = self.notify(records)
        self._append_log(results, log_path)
        return results

    @staticmethod
    def _append_log(results: list, log_path: str):
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        file_exists = os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "name", "district", "phone_number", "email",
                                  "channel", "status", "detail"])
            for r in results:
                writer.writerow([
                    r.timestamp, r.record.name, r.record.district,
                    r.record.phone_number, r.record.email,
                    r.channel, r.status, r.detail
                ])


# ─── Helper: build records from eligible-missing citizens ───────────────────
def find_eligible_missing_voters_from_df(df) -> list:
    """
    df: a pandas DataFrame with the same shape as the survey CSV
    (age, gender, district, has_voter_id, phone_number, nearest_seva_camp, ...).
    Returns a list of VoterRecord for everyone age>=18 with has_voter_id == "No".
    """
    subset = df[(df["age"] >= 18) & (df["has_voter_id"] == "No")]
    records = []
    for _, row in subset.iterrows():
        records.append(VoterRecord(
            name=row.get("name", "Citizen"),
            age=int(row["age"]),
            gender=row.get("gender", ""),
            district=row.get("district", ""),
            phone_number=str(row["phone_number"]) if "phone_number" in row and not pd_isna(row["phone_number"]) else None,
            email=row.get("email") if "email" in row and not pd_isna(row.get("email")) else None,
            nearest_seva_camp=row.get("nearest_seva_camp"),
            risk_level=row.get("risk_level"),
        ))
    return records


def pd_isna(value) -> bool:
    try:
        import pandas as pd
        return pd.isna(value)
    except Exception:
        return value is None
