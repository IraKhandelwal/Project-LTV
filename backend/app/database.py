import json
import os
import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

DATABASE_PATH = Path(os.getenv("LTV_DATABASE_PATH", Path(__file__).resolve().parents[1] / "ltv.sqlite3"))


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS zones (
                zone_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                density_persons_m2 REAL NOT NULL,
                headcount INTEGER NOT NULL,
                capacity_percent REAL NOT NULL,
                level TEXT NOT NULL,
                confidence REAL NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                zone_id TEXT NOT NULL,
                level TEXT NOT NULL,
                reading TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                acknowledged INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                object_id TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                details TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('operator', 'supervisor', 'administrator'))
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS retention_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS watchlist_authorizations (
                authorization_id TEXT PRIMARY KEY,
                reference TEXT NOT NULL,
                authorized_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                revoked_at TEXT
            );
            CREATE TABLE IF NOT EXISTS camera_setup (
                setup_id TEXT PRIMARY KEY,
                camera_name TEXT NOT NULL,
                source_name TEXT NOT NULL,
                frame_width INTEGER NOT NULL,
                frame_height INTEGER NOT NULL,
                ground_reference TEXT,
                zones TEXT NOT NULL,
                updated_by TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        db.executemany(
            "INSERT OR IGNORE INTO retention_settings VALUES (?, ?)",
            [("logs_days", 7), ("clips_days", 7), ("unmatched_face_hours", 24)],
        )
        current_settings = {row["setting_key"]: row["setting_value"] for row in
                            db.execute("SELECT setting_key, setting_value FROM retention_settings").fetchall()}
        audit_cutoff = (datetime.now(timezone.utc) - timedelta(days=current_settings["logs_days"])).isoformat()
        db.execute("DELETE FROM audit_events WHERE occurred_at < ?", (audit_cutoff,))
        if db.execute("SELECT COUNT(*) FROM zones").fetchone()[0] == 0:
            now = datetime.now(timezone.utc).isoformat()
            db.executemany(
                "INSERT INTO zones VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    ("north-gate", "North Gate", 2.6, 1480, 62, "yellow", 0.91, now),
                    ("main-plaza", "Main Plaza", 1.4, 920, 38, "green", 0.96, now),
                ],
            )
        if db.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 0:
            db.execute(
                "INSERT INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "demo-001",
                    "north-gate",
                    "yellow",
                    "2.6 persons/m²",
                    "Review flow and consider approved guidance.",
                    0,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        if db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            password = os.getenv("LTV_DEMO_PASSWORD", "demo-only-change-me")
            for username, role in (
                ("operator", "operator"),
                ("supervisor", "supervisor"),
                ("administrator", "administrator"),
            ):
                salt = secrets.token_bytes(16)
                digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
                db.execute(
                    "INSERT INTO users VALUES (?, ?, ?)",
                    (username, f"{salt.hex()}:{digest.hex()}", role),
                )


def get_zones() -> list["ZoneReading"]:
    from .main import AlertLevel, ZoneReading

    with connect() as db:
        rows = db.execute("SELECT * FROM zones ORDER BY zone_id").fetchall()
    return [
        ZoneReading(
            zone_id=row["zone_id"], name=row["name"],
            density_persons_m2=row["density_persons_m2"], headcount=row["headcount"],
            capacity_percent=row["capacity_percent"], level=AlertLevel(row["level"]),
            confidence=row["confidence"], updated_at=datetime.fromisoformat(row["updated_at"]),
        )
        for row in rows
    ]


def get_alerts() -> list["Alert"]:
    from .main import Alert, AlertLevel

    with connect() as db:
        rows = db.execute("SELECT * FROM alerts ORDER BY created_at DESC").fetchall()
    return [
        Alert(
            id=row["id"], zone_id=row["zone_id"], level=AlertLevel(row["level"]),
            reading=row["reading"], recommendation=row["recommendation"],
            acknowledged=bool(row["acknowledged"]), created_at=datetime.fromisoformat(row["created_at"]),
        )
        for row in rows
    ]


def authenticate(username: str, password: str) -> dict[str, str] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        return None
    salt_hex, expected_hex = row["password_hash"].split(":", maxsplit=1)
    actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 310_000)
    if not secrets.compare_digest(actual.hex(), expected_hex):
        return None
    return {"username": row["username"], "role": row["role"]}


def create_session(username: str, expires_at: datetime) -> str:
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with connect() as db:
        db.execute("DELETE FROM sessions WHERE expires_at <= ?", (datetime.now(timezone.utc).isoformat(),))
        db.execute("INSERT INTO sessions VALUES (?, ?, ?)", (token_hash, username, expires_at.isoformat()))
        db.execute(
            "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid4()), username, "auth.login", username, datetime.now(timezone.utc).isoformat(), "{}"),
        )
    return token


def get_session(token: str) -> dict[str, str] | None:
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        row = db.execute(
            "SELECT users.username, users.role FROM sessions "
            "JOIN users ON users.username = sessions.username "
            "WHERE sessions.token_hash = ? AND sessions.expires_at > ?",
            (token_hash, now),
        ).fetchone()
    return {"username": row["username"], "role": row["role"]} if row else None


def delete_session(token: str, username: str) -> None:
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with connect() as db:
        db.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
        db.execute(
            "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid4()), username, "auth.logout", username, datetime.now(timezone.utc).isoformat(), "{}"),
        )


def record_acknowledgement(alert_id: str, username: str) -> "Alert | None":
    from .main import Alert, AlertLevel

    with connect() as db:
        row = db.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        if row is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        db.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
        db.execute(
            "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid4()), username, "alert.acknowledge", alert_id, now,
             json.dumps({"zone_id": row["zone_id"], "level": row["level"]})),
        )
        row = db.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    return Alert(
        id=row["id"], zone_id=row["zone_id"], level=AlertLevel(row["level"]),
        reading=row["reading"], recommendation=row["recommendation"],
        acknowledged=bool(row["acknowledged"]), created_at=datetime.fromisoformat(row["created_at"]),
    )


def get_audit_events(limit: int = 100) -> list[dict[str, str]]:
    with connect() as db:
        rows = db.execute(
            "SELECT event_id, actor, action, object_id, occurred_at, details "
            "FROM audit_events ORDER BY occurred_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def record_audit_action(actor: str, action: str, object_id: str, details: dict[str, object]) -> None:
    with connect() as db:
        db.execute(
            "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid4()), actor, action, object_id, datetime.now(timezone.utc).isoformat(), json.dumps(details)),
        )


def get_camera_setup() -> dict[str, object] | None:
    from .calibration import describe_calibration

    with connect() as db:
        row = db.execute("SELECT * FROM camera_setup WHERE setup_id = 'local-replay'").fetchone()
    if row is None:
        return None
    result = dict(row)
    result["ground_reference"] = json.loads(result["ground_reference"]) if result["ground_reference"] else None
    result["zones"] = json.loads(result["zones"])
    result["calibration"] = describe_calibration(result)
    result["calibration_status"] = (
        "draft" if result["calibration"]["status"] == "experimental-area-mapping"
        else "uncalibrated"
    )
    return result


def save_camera_setup(setup: dict[str, object], actor: str) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    reference = setup.get("ground_reference")
    with connect() as db:
        db.execute(
            "INSERT INTO camera_setup VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(setup_id) DO UPDATE SET camera_name=excluded.camera_name, "
            "source_name=excluded.source_name, frame_width=excluded.frame_width, "
            "frame_height=excluded.frame_height, ground_reference=excluded.ground_reference, "
            "zones=excluded.zones, updated_by=excluded.updated_by, updated_at=excluded.updated_at",
            ("local-replay", setup["camera_name"], setup["source_name"], setup["frame_width"],
             setup["frame_height"], json.dumps(reference) if reference else None,
             json.dumps(setup["zones"]), actor, now),
        )
    return get_camera_setup() or {}


def search_audit_events(query: str = "", action: str = "", start: str = "", end: str = "",
                        limit: int = 5000) -> list[dict[str, str]]:
    clauses: list[str] = []
    values: list[str | int] = []
    if query:
        clauses.append("(actor LIKE ? OR action LIKE ? OR object_id LIKE ? OR details LIKE ?)")
        term = f"%{query}%"
        values.extend([term, term, term, term])
    if action:
        clauses.append("action = ?")
        values.append(action)
    if start:
        clauses.append("occurred_at >= ?")
        values.append(start)
    if end:
        clauses.append("occurred_at <= ?")
        values.append(end)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect() as db:
        setting = db.execute(
            "SELECT setting_value FROM retention_settings WHERE setting_key = 'logs_days'"
        ).fetchone()
        if setting:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=setting[0])).isoformat()
            db.execute("DELETE FROM audit_events WHERE occurred_at < ?", (cutoff,))
        rows = db.execute(
            f"SELECT event_id, actor, action, object_id, occurred_at, details FROM audit_events "
            f"{where} ORDER BY occurred_at DESC LIMIT ?", (*values, limit)
        ).fetchall()
    return [dict(row) for row in rows]


def get_retention_settings() -> dict[str, int]:
    with connect() as db:
        return {row["setting_key"]: row["setting_value"] for row in
                db.execute("SELECT setting_key, setting_value FROM retention_settings").fetchall()}


def set_retention_settings(settings: dict[str, int], actor: str) -> dict[str, int]:
    with connect() as db:
        for key, value in settings.items():
            db.execute("UPDATE retention_settings SET setting_value = ? WHERE setting_key = ?", (value, key))
        db.execute(
            "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid4()), actor, "retention.update", "retention_settings", datetime.now(timezone.utc).isoformat(), json.dumps(settings)),
        )
        logs_days = settings.get("logs_days")
        if logs_days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=logs_days)).isoformat()
            db.execute("DELETE FROM audit_events WHERE occurred_at < ?", (cutoff,))
    return get_retention_settings()


def list_watchlist_authorizations() -> list[dict[str, str | bool]]:
    now = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM watchlist_authorizations ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
    return [{**dict(row), "active": not row["revoked_at"] and row["expires_at"] > now} for row in rows]


def create_watchlist_authorization(reference: str, actor: str, expires_at: datetime) -> dict[str, str | bool]:
    authorization_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    expiry = expires_at.astimezone(timezone.utc).isoformat()
    with connect() as db:
        db.execute(
            "INSERT INTO watchlist_authorizations VALUES (?, ?, ?, ?, ?, NULL)",
            (authorization_id, reference, actor, created_at, expiry),
        )
        db.execute(
            "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid4()), actor, "watchlist.authorization.create", authorization_id, created_at,
             json.dumps({"reference": reference, "expires_at": expiry})),
        )
    return {"authorization_id": authorization_id, "reference": reference, "authorized_by": actor,
            "created_at": created_at, "expires_at": expiry, "revoked_at": "", "active": True}


def revoke_watchlist_authorization(authorization_id: str, actor: str) -> bool:
    now = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        cursor = db.execute(
            "UPDATE watchlist_authorizations SET revoked_at = ? WHERE authorization_id = ? AND revoked_at IS NULL",
            (now, authorization_id),
        )
        if cursor.rowcount:
            db.execute(
                "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid4()), actor, "watchlist.authorization.revoke", authorization_id, now, "{}"),
            )
    return bool(cursor.rowcount)
