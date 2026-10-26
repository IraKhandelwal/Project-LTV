from datetime import datetime, timedelta, timezone
from enum import StrEnum
import csv
from io import BytesIO, StringIO
from xml.sax.saxutils import escape

from fastapi import Cookie, Depends, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

from . import database, simulation


class AlertLevel(StrEnum):
    green = "green"
    yellow = "yellow"
    amber = "amber"
    red = "red"


class ZoneReading(BaseModel):
    zone_id: str
    name: str
    density_persons_m2: float
    headcount: int
    capacity_percent: float
    level: AlertLevel
    confidence: float
    updated_at: datetime


class Alert(BaseModel):
    id: str
    zone_id: str
    level: AlertLevel
    reading: str
    recommendation: str
    acknowledged: bool = False
    escalated: bool = False
    created_at: datetime


class LoginRequest(BaseModel):
    username: str
    password: str


class RetentionSettings(BaseModel):
    logs_days: int
    clips_days: int
    unmatched_face_hours: int


class WatchlistAuthorizationRequest(BaseModel):
    reference: str
    expires_at: datetime


class DemoScenarioRequest(BaseModel):
    scenario: str


class NormalizedPoint(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class ZonePolygon(BaseModel):
    zone_id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=80)
    points: list[NormalizedPoint] = Field(min_length=3, max_length=64)


class GroundReference(BaseModel):
    corners: list[NormalizedPoint] = Field(min_length=4, max_length=4)
    width_m: float = Field(gt=0, le=5000)
    depth_m: float = Field(gt=0, le=5000)


class CameraSetupRequest(BaseModel):
    camera_name: str = Field(min_length=1, max_length=100)
    source_name: str = Field(default="", max_length=255)
    frame_width: int = Field(ge=1, le=20000)
    frame_height: int = Field(ge=1, le=20000)
    ground_reference: GroundReference | None = None
    zones: list[ZonePolygon] = Field(default_factory=list, max_length=20)


app = FastAPI(title="LTV Operations API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.on_event("startup")
def initialize_database() -> None:
    database.initialize()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "simulated"}


def current_user(ltv_session: str | None = Cookie(default=None)) -> dict[str, str]:
    user = database.get_session(ltv_session) if ltv_session else None
    if user is None:
        raise HTTPException(status_code=401, detail="Sign-in required")
    return user


def require_roles(*allowed: str):
    def check(user: dict[str, str] = Depends(current_user)) -> dict[str, str]:
        if user["role"] not in allowed:
            raise HTTPException(status_code=403, detail="This role cannot access this action")
        return user
    return check


@app.post("/api/auth/login")
def login(payload: LoginRequest, response: Response) -> dict[str, str]:
    user = database.authenticate(payload.username, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    expires_at = datetime.now(timezone.utc) + timedelta(hours=8)
    token = database.create_session(user["username"], expires_at)
    response.set_cookie(
        "ltv_session", token, httponly=True, samesite="lax", secure=False,
        max_age=8 * 60 * 60, path="/",
    )
    return user


@app.post("/api/auth/logout")
def logout(response: Response, user: dict[str, str] = Depends(current_user),
           ltv_session: str = Cookie()) -> dict[str, str]:
    database.delete_session(ltv_session, user["username"])
    response.delete_cookie("ltv_session", path="/")
    return {"status": "signed out"}


@app.get("/api/auth/me")
def me(user: dict[str, str] = Depends(current_user)) -> dict[str, str]:
    return user


@app.get("/api/zones", response_model=list[ZoneReading])
def list_zones(_: dict[str, str] = Depends(current_user)) -> list[ZoneReading]:
    return simulation.get_state()["zones"]


@app.get("/api/alerts", response_model=list[Alert])
def list_alerts(_: dict[str, str] = Depends(current_user)) -> list[Alert]:
    return simulation.get_state()["alerts"]


@app.get("/api/demo/state")
def demo_state(_: dict[str, str] = Depends(current_user)) -> dict:
    return simulation.get_state()


@app.get("/api/camera/setup")
def get_camera_setup(_: dict[str, str] = Depends(current_user)) -> dict[str, object] | None:
    return database.get_camera_setup()


@app.put("/api/camera/setup")
def save_camera_setup(
    setup: CameraSetupRequest,
    user: dict[str, str] = Depends(require_roles("operator", "supervisor", "administrator")),
) -> dict[str, object]:
    if (setup.ground_reference is not None
            and (setup.frame_width <= 0 or setup.frame_height <= 0)):
        raise HTTPException(status_code=422, detail="A video frame size is required for calibration points")
    saved = database.save_camera_setup(setup.model_dump(mode="json"), user["username"])
    database.record_audit_action(
        user["username"], "camera.setup.save", "local-replay",
        {"zone_count": len(setup.zones), "ground_reference_saved": setup.ground_reference is not None},
    )
    return saved


def record_synthetic_alerts(alerts: list[dict]) -> None:
    for alert in alerts:
        database.record_audit_action(
            "synthetic-engine", "alert.created", alert["id"],
            {"zone_id": alert["zone_id"], "level": alert["level"], "reading": alert["reading"], "synthetic": True},
        )


@app.post("/api/demo/scenario")
def start_demo_scenario(
    payload: DemoScenarioRequest,
    user: dict[str, str] = Depends(require_roles("operator", "supervisor", "administrator")),
) -> dict:
    try:
        state = simulation.start_scenario(payload.scenario)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    record_synthetic_alerts(state["alerts"])
    database.record_audit_action(user["username"], "demo.scenario.start", payload.scenario,
                                 {"synthetic": True})
    return state


@app.post("/api/demo/stop")
def stop_demo_scenario(
    user: dict[str, str] = Depends(require_roles("operator", "supervisor", "administrator")),
) -> dict:
    state = simulation.stop_scenario()
    database.record_audit_action(user["username"], "demo.scenario.stop", "synthetic",
                                 {"synthetic": True})
    return state


@app.post("/api/demo/tick")
def advance_demo_scenario(_: dict[str, str] = Depends(current_user)) -> dict:
    previous_alerts = simulation.get_state()["alerts"]
    previous_ids = {alert["id"] for alert in previous_alerts}
    previously_escalated = {alert["id"] for alert in previous_alerts if alert["escalated"]}
    state = simulation.advance()
    record_synthetic_alerts([alert for alert in state["alerts"] if alert["id"] not in previous_ids])
    for alert in state["alerts"]:
        if alert["escalated"] and alert["id"] not in previously_escalated:
            database.record_audit_action(
                "synthetic-engine", "alert.escalated", alert["id"],
                {"zone_id": alert["zone_id"], "level": alert["level"], "synthetic": True},
            )
    return state


@app.post("/api/alerts/{alert_id}/acknowledge", response_model=Alert)
def acknowledge_alert(alert_id: str, user: dict[str, str] = Depends(require_roles("operator", "supervisor", "administrator"))) -> Alert:
    simulated = simulation.acknowledge(alert_id)
    if simulated is not None:
        database.record_audit_action(user["username"], "alert.acknowledge", alert_id,
                                     {"zone_id": simulated["zone_id"], "level": simulated["level"], "synthetic": True})
        return Alert(**simulated)
    alert = database.record_acknowledgement(alert_id, user["username"])
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@app.get("/api/audit")
def list_audit_events(
    q: str = Query(default="", max_length=200), action: str = "",
    start: str = "", end: str = "",
    user: dict[str, str] = Depends(require_roles("supervisor", "administrator")),
) -> list[dict[str, str]]:
    database.record_audit_action(user["username"], "audit.search", "audit_events",
                                 {"q": q, "action": action, "start": start, "end": end})
    return database.search_audit_events(q, action, start, end)


def audit_filters(q: str, action: str, start: str, end: str) -> list[dict[str, str]]:
    return database.search_audit_events(q, action, start, end)


@app.get("/api/audit/export.csv")
def export_audit_csv(
    q: str = "", action: str = "", start: str = "", end: str = "",
    user: dict[str, str] = Depends(require_roles("supervisor", "administrator")),
) -> StreamingResponse:
    database.record_audit_action(user["username"], "audit.export.csv", "audit_events",
                                 {"q": q, "action": action, "start": start, "end": end})
    output = StringIO(newline="")
    fields = ["event_id", "occurred_at", "actor", "action", "object_id", "details"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for event in audit_filters(q, action, start, end):
        safe_event = {
            key: (f"'{value}" if isinstance(value, str) and value[:1] in ("=", "+", "-", "@") else value)
            for key, value in event.items()
        }
        writer.writerow(safe_event)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=ltv-audit-log.csv"})


@app.get("/api/audit/export.pdf")
def export_audit_pdf(
    q: str = "", action: str = "", start: str = "", end: str = "",
    user: dict[str, str] = Depends(require_roles("supervisor", "administrator")),
) -> StreamingResponse:
    database.record_audit_action(user["username"], "audit.export.pdf", "audit_events",
                                 {"q": q, "action": action, "start": start, "end": end})
    events = audit_filters(q, action, start, end)
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=landscape(letter), title="LTV Audit Log")
    styles = getSampleStyleSheet()
    body_style = styles["BodyText"]
    body_style.fontSize = 7
    body_style.leading = 9
    rows = [[Paragraph(label, styles["Heading5"]) for label in
             ["Time (UTC)", "User", "Action", "Record", "Details"]]]
    rows.extend([[Paragraph(escape(event[key]), body_style) for key in
                  ["occurred_at", "actor", "action", "object_id", "details"]]
                 for event in events])
    table = Table(rows, repeatRows=1, colWidths=[108, 80, 115, 105, 220])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263b37")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c8d2ce")),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f4")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    document.build([Paragraph("LTV Audit Log", styles["Title"]), Spacer(1, 10), table])
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf",
                             headers={"Content-Disposition": "attachment; filename=ltv-audit-log.pdf"})


@app.get("/api/admin/retention", response_model=RetentionSettings)
def get_retention(_: dict[str, str] = Depends(require_roles("administrator"))) -> dict[str, int]:
    return database.get_retention_settings()


@app.put("/api/admin/retention", response_model=RetentionSettings)
def update_retention(settings: RetentionSettings,
                     user: dict[str, str] = Depends(require_roles("administrator"))) -> dict[str, int]:
    if not (1 <= settings.logs_days <= 30 and 1 <= settings.clips_days <= 30
            and 1 <= settings.unmatched_face_hours <= 168):
        raise HTTPException(status_code=422, detail="Retention values are outside allowed limits")
    return database.set_retention_settings(settings.model_dump(), user["username"])


@app.get("/api/admin/watchlist-authorizations")
def get_watchlist_authorizations(
    _: dict[str, str] = Depends(require_roles("administrator")),
) -> list[dict[str, str | bool]]:
    return database.list_watchlist_authorizations()


@app.post("/api/admin/watchlist-authorizations")
def authorize_watchlist(
    authorization: WatchlistAuthorizationRequest,
    user: dict[str, str] = Depends(require_roles("administrator")),
) -> dict[str, str | bool]:
    now = datetime.now(timezone.utc)
    expiry = authorization.expires_at
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    if not authorization.reference.strip() or expiry <= now:
        raise HTTPException(status_code=422, detail="Provide a written authorization reference and future expiry")
    return database.create_watchlist_authorization(authorization.reference.strip(), user["username"],
                                                   expiry)


@app.delete("/api/admin/watchlist-authorizations/{authorization_id}")
def revoke_watchlist(
    authorization_id: str,
    user: dict[str, str] = Depends(require_roles("administrator")),
) -> dict[str, str]:
    if not database.revoke_watchlist_authorization(authorization_id, user["username"]):
        raise HTTPException(status_code=404, detail="Active authorization not found")
    return {"status": "authorization revoked"}
