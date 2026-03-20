from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional, List
from api.auth import get_current_user
from api.database import alerts_collection
from api import detection_engine as engine

router = APIRouter(prefix="/api/detection", tags=["Detection"])

# Global cache for heavy DB counts to keep API responsive
_db_stats_cache = {"total": 0, "shoplifting": 0, "last_updated": 0}

@router.get("/state")
async def get_system_state(
    limit_alerts: int = Query(default=5, le=20),
    current_user: dict = Depends(get_current_user)
):
    try:
        now = datetime.now().timestamp()
        if now - _db_stats_cache["last_updated"] > 10:
            _db_stats_cache["total"] = await alerts_collection.estimated_document_count()
            _db_stats_cache["shoplifting"] = await alerts_collection.count_documents({"status": "Shoplifting"})
            _db_stats_cache["last_updated"] = now

        mem_alerts = list(engine.latest_alerts)[:limit_alerts]
        for a in mem_alerts:
            if isinstance(a.get("timestamp"), datetime):
                a["timestamp"] = a["timestamp"].isoformat()

        return {
            "engine": engine.get_status(),
            "db": {
                "total_alerts": _db_stats_cache["total"],
                "shoplifting_alerts": _db_stats_cache["shoplifting"]
            },
            "recent_alerts": mem_alerts,
            "server_time": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        print(f"State API Error: {e}")
        return {"error": str(e), "engine": engine.get_status(), "recent_alerts": []}

# ─── Legacy Endpoints for Compatibility ───────────────────

@router.get("/status")
async def get_status(current_user: dict = Depends(get_current_user)):
    return engine.get_status()

@router.get("/alerts")
async def get_alerts_legacy(
    limit: int = Query(default=20, le=100),
    status: Optional[str] = Query(default=None),
    current_user: dict = Depends(get_current_user)
):
    mem = list(engine.latest_alerts)
    if status:
        mem = [a for a in mem if a.get("status") == status]
    
    for a in mem:
        if isinstance(a.get("timestamp"), datetime):
            a["timestamp"] = a["timestamp"].isoformat()
            
    return mem[:limit]

@router.get("/stats")
async def get_stats_legacy(current_user: dict = Depends(get_current_user)):
    return {
        "engine": engine.get_status(),
        "db": {
            "total_alerts": _db_stats_cache["total"],
            "shoplifting_alerts": _db_stats_cache["shoplifting"]
        }
    }

# ─── Control ──────────────────────────────────────────────

@router.post("/start")
async def start_detection(source: str = Query(default="0"), user = Depends(get_current_user)):
    started = engine.start(source)
    return {"message": "Started" if started else "Running", "source": source}

@router.post("/stop")
async def stop_detection(user = Depends(get_current_user)):
    engine.stop()
    return {"message": "Stopped"}

@router.get("/stream")
async def video_stream():
    def _gen():
        while True:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + engine.get_frame() + b"\r\n")
            import time; time.sleep(0.05)
    return StreamingResponse(_gen(), media_type="multipart/x-mixed-replace; boundary=frame")
