"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from karuth.api.routes.planner import router as planner_router

app = FastAPI(title="Karuth", version="0.1.0")
app.include_router(planner_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
