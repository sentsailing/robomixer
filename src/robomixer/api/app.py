"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI

from robomixer.api.routes import songs, transitions

app = FastAPI(
    title="Robomixer",
    description="AI-powered DJ transition detection and mixing",
    version="0.1.0",
)

app.include_router(songs.router, prefix="/songs", tags=["songs"])
app.include_router(transitions.router, prefix="/transitions", tags=["transitions"])
