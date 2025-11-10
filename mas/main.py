import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mas.config.logging import configure_logging
from mas.routers import projects

# Configure logging on application startup
configure_logging()
log = structlog.get_logger()

app = FastAPI(
    title="Methodology Automation System (MAS)",
    description="An AI-powered system to automate the software development lifecycle.",
    version="0.1.0",
)

@app.middleware("http")
async def log_requests_and_handle_errors(request: Request, call_next):
    log.info("Request received", method=request.method, path=request.url.path)
    try:
        response = await call_next(request)
        log.info("Request successful", method=request.method, path=request.url.path, status_code=response.status_code)
        return response
    except Exception as e:
        log.exception("Unhandled exception", error=str(e))
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred."},
        )

@app.get("/health", summary="Health Check")
def health_check():
    """
    Simple health check endpoint to confirm the service is running.
    """
    return {"status": "ok"}

# Include the routers for different parts of the API
app.include_router(projects.router, prefix="/projects", tags=["Projects"])
