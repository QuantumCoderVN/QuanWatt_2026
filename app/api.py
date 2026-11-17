"""ASGI / Vercel entrypoint — re-exports the FastAPI app from ``backend``."""
from backend.api import app

try:
    from mangum import Mangum

    handler = Mangum(app)
except ImportError:
    handler = None
