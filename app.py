
"""SplitSnap API (FastAPI). Run:  python app.py   ->  http://localhost:8000   (docs at /docs)
Env: SPLITSNAP_CKPT=<ckpt_best dir> for the real model, SPLITSNAP_REMOTE=<url> for a model served elsewhere."""
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import threading
from core import MODE, SplitError, scan_bill, run_split, warmup

app = FastAPI(title="SplitSnap")
HERE = os.path.dirname(os.path.abspath(__file__))


@app.exception_handler(SplitError)
async def _split_err(_, e): return JSONResponse({"error": str(e)}, status_code=400)


@app.exception_handler(Exception)
async def _any_err(_, e):
    import traceback; traceback.print_exc()          
    return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.get("/api/health")
def health(): return {"ok": True, "extractor": MODE}


@app.post("/api/extract")
async def extract_route(image: UploadFile = File(None)):
    data = await image.read() if image else b""
    if MODE != "mock" and not data:
        raise HTTPException(400, "No image received.")
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "Image is over 15 MB.")
    return scan_bill(data)


@app.get("/api/sample")
def sample_route(): return scan_bill(b"", sample=True)   


@app.post("/api/split")
async def split_route(request: Request): return run_split(await request.json())


@app.get("/")
def index(): return FileResponse(os.path.join(HERE, "static", "index.html"))


if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=warmup, daemon=True).start()
    print(f"SplitSnap · extractor = {MODE}")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))