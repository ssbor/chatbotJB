from __future__ import annotations

import os
import io
import uuid
import threading
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import hashlib
import urllib.request
import urllib.error

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- Paths & Storage ----------
BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
CHROMA_DIR = STORAGE_DIR / "chroma"
TEXTS_DIR = STORAGE_DIR / "texts"
INBOX_PDFS_DIR = STORAGE_DIR / "pdfs"
FRONTEND_DIR = BASE_DIR.parent / "frontend"
INGEST_LOG = STORAGE_DIR / "ingested.json"
CONVERT_LOG = STORAGE_DIR / "converted_pdfs.json"
INGEST_CONFIG = Path(__file__).resolve().parent / "ingest_paths.json"

# Optional generation backends (OpenAI via env or local Ollama)
GEN_BACKEND = (os.getenv("GEN_BACKEND", "").strip().lower() or "")  # values: "", "ollama"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()

for p in [STORAGE_DIR, UPLOADS_DIR, CHROMA_DIR, TEXTS_DIR, INBOX_PDFS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
if not INGEST_LOG.exists():
    INGEST_LOG.write_text(json.dumps({"files": {}}, ensure_ascii=False, indent=2), encoding="utf-8")
if not CONVERT_LOG.exists():
    CONVERT_LOG.write_text(json.dumps({"files": {}}, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- Embeddings & Vector Store ----------
# Multilingual model (lepší pro češtinu)
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"


class LocalVectorStore:
    def __init__(self, dir_path: Path, model_name: str):
        self.dir = dir_path
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_json = self.dir / "index.json"
        self.emb_path = self.dir / "embeddings.npy"
        self.model_name = model_name

        self._model: Optional[SentenceTransformer] = None
        self.ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

        self._load()

    def _load(self) -> None:
        if self.index_json.exists() and self.emb_path.exists():
            try:
                data = json.loads(self.index_json.read_text(encoding="utf-8"))
                if data.get("model_name") == self.model_name:
                    self.ids = data.get("ids", [])
                    self.documents = data.get("documents", [])
                    self.metadatas = data.get("metadatas", [])
                    self.embeddings = np.load(self.emb_path)
            except Exception:
                # start fresh
                self.ids, self.documents, self.metadatas, self.embeddings = [], [], [], None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _persist(self) -> None:
        meta = {
            "model_name": self.model_name,
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
        }
        self.index_json.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        if self.embeddings is None:
            np.save(self.emb_path, np.zeros((0, 384), dtype=np.float32))
        else:
            np.save(self.emb_path, self.embeddings)

    def _maybe_prefix_docs(self, docs: List[str]) -> List[str]:
        name = (self.model_name or "").lower()
        if "e5" in name:
            return [f"passage: {d}" for d in docs]
        return docs

    def _maybe_prefix_queries(self, queries: List[str]) -> List[str]:
        name = (self.model_name or "").lower()
        if "e5" in name:
            return [f"query: {q}" for q in queries]
        return queries

    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]):
        if not ids:
            return
        docs_pref = self._maybe_prefix_docs(documents)
        embs = self.model.encode(docs_pref, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        # Robust převod na numpy float32 (někdy se může vrátit torch.Tensor)
        try:
            import torch  # type: ignore
            if isinstance(embs, torch.Tensor):
                embs = embs.detach().cpu().numpy()
        except Exception:
            pass
        embs = np.asarray(embs, dtype=np.float32)
        if self.embeddings is None or getattr(self.embeddings, "size", 0) == 0:
            self.embeddings = embs
        else:
            self.embeddings = np.vstack([self.embeddings, embs])
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self._persist()

    def query(self, query_texts: List[str], n_results: int = 5) -> Dict[str, Any]:
        if self.embeddings is None or len(self.documents) == 0:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}
        q_pref = self._maybe_prefix_queries(query_texts)
        q_embs = self.model.encode(q_pref, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        try:
            import torch  # type: ignore
            if isinstance(q_embs, torch.Tensor):
                q_embs = q_embs.detach().cpu().numpy()
        except Exception:
            pass
        q_embs = np.asarray(q_embs, dtype=np.float32)
        # cosine similarity equals dot product due to normalization
        sims = np.dot(q_embs, self.embeddings.T)
        topk_idx = np.argpartition(-sims, kth=min(n_results, sims.shape[1]-1), axis=1)[:, :n_results]
        # sort within top-k
        results_docs, results_metas, results_ids = [], [], []
        for i in range(sims.shape[0]):
            idxs = topk_idx[i]
            sorted_local = idxs[np.argsort(-sims[i, idxs])]
            docs = [self.documents[j] for j in sorted_local]
            metas = [self.metadatas[j] for j in sorted_local]
            ids = [self.ids[j] for j in sorted_local]
            results_docs.append(docs)
            results_metas.append(metas)
            results_ids.append(ids)
        return {"documents": results_docs, "metadatas": results_metas, "ids": results_ids}


# Initialize local vector store
vector_store = LocalVectorStore(CHROMA_DIR, EMBED_MODEL_NAME)

# ---------- FastAPI App ----------
app = FastAPI(title="PDF QA Chatbot", version="0.1.0")

# Allow local dev origins (frontend served by this app too)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pozn.: statický frontend namontujeme až po definici API rout, aby je nepřekryl


# ---------- Helpers ----------
def read_pdf_text(file_path: Path) -> str:
    try:
        reader = PdfReader(str(file_path))
        texts: List[str] = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            if page_text:
                texts.append(page_text)
        return "\n".join(texts)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Chyba při čtení PDF: {e}")


def read_text_file(file_path: Path) -> str:
    # Načti .txt s detekcí běžných kódování na Windows (UTF-8, UTF-8-SIG, CP1250, Latin-1)
    encodings = ["utf-8", "utf-8-sig", "cp1250", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            return file_path.read_text(encoding=enc, errors="ignore")
        except Exception as e:  # pragma: no cover
            last_err = e
            continue
    raise HTTPException(status_code=400, detail=f"Chyba při čtení TXT: {last_err}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    # Simple character-based chunking with overlap
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def add_chunks_to_collection(doc_id: str, source_name: str, chunks: List[str]) -> int:
    if not chunks:
        return 0
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
    vector_store.add(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def _load_ingest_log() -> Dict[str, Any]:
    try:
        return json.loads(INGEST_LOG.read_text(encoding="utf-8"))
    except Exception:
        return {"files": {}}


def _save_ingest_log(data: Dict[str, Any]) -> None:
    INGEST_LOG.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _fingerprint(path: Path) -> str:
    try:
        st = path.stat()
        h = hashlib.sha1()
        h.update(str(path).encode("utf-8", errors="ignore"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
        return h.hexdigest()
    except Exception:
        return hashlib.sha1(str(path).encode("utf-8", errors="ignore")).hexdigest()


# --- Optional: local generation via Ollama ---
def _ollama_generate(prompt: str, *, model: Optional[str] = None, host: Optional[str] = None, timeout: float = 30.0) -> Optional[str]:
    """Call local Ollama HTTP API to generate text.
    Returns the generated response text or None on failure.
    """
    try:
        m = (model or OLLAMA_MODEL or "mistral").strip()
        h = (host or OLLAMA_HOST or "http://127.0.0.1:11434").rstrip("/")
        url = f"{h}/api/generate"
        payload = {
            "model": m,
            "prompt": prompt,
            "stream": False,
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(raw)
            out = (data.get("response") or "").strip()
            return out or None
    except Exception:
        return None


# --- Auto-convert PDFs dropped into INBOX_PDFS_DIR ---
def _load_convert_log() -> Dict[str, Any]:
    try:
        return json.loads(CONVERT_LOG.read_text(encoding="utf-8"))
    except Exception:
        return {"files": {}}


def _save_convert_log(data: Dict[str, Any]) -> None:
    CONVERT_LOG.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _convert_pdf_to_txt(pdf_path: Path) -> Optional[Path]:
    """Convert a PDF to TXT in TEXTS_DIR; returns path to TXT or None if empty."""
    try:
        text = read_pdf_text(pdf_path)
    except HTTPException:
        return None
    text = (text or "").strip()
    # Skip writing empty outputs
    if not text:
        return None
    out_name = pdf_path.stem + ".txt"
    out_path = TEXTS_DIR / out_name
    try:
        out_path.write_text(text, encoding="utf-8")
        return out_path
    except Exception:
        return None


def _process_pdf_if_needed(pdf_path: Path) -> Optional[Path]:
    """Convert and ingest if new/changed; return TXT path if ingested."""
    log = _load_convert_log()
    files_log: Dict[str, Any] = log.get("files", {})
    fp = _fingerprint(pdf_path)
    prev = files_log.get(str(pdf_path))
    if prev and prev.get("fingerprint") == fp:
        return None  # up-to-date

    txt_path = _convert_pdf_to_txt(pdf_path)
    if not txt_path or not txt_path.exists():
        # Still update fingerprint to avoid tight loops on unreadable PDFs
        files_log[str(pdf_path)] = {"fingerprint": fp, "status": "empty-or-failed"}
        log["files"] = files_log
        _save_convert_log(log)
        return None

    # Update convert log then ingest the new TXT
    files_log[str(pdf_path)] = {
        "fingerprint": fp,
        "txt": str(txt_path),
        "converted_at": int(time.time()),
    }
    log["files"] = files_log
    _save_convert_log(log)

    try:
        ingest_local_paths([str(txt_path)])
    except Exception:
        # Ingestion failure shouldn't crash watcher
        pass
    return txt_path


def _watch_pdfs_folder(interval_sec: float = 5.0) -> None:
    """Periodic scanner for new/changed PDFs in INBOX_PDFS_DIR."""
    while True:
        try:
            for pdf in INBOX_PDFS_DIR.glob("*.pdf"):
                _process_pdf_if_needed(pdf)
        except Exception:
            # Never crash the loop
            pass
        time.sleep(max(1.0, interval_sec))


def ingest_local_paths(paths: List[str]) -> Dict[str, Any]:
    log = _load_ingest_log()
    files_log: Dict[str, Any] = log.get("files", {})

    total_chunks = 0
    ingested: List[Dict[str, Any]] = []

    for p in paths:
        pth = Path(p)
        if not pth.exists() or not pth.is_file():
            ingested.append({"path": str(p), "status": "skipped", "reason": "soubor nenalezen"})
            continue

        fp = _fingerprint(pth)
        prev = files_log.get(str(pth))
        # Reingest, pokud se změnil soubor NEBO změnil embeddingový model
        if prev and prev.get("fingerprint") == fp and prev.get("model_name") == EMBED_MODEL_NAME:
            ingested.append({"path": str(pth), "status": "up-to-date", "doc_id": prev.get("doc_id"), "chunks": prev.get("chunks", 0)})
            continue

        # (Re)ingest – podporujeme PDF i TXT
        ext = pth.suffix.lower()
        if ext == ".pdf":
            text = read_pdf_text(pth)
        elif ext == ".txt":
            text = read_text_file(pth)
        else:
            ingested.append({"path": str(pth), "status": "skipped", "reason": "podporované jsou pouze .pdf a .txt"})
            continue
        chunks = chunk_text(text)
        # Stable doc_id based on path (not fingerprint) so links remain stable across small edits
        stable_hash = hashlib.sha1(str(pth).encode("utf-8", errors="ignore")).hexdigest()[:16]
        doc_id = f"local_{stable_hash}"
        added = add_chunks_to_collection(doc_id, pth.name, chunks)
        files_log[str(pth)] = {
            "fingerprint": fp,
            "doc_id": doc_id,
            "chunks": added,
            "name": pth.name,
            "model_name": EMBED_MODEL_NAME,
        }
        total_chunks += added
        ingested.append({"path": str(pth), "status": "ingested", "doc_id": doc_id, "chunks": added})

    # persist
    log["files"] = files_log
    _save_ingest_log(log)

    return {"status": "ok", "ingested": ingested, "total_chunks": total_chunks}


# ---------- Schemas ----------
class AskRequest(BaseModel):
    question: str
    k: int = 5
    openai_api_key: Optional[str] = None
    model: Optional[str] = "gpt-4o-mini"
    max_tokens: int = 400


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    generation_backend: Optional[str] = None  # 'openai' | 'ollama' | 'none'
    used_model: Optional[str] = None
    generation_status: Optional[str] = None   # 'ok' | 'fallback'
    generation_error: Optional[str] = None


# ---------- Routes ----------
@app.get("/health")
def health() -> Dict[str, Any]:
    """Basic health plus generation/backend status so users can verify OpenAI/Ollama wiring."""
    openai_key_present = bool((os.getenv("OPENAI_API_KEY") or "").strip())
    if openai_key_present:
        gen = "openai"
    elif GEN_BACKEND == "ollama":
        gen = "ollama"
    else:
        gen = "none"

    return {
        "status": "ok",
        "embedding_model": EMBED_MODEL_NAME,
        "generation_backend": gen,
        "openai_key_present": openai_key_present,
        "ollama": {"host": OLLAMA_HOST, "model": OLLAMA_MODEL} if GEN_BACKEND == "ollama" else None,
    }


@app.on_event("startup")
def on_startup() -> None:
    # Automaticky načíst cesty z konfiguračního souboru, pokud existuje
    try:
        if INGEST_CONFIG.exists():
            data = json.loads(INGEST_CONFIG.read_text(encoding="utf-8"))
            paths = data.get("paths") or []
            if paths:
                ingest_local_paths(paths)
        # Navíc automaticky naindexuj všechny .txt v TEXTS_DIR (pokud existují)
        txt_paths = [str(p) for p in TEXTS_DIR.glob("*.txt")]
        if txt_paths:
            ingest_local_paths(txt_paths)
    except Exception:
        # Nezastavuj aplikaci, pokud ingest selže
        pass
    # Spusť pozadí skener PDF složky (auto-konverze -> TXT -> ingest)
    try:
        t = threading.Thread(target=_watch_pdfs_folder, kwargs={"interval_sec": 5.0}, daemon=True)
        t.start()
    except Exception:
        # Pokud se watcher nepodaří spustit, ignoruj
        pass


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Nahrajte prosím alespoň jeden PDF soubor.")

    total_chunks = 0
    ingested_docs: List[Dict[str, Any]] = []

    for uf in files:
        if not uf.filename or not uf.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Soubor musí být PDF: {uf.filename}")

        # Save upload to disk
        raw = await uf.read()
        doc_id = str(uuid.uuid4())
        save_path = UPLOADS_DIR / f"{doc_id}_{uf.filename}"
        with open(save_path, "wb") as f:
            f.write(raw)
        text = read_pdf_text(save_path)
        chunks = chunk_text(text)
        added = add_chunks_to_collection(doc_id, uf.filename, chunks)
        total_chunks += added
        ingested_docs.append({"filename": uf.filename, "doc_id": doc_id, "chunks": added})

    return {"status": "ok", "ingested": ingested_docs, "total_chunks": total_chunks}


@app.post("/ingest_local")
async def ingest_local(body: Dict[str, Any]) -> Dict[str, Any]:
    paths = body.get("paths") if isinstance(body, dict) else None
    if not paths or not isinstance(paths, list):
        raise HTTPException(status_code=400, detail="Pošlete pole 'paths' s cestami k souborům (.pdf nebo .txt).")
    return ingest_local_paths(paths)


@app.post("/ingest_texts")
def ingest_texts() -> Dict[str, Any]:
    """
    Načte všechny .txt soubory z adresáře backend/storage/texts a přidá je do indexu.
    Není potřeba uvádět absolutní cesty.
    """
    paths = [str(p) for p in TEXTS_DIR.glob("*.txt")]
    if not paths:
        return {"status": "ok", "ingested": [], "total_chunks": 0, "note": "Ve složce /storage/texts nejsou žádné .txt soubory."}
    return ingest_local_paths(paths)


@app.get("/ingested")
def list_ingested() -> Dict[str, Any]:
    log = _load_ingest_log()
    files = [
        {"path": k, **(v if isinstance(v, dict) else {})} for k, v in (log.get("files", {}) or {}).items()
    ]
    return {"files": files, "count": len(files)}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Chybí dotaz.")

    # Retrieve relevant chunks
    try:
        result = vector_store.query(query_texts=[req.question], n_results=max(1, min(req.k, 10)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba ve vyhledávání: {e}")

    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]

    # Build context
    context = "\n\n".join(docs)

    # Try to generate an answer with a model (OpenAI by API key or local Ollama), else fallback
    answer_text = None
    generation_backend = "none"
    used_model: Optional[str] = None
    generation_status = "fallback"
    generation_error: Optional[str] = None
    openai_key = (req.openai_api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if openai_key:
        try:
            from openai import OpenAI

            client_oa = OpenAI(api_key=openai_key)
            system_prompt = (
                "Jsi asistent, který odpovídá stručně,lidsky a přesně na základě poskytnutého kontextu. "
                ""
            )
            user_prompt = (
                f"Kontext:\n{context}\n\nOtázka: {req.question}\n"
                "Odpověz česky, buď věcný a uveď, pokud je odpověď nejistá."
            )
            chat = client_oa.chat.completions.create(
                model=req.model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=req.max_tokens,
            )
            answer_text = (chat.choices[0].message.content or "").strip()
            generation_backend = "openai"
            used_model = (req.model or "gpt-4o-mini")
            generation_status = "ok"
        except Exception as e:
            answer_text = None
            generation_backend = "openai"
            used_model = (req.model or "gpt-4o-mini")
            generation_error = f"OpenAI error: {e}"
    elif GEN_BACKEND == "ollama":
        # Local lightweight generation via Ollama (no cloud key required)
        system_prompt = (
            "Jsi asistent, který odpovídá stručně a přesně pouze na základě poskytnutého kontextu. "
            "Pokud odpověď nelze najít v kontextu, řekni, že ji v dokumentu nemáš."
        )
        full_prompt = (
            f"{system_prompt}\n\n"
            f"Kontext:\n{context}\n\nOtázka: {req.question}\nOdpověď:"
        )
        try:
            answer_text = _ollama_generate(full_prompt)
            if answer_text:
                generation_backend = "ollama"
                used_model = OLLAMA_MODEL
                generation_status = "ok"
            else:
                generation_backend = "ollama"
                used_model = OLLAMA_MODEL
                generation_error = "Ollama returned empty response"
        except Exception as e:
            generation_backend = "ollama"
            used_model = OLLAMA_MODEL
            generation_error = f"Ollama error: {e}"

    if not answer_text:
        # Jednoduchý extraktivní fallback: vyber nejrelevantnější věty z top dokumentů
        import re
        def sentences(text: str) -> List[str]:
            parts = re.split(r"(?<=[\.!?])\s+", text)
            return [p.strip() for p in parts if p.strip()]

        q_terms = set(re.findall(r"\w{3,}", req.question.lower()))
        best_sent = None
        best_score = -1
        for d in docs[:5]:
            for s in sentences(d)[:30]:
                s_terms = set(re.findall(r"\w{3,}", s.lower()))
                # skóre: prostý překryv tokenů + délková penalizace
                overlap = len(q_terms & s_terms)
                score = overlap / max(1, len(q_terms)) + min(len(s), 300) / 1000.0
                if score > best_score:
                    best_score = score
                    best_sent = s
        if best_sent:
            answer_text = best_sent
        else:
            # fallback na první chunk
            top = docs[0] if docs else ""
            preview = (top[:700] + "…") if len(top) > 700 else top
            answer_text = preview or "Nebyly nalezeny žádné relevantní informace v dokumentech."

    sources = []
    for i, meta in enumerate(metas):
        sources.append({
            "source": meta.get("source"),
            "doc_id": meta.get("doc_id"),
            "chunk_index": meta.get("chunk_index"),
            "id": ids[i] if i < len(ids) else None,
        })

    return AskResponse(
        answer=answer_text,
        sources=sources,
        generation_backend=generation_backend,
        used_model=used_model,
        generation_status=generation_status,
        generation_error=generation_error,
    )


# --- Mount static frontend last so it doesn't overshadow API routes ---
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")



