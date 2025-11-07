# Chatbot nad dokumenty (CZ)

Jednoduchý chatbot, který umí odpovídat na dotazy na základě obsahu vašich dokumentů.
Backend: FastAPI + sentence-transformers (lokální vektorový index na disku), Frontend: statická stránka se středovým vyhledávacím polem.

## Co umí
- Automaticky naindexovat PDF soubory ze seznamu cest (viz `backend/app/ingest_paths.json`)
- Nebo jednoduše naindexovat všechny `.txt` soubory vložené do `backend/storage/texts/` (bez absolutních cest)
- Zeptat se na libovolný dotaz; systém najde relevantní úryvky
- Volitelně použít OpenAI API klíč pro generování přirozené odpovědi (jinak vrátí extrakt z dokumentu)
- Volitelně lehké generování odpovědí bez cloudu přes lokální Ollama (pokud je nainstalována)

## Požadavky
- Python 3.10+
- Internet (pouze pro stažení embedding modelu při prvním spuštění; OpenAI je volitelné)

## Rychlý start (Windows PowerShell)

```powershell
# 1) Vytvoření a aktivace virtuálního prostředí (doporučeno)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Instalace závislostí
pip install -r backend\requirements.txt

# 3) (Volitelně) Upravte seznam PDF k indexaci
#    Zadejte absolutní cesty k PDF v souboru backend\app\ingest_paths.json

# 4) Spuštění serveru (ponechte terminál otevřený)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --app-dir backend

# 5) Otevřete v prohlížeči
# http://127.0.0.1:8001

## Dva způsoby ingestu (indexace)

1) PDF přes absolutní cesty (původní varianta)
	- Upravte `backend/app/ingest_paths.json` (pole `paths` s cestami k PDF souborům).
	- Při startu serveru se nové/změněné soubory automaticky naindexují.
	- Alternativně lze kdykoli POSTnout na `/ingest_local` s JSON `{ "paths": ["C:\\cesta\\soubor.pdf"] }`.

2) TXT soubory bez cest (doporučeno, jednodušší)
	- Vložte libovolné `.txt` soubory do `backend/storage/texts/` (UTF-8 doporučeno; podporovány i CP1250/Windows-1250 a Latin-1).
	- Při startu serveru se automaticky naindexují; případně zavolejte `POST /ingest_texts`.
	- Stav zkontrolujete na `GET /ingested`.
```

### (Volitelné) Smysluplnější odpovědi – generování přes OpenAI nebo Ollama

- OpenAI (cloud): v UI není potřeba nic měnit – stačí poslat dotaz, pokud backend zná klíč. Backend použije OpenAI jen tehdy, když najde klíč v proměnné prostředí `OPENAI_API_KEY` (nebo když klient pošle `openai_api_key` v JSON na `/ask`).
- Ollama (lokálně, bez cloudu): nainstalujte Ollama pro Windows, stáhněte menší model (např. `llama3.2:3b`) a při spuštění backendu nastavte proměnné prostředí:
	- `GEN_BACKEND=ollama`
	- (volitelně) `OLLAMA_MODEL=llama3.2:3b` a `OLLAMA_HOST=http://127.0.0.1:11434`

Když je aktivní OpenAI (klíč) nebo Ollama (GEN_BACKEND=ollama), endpoint `/ask` zkombinuje nalezený kontext z dokumentů a vygeneruje stručnou českou odpověď. Pokud generování není k dispozici, vrací se nejrelevantnější úryvek jako doposud.

## Použití
1. Vyberte si způsob ingestu (PDF cesty nebo TXT ve složce) – viz výše.
2. Spusťte server; při startu dojde k automatické indexaci (PDF z konfigurace, TXT ze složky `storage/texts`).
3. Otevřete aplikaci v prohlížeči, zadejte dotaz do pole uprostřed a klikněte na „Zeptat se“.
4. (API) Pro manuální spuštění ingestu bez restartu serveru lze volat `POST /ingest_local` (PDF/TXT cesty) nebo `POST /ingest_texts` (všechny TXT ve složce).

## Struktura projektu
- `backend/app/main.py` – FastAPI server, endpointy `/ask`, `/ingest_local`, `/ingest_texts`, `/ingested`, statický frontend; automatický ingest ze `ingest_paths.json` a `storage/texts`
- `backend/app/ingest_paths.json` – Seznam lokálních PDF k indexaci
- `backend/storage/` – Persistovaná data (lokální vektorový index a log ingestu)
- `backend/storage/texts/` – Sem vkládejte `.txt` soubory pro snadný ingest bez absolutních cest
- `frontend/index.html` – Jednoduchý moderní frontend

## Poznámky
- Poprvé se stáhne embedding model `intfloat/multilingual-e5-small` (cca ~120 MB).
- Pokud nepoužijete OpenAI klíč (není vyžadován), aplikace vrátí nejrelevantnější úryvek místo syntetizované odpovědi.
- Zdrojové soubory jsou uvedeny u odpovědi, abyste věděli, odkud informace pochází.

## Licence
MIT
