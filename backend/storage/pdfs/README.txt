Sem vkládejte vaše PDF soubory.

Co se stane automaticky (když běží server):
- Backend každých pár sekund zkontroluje tuto složku.
- Nové nebo změněné PDF převede do UTF-8 .txt do složky ../texts.
- Nově vzniklé .txt okamžitě naindexuje (není třeba ručně volat /ingest_texts).

Kde jsou výstupy:
- TXT soubory: backend/storage/texts
- Stav indexu: GET http://127.0.0.1:8001/ingested

Poznámky:
- Pokud je PDF naskenovaný obrázek bez textové vrstvy, převod bude prázdný. V takovém
  případě je potřeba OCR (např. OCRmyPDF / Tesseract) a znovu vložit výsledek.
- První spuštění může chvíli trvat (stahuje se embedding model).