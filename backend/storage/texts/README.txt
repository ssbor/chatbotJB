Sem se ukládají .txt soubory, které se indexují pro vyhledávání (můžete je sem vložit ručně, nebo vznikají automaticky z PDF).

Dva způsoby práce:
1) Auto-konverze z PDF (doporučeno)
	- PDF soubor stačí vložit do složky ../pdfs
	- Běží-li server, PDF se během pár sekund převede do .txt do této složky a hned se naindexuje.
	- Stav indexu: GET /ingested

2) Ruční .txt
	- Každý dokument uložte jako samostatný .txt (UTF-8 doporučeno; fungují i CP1250/Windows-1250 a Latin-1).
	- Poté zavolejte endpoint /ingest_texts (POST) nebo restartujte server – při startu se .txt automaticky naindexují.
	- Stav indexu lze ověřit na /ingested.

Poznámky:
- Pokud je PDF naskenovaný obrázek bez textu, převod bude prázdný; je potřeba nejprve použít OCR.
- Ingest přes cesty k PDF/TXT souborům z disku zůstává funkční (/ingest_local).
