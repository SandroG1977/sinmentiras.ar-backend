import json
import re
import hashlib
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from app.services.rag_service import LocalRAGService

nb = json.loads(Path("app/preparacion.ipynb").read_text(encoding="utf-8"))
cell_src = next(
    "".join(c.get("source", []))
    for c in nb["cells"]
    if "def split_ley_semantico_articulos" in "".join(c.get("source", []))
)

start = cell_src.find("heading_start_re = re.compile(")
end = cell_src.find(
    "chunks_estructurados, notes_by_ref = split_ley_semantico_articulos(texto_fuente)"
)
assert start != -1 and end != -1, "No se pudo extraer bloque del splitter"

ns = {"re": re}
exec(cell_src[start:end], ns)
nb_split = ns["split_ley_semantico_articulos"]

url = "https://servicios.infoleg.gob.ar/infolegInternet/anexos/25000-29999/25552/texact.htm"
r = requests.get(url, timeout=30)
r.raise_for_status()
r.encoding = r.apparent_encoding or "utf-8"
texto = re.sub(
    r"\s+", " ", BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
).strip()

nb_chunks, nb_notes = nb_split(texto)
be_chunks, be_notes = LocalRAGService()._split_ley_semantico_articulos(texto)


def norm_chunks(items):
    return [
        {
            "kind": c.get("kind"),
            "title": c.get("title"),
            "text": c.get("text"),
            "path": c.get("path"),
            "articulo_nro": c.get("articulo_nro"),
            "articulo_suffix": c.get("articulo_suffix"),
            "articulo_ref": c.get("articulo_ref"),
            "capitulo": c.get("capitulo"),
        }
        for c in items
    ]


def norm_notes(items):
    return {str(k): list(v) for k, v in sorted(items.items(), key=lambda x: str(x[0]))}


nbc = norm_chunks(nb_chunks)
bec = norm_chunks(be_chunks)
nbn = norm_notes(nb_notes)
ben = norm_notes(be_notes)

report = {
    "ley": 20744,
    "same_chunks": nbc == bec,
    "same_notes": nbn == ben,
    "nb_chunks": len(nbc),
    "be_chunks": len(bec),
    "nb_notes_refs": len(nbn),
    "be_notes_refs": len(ben),
    "nb_chunks_sha256": hashlib.sha256(
        json.dumps(nbc, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest(),
    "be_chunks_sha256": hashlib.sha256(
        json.dumps(bec, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest(),
}

print(json.dumps(report, ensure_ascii=False, indent=2))

if report["same_chunks"] and report["same_notes"]:
    raise SystemExit(0)

max_len = min(len(nbc), len(bec))
for i in range(max_len):
    if nbc[i] != bec[i]:
        print("primer_diff_chunk_idx:", i)
        break
else:
    if len(nbc) != len(bec):
        print("primer_diff_chunk_idx:", max_len)

nb_keys = set(nbn.keys())
be_keys = set(ben.keys())
print("keys_solo_notebook:", sorted(nb_keys - be_keys)[:20])
print("keys_solo_backend:", sorted(be_keys - nb_keys)[:20])
raise SystemExit(1)
