import json
from pathlib import Path
src=Path("data/notes.jsonl")
dst=Path("data/notes_clear.jsonl")
dst.parent.mkdir(parents=True,exist_ok=True)
w=dst.open("w",encoding="utf-8")
for line in src.open("r",encoding="utf-8"):
    line=line.strip()
    if not line: 
        continue
    o=json.loads(line)
    t=o.get("text") or o.get("note") or o.get("note_text") or ""
    o["text"]=t
    o["note"]=t
    o["note_text"]=t
    o["patient_id"]=o.get("id") or o.get("patient_id") or ""
    w.write(json.dumps(o,ensure_ascii=False)+"\n")
w.close()
print("ok")
