import os,json,csv
from pathlib import Path
from anthropic import Anthropic

def choose_notes():
    a=Path("data/notes_clear.jsonl"); b=Path("data/notes.jsonl")
    return a if a.exists() else b

def coerce(o):
    def b(x):
        s=str(x).strip().lower()
        return 1 if s in {"1","true","yes","y"} else 0
    return {
        "admission": str(o.get("admission","")),
        "exit": str(o.get("exit","")),
        "es.2": o.get("es.2",""),
        "stroke": b(o.get("stroke",0)),
        "reoperation": b(o.get("reoperation",0)),
        "pm": b(o.get("pm",0)),
        "a.fib": b(o.get("a.fib",0)),
        "pleural.tap": b(o.get("pleural.tap",0)),
    }

notes_path=choose_notes()
schema=json.loads(Path("data/schema_8fields.json").read_text(encoding="utf-8"))
out_dir=Path("data/clear"); out_dir.mkdir(parents=True,exist_ok=True)
out_json=out_dir/"output.jsonl"
out_csv=out_dir/"output.csv"
dump_path=out_dir/"prompt_dump_assert.jsonl"

cli=Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
rows=[]
with notes_path.open("r",encoding="utf-8") as fin, out_json.open("w",encoding="utf-8") as fj, dump_path.open("w",encoding="utf-8") as fd:
    for line in fin:
        ex=json.loads(line)
        pid=ex.get("id") or ex.get("patient_id") or ""
        text=ex.get("text") or ex.get("note") or ex.get("note_text") or ""
        sysmsg="You extract 8 clinical variables and respond with only minified JSON matching the given schema. Dates keep original format. Booleans must be 0 or 1. If unavailable, pick the most reasonable value from the note; never return null."
        user=("Schema:\n"+json.dumps(schema,ensure_ascii=False)+"\nNote:\n"+text)
        resp=cli.messages.create(model=os.getenv("ANTHROPIC_MODEL","claude-3-5-haiku-latest"),max_tokens=512,temperature=0,system=sysmsg,messages=[{"role":"user","content":user}])
        content=resp.content[0].text if resp.content else ""
        fd.write(json.dumps({"patient_id":pid,"prompt":user,"response":content},ensure_ascii=False)+"\n")
        try:
            o=json.loads(content.strip())
        except Exception:
            s=content.find("{"); e=content.rfind("}")
            o=json.loads(content[s:e+1]) if s!=-1 and e!=-1 else {}
        o=coerce(o); o["patient_id"]=pid
        fj.write(json.dumps(o,ensure_ascii=False)+"\n")
        rows.append(o)

with out_csv.open("w",newline="",encoding="utf-8") as fc:
    w=csv.DictWriter(fc,fieldnames=["patient_id","admission","exit","es.2","stroke","reoperation","pm","a.fib","pleural.tap"])
    w.writeheader(); w.writerows(rows)

print("OK", len(rows), "rows ->", str(out_json), "and", str(out_csv))
