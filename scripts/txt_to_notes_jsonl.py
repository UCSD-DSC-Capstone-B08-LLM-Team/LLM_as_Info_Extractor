import argparse,re,json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--txt",required=True)
p.add_argument("--out",default="data/notes.jsonl")
p.add_argument("--id_prefix",default="P")
p.add_argument("--pad",type=int,default=3)
p.add_argument("--min_chars",type=int,default=80)
a=p.parse_args()

t=Path(a.txt).read_text(encoding="utf-8-sig")
t=t.replace("\r\n","\n").replace("\r","\n")

ms=list(re.finditer(r"(?m)^\s*Patient\s+(\d+)\s*$",t))
rows=[]
for i,m in enumerate(ms):
    num=int(m.group(1))
    s=m.end()
    e=ms[i+1].start() if i+1<len(ms) else len(t)
    chunk=t[s:e]
    chunk=re.sub(r"(?m)^\s*-{3,}\s*$","",chunk)
    chunk=chunk.strip()
    if len(chunk)<a.min_chars: 
        continue
    rid=f"{a.id_prefix}{str(num).zfill(a.pad)}"
    rows.append({"id":rid,"text":chunk})

Path(Path(a.out).parent).mkdir(parents=True,exist_ok=True)
with open(a.out,"w",encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(len(rows))
