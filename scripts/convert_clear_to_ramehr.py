import json,pandas as pd
from pathlib import Path
CLEAR_CSV=Path("data/clear/output.csv")
CLEAR_JSONL=Path("data/clear/output.jsonl")
OUT_DIR=Path("data/ramehr");OUT_DIR.mkdir(parents=True,exist_ok=True)
code_name={"D_AF":"atrial fibrillation","D_STROKE":"stroke","P_PM":"pacemaker implantation","P_THOR":"thoracentesis (pleural tap)","P_REOP":"reoperation for bleeding or tamponade","RISK_LOW":"EuroSCORE II low risk","RISK_MED":"EuroSCORE II medium risk","RISK_HIGH":"EuroSCORE II high risk"}
(OUT_DIR/"code_name_map.json").write_text(json.dumps(code_name,indent=2))
def risk_bin(es2):
    try:v=float(str(es2).replace("%",""))
    except:return None
    if v<2:return "RISK_LOW"
    if v<6:return "RISK_MED"
    return "RISK_HIGH"
if CLEAR_CSV.exists():
    df=pd.read_csv(CLEAR_CSV)
elif CLEAR_JSONL.exists():
    rows=[json.loads(x) for x in CLEAR_JSONL.read_text(encoding="utf-8").splitlines() if x.strip()]
    df=pd.DataFrame(rows)
else:
    raise SystemExit("no output.csv or output.jsonl found in data/clear/")
visits,aug=[],[]
for _,r in df.iterrows():
    pid=str(r.get("patient_id",r.get("id","UNK")))
    adm=str(r.get("admission","")).replace(".","-").replace("/","-")
    dis=str(r.get("exit","")).replace(".","-").replace("/","-")
    diagnoses,procedures=[],[]
    if str(r.get("a.fib","0"))=='1':diagnoses.append("D_AF")
    if str(r.get("stroke","0"))=='1':diagnoses.append("D_STROKE")
    if str(r.get("pm","0"))=='1':procedures.append("P_PM")
    if str(r.get("pleural.tap","0"))=='1':procedures.append("P_THOR")
    if str(r.get("reoperation","0"))=='1':procedures.append("P_REOP")
    rb=risk_bin(r.get("es.2",""))
    if rb:diagnoses.append(rb)
    visits.append({"patient_id":pid,"visit_id":adm,"time":{"admission":adm,"discharge":dis},"diagnoses":diagnoses,"medications":[],"procedures":procedures})
    ctx=(f"Admitted {adm}, discharged {dis}. EuroSCORE II {r.get('es.2','')}. "+("Stroke; " if str(r.get('stroke','0'))=='1' else "No stroke; ")+("AF; " if str(r.get('a.fib','0'))=='1' else "No AF; ")+("pacemaker; " if str(r.get('pm','0'))=='1' else "")+("pleural tap; " if str(r.get('pleural.tap','0'))=='1' else "")+("reoperation; " if str(r.get('reoperation','0'))=='1' else "")).strip()
    aug.append({"patient_id":pid,"visit_id":adm,"context_text":ctx})
with open(OUT_DIR/"visits.jsonl","w",encoding="utf-8") as f:
    for v in visits:f.write(json.dumps(v,ensure_ascii=False)+"\n")
with open(OUT_DIR/"patient_aug_text.jsonl","w",encoding="utf-8") as f:
    for a in aug:f.write(json.dumps(a,ensure_ascii=False)+"\n")
print("OK -> data/ramehr/visits.jsonl, patient_aug_text.jsonl, code_name_map.json")
