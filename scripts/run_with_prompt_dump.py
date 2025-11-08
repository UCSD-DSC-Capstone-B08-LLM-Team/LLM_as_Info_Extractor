import os,sys,runpy,importlib,types,json,time
from pathlib import Path
log_path=os.getenv("PROMPT_LOG","data/clear/prompt_dump.jsonl")
Path(log_path).parent.mkdir(parents=True,exist_ok=True)
sp=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(sp.parent))
try:
    import helper_functions as hf
    hf.send_error_email=lambda *a,**k: None
except Exception:
    pass
class AKD(dict):
    def __getitem__(self,k): return [os.getenv(k,"")]
g={'__name__':'__main__','auth_keys_data':AKD()}
real=importlib.import_module("anthropic")
m=types.ModuleType("anthropic")
for k,v in real.__dict__.items(): setattr(m,k,v)
class _Msgs:
    def __init__(self,rm): self._rm=rm
    def create(self,*a,**kw):
        try:
            out={"ts":time.time(),"model":kw.get("model"),"messages":kw.get("messages"),"extra":{k:kw[k] for k in kw if k!="messages"}}
            with open(log_path,"a",encoding="utf-8") as f: f.write(json.dumps(out,ensure_ascii=False)+"\n")
        except Exception: pass
        return self._rm.create(*a,**kw)
class AnthropicShim:
    def __init__(self,*a,**kw):
        self._c=real.Anthropic(*a,**kw)
        self.messages=_Msgs(self._c.messages)
setattr(m,"Anthropic",AnthropicShim)
sys.modules["anthropic"]=m
sys.argv=[str(sp)]+sys.argv[2:]
runpy.run_path(str(sp),init_globals=g)
