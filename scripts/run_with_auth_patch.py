import os,sys,runpy
from pathlib import Path
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
sys.argv=[str(sp)]+sys.argv[2:]
runpy.run_path(str(sp),init_globals=g)
