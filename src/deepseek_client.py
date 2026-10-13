"""Portable opt-in LLM transport. No credentials or historical request ledger included."""
import json,math,os,time,urllib.request,urllib.error
from pathlib import Path
from datetime import datetime,timezone
OUT=Path('runs/llm')
REQUEST_LIMIT=0
class ApiFailure(RuntimeError):pass
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ApiFailure('API redirect rejected')
def validate_vector(result):
    v=result.get('rod_length_scales')
    if not isinstance(v,list) or len(v)!=8 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not .9<=x<=1.1 for x in v):
        raise ApiFailure('Expected eight finite rod scales in [0.9, 1.1]')
    return list(map(float,v))
def completion(messages,purpose='morphology_proposal',max_tokens=768):
    if REQUEST_LIMIT<=0:raise ApiFailure('API disabled. Supply --allow-api and a positive --max-api-calls.')
    key=os.environ.get('DEEPSEEK_API_KEY','')
    if not key:raise ApiFailure('Set DEEPSEEK_API_KEY locally; do not put it in code.')
    if not 1<=max_tokens<=768 or len(json.dumps(messages))>30000:raise ApiFailure('Prompt/token budget exceeded')
    OUT.mkdir(parents=True,exist_ok=True);ledger=OUT/'requests.jsonl';lock=OUT/'request.lock'
    try:fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:raise ApiFailure('Another API request is active; inspect the request lock') from None
    def log(**row):
        with ledger.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),**row))+'\n')
    try:
        used=sum(json.loads(s).get('event')=='started' for s in ledger.read_text().splitlines()) if ledger.exists() else 0
        if used>=REQUEST_LIMIT:raise ApiFailure('Explicit API request cap reached; no retry or algorithm substitution')
        model=os.environ.get('DEEPSEEK_MODEL','deepseek-chat')
        payload=dict(model=model,messages=messages,max_tokens=max_tokens,response_format={'type':'json_object'},stream=False)
        req=urllib.request.Request('https://api.deepseek.com/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
        log(event='started',number=used+1,purpose=purpose,model=model);start=time.monotonic()
        try:
            with urllib.request.build_opener(NoRedirect).open(req,timeout=60) as r:reply=json.loads(r.read(2*1024*1024))
            result=json.loads(reply['choices'][0]['message']['content'])
            if not isinstance(result,dict):raise ValueError('Expected JSON object')
        except (OSError,ValueError,KeyError,TypeError,ApiFailure) as e:
            log(event='failed',number=used+1,error_type=type(e).__name__)
            raise ApiFailure('LLM request failed; no automatic retry; credential/response body not logged') from None
        log(event='completed',number=used+1,model=reply.get('model'),usage=reply.get('usage'),elapsed_seconds=time.monotonic()-start)
        return result,reply.get('usage',{}),reply.get('model',model)
    finally:os.close(fd);lock.unlink()
