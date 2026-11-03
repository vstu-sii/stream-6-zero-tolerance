"""Run from repository root. No automatic retries or hidden model calls."""
import argparse, json, platform, time, uuid
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
import sympy as sp
ROOT=Path(__file__).resolve().parent

def read(name): return json.loads((ROOT/name).read_text(encoding='utf-8'))
def write(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
class OllamaAdapter:
    def __init__(self,config): self.config=config
    def request(self,path,body=None):
        req=Request(self.config['base_url']+path,data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=self.config['timeout_seconds']) as r: return json.load(r)
    def metadata(self):
        tags=self.request('/api/tags')
        found=[m for m in tags['models'] if m['name']==self.config['model']]
        if len(found)!=1: raise ValueError('Configured model not found in /api/tags; no download attempted')
        return {'installed_model':found[0],'server':self.request('/api/version')}
    def prepare(self,messages,seed):
        return {'model':self.config['model'],'messages':messages,'think':False,'stream':False,'format':'json','options':dict(self.config['options'],seed=seed)}
    def generate(self,request):
        raw=self.request('/api/chat',request)
        return {'text':raw['message']['content'],'model':raw['model'],'metrics':{k:raw.get(k) for k in ['prompt_eval_count','eval_count','total_duration','load_duration','prompt_eval_duration','eval_duration','done_reason']},'raw':raw}

def validate_cases(cases):
    x=sp.Symbol('x'); env={'x':x,'Matrix':sp.Matrix,'diff':sp.diff,'limit':sp.limit}
    results=[]
    for c in cases:
        # Only trusted, checked-in reference expressions; never evaluate model output.
        a=sp.sympify(c['reference']['expression'],locals=env); b=sp.sympify(c['reference']['expected'],locals=env)
        ok=(a-b).applyfunc(sp.simplify)==sp.zeros(*a.shape) if isinstance(a,sp.MatrixBase) else sp.simplify(a-b)==0
        if not ok: raise AssertionError(c['id'])
        results.append({'case_id':c['id'],'reference_verified':bool(ok),'value':str(a)})
    return results

def validate_output(obj,stage,step_count):
    if not isinstance(obj,dict) or not isinstance(obj.get('internal'),dict): raise ValueError('Missing internal object')
    d=obj['internal']; step=d.get('first_error_step'); kind=d.get('error_type')
    if 'first_error_step' not in d or (step is not None and (type(step) is not int or not 1<=step<=step_count)): raise ValueError('Invalid step')
    if kind not in ('conceptual','arithmetic','none'): raise ValueError('Invalid error_type')
    if (kind=='none') != (step is None): raise ValueError('Inconsistent step/type')
    if not isinstance(d.get('diagnosis'),str) or not d['diagnosis'].strip(): raise ValueError('Missing diagnosis')
    if stage!='C_diagnose' and (not isinstance(obj.get('student_message'),str) or not obj['student_message'].strip()): raise ValueError('Missing student_message')

def run(case_ids=('L1','M1'),repeat=1,start_repeat=0):
    cfg=read('protocols/v2/config.json'); cases=read('cases.json'); prompts=read('protocols/v2/prompts.json')
    if cfg['think'] is not False: raise ValueError('This experiment requires think=false')
    adapters={'ollama':OllamaAdapter}; adapter=adapters[cfg['adapter']](cfg)
    selected=[next(c for c in cases if c['id']==i) for i in case_ids]
    references=validate_cases(cases); metadata=adapter.metadata()
    folder=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]); folder.mkdir(parents=True)
    write(folder/'manifest.json',{'protocol_version':'v2','runner_sha256':__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),'config':cfg,'metadata':metadata,'python':platform.python_version(),'sympy':sp.__version__,'platform':platform.platform(),'hardware_user_reported':{'gpu':'RTX 5060 Ti 16 GB','ram':'32 GB DDR5','cpu':'i5-14400F'},'case_ids':list(case_ids),'repeat':repeat,'start_repeat':start_repeat,'cases':cases,'prompts':prompts,'references':references,'duration_unit':'nanoseconds for Ollama durations; seconds for wall_seconds','cost':'No token tariff; electricity and hardware cost not measured'})
    summaries=[]
    for case in selected:
      context=json.dumps({k:case[k] for k in ['task','student_steps']},ensure_ascii=False)
      for n in range(start_repeat,start_repeat+repeat):
       for variant in ['A','B','C']:
        diagnosis=None; records=[]
        for stage in (['C_diagnose','C_explain'] if variant=='C' else [variant]):
         user=prompts[stage]+'\n'+context
         if stage=='B': user+='\nРџР°РјСЏС‚РєР°: '+case['material']
         if stage=='C_explain': user+='\nР”РёР°РіРЅРѕСЃС‚РёРєР°: '+json.dumps(diagnosis,ensure_ascii=False)
         messages=[{'role':'system','content':prompts['system']},{'role':'user','content':user}]
         record={'case_id':case['id'],'variant':variant,'stage':stage,'repeat':n+1,'timestamp':datetime.now(timezone.utc).isoformat(),'request':adapter.prepare(messages,cfg['options']['seed']+n)}
         # Persist exact request BEFORE dispatch, including interrupted calls.
         with (folder/'requests.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record,ensure_ascii=False)+'\n')
         start=time.perf_counter()
         try:
          record['response']=adapter.generate(record['request'])
          record['parsed']=json.loads(record['response']['text'])
          obj=record['parsed']
          validate_output(obj,stage,len(case['student_steps']))
          record['error']=None; diagnosis=obj['internal']
         except Exception as exc: record['error']={'type':type(exc).__name__,'message':str(exc)}
         record['wall_seconds']=time.perf_counter()-start
         with (folder/'responses.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record,ensure_ascii=False)+'\n')
         records.append(record)
         if record['error']: break
        final=records[-1]; parsed=final.get('parsed',{}); obj=parsed.get('internal',{})
        summaries.append({'case_id':case['id'],'variant':variant,'repeat':n+1,'success':all(r['error'] is None for r in records),'first_error_correct':obj.get('first_error_step')==case['reference']['first_error_step'] if not final['error'] else None,'error_type_correct':obj.get('error_type')==case['reference']['error_type'] if not final['error'] else None,'protocol_version':'v2','student_message':parsed.get('student_message'),'manual_review':None,'wall_seconds':sum(r['wall_seconds'] for r in records),'prompt_tokens':sum(r.get('response',{}).get('metrics',{}).get('prompt_eval_count',0) or 0 for r in records),'output_tokens':sum(r.get('response',{}).get('metrics',{}).get('eval_count',0) or 0 for r in records)})
        write(folder/'summary.json',summaries)
        print(case['id'],variant,json.dumps(summaries[-1],ensure_ascii=False),flush=True)
    return folder
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--cases',nargs='+',default=['L1','M1']); parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--start-repeat',type=int,default=0)
    args=parser.parse_args(); print(run(args.cases,args.repeat,args.start_repeat))
