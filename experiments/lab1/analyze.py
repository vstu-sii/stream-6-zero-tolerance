"""Offline aggregation only: never calls the model."""
import csv,json,itertools,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def collect():
    rows=[]; seen=set(); baseline=json.loads((ROOT/'protocols/v2/prompts.json').read_text(encoding='utf-8')); cfg=json.loads((ROOT/'protocols/v2/config.json').read_text()); cases=json.loads((ROOT/'cases.json').read_text(encoding='utf-8'))
    refs={c['id']:c['reference'] for c in cases}
    for run_name in json.loads((ROOT/'selection.json').read_text(encoding='utf-8')):
        folder=ROOT/'runs'/run_name
        m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if m.get('protocol_version')!='v2': continue
        assert m['prompts']==baseline and m['config']==cfg and m['cases']==cases
        assert m['metadata']['installed_model']['digest']=='500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41'
        records=[json.loads(x) for x in (folder/'responses.jsonl').read_text(encoding='utf-8').splitlines()]
        requests=[json.loads(x) for x in (folder/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(records)==len(requests)
        for a,b in zip(requests,records): assert a['request']==b['request'] and a['request']['think'] is False
        for s in json.loads((folder/'summary.json').read_text(encoding='utf-8')):
            calls=[r for r in records if all(r[k]==s[k] for k in ['case_id','variant','repeat'])]
            seed=calls[0]['request']['options']['seed']; key=(s['case_id'],s['variant'],seed)
            assert key not in seen, key
            seen.add(key); final=calls[-1]; internal=final.get('parsed',{}).get('internal',{})
            row=dict(s,seed=seed,run=folder.name,calls=len(calls),expected_type=refs[s['case_id']]['error_type'],predicted_type=internal.get('error_type'),predicted_step=internal.get('first_error_step'),diagnosis=internal.get('diagnosis'),errors=[r['error'] for r in calls if r['error']],stop_reasons=[r.get('response',{}).get('metrics',{}).get('done_reason') for r in calls])
            for r in calls:
                assert r['request']['options']==dict(cfg['options'],seed=seed)
                assert all(r.get('response',{}).get('metrics',{}).get(k) is not None for k in ['eval_count','prompt_eval_count'])
            rows.append(row)
    expected=set(itertools.product(refs,['A','B','C'],[42,43])); assert seen==expected,(expected-seen,seen-expected)
    hashes=json.loads((ROOT/'protocols/v2/pre_completion_hashes.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    return sorted(rows,key=lambda r:(r['case_id'],r['seed'],r['variant']))

def main():
    rows=collect(); reviews=json.loads((ROOT/'evaluation.json').read_text(encoding='utf-8')); lookup={(r['case_id'],r['variant'],r['seed']):r for r in reviews}
    assert len(lookup)==len(rows)==54
    for r in rows:r.update(lookup[r['case_id'],r['variant'],r['seed']])
    out=ROOT/'results';out.mkdir(exist_ok=True)
    (out/'evaluated.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    totals=[]
    for v in 'ABC':
        group=[r for r in rows if r['variant']==v]
        t={'variant':v,'results':len(group)}
        for k in ['first_error_correct','error_type_correct','hint_correct','no_leak','calls','prompt_tokens','output_tokens','wall_seconds']:t[k]=sum(r[k] for r in group)
        t['false_alarms']=sum(r['predicted_type']!='none' or r['predicted_step'] is not None for r in group if r['expected_type']=='none')
        totals.append(t)
    (out/'aggregate.json').write_text(json.dumps(totals,ensure_ascii=False,indent=2),encoding='utf-8')
    keys=['case_id','variant','seed','first_error_correct','error_type_correct','hint_correct','no_leak','calls','prompt_tokens','output_tokens','wall_seconds','student_message','review_note','run']
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print(json.dumps(totals,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
