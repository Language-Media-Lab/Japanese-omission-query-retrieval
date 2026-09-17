"""Verify public frozen results without API access."""
from pathlib import Path
import json, hashlib
R=Path(__file__).resolve().parent.parent
read=lambda p:json.loads((R/p).read_text())
e=read('results/review/experiments.json');c=read('results/review/candidate_ranks.json');a=read('results/review/analysis.json')
data=[json.loads(x) for x in (R/'data/annotation_filtered_276.jsonl').read_text().splitlines()]
assert data==e['dataset'] and len(data)==276
assert sorted(i for f in e['folds'] for i in f)==list(range(276))
checked=0
for method,result in e['tuned']['count'].items():
 for f,test in enumerate(e['folds']):
  dev=sorted(set(range(276))-set(test))
  assert not ({data[i]['gold_chunk_id'] for i in test}&{data[i]['gold_chunk_id'] for i in dev})
  candidates=c['count'][method]
  scores=[sum(1/x['ranks']['omission'][i] for i in dev)/len(dev) for x in candidates]
  selected=next(i for i,x in enumerate(scores) if x>=max(scores)-1e-12)
  assert selected==result['selections'][f]['candidate_index']
  for v in ['original','omission']:
   if v not in result:continue
   assert all(result[v]['ranks'][i]==candidates[selected]['ranks'][v][i] for i in test)
   checked+=len(test)
 for v in ['original','omission']:
  if v not in result:continue
  ranks=result[v]['ranks'];s=result[v]['summary']
  assert abs(sum(1/x for x in ranks)/276-s['mrr'])<1e-12
  for k in [1,3,5,10]:assert abs(sum(x<=k for x in ranks)/276-s['recall'][f'R@{k}'])<1e-12
for name in ['single','category','oracle']:
 x=a['routing']['count'][name];assert abs(sum(1/r for r in x['ranks'])/276-x['summary']['mrr'])<1e-12
for mode in ['generic','category_conditioned']:
 rows=[json.loads(x) for x in (R/f'data/query2doc/{mode}.jsonl').read_text().splitlines()]
 assert len(rows)==276 and all('response_id' not in x for x in rows)
 assert {x['eval_id']:x['omission_query'] for x in rows}=={x['eval_id']:x['omission_query'] for x in data}
tex=(R/'paper/paper_lualatex_v4.tex').read_text();assert 'RQ4' not in tex
for method in ['Dense','BM25','RM3','RRF','Hybrid','Query2doc','CategoryQuery2doc']:
 assert format(e['tuned']['count'][method]['omission']['summary']['mrr'],'.4f') in tex
manifest=read('release_manifest.json')
for path,h in manifest['sha256'].items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==h,path
print('OK:',checked,'selected ranks; 276 queries; 552 generated texts; all main MRRs;',len(manifest['sha256']),'hashes')
