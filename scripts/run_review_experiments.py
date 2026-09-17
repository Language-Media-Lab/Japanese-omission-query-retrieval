"""Offline review experiments; preserve submission artifacts and reuse caches."""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from run_bm25_retrieval_baseline import BM25, read_jsonl, tokenize
from run_rm3_retrieval import rm3_query_model
from run_rrf_retrieval import load_cached_embeddings
from run_weighted_hybrid_retrieval import make_group_folds, minmax_rows

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'omission_query_dataset'
OUT = ROOT / 'outputs/review'
GRID = sorted(itertools.product([1.2, 0.6, 1.8, 2.4], [0.75, 0, 0.25, 0.5, 1]), key=lambda p:(abs(p[0]-1.2),abs(p[1]-.75),p))
ALPHAS = sorted([i/10 for i in range(11)],key=lambda a:(abs(a-.5),a))


def expansion_weights(query, pseudo, repetitions):
    """No artificial character bigrams across repetitions or component boundaries."""
    weights = Counter({t: n * repetitions for t, n in Counter(tokenize(query)).items()})
    weights.update(tokenize(pseudo))
    return weights


def query_matrix(weights, vocabulary):
    rows, cols, values = [], [], []
    for i, counts in enumerate(weights):
        for t, w in counts.items():
            if t in vocabulary:
                rows.append(i); cols.append(vocabulary[t]); values.append(w)
    return csr_matrix((values, (rows, cols)), shape=(len(weights), len(vocabulary)))


def contribution_matrix(model, vocabulary):
    rows, cols, values = [], [], []
    for term, postings in model.postings.items():
        for j, tf in postings:
            norm = 1-model.b + model.b*model.lengths[j]/model.average_length
            rows.append(vocabulary[term]); cols.append(j)
            values.append(model.idf[term]*tf*(model.k1+1)/(tf+model.k1*norm))
    return csr_matrix((values, (rows, cols)), shape=(len(vocabulary),len(model.lengths)))


def rankings(scores):
    # Corpus is sorted by ID before scoring; stable sort implements the tie rule.
    order = np.argsort(-scores, axis=1, kind='stable')
    inverse = np.empty_like(order)
    np.put_along_axis(inverse, order, np.arange(1,scores.shape[1]+1)[None,:], axis=1)
    return order, inverse


def metrics(ranks):
    r = np.asarray(ranks)
    return dict(mrr=float(np.mean(1/r)), recall={f'R@{k}':float(np.mean(r<=k)) for k in [1,3,5,10]})


def choose(candidate_ranks, indices):
    scores = (1/candidate_ranks[:,indices]).mean(axis=1)
    return int(np.flatnonzero(scores >= scores.max()-1e-12)[0])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = read_jsonl(ROOT/'data/annotation_filtered_276.jsonl')
    corpus = sorted(read_jsonl(DATA/'jdocqa_510_corpus.jsonl'),key=lambda r:r['chunk_id'])
    ids = [r['eval_id'] for r in dataset]
    chunk_ids = [r['chunk_id'] for r in corpus]
    gold = np.array([chunk_ids.index(r['gold_chunk_id']) for r in dataset])
    folds = make_group_folds(dataset,5,20260802)
    old = json.loads((ROOT/'results/submission/weighted_hybrid_retrieval_results.json').read_text())
    old_fold = {r['eval_id']:r['fold'] for r in old['omission']['details']}
    assert all(old_fold[ids[i]] == f+1 for f,ix in enumerate(folds) for i in ix)
    assert len(set(ids))==276 and len(set(chunk_ids))==510
    tokens = {v:[tokenize(r[field]) for r in dataset] for v,field in [('original','original_question'),('omission','omission_query')]}
    document_tokens = [tokenize(r['text']) for r in corpus]
    base = BM25(document_tokens)
    vocabulary = {t:i for i,t in enumerate(sorted(base.idf))}
    collection_counts = sum(base.term_frequencies, Counter())
    collection_length = sum(collection_counts.values())
    # Load original ordering of cached documents, then reorder by ID.
    source_corpus = read_jsonl(DATA/'jdocqa_510_corpus.jsonl')
    vectors = load_cached_embeddings(DATA/'embedding_cache','text-embedding-3-small',[r['text'] for r in source_corpus])
    position = {r['chunk_id']:i for i,r in enumerate(source_corpus)}
    vectors = vectors[[position[k] for k in chunk_ids]]
    dense = {}
    for v,field in [('original','original_question'),('omission','omission_query')]:
        q = load_cached_embeddings(DATA/'embedding_cache','text-embedding-3-small',[r[field] for r in dataset])
        dense[v] = q @ vectors.T
    dense_norm = {v:minmax_rows(s) for v,s in dense.items()}
    dense_ranks = {v:rankings(s)[1] for v,s in dense.items()}
    pseudo = {}
    for name,file in [('Query2doc','generic.jsonl'),('CategoryQuery2doc','category_conditioned.jsonl')]:
        rows = read_jsonl(ROOT/'data/query2doc'/file)
        assert len(rows)==len(ids) and {r['eval_id'] for r in rows}==set(ids)
        keyed = {r['eval_id']:r for r in rows}
        assert all(keyed[r['eval_id']]['omission_query']==r['omission_query'] for r in dataset)
        pseudo[name] = [keyed[i]['pseudo_document'] for i in ids]
    qmat = {(mode,v):query_matrix([Counter(t) if mode=='count' else dict.fromkeys(t,1) for t in ts],vocabulary)
            for mode in ['binary','count'] for v,ts in tokens.items()}
    expanded = {(name,n):query_matrix([expansion_weights(r['omission_query'],p,n) for r,p in zip(dataset,ps)],vocabulary)
                for name,ps in pseudo.items() for n in [1,5]}
    candidates = {mode:{name:[] for name in ['Dense','BM25','RM3','RRF','Hybrid','Query2doc','CategoryQuery2doc']}
                  for mode in ['binary','count']}
    fixed = {}
    def save_candidate(mode,name,params,scores, fixed_name=None):
        ranks = {}
        for v,s in scores.items():
            order,inv = rankings(s)
            ranks[v] = inv[np.arange(len(ids)),gold].tolist()
            if fixed_name:
                fixed.setdefault(fixed_name,dict(configuration=params))[v] = dict(summary=metrics(ranks[v]),details=[
                    dict(eval_id=ids[i],gold_chunk_id=chunk_ids[gold[i]],gold_rank=int(ranks[v][i]),
                         top_chunk_ids=[chunk_ids[j] for j in order[i,:10]],top_scores=s[i,order[i,:10]].tolist()) for i in range(len(ids))])
        if mode:
            candidates[mode][name].append(dict(parameters=params,ranks=ranks))
    for mode in candidates:
        save_candidate(mode,'Dense',{},dense,mode+'_Dense')
    for grid_index,(k1,b) in enumerate(GRID):
        print(f'BM25 grid {grid_index+1}/{len(GRID)}: k1={k1}, b={b}',flush=True)
        model = BM25(document_tokens,k1,b)
        term_scores = contribution_matrix(model,vocabulary)
        for mode in candidates:
            params = dict(k1=k1,b=b,query_term_weight=mode)
            bm = {v:(qmat[mode,v]@term_scores).toarray() for v in tokens}
            save_candidate(mode,'BM25',params,bm,mode+'_BM25' if grid_index==0 else None)
            norm = {v:minmax_rows(s) for v,s in bm.items()}
            for alpha in ALPHAS:
                scores = {v:alpha*dense_norm[v]+(1-alpha)*norm[v] for v in tokens}
                save_candidate(mode,'Hybrid',dict(**params,alpha=alpha),scores)
            branks = {v:rankings(s)[1] for v,s in bm.items()}
            for k in [60,10,30,100]:
                scores = {v:1/(k+dense_ranks[v])+1/(k+branks[v]) for v in tokens}
                save_candidate(mode,'RRF',dict(**params,rrf_k=k),scores,mode+'_RRF' if grid_index==0 and k==60 else None)
            # Reuse feedback models across interpolation weights.
            for fb_terms in [10,5,20]:
                models = {v:[rm3_query_model(ts,bm[v][i].tolist(),model,10,fb_terms,0.,1000.,collection_counts,collection_length)[0]
                              for i,ts in enumerate(tokens[v])] for v in tokens}
                fb_scores = {v:(query_matrix(ms,vocabulary)@term_scores).toarray() for v,ms in models.items()}
                original_scores = {v:(qmat['count',v]@term_scores).toarray()/np.array([len(ts) for ts in tokens[v]])[:,None] for v in tokens}
                for weight in [.5,.25,.75]:
                    scores = {v:weight*original_scores[v]+(1-weight)*fb_scores[v] for v in tokens}
                    save_candidate(mode,'RM3',dict(**params,fb_docs=10,fb_terms=fb_terms,original_weight=weight,mu=1000),scores,
                                   mode+'_RM3' if grid_index==0 and fb_terms==10 and weight==.5 else None)
            for name in pseudo:
                # binary panel includes legacy BM25 comparator, corrected expansion is count in both panels.
                scores = {'omission':(expanded[name,5]@term_scores).toarray()}
                save_candidate(mode,name,dict(k1=k1,b=b,query_term_weight='count',repetitions=5,composition='separate_token_bags'),scores,
                               mode+'_'+name if grid_index==0 else None)
        if grid_index==0:
            for name,ps in pseudo.items():
                save_candidate(None,name,dict(repetitions=1,composition='separate_token_bags'),
                               {'omission':(expanded[name,1]@term_scores).toarray()},'count_'+name+'_1repeat')
                legacy = query_matrix([Counter(tokenize(' '.join([r['omission_query']]*5)+' [SEP] '+p)) for r,p in zip(dataset,ps)],vocabulary)
                save_candidate(None,name,dict(repetitions=5,composition='legacy_string_with_SEP',query_term_weight='count'),
                               {'omission':(legacy@term_scores).toarray()},'count_'+name+'_legacy_concat')
    # Restore submission alpha-only protocol for both query-frequency panels.
    for mode in candidates:
        subset = [c for c in candidates[mode]['Hybrid'] if c['parameters']['k1']==1.2 and c['parameters']['b']==.75]
        candidates[mode]['HybridAlphaOnly'] = subset
    outputs = {}
    all_indices = set(range(len(ids)))
    for mode,methods in candidates.items():
        outputs[mode] = {}
        for name,cs in methods.items():
            ranks = np.array([c['ranks']['omission'] for c in cs])
            selections, result = [],{v:np.zeros(len(ids),dtype=int) for v in cs[0]['ranks']}
            for f,test in enumerate(folds):
                dev = sorted(all_indices-set(test))
                selected = choose(ranks,dev)
                selections.append(dict(fold=f+1,parameters=cs[selected]['parameters'],candidate_index=selected,
                                       development_mrr=float((1/ranks[selected,dev]).mean())))
                for v in result:
                    result[v][test] = np.array(cs[selected]['ranks'][v])[test]
            outputs[mode][name] = dict(candidate_count=len(cs),selections=selections,**{
                v:dict(summary=metrics(r),ranks=r.tolist()) for v,r in result.items()})
    # Validate exact reproduction of the key submitted baselines before inference.
    checks = {}
    for name,file in [('BM25','bm25_retrieval_results.json'),('RM3','rm3_retrieval_results.json')]:
        submitted=json.loads((ROOT/'results/submission'/file).read_text())
        for v in tokens:
            expected={r['eval_id']:r['gold_rank'] for r in submitted[v]['details']}
            got=fixed['binary_'+name][v]['details']
            mismatches=sum(r['gold_rank']!=expected[r['eval_id']] for r in got)
            checks[name+'_'+v]=mismatches
            assert mismatches==0,(name,v,mismatches)
    for v in tokens:
        expected={r['eval_id']:r['gold_rank'] for r in old[v]['details']}
        mismatches=sum(r!=expected[i] for i,r in zip(ids,outputs['binary']['HybridAlphaOnly'][v]['ranks']))
        checks['HybridAlphaOnly_'+v]=mismatches
        assert mismatches==0,checks
    source_files=['data/annotation_filtered_276.jsonl','omission_query_dataset/jdocqa_510_corpus.jsonl','data/query2doc/generic.jsonl','data/query2doc/category_conditioned.jsonl']
    output=dict(configuration=dict(protocol='docs/review_20260911/experiment_protocol.md',seed=20260802,
                inputs_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files}),
                dataset=dataset,folds=folds,fixed=fixed,tuned=outputs,reproduction_checks=checks)
    (OUT/'experiments.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    (OUT/'candidate_ranks.json').write_text(json.dumps(candidates,ensure_ascii=False)+'\n')
    for mode,methods in outputs.items():
        print(mode,{n:d['omission']['summary'] for n,d in methods.items()},flush=True)


if __name__=='__main__':
    main()
