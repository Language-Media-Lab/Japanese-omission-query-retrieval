"""Cluster-aware inference and nested routing for the review experiments."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from run_review_experiments import metrics, choose
from run_weighted_hybrid_retrieval import make_group_folds

ROOT=Path(__file__).resolve().parent.parent
import argparse
_parser=argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--input-dir',type=Path,default=ROOT/'results/review')
_parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/review')
DATA=ROOT/'results/review'
OUTPUT=ROOT/'outputs/review'
DOCS=ROOT/'outputs/review'
N=100_000
SEED=20260911


def inference(delta, groups):
    grouped=defaultdict(list)
    for x,g in zip(delta,groups): grouped[g].append(x)
    sums=np.array([sum(xs) for xs in grouped.values()])
    sizes=np.array([len(xs) for xs in grouped.values()])
    mean=float(np.mean(delta))
    rng=np.random.default_rng(SEED)
    extreme=0; samples=[]
    for start in range(0,N,2000):
        m=min(2000,N-start)
        perm=rng.choice([-1.,1.],size=(m,len(sums)))@sums/len(delta)
        extreme+=int((np.abs(perm)>=abs(mean)-1e-15).sum())
        ix=rng.integers(0,len(sums),size=(m,len(sums)))
        samples.extend((sums[ix].sum(axis=1)/sizes[ix].sum(axis=1)).tolist())
    return dict(difference=mean,p=(extreme+1)/(N+1),ci95=np.quantile(samples,[.025,.975]).tolist())


def holm(tests):
    names=sorted(tests,key=lambda k:tests[k]['p'])
    largest=0
    for i,name in enumerate(names):
        largest=max(largest,min(1.,(len(names)-i)*tests[name]['p']))
        tests[name]['p_holm']=largest


def nested_routing(dataset,folds,candidates,method_names):
    # All candidate scores are label-independent. Inner selection is performed
    # solely on the outer training set, never on other outer-fold OOF predictions.
    ranks={name:np.array([c['ranks']['omission'] for c in candidates[name]]) for name in method_names}
    all_indices=set(range(len(dataset)))
    routed=np.zeros(len(dataset),int); single=routed.copy(); oracle=routed.copy()
    selections=[]
    categories=np.array([r['omission_category_id'] for r in dataset])
    for f,test in enumerate(folds):
        dev=sorted(all_indices-set(test)); devset=set(dev)
        assert not ({dataset[i]['gold_chunk_id'] for i in dev}&{dataset[i]['gold_chunk_id'] for i in test})
        inner=make_group_folds([dataset[i] for i in dev],4,SEED+f)
        inner_ranks={name:np.zeros(len(dataset),int) for name in method_names}
        outer_choice={}
        for name in method_names:
            outer_choice[name]=choose(ranks[name],dev)
            for local in inner:
                val=[dev[i] for i in local]
                train=sorted(devset-set(val))
                selected=choose(ranks[name],train)
                inner_ranks[name][val]=ranks[name][selected,val]
        def best(indices):
            values=[(1/inner_ranks[name][indices]).mean() for name in method_names]
            return method_names[int(np.argmax(values))]
        global_best=best(dev)
        by_category={c:best([i for i in dev if categories[i]==c]) for c in sorted(set(categories))}
        outer_eval=np.array([ranks[name][outer_choice[name],test] for name in method_names])
        oracle[test]=outer_eval.min(axis=0)
        single[test]=ranks[global_best][outer_choice[global_best],test]
        for i in test:
            name=by_category[categories[i]]
            routed[i]=ranks[name][outer_choice[name],i]
        selections.append(dict(fold=f+1,global_best=global_best,by_category=by_category,
                               candidate_indices=outer_choice,inner_fold_count=4))
    return dict(methods=method_names,selections=selections,**{
        name:dict(summary=metrics(r),ranks=r.tolist()) for name,r in [('single',single),('category',routed),('oracle',oracle)]})


def main():
    global DATA, OUTPUT, DOCS
    args=_parser.parse_args(); DATA=args.input_dir; OUTPUT=args.output_dir; DOCS=OUTPUT
    OUTPUT.mkdir(parents=True,exist_ok=True)
    data=json.loads((DATA/'experiments.json').read_text())
    candidates=json.loads((DATA/'candidate_ranks.json').read_text())
    groups=[r['gold_chunk_id'] for r in data['dataset']]
    result=dict(configuration=dict(seed=SEED,permutations=N,bootstrap=N,grouping='gold_chunk_id',
               uncertainty='conditional on stored OOF predictions; model selection not refitted in bootstrap'),panels={},routing={})
    lines=['# 査読追加実験結果','', '生成APIを使わず、276質問・510文書と保存済み擬似文書で実施。提出時結果は保存したまま。',
           '質問語頻度countは出現回数による線形重み、binaryは出現の有無。追加実験の探索範囲はexperiment_protocol.md参照。','']
    for mode in ['binary','count']:
        for panel in ['fixed','tuned']:
            names=['Dense','BM25','RM3','RRF','Hybrid','Query2doc','CategoryQuery2doc']
            methods={}
            for name in names:
                if panel=='tuned':
                    methods[name]={v:np.array(d['ranks']) for v,d in data['tuned'][mode][name].items() if v in ['original','omission']}
                elif name=='Hybrid':
                    methods[name]={v:np.array(data['tuned'][mode]['HybridAlphaOnly'][v]['ranks']) for v in ['original','omission']}
                else:
                    methods[name]={v:np.array([r['gold_rank'] for r in d['details']]) for v,d in data['fixed'][mode+'_'+name].items() if v in ['original','omission']}
            bm=methods['BM25']; tests={}
            for name,vs in methods.items():
                if name!='BM25': tests[name]=inference(1/vs['omission']-1/bm['omission'],groups)
            holm(tests)
            interactions={name:inference((1/vs['omission']-1/bm['omission'])-(1/vs['original']-1/bm['original']),groups)
                          for name,vs in methods.items() if name!='BM25' and 'original' in vs}
            holm(interactions)
            key=mode+'_'+panel
            result['panels'][key]=dict(tests_mrr_vs_bm25=tests,omission_interactions=interactions,
                summaries={name:{v:metrics(r) for v,r in vs.items()} for name,vs in methods.items()})
            lines+=['## '+key,'','|手法|MRR 元|MRR 省略|R@1 省略 (%)|対BM25 p|Holm p|','|---|---:|---:|---:|---:|---:|']
            for name,vs in methods.items():
                o=f"{metrics(vs['original'])['mrr']:.4f}" if 'original' in vs else '未生成'
                q=metrics(vs['omission']); t=tests.get(name)
                p=f"{t['p']:.4f}" if t else '—'; h=f"{t['p_holm']:.4f}" if t else '—'
                lines.append(f"|{name}|{o}|{q['mrr']:.4f}|{100*q['recall']['R@1']:.1f}|{p}|{h}|")
            h=interactions['Hybrid']
            lines+=['',f"疎密統合の利得差（省略時のBM25との差 − 元質問時の差）={h['difference']:.4f}、95% CI [{h['ci95'][0]:.4f}, {h['ci95'][1]:.4f}]、p={h['p']:.4f}、4比較Holm p={h['p_holm']:.4f}。",'']
            baseline=bm['omission']; transitions={}
            for name,vs in methods.items():
                r=vs['omission']
                transitions[name]=dict(improved=int((r<baseline).sum()),worsened=int((r>baseline).sum()),
                                       recovered=int(((r==1)&(baseline>1)).sum()),lost=int(((r>1)&(baseline==1)).sum()))
            expansions=np.array([methods[name]['omission'] for name in ['RM3','Query2doc','CategoryQuery2doc']])
            result['panels'][key]['transitions']=transitions
            result['panels'][key]['expansion_union']=dict(recovered=int(((expansions==1).any(axis=0)&(baseline>1)).sum()),
                lost=int(((expansions>1).any(axis=0)&(baseline==1)).sum()))
    lines+=['## Query2doc実装の切り分け','','|条件|MRR|R@1 (%)|','|---|---:|---:|']
    for name in ['BM25','Query2doc_1repeat','Query2doc','Query2doc_legacy_concat','CategoryQuery2doc_1repeat','CategoryQuery2doc','CategoryQuery2doc_legacy_concat']:
        q=data['fixed']['count_'+name]['omission']['summary']
        lines.append(f"|count {name}|{q['mrr']:.4f}|{100*q['recall']['R@1']:.1f}|")
    for mode in ['binary','count']:
        routing=nested_routing(data['dataset'],data['folds'],candidates[mode],['BM25','RM3','Query2doc','CategoryQuery2doc','Hybrid'])
        routing['category_vs_single']=inference(1/np.array(routing['category']['ranks'])-1/np.array(routing['single']['ranks']),groups)
        result['routing'][mode]=routing
        lines+=['','## 内側4-foldを用いたルーティング '+mode,'','|方式|MRR|R@1 (%)|','|---|---:|---:|']
        for name in ['single','category','oracle']:
            s=routing[name]['summary']; lines.append(f"|{name}|{s['mrr']:.4f}|{100*s['recall']['R@1']:.1f}|")
    lines+=['','## 解釈上の制約','',
        '- 調整の機会を設けたが、手法ごとの探索候補数は異なる。等予算の比較ではない。',
        '- fixedパネルでもHybridだけはalpha選択を行う。tunedパネルではBM25を含む改善手法を開発側で選択する。',
        '- binaryパネルのQuery2docはcountによる修正実装である。公平な語頻度比較はcountパネルで確認する。',
        '- 信頼区間とp値は保存されたOOF予測に条件付けた分析。パラメータ選択全体を再実行する不確実性は含めない。',
        '- Query2docの元質問用擬似文書は生成していないため、省略前後の比較は他の5手法に限定。',
        '- プロンプトによる推測の断定抑制は維持。新たなプロンプトの有効性は検証していない。',
        '- Oracleは5手法の候補集合に依存する事後上限であり、候補追加で非減少。',
        '- 提出版の数値を変更する場合は本文・図表・エラー分析・ルーティングを一緒に更新する。','']
    (OUTPUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (DOCS/'experiment_results.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__=='__main__': main()
