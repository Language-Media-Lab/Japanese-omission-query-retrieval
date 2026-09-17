"""Regression checks for query weighting, held-out selection, and ordinal alpha."""
import unittest
from collections import Counter
import numpy as np
from run_bm25_retrieval_baseline import BM25, tokenize
from run_rm3_retrieval import weighted_bm25_scores
from run_review_experiments import contribution_matrix, query_matrix, expansion_weights, choose, rankings
from annotation_agreement import agreement


class ReviewChecks(unittest.TestCase):
    def test_standalone_bm25_preserves_frequency(self):
        model=BM25([['a','a','b'],['b','c'],['a','c']])
        np.testing.assert_allclose(model.scores(['a','a','b']),weighted_bm25_scores(model,{'a':2,'b':1}))

    def test_weighting_matches_independent_loop(self):
        model=BM25([['a','a'],['c','c'],['b','b']])
        vocab={t:i for i,t in enumerate(sorted(model.idf))}
        weights=[Counter({'a':5,'c':1}),Counter({'a':1,'c':5})]
        got=(query_matrix(weights,vocab)@contribution_matrix(model,vocab)).toarray()
        np.testing.assert_allclose(got,[weighted_bm25_scores(model,w) for w in weights],rtol=1e-14)
        self.assertNotEqual(int(np.argmax(got[0])),int(np.argmax(got[1])))

    def test_repetition_changes_only_question_weights(self):
        q,p='甲乙甲乙','乙丙丁'
        one=expansion_weights(q,p,1); five=expansion_weights(q,p,5)
        difference=Counter({t:five[t]-one[t] for t in set(one)|set(five)})
        self.assertEqual(+difference,Counter({t:4*n for t,n in Counter(tokenize(q)).items()}))
        self.assertEqual(five['乙丙'],1)
        self.assertEqual(five['乙乙'],0)

    def test_repeating_whole_query_scales_score(self):
        model=BM25([['a','a','b'],['b','c'],['a','c']])
        np.testing.assert_allclose(weighted_bm25_scores(model,{'a':5,'b':10}),
            5*np.array(weighted_bm25_scores(model,{'a':1,'b':2})))

    def test_selection_ignores_held_out_labels(self):
        ranks=np.array([[1,1,100],[2,2,1]])
        self.assertEqual(choose(ranks,[0,1]),0)
        ranks[:,2]=[510,1]
        self.assertEqual(choose(ranks,[0,1]),0)

    def test_ties_use_sorted_corpus_order(self):
        order,inv=rankings(np.array([[2.,2.,1.]]))
        self.assertEqual(order.tolist(),[[0,1,2]])
        self.assertEqual(inv.tolist(),[[1,2,3]])

    def test_ordinal_alpha_independent_pair_formula(self):
        items=[[1,1,2],[1,2,3,3,3],[2,2,2],[1,3,3]]
        counts=Counter(x for xs in items for x in xs)
        def d(a,b):
            lo,hi=sorted([a,b])
            return (sum(counts[k] for k in range(lo,hi+1))-(counts[lo]+counts[hi])/2)**2
        n=sum(map(len,items))
        observed=sum(sum(d(a,b) for a in xs for b in xs)/(len(xs)-1) for xs in items)/n
        population=[x for xs in items for x in xs]
        expected=sum(d(a,b) for i,a in enumerate(population) for j,b in enumerate(population) if i!=j)/(n*(n-1))
        self.assertAlmostEqual(agreement(items)['ordinal_alpha'],1-observed/expected,places=12)
        self.assertEqual(agreement([[1,1,1],[2,2,2]])['ordinal_alpha'],1.)


if __name__=='__main__':
    unittest.main()
