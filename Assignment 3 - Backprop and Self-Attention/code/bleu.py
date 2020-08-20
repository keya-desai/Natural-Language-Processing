import math
import sys

from collections import Counter
from functools import reduce


def compute_bleu(reflists, hyps, n_max=4, use_shortest_ref=False):
    assert len(reflists) == len(hyps)

    prec_mean = 0  # TODO: Implement

    # reduce()

    for n in range(1,n_max+1):
    	a_n_sum = 0
    	b_n_sum = 0
    	for l in range(len(hyps)):
    		num_hyp_ngrams_in_refs_clipped, num_hyp_ngrams = get_ngram_counts(reflists[l],hyps[l],n)
    		a_n_sum += num_hyp_ngrams_in_refs_clipped
    		b_n_sum += num_hyp_ngrams

    	prec_mean += math.log(a_n_sum/b_n_sum)

    prec_mean = prec_mean/n
    prec_mean = math.exp(prec_mean)

    R = 0
    H = 0
    for l in range(len(hyps)):
    	R += min(len(x) for x in reflists[l])
    	H += len(hyps[l])

    brevity_penalty = min(1, math.exp(1 - (R/H)))  # TODO:Implement

    bleu = brevity_penalty * prec_mean

    return bleu


def get_ngram_counts(refs, hyp, n):
    hyp_ngrams = [tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1)]
    num_hyp_ngrams = max(1, len(hyp_ngrams))  # Avoid empty

    num_hyp_ngrams_in_refs_clipped = 0  # TODO: Implement

    gc = Counter(hyp_ngrams)
    ref_ngrams = [[tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)] for ref in refs]
    ref_ngrams_count_list = [Counter(ngrams) for ngrams in ref_ngrams]

    for g,c in gc.items():
    	num_hyp_ngrams_in_refs_clipped += min(c, max(ref[g] for ref in ref_ngrams_count_list))

    return num_hyp_ngrams_in_refs_clipped, num_hyp_ngrams
