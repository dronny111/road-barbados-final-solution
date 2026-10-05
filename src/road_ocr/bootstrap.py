"""Paired bootstrap of two prediction sets on the same labelled rows (exact-label groups are resampled)."""
from collections import defaultdict
import random

from road_ocr.metrics import edit_distance


def row_costs(references,predictions):
    """Per-row edit totals; corpus scores over any subset are sums of these."""
    return [(edit_distance(r,p),len(r),edit_distance(r.split(),p.split()),len(r.split()))
            for r,p in zip(references,predictions,strict=True)]


def corpus_from_costs(costs,indices):
    ce=sum(costs[i][0] for i in indices);rc=sum(costs[i][1] for i in indices)
    we=sum(costs[i][2] for i in indices);rw=sum(costs[i][3] for i in indices)
    if not rc or not rw:
        raise ValueError('Empty bootstrap denominator')
    return ce/rc,we/rw,0.5*(ce/rc)+0.5*(we/rw)


def paired_bootstrap(references,baseline,candidate,replicates,seed):
    """Resample exact-label groups; corpus scoring is recomputed inside each resample."""
    groups=defaultdict(list)
    for index,reference in enumerate(references):
        groups[reference].append(index)
    keys=sorted(groups)
    base_costs,candidate_costs=row_costs(references,baseline),row_costs(references,candidate)
    rng=random.Random(seed)
    gains={'cer':[],'wer':[],'combined':[]}
    for _ in range(replicates):
        indices=[i for _ in keys for i in groups[keys[rng.randrange(len(keys))]]]
        base=corpus_from_costs(base_costs,indices)
        other=corpus_from_costs(candidate_costs,indices)
        for name,position in (('cer',0),('wer',1),('combined',2)):
            gains[name].append(base[position]-other[position])
    def interval(values):
        values=sorted(values)
        pick=lambda f:values[min(len(values)-1,max(0,round(f*(len(values)-1))))]
        return dict(low=pick(0.025),median=pick(0.5),high=pick(0.975),replicates=len(values))
    return {name:interval(values) for name,values in gains.items()}
