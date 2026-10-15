"""Smallest end to end example: calibrate, then decide two real dispatch sites.

Run with:  python demo.py
"""
from dispatchfold import lowering as L
from dispatchfold import profiles
from dispatchfold.bench import cpu_name, measure_site
from dispatchfold.model import calibrate

PICK = ('compressed-bytes', 'json-classes')


def main():
    print('host: %s' % cpu_name())
    print()
    model, _rows = calibrate(n=1 << 16)
    print()

    for loader in profiles.ALL:
        work = loader()
        if work.name not in PICK:
            continue
        stream = work.stream[:1 << 17]
        site = L.Site(work.keys, work.lo, work.hi)
        decision = model.decide(stream, site.n_slots)
        timed = measure_site(site, stream)
        table = timed['jump-table']['ns_per_key']
        folded = timed['fold']['ns_per_key']

        print('%s: %s' % (work.name, work.what))
        print('  source           %s' % work.source)
        print('  keys             %d over %d cases' % (len(stream), len(work.keys)))
        print('  mispredict rate  %.3f   repeat rate %.3f'
              % (decision['miss_rate'], decision['repeat_rate']))
        print('  predicted        jump table %.2f ns, folded %.2f ns, so pick %s'
              % (decision['predicted_direct_ns'],
                 decision['predicted_fold_ns'], decision['choice']))
        print('  measured         jump table %.2f ns, folded %.2f ns'
              % (table, folded))
        faster = 'fold' if folded < table else 'table'
        print('  faster in fact   %s  (model was %s)'
              % (faster, 'right' if faster == decision['choice'] else 'wrong'))
        print('  every lowering returned the same accumulator: %s'
              % all(v['verified'] for v in timed.values()))
        print()


if __name__ == '__main__':
    main()
