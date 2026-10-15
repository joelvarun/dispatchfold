"""Measure every lowering on every real workload and write docs/results.json."""
import json
import os
import platform
import time

from . import lowering as L
from . import profiles
from .jit import Buffer, Kernel, time_kernel
from .model import calibrate

MAX_KEYS = 1 << 17
CHAIN_LIMIT = 16          # an if-else chain over 256 cases is not a real option


def cpu_name():
    """The marketing name of the CPU, which is what a reader recognises."""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            'HARDWARE' + chr(92) + 'DESCRIPTION' + chr(92) + 'System'
            + chr(92) + 'CentralProcessor' + chr(92) + '0')
        try:
            return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        finally:
            winreg.CloseKey(key)
    except Exception:
        return os.environ.get('PROCESSOR_IDENTIFIER', platform.processor())


def host():
    return {
        'cpu': cpu_name(),
        'machine': platform.machine(),
        'system': '%s %s' % (platform.system(), platform.release()),
        'python': platform.python_version(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


def measure_site(site, stream, block=L.DEFAULT_BLOCK, reps=15):
    """Time every lowering on one site and check them all against each other."""
    ref = site.reference(stream)
    buf = Buffer(stream, L.scratch_words(site, block))
    variants = [('jump-table', L.jump_table(site)),
                ('binary-search', L.binary_search(site)),
                ('fold', L.fold(site, block))]
    if len(site.keys) <= CHAIN_LIMIT:
        variants.insert(0, ('if-chain', L.chain(site)))
    out = {}
    for name, builder in variants:
        kern = Kernel(name, builder)
        ns, acc = time_kernel(kern, buf, reps=reps)
        if acc != ref:
            raise AssertionError(
                'lowering %s returned %d, reference is %d' % (name, acc, ref))
        out[name] = {'ns_per_key': ns, 'code_bytes': kern.code_size,
                     'verified': True}
        kern.free()
    return out


def run(body_ops=0, block=L.DEFAULT_BLOCK, out_path='docs/results.json',
        sweep_workload='compressed-bytes'):
    info = host()
    print('dispatchfold benchmark')
    print('  host: %s' % info['cpu'])
    print('  %s, python %s' % (info['system'], info['python']))
    print()
    model, cal_rows = calibrate(block=block)
    print()
    print('loading real dispatch streams from this machine')
    workloads = profiles.load_all()
    print()

    results = []
    for w in workloads:
        stream = w.stream[:MAX_KEYS]
        site = L.Site(w.keys, w.lo, w.hi, body_ops=body_ops)
        decision = model.decide(stream, site.n_slots, block)
        timed = measure_site(site, stream, block)
        table_ns = timed['jump-table']['ns_per_key']
        fold_ns = timed['fold']['ns_per_key']
        chosen = decision['choice']
        chosen_ns = fold_ns if chosen == 'fold' else table_ns
        best = min(table_ns, fold_ns)
        results.append({
            'workload': w.name,
            'what': w.what,
            'source': w.source,
            'keys': len(stream),
            'cases': len(w.keys),
            'slots': site.n_slots,
            'miss_rate': decision['miss_rate'],
            'repeat_rate': decision['repeat_rate'],
            'predicted_direct_ns': decision['predicted_direct_ns'],
            'predicted_fold_ns': decision['predicted_fold_ns'],
            'choice': chosen,
            'timings': timed,
            'baseline_ns': table_ns,
            'fold_ns': fold_ns,
            'chosen_ns': chosen_ns,
            'speedup_vs_baseline': table_ns / chosen_ns,
            'fold_speedup': table_ns / fold_ns,
            'decision_correct': (chosen == 'fold') == (fold_ns < table_ns),
            'regret_ns': chosen_ns - best,
        })
        print('  %-17s miss %.3f  table %6.2f  fold %6.2f  pick %-5s  %.2fx'
              % (w.name, decision['miss_rate'], table_ns, fold_ns,
                 chosen, table_ns / chosen_ns))

    sweep = []
    target = next((w for w in workloads if w.name == sweep_workload), workloads[0])
    print()
    print('handler work sweep on %s' % target.name)
    for ops in (0, 2, 4, 8, 16):
        site = L.Site(target.keys, target.lo, target.hi, body_ops=ops)
        timed = measure_site(site, target.stream[:MAX_KEYS], block, reps=9)
        t, f = timed['jump-table']['ns_per_key'], timed['fold']['ns_per_key']
        sweep.append({'body_ops': ops, 'table_ns': t, 'fold_ns': f,
                      'speedup': t / f})
        print('  handler ops %2d  table %6.2f  fold %6.2f  %.2fx' % (ops, t, f, t / f))

    totals = {
        'always_table_ns': sum(r['baseline_ns'] for r in results),
        'always_fold_ns': sum(r['fold_ns'] for r in results),
        'dispatchfold_ns': sum(r['chosen_ns'] for r in results),
    }
    totals['speedup_vs_always_table'] = (totals['always_table_ns']
                                         / totals['dispatchfold_ns'])
    totals['speedup_vs_always_fold'] = (totals['always_fold_ns']
                                        / totals['dispatchfold_ns'])
    totals['decisions_correct'] = sum(1 for r in results if r['decision_correct'])
    totals['decisions'] = len(results)

    payload = {'host': info, 'block': block, 'body_ops': body_ops,
               'cost_model': model.as_dict(),
               'calibration': [{'stream': r[0], 'miss_rate': r[1],
                                'repeat_rate': r[2], 'slots': r[3],
                                'table_ns': r[4], 'fold_ns': r[5]}
                               for r in cal_rows],
               'workloads': results, 'handler_sweep': sweep, 'totals': totals}
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(payload, fh, indent=2)

    print()
    print('summed over %d real workloads, ns per dispatch:' % len(results))
    print('  compiler jump table everywhere : %6.2f ns' % totals['always_table_ns'])
    print('  fold everywhere                : %6.2f ns' % totals['always_fold_ns'])
    print('  dispatchfold decides per site  : %6.2f ns  (%.2fx better than the '
          'jump table)' % (totals['dispatchfold_ns'],
                           totals['speedup_vs_always_table']))
    print('  model picked the faster kernel on %d of %d sites'
          % (totals['decisions_correct'], totals['decisions']))
    print()
    print('wrote %s' % out_path)
    return payload
