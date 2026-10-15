"""Command line entry point: python -m dispatchfold <command>"""
import argparse
import sys

from . import bench as bench_mod
from . import lowering as L


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='python -m dispatchfold',
        description='Decide, per dispatch site, whether folding a switch into '
                    'per handler batches beats the jump table a compiler emits.')
    ap.add_argument('command',
                    choices=['calibrate', 'bench', 'chart', 'all'],
                    help='calibrate measures this CPU only; bench also times '
                         'the real workloads and writes docs/results.json; '
                         'chart renders docs/before_after.png from that file; '
                         'all runs bench then chart')
    ap.add_argument('--body-ops', type=int, default=0,
                    help='extra dependent operations inside every handler, '
                         'to model heavier handler bodies (default 0)')
    ap.add_argument('--block', type=int, default=L.DEFAULT_BLOCK,
                    help='items the folded kernel buckets at a time '
                         '(default %d)' % L.DEFAULT_BLOCK)
    ap.add_argument('--results', default='docs/results.json')
    ap.add_argument('--out', default='docs/before_after.png')
    args = ap.parse_args(argv)

    if args.command == 'calibrate':
        from .model import calibrate
        calibrate(block=args.block)
        return 0
    if args.command in ('bench', 'all'):
        bench_mod.run(body_ops=args.body_ops, block=args.block,
                      out_path=args.results)
    if args.command in ('chart', 'all'):
        from .chart import make_chart
        print('wrote %s' % make_chart(args.results, args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
