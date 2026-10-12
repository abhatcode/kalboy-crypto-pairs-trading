"""Command line entry point: python -m kalboy <command>."""
import argparse
import json
import os
import sys

import pandas as pd


def main(argv=None):
    p = argparse.ArgumentParser(prog='kalboy', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    f = sub.add_parser('fetch')
    f.add_argument('--timeframe', required=True)
    f.add_argument('--start', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--top_n', type=int, default=80)
    f.add_argument('--workers', type=int, default=3)

    w = sub.add_parser('walkforward')
    w.add_argument('--config', required=True)
    w.add_argument('--set', action='append', default=[], help='override, e.g. strategy.stop_z=3.5')
    w.add_argument('--segment', default='dev', choices=['dev', 'holdout'])
    w.add_argument('--confirm-holdout', action='store_true', help='required to run on the sealed holdout')
    w.add_argument('--label', default=None)
    w.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    w.add_argument('--purpose', default='trial', choices=['trial', 'sensitivity', 'reproduction'],
                   help='trial is a candidate strategy. sensitivity re-runs one under other assumptions, and '
                        'reproduction re-runs one to check the code still gives the same result. '
                        'Both are logged but never counted as trials.')
    w.add_argument('--no-cache', action='store_true', help='recompute pair selection without the cache in cache/selection')

    m = sub.add_parser('metalabel')
    m.add_argument('run_dir')

    pp = sub.add_parser('paper')
    pp.add_argument('--config', required=True)
    pp.add_argument('--set', action='append', default=[])
    pp.add_argument('--label', required=True)
    pp.add_argument('--once', action='store_true', help='process closed bars once and exit')
    pp.add_argument('--summary', action='store_true', help='write a summary page for the paper run and exit')

    r = sub.add_parser('report')
    r.add_argument('run_dirs', nargs='+')
    r.add_argument('--compare', default=None, help='also write a comparison CSV here')
    r.add_argument('--sensitivity', default=None, help='cost-sensitivity table and figure, e.g. results/cost_sensitivity.csv')

    c = sub.add_parser('check')
    c.add_argument('data_dir')

    x = sub.add_parser('crosscheck')
    x.add_argument('data_dir')
    x.add_argument('--start', default=None)
    x.add_argument('--end', default=None)

    sub.add_parser('tables')
    sub.add_parser('ledger')
    sub.add_parser('freeze-check')
    args = p.parse_args(argv)

    if args.cmd == 'fetch':
        from kalboy.data import fetch_dataset
        fetch_dataset(args.out, args.timeframe, args.start, top_n=args.top_n, workers=args.workers)

    elif args.cmd == 'walkforward':
        from kalboy import ledger, report, walkforward
        from kalboy.config import load_config
        cfg = load_config(args.config, args.set)
        if args.segment == 'holdout':
            looks = ledger.holdout_looks(cfg.walkforward.data_dir)
            if not args.confirm_holdout:
                sys.exit(f"The holdout from {cfg.walkforward.holdout_start} is sealed ({len(looks)} previous looks "
                         f"logged). Re-run with --confirm-holdout only for the final, frozen config.")
        label = args.label or f"{cfg.name}_{args.segment}"
        out = os.path.join('results', label)
        result = walkforward.run(cfg, args.segment, workers=args.workers, purpose=args.purpose,
                                 cache_dir=None if args.no_cache else os.path.join('cache', 'selection'))
        walkforward.save(result, cfg, out)
        walkforward.log_run(result, out)
        walkforward.print_summary(result['summary'])
        report.charts(out)
        report.markdown(out)
        print(f"Saved to {out}")

    elif args.cmd == 'metalabel':
        from kalboy import metalabel
        trades = pd.read_csv(os.path.join(args.run_dir, 'trades.csv'))
        windows = pd.read_csv(os.path.join(args.run_dir, 'windows.csv'))
        result, scored = metalabel.run(trades, windows)
        with open(os.path.join(args.run_dir, 'metalabel.json'), 'w') as fh:
            json.dump(result, fh, indent=2)
        scored.to_csv(os.path.join(args.run_dir, 'metalabel_trades.csv'), index=False)
        for name, res in result.items():
            a, t = res['all_scored'], res['taken']
            print(f"{name}: AUC {res['auc'] or float('nan'):.3f}, {t['n_trades']} of {res['scored_trades']} trades taken, "
                  f"mean {t.get('mean_return', float('nan')):.4%} vs {a.get('mean_return', float('nan')):.4%} for all")

    elif args.cmd == 'paper':
        from kalboy.config import load_config
        from kalboy.paper import LiveSource, PaperTrader
        cfg = load_config(args.config, args.set)
        if args.summary:
            from kalboy.report import paper_markdown
            print(open(paper_markdown(os.path.join('paper', args.label))).read())
            return
        from kalboy.notify import Alerts, Notifier
        notifier = Notifier()
        trader = PaperTrader(cfg, os.path.join('paper', args.label), LiveSource(cfg.walkforward.data_dir),
                             alerts=Alerts(notifier, args.label), dashboard=True)
        if args.once:
            print(f"{trader.process_new_bars()} new bars")
        else:
            trader.run_forever()

    elif args.cmd == 'report':
        from kalboy import report
        for d in args.run_dirs:
            report.charts(d)
            print(report.markdown(d))
        if args.compare:
            print(report.compare(args.run_dirs, args.compare).to_string(index=False))
        if args.sensitivity:
            print(report.sensitivity(args.run_dirs, args.sensitivity).to_string(index=False))

    elif args.cmd == 'check':
        from kalboy.data import validate
        report = validate(args.data_dir)
        report.to_csv(os.path.join(args.data_dir, 'validation.csv'), index=False)
        print(report.describe().T[['mean', 'min', 'max']].to_string())

    elif args.cmd == 'crosscheck':
        from kalboy import crosscheck
        rep = crosscheck.run(args.data_dir, args.start, args.end)
        print(f"{len(rep)} coins, median gap {rep['median_abs_gap_bps'].median():.1f} bps, "
              f"median return correlation {rep['return_corr'].median():.3f}")

    elif args.cmd == 'tables':
        from kalboy.tables import write_all
        write_all()

    elif args.cmd == 'freeze-check':
        import hashlib
        with open(os.path.join('results', 'freeze.json')) as fh:
            frozen = json.load(fh)
        changed = [f for f, h in frozen['sha256'].items()
                   if not os.path.exists(f) or hashlib.sha256(open(f, 'rb').read()).hexdigest() != h]
        presentation = set(frozen.get('presentation_only', []))
        logged = {c['file'] for c in frozen.get('changes_after_freeze', [])}
        bad = [f for f in changed if f not in presentation]
        unlogged = [f for f in changed if f in presentation and f not in logged]
        n_result = len(frozen['sha256']) - len(presentation)
        print(f"{n_result} result files match freeze.json" if not bad else f"Changed: {', '.join(bad)}")
        if unlogged:
            print(f"Changed without a logged reason: {', '.join(unlogged)}")
        if bad or unlogged:
            sys.exit(1)

    elif args.cmd == 'ledger':
        from kalboy import ledger
        rows = ledger.read()
        print(pd.DataFrame(rows).to_string(index=False) if rows else 'Ledger is empty')


if __name__ == '__main__':
    main()
