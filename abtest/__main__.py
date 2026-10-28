import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from .core import plan, save_json, simulate, analyze_file, monte_carlo, validate_plan
from .report import write_report


def main():
    p = argparse.ArgumentParser(description='Учебный A/B-тест бинарной конверсии, Python + statsmodels')
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('demo', help='Полный воспроизводимый эксперимент с синтетическими данными')
    d.add_argument('--out', default='local/demo'); d.add_argument('--seed', type=int, default=42)
    pl = sub.add_parser('plan', help='Рассчитать выборку до запуска теста')
    pl.add_argument('--baseline', type=float, default=.10); pl.add_argument('--mde', type=float, default=.02)
    pl.add_argument('--alpha', type=float, default=.05); pl.add_argument('--power', type=float, default=.80)
    pl.add_argument('--daily-users', type=int, default=1000)
    pl.add_argument('--practical-threshold', type=float, default=.01)
    pl.add_argument('--out', required=True)
    s = sub.add_parser('simulate', help='Симуляция данных по заранее заданному плану')
    s.add_argument('--plan', required=True); s.add_argument('--p-a', type=float, default=.10)
    s.add_argument('--p-b', type=float, default=.12); s.add_argument('--seed', type=int, default=42)
    s.add_argument('--out', required=True)
    a = sub.add_parser('analyze', help='Проанализировать CSV')
    a.add_argument('--input', required=True); a.add_argument('--plan', required=True); a.add_argument('--out', required=True)
    args = p.parse_args()
    try:
        if args.command == 'plan':
            result = plan(args.baseline, args.mde, args.alpha, args.power, args.daily_users, args.practical_threshold)
            save_json(args.out, result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
        if args.command == 'demo':
            root = Path(args.out)
            if root.exists(): raise ValueError('Demo output already exists; choose a new --out directory')
            root.mkdir(parents=True)
            exp = plan(); save_json(root/'plan.json', exp)
            meta = simulate(root/'experiment.csv', exp['n_per_group'], exp['baseline'], exp['target'], args.seed)
            save_json(root/'simulation.json', meta)
            result = analyze_file(root/'experiment.csv', exp); result['source'] = 'synthetic_simulation'
            mc = dict(null=monte_carlo(exp['n_per_group'], .10, .10, seed=123),
                      alternative=monte_carlo(exp['n_per_group'], .10, .12, seed=456))
            write_report(root/'analysis', result, exp, meta, mc)
            env = dict(python=platform.python_version(), packages={name: importlib.metadata.version(name) for name in ['numpy','scipy','statsmodels','matplotlib']})
            save_json(root/'environment.json', env)
            print(f"Готово: {root/'analysis'/'REPORT.md'}"); return
        exp = validate_plan(json.loads(Path(args.plan).read_text(encoding='utf-8')))
        if args.command == 'simulate':
            if Path(args.out+'.meta.json').exists(): raise ValueError('Metadata output already exists')
            meta = simulate(args.out, exp['n_per_group'], args.p_a, args.p_b, args.seed)
            save_json(args.out+'.meta.json', meta); print(args.out); return
        result = analyze_file(args.input, exp)
        write_report(args.out, result, exp)
        print(f"Готово: {Path(args.out)/'REPORT.md'}")
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, f'Ошибка: {exc}\n')

if __name__ == '__main__': main()
