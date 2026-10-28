import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, fisher_exact
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    proportion_effectsize, proportions_ztest, confint_proportions_2indep,
)


def probability(x, name, *, boundaries=False):
    if not math.isfinite(x) or not (0 <= x <= 1 if boundaries else 0 < x < 1):
        raise ValueError(f'{name} must be a finite probability')


def plan(baseline=.10, mde=.02, alpha=.05, power=.80, daily_users=1000, practical_threshold=.01):
    """mde is an absolute positive difference: .02 means two percentage points."""
    probability(baseline, 'baseline')
    probability(alpha, 'alpha'); probability(power, 'power')
    if not math.isfinite(mde) or mde <= 0 or baseline + mde >= 1:
        raise ValueError('mde must be positive and baseline + mde must be below 1')
    if daily_users <= 0 or not isinstance(daily_users, int):
        raise ValueError('daily_users must be a positive integer')
    if not math.isfinite(practical_threshold) or practical_threshold <= 0 or practical_threshold >= 1:
        raise ValueError('practical_threshold must be in (0,1)')
    effect = abs(float(proportion_effectsize(baseline, baseline + mde)))
    n = math.ceil(NormalIndPower().solve_power(effect_size=effect, alpha=alpha,
                   power=power, ratio=1, alternative='two-sided'))
    days = math.ceil(2 * n / daily_users)
    return dict(metric='user_conversion', baseline=baseline, target=baseline + mde,
        mde_absolute=mde, mde_relative=mde / baseline, practical_threshold=practical_threshold, alpha=alpha, power=power,
        alternative='two-sided', allocation_a=.5, allocation_b=.5,
        method='NormalIndPower with Cohen h (normal approximation)',
        n_per_group=n, n_total=n * 2, daily_eligible_users=daily_users,
        estimated_enrollment_days=days, suggested_full_week_days=max(7, math.ceil(days / 7) * 7),
        conversion_window_days=7,
        hypothesis_null='p_B - p_A = 0', hypothesis_alternative='p_B - p_A != 0',
        assumptions=['Independent users, one row per randomized user, binary outcome.',
            'Equal allocation; one primary metric; single final analysis.',
            'Do not stop when significance first appears.',
            'Time estimate excludes the final 7-day conversion maturation window.'])


def validate_plan(p):
    expected = plan(p['baseline'], p['mde_absolute'], p['alpha'], p['power'], p['daily_eligible_users'], p['practical_threshold'])
    for key in ('n_per_group', 'n_total', 'allocation_a', 'allocation_b', 'alternative', 'metric'):
        if p.get(key) != expected[key]:
            raise ValueError(f'Invalid or unsupported plan field: {key}')
    return p


def save_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f'Output exists: {path}; choose a new path')
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def simulate(path, n_per_group, p_a=.10, p_b=.12, seed=42):
    probability(p_a, 'p_a', boundaries=True); probability(p_b, 'p_b', boundaries=True)
    if not isinstance(n_per_group, int) or n_per_group < 1:
        raise ValueError('n_per_group must be a positive integer')
    path = Path(path)
    if path.exists():
        raise ValueError(f'Output exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for group, p in [('A', p_a), ('B', p_b)]:
        for i, converted in enumerate(rng.binomial(1, p, n_per_group)):
            rows.append((f'sim-{group}-{i:07d}', group, int(converted)))
    rng.shuffle(rows)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); writer.writerow(['user_id', 'group', 'converted']); writer.writerows(rows)
    return dict(source='synthetic_simulation', seed=seed, n_per_group=n_per_group,
                true_p_a=p_a, true_p_b=p_b, true_effect=p_b-p_a,
                assignment='exact equal group sizes, randomized row order')


def load_data(path):
    counts = {'A': [0, 0], 'B': [0, 0]}; seen = set()
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['user_id', 'group', 'converted']:
            raise ValueError('CSV header must be user_id,group,converted in that order')
        for line, row in enumerate(reader, 2):
            uid, group, outcome = row['user_id'], row['group'], row['converted']
            if not uid or not uid.strip() or uid in seen:
                raise ValueError(f'Line {line}: empty or duplicate user_id')
            if group not in counts or outcome not in ('0', '1') or None in row:
                raise ValueError(f'Line {line}: group must be A/B and converted must be 0/1')
            seen.add(uid); counts[group][0] += 1; counts[group][1] += int(outcome)
    if any(n == 0 for n, x in counts.values()):
        raise ValueError('Both groups must contain at least one user')
    return counts


def analyze_counts(n_a, x_a, n_b, x_b, *, alpha=.05, practical_threshold=.02, planned_n=None):
    probability(alpha, 'alpha')
    if not math.isfinite(practical_threshold) or practical_threshold <= 0:
        raise ValueError('practical_threshold must be positive and finite')
    for n, x in [(n_a, x_a), (n_b, x_b)]:
        if not isinstance(n, int) or not isinstance(x, int) or n < 1 or not 0 <= x <= n:
            raise ValueError('Counts must be integers with n > 0 and 0 <= conversions <= n')
    pa, pb = x_a / n_a, x_b / n_b
    pooled = (x_a + x_b) / (n_a + n_b)
    expected_cells = [n_a * pooled, n_a * (1-pooled), n_b * pooled, n_b * (1-pooled)]
    if min(expected_cells) < 5:
        _, pvalue = fisher_exact([[x_b, n_b - x_b], [x_a, n_a - x_a]], alternative='two-sided')
        statistic, method = None, 'Fisher exact (small expected counts)'
    else:
        statistic, pvalue = proportions_ztest([x_b, x_a], [n_b, n_a], alternative='two-sided')
        statistic, method = float(statistic), 'Pooled two-proportion z-test'
    low, high = confint_proportions_2indep(x_b, n_b, x_a, n_a, compare='diff', method='newcomb', alpha=alpha)
    srm_pvalue = float(binomtest(n_b, n_a + n_b, p=.5, alternative='two-sided').pvalue)
    srm = srm_pvalue < .001
    enough = None if planned_n is None else n_a >= planned_n and n_b >= planned_n
    significant = bool(pvalue < alpha)
    if srm:
        decision = 'invalid_allocation_check_randomization_and_missing_data'
    elif enough is False:
        decision = 'incomplete_collect_planned_sample_without_peeking'
    elif float(low) >= practical_threshold and significant:
        decision = 'evidence_of_positive_effect_at_least_practical_threshold'
    elif float(high) < 0 and significant:
        decision = 'evidence_of_negative_effect'
    elif significant and pb > pa:
        decision = 'positive_statistical_effect_practical_size_not_confirmed'
    elif significant and pb < pa:
        decision = 'negative_statistical_effect'
    else:
        decision = 'inconclusive_not_evidence_of_no_effect'
    return dict(n_a=n_a, conversions_a=x_a, conversion_a=pa, n_b=n_b, conversions_b=x_b,
        conversion_b=pb, difference_b_minus_a=pb-pa, difference_percentage_points=(pb-pa)*100,
        relative_uplift=None if pa == 0 else (pb-pa)/pa,
        confidence_interval_difference=[float(low), float(high)], confidence_level=1-alpha,
        ci_method='Newcombe difference of independent proportions',
        alpha=alpha, p_value=float(pvalue), statistic=statistic, test_method=method,
        statistically_significant=significant, practical_threshold=practical_threshold,
        practical_threshold_confirmed=bool(significant and low >= practical_threshold),
        srm_p_value=srm_pvalue, srm_alpha=.001, srm_flag=srm,
        planned_n_per_group=planned_n, planned_sample_reached=enough, decision=decision)


def analyze_file(path, experiment_plan=None):
    counts = load_data(path)
    p = validate_plan(experiment_plan) if experiment_plan is not None else None
    result = analyze_counts(*counts['A'], *counts['B'], alpha=p['alpha'] if p else .05,
        practical_threshold=p['practical_threshold'] if p else .01, planned_n=p['n_per_group'] if p else None)
    result.update(data_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  source='csv_input_not_verified_by_the_program')
    return result


def monte_carlo(n_per_group, p_a=.10, p_b=.12, repetitions=1000, seed=123, alpha=.05):
    """Empirical power/false positive frequency; simulation, not observed power."""
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError('repetitions must be positive')
    if not isinstance(n_per_group, int) or n_per_group < 1:
        raise ValueError('n_per_group must be positive')
    probability(p_a, 'p_a', boundaries=True); probability(p_b, 'p_b', boundaries=True)
    probability(alpha, 'alpha')
    rng = np.random.default_rng(seed); rejected = 0
    # The test follows the same predeclared expected-count rule as analyze_counts.
    for a, b in zip(rng.binomial(n_per_group, p_a, repetitions), rng.binomial(n_per_group, p_b, repetitions)):
        pooled = (a + b) / (2 * n_per_group)
        if min(n_per_group * pooled, n_per_group * (1-pooled)) < 5:
            _, pv = fisher_exact([[b, n_per_group-b], [a, n_per_group-a]], alternative='two-sided')
        else:
            _, pv = proportions_ztest([b, a], [n_per_group, n_per_group], alternative='two-sided')
        rejected += bool(pv < alpha)
    frequency = rejected / repetitions
    return dict(source='monte_carlo_simulation', n_per_group=n_per_group, true_p_a=p_a,
        true_p_b=p_b, repetitions=repetitions, seed=seed, alpha=alpha,
        rejection_frequency=frequency, monte_carlo_standard_error=math.sqrt(frequency*(1-frequency)/repetitions),
        interpretation='false_positive_frequency' if p_a == p_b else 'empirical_power')
