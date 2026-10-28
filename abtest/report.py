import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .core import save_json

DECISIONS = {
    'invalid_allocation_check_randomization_and_missing_data': 'Проверьте рандомизацию и пропуски: найден дисбаланс размеров групп (SRM). Итоговый вывод заблокирован.',
    'incomplete_collect_planned_sample_without_peeking': 'Плановая выборка не набрана. Это промежуточный результат; решение об эффекте не принимается.',
    'evidence_of_positive_effect_at_least_practical_threshold': 'Данные поддерживают положительный эффект не ниже заданного практического порога при принятых предпосылках.',
    'evidence_of_negative_effect': 'Данные указывают на отрицательный эффект.',
    'positive_statistical_effect_practical_size_not_confirmed': 'Статистически значимый рост обнаружен, но величина эффекта не подтверждена на уровне практического порога.',
    'negative_statistical_effect': 'Обнаружена статистически значимая отрицательная разница.',
    'inconclusive_not_evidence_of_no_effect': 'Недостаточно оснований отвергнуть равенство конверсий. Это не доказывает отсутствие эффекта.',
}


def write_report(directory, result, experiment_plan=None, simulation=None, monte_carlo=None):
    directory = Path(directory)
    if directory.exists():
        raise ValueError(f'Output directory exists: {directory}; choose a new one')
    directory.mkdir(parents=True)
    save_json(directory / 'result.json', result)
    if experiment_plan: save_json(directory / 'plan.json', experiment_plan)
    if simulation: save_json(directory / 'simulation.json', simulation)
    if monte_carlo: save_json(directory / 'monte_carlo.json', monte_carlo)
    low, high = result['confidence_interval_difference']
    relative = result['relative_uplift']
    uplift = 'не определён при нулевой конверсии A' if relative is None else f'{relative * 100:+.2f}%'
    lines = ['# A/B-эксперимент: анализ конверсии', '',
        '**Учебная симуляция. Данные не являются результатом реального продуктового эксперимента.**' if simulation else '**Анализ входного CSV. Происхождение данных, рандомизация и созревание конверсий требуют отдельной проверки.**', '',
        '| Группа | Пользователи | Конверсии | Конверсия |', '|---|---:|---:|---:|',
        f"| A | {result['n_a']} | {result['conversions_a']} | {result['conversion_a']:.2%} |",
        f"| B | {result['n_b']} | {result['conversions_b']} | {result['conversion_b']:.2%} |", '',
        f"Разница B − A: **{result['difference_percentage_points']:+.2f} п.п.** Относительный uplift: **{uplift}**.",
        f"{result['confidence_level']:.0%} доверительный интервал разницы: **[{low*100:+.2f}; {high*100:+.2f}] п.п.** Метод: Newcombe.",
        f"Двусторонний тест: {result['test_method']}; p-value = **{result['p_value']:.6g}**, α = {result['alpha']:.3g}.", '',
        f"Практический порог: +{result['practical_threshold']*100:.2f} п.п. Он отличается по смыслу от MDE для планирования мощности.",
        f"SRM: p-value = {result['srm_p_value']:.6g}, порог проверки = {result['srm_alpha']}; флаг дисбаланса: {result['srm_flag']}.", '',
        '**Интерпретация:** ' + DECISIONS[result['decision']], '']
    if experiment_plan:
        lines += [f"План: baseline {experiment_plan['baseline']:.1%}, MDE +{experiment_plan['mde_absolute']*100:.1f} п.п., мощность {experiment_plan['power']:.0%}, по {experiment_plan['n_per_group']} пользователей на группу. Выборка набрана: {result['planned_sample_reached']}.",
            f"Оценка набора: {experiment_plan['estimated_enrollment_days']} дней; с округлением до полных недель — {experiment_plan['suggested_full_week_days']}. После последнего включённого пользователя нужно дождаться 7-дневного окна конверсии.", '']
    if monte_carlo:
        lines += ['## Monte Carlo', '', '| Сценарий | Частота отклонения H0 | Стандартная ошибка Monte Carlo |', '|---|---:|---:|']
        for name, r in monte_carlo.items():
            lines.append(f"| {name} | {r['rejection_frequency']:.2%} | {r['monte_carlo_standard_error']:.4f} |")
        lines += ['', 'Это частоты в повторных симуляциях, а не «наблюдаемая мощность» единственного теста.', '']
    lines += ['## Допущения и ограничения', '',
        '- Один независимый пользователь — одна строка; экспериментальная единица и единица анализа совпадают.',
        '- Одна основная бинарная метрика, две группы, один заранее выбранный финальный анализ.',
        '- Нет подглядывания, досрочной остановки, перебора сегментов и выбора удачного seed.',
        '- Окно конверсии одинаковое, неполные наблюдения не исключаются после знакомства с результатом.',
        '- Статистическая значимость не гарантирует бизнес-ценность и не является вероятностью истинности H0.',
        '- MDE — эффект, для которого планируется заданная мощность; это не минимальная наблюдаемая разница и не обязательный бизнес-порог.',
        '- CI Newcombe и выбранный тест не являются точными взаимными обращениями; возле границы значимости возможны небольшие расхождения.',
        '- Корректировки на множественные сравнения, кластеризацию, CUPED и последовательный анализ не реализованы.', '',
        '![Оценка эффекта](effect.png)', '']
    (directory / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8, 3.0))
    effect = result['difference_percentage_points']
    ax.errorbar([effect], [0], xerr=[[max(0,effect-low*100)], [max(0,high*100-effect)]], fmt='o', capsize=8, color='#2563eb')
    ax.axvline(0, color='#64748b', linestyle='--', label='No difference')
    ax.axvline(result['practical_threshold']*100, color='#059669', linestyle=':', label='Practical threshold')
    ax.set_xlabel('Conversion difference B - A (percentage points)')
    ax.set_yticks([]); ax.set_ylim(-1, 1); ax.set_title('Effect estimate and confidence interval')
    ax.legend(loc='upper left', fontsize=8); ax.grid(axis='x', alpha=.2)
    fig.tight_layout(); fig.savefig(directory/'effect.png', dpi=150); plt.close(fig)
