r"""Regenerate V21 manuscript Figures 2--7 from completed evaluation CSVs.

Usage (from Sensors_V21_Final_LongRun folder):
    py -3.13 .\make_figures.py --out outputs\final_200

Inputs (all created by run_final.py): episodes.csv, summary.csv, paired_ci.csv,
protocol.json. Outputs: figures/Fig02_... through Fig07_..., as PDF and PNG.
Only figure presentation changes. No experiment is rerun and no results are altered.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

COLORS = {'Guarded': '#2877ad', 'Fixed9dB': '#7f8791', 'Unguarded': '#d98c39'}
LABELS = {'Guarded': 'Guarded contextual', 'Fixed9dB': 'Fixed 9-dB threshold', 'Unguarded': 'Unguarded contextual'}
SHORT = {'nominal': 'Nominal', 'light_traffic': 'Light traffic', 'heavy_traffic': 'Heavy traffic', 'weaker_channel': 'Weaker channel', 'stronger_channel': 'Stronger channel', 'heavy_weak': 'Heavy + weak'}
METHODS = list(COLORS)

plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})


def save_figure(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight', pad_inches=0.12)
    fig.savefig(output / f'{name}.png', dpi=250, bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)


def shared_legend(fig, handles, labels, y=0.975):
    """A true figure-level legend, centered independently of any subplot."""
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, y), ncol=len(labels), frameon=False, fontsize=9, columnspacing=2.4, handletextpad=0.6, borderaxespad=0.0)


def make_plots(rows, summary, pairs, seeds, scenarios, out, cap):
    out = Path(out)
    folder = out / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    index = {(r['scenario'], r['method']): r for r in summary}
    missing = [(s, m) for s in scenarios for m in METHODS if (s, m) not in index]
    if missing:
        raise ValueError(f'Missing scenario/method summary rows: {missing}')
    names = [SHORT.get(s, s) for s in scenarios]
    x = np.arange(len(scenarios))
    bar_width = 0.24

    # Figure 2: Throughput across scenarios.
    fig, ax = plt.subplots(figsize=(11.2, 4.35))
    for j, method in enumerate(METHODS):
        ax.bar(x + (j - 1) * bar_width, [index[s, method]['throughput_mbps'] for s in scenarios], width=bar_width, color=COLORS[method], label=LABELS[method])
    ax.set_xticks(x, names)
    ax.set_ylabel('Mean throughput (Mbps)')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.14)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    shared_legend(fig, handles, labels)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.83, bottom=0.16)
    save_figure(fig, folder, 'Fig02_Throughput_Across_Scenarios')

    # Figure 3: Paired seed-cluster throughput effects, not independent episodes.
    effects = [next(p for p in pairs if p['scenario'] == s and p['comparison'] == 'Guarded_minus_Fixed9dB') for s in scenarios]
    means = np.array([p['diff_mbps'] for p in effects])
    low = np.array([p['ci_low'] for p in effects])
    high = np.array([p['ci_high'] for p in effects])
    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    ax.errorbar(means, x, xerr=np.vstack([np.maximum(0, means - low), np.maximum(0, high - means)]), fmt='o', capsize=4, color=COLORS['Guarded'], ms=5.5, lw=1.7)
    ax.axvline(0, color='#888888', linestyle='--', linewidth=1)
    ax.set_yticks(x, names)
    ax.invert_yaxis()
    ax.set_xlabel('Guarded minus fixed-threshold throughput (Mbps)')
    ax.grid(axis='x', alpha=0.13)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.95, bottom=0.16)
    save_figure(fig, folder, 'Fig03_Paired_Throughput_Effects')

    # Figure 4: Corrected centered legend, safely ABOVE panel headings.
    selected = [s for s in ('nominal', 'stronger_channel') if s in scenarios]
    if selected:
        fig, axes = plt.subplots(1, len(selected), figsize=(11.8 if len(selected) == 2 else 6.5, 4.8), squeeze=False)
        for ax, scenario in zip(axes[0], selected):
            for method in METHODS:
                ax.scatter(index[scenario, method]['energy_mj_step'], index[scenario, method]['throughput_mbps'], s=80, color=COLORS[method], label=LABELS[method], zorder=4)
            ax.axvline(cap, color='#bf4444', linestyle='--', linewidth=1.2, zorder=1)
            ax.set_title(SHORT[scenario], pad=12, fontsize=11)
            ax.set_xlabel('Energy (mJ per environment-step)', labelpad=9)
            ax.set_ylabel('Throughput (Mbps)')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.tick_params(axis='x', labelrotation=0, labelsize=9)
            ax.grid(alpha=0.16, zorder=0)
            ax.margins(x=0.11, y=0.12)
        handles, labels = axes[0][0].get_legend_handles_labels()
        shared_legend(fig, handles, labels, y=0.985)
        # These reserved regions are intentional. Do not use plt.tight_layout() here.
        fig.subplots_adjust(left=0.085, right=0.985, top=0.76, bottom=0.22, wspace=0.37)
        save_figure(fig, folder, 'Fig04_Throughput_Energy_Operating_Points')

    # Figure 5: Energy violations and seed-average compliance.
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.6))
    for j, method in enumerate(METHODS):
        axes[0].bar(x + (j-1)*bar_width, [100*index[s, method]['episode_violations']/index[s, method]['n_episodes'] for s in scenarios], bar_width, label=LABELS[method], color=COLORS[method])
        axes[1].bar(x + (j-1)*bar_width, [100*index[s, method]['seed_feasible']/index[s, method]['n_seeds'] for s in scenarios], bar_width, color=COLORS[method])
    axes[0].set_ylabel('Episodes exceeding cap (%)')
    axes[1].set_ylabel('Feasible seed clusters (%)')
    for ax in axes:
        ax.set_xticks(x, names, rotation=25, ha='right')
        ax.tick_params(axis='x', labelsize=8)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.12)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    shared_legend(fig, handles, labels)
    fig.subplots_adjust(left=0.075, right=0.987, top=0.80, bottom=0.27, wspace=0.35)
    save_figure(fig, folder, 'Fig05_Energy_Constraint_Compliance')

    # Figure 6: Monte Carlo estimate stabilization over independent seed clusters.
    if 'nominal' in scenarios:
        fig, ax = plt.subplots(figsize=(10.3, 4.3))
        row_map = {(r['scenario'], r['method'], r['seed']): [] for r in rows if r['scenario'] == 'nominal'}
        for row in rows:
            if row['scenario'] == 'nominal':
                row_map[row['scenario'], row['method'], row['seed']].append(row['throughput_mbps'])
        for method in METHODS:
            values = np.array([np.mean(row_map['nominal', method, s]) for s in seeds])
            running = np.cumsum(values)/np.arange(1, len(values)+1)
            ax.plot(np.arange(1,len(running)+1), running, color=COLORS[method], label=LABELS[method], linewidth=1.8)
        ax.set_xlabel('Cumulative independent seed clusters')
        ax.set_ylabel('Running mean throughput (Mbps)')
        ax.grid(alpha=0.12)
        handles, labels = ax.get_legend_handles_labels()
        shared_legend(fig, handles, labels)
        fig.subplots_adjust(left=0.10, right=0.985, top=0.82, bottom=0.17)
        save_figure(fig, folder, 'Fig06_MonteCarlo_Stabilization')

    # Figure 7: Joint scenario-metric heatmap, no global legend.
    panels = [('Mean throughput (Mbps)', lambda r: r['throughput_mbps'], '.2f'), ('Energy budget used (%)', lambda r: 100*r['energy_mj_step']/cap, '.1f'), ('Episodes above cap (%)', lambda r: 100*r['episode_violations']/r['n_episodes'], '.0f'), ('Feasible seeds (%)', lambda r: 100*r['seed_feasible']/r['n_seeds'], '.0f')]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.9))
    for ax, (title, selector, fmt) in zip(axes.flat, panels):
        values = np.array([[selector(index[s, m]) for m in METHODS] for s in scenarios])
        ax.imshow(values, cmap='YlGnBu', aspect='auto')
        ax.set_title(title, fontsize=11, pad=9)
        ax.set_xticks(range(len(METHODS)), ['Guarded', 'Fixed 9 dB', 'Unguarded'])
        ax.set_yticks(range(len(scenarios)), names)
        ax.tick_params(axis='both', length=0, labelsize=9)
        minimum, maximum = np.min(values), np.max(values)
        for i in range(len(scenarios)):
            for j in range(len(METHODS)):
                fraction = (values[i,j]-minimum)/(maximum-minimum) if maximum > minimum else 0
                ax.text(j, i, format(values[i,j], fmt), ha='center', va='center', fontsize=9, color='white' if fraction > 0.65 else '#171717')
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.08, top=0.93, wspace=0.41, hspace=0.38)
    save_figure(fig, folder, 'Fig07_Scenario_Metric_Heatmap')
    print(f'FIGURES COMPLETE: {folder.resolve()}', flush=True)


def read_csv(directory, filename):
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f'Required completed-run file is missing: {path}')
    with path.open(newline='', encoding='utf-8-sig') as source:
        return list(csv.DictReader(source))


def main():
    parser = argparse.ArgumentParser(description='Regenerate publication figures from frozen V21 simulation outputs without rerunning experiments.')
    parser.add_argument('--out', default='outputs/final_200', help='Completed output directory containing episodes.csv, summary.csv, paired_ci.csv and protocol.json')
    args = parser.parse_args()
    directory = Path(args.out)
    with (directory/'protocol.json').open(encoding='utf-8') as source:
        protocol = json.load(source)
    episodes = read_csv(directory, 'episodes.csv')
    summary = read_csv(directory, 'summary.csv')
    paired = read_csv(directory, 'paired_ci.csv')
    for row in episodes:
        row['seed'] = int(row['seed'])
        row['episode'] = int(row['episode'])
        row['throughput_mbps'] = float(row['throughput_mbps'])
        row['energy_mj_step'] = float(row['energy_mj_step'])
        row['low_fraction'] = float(row['low_fraction'])
    for row in summary:
        for field in ('throughput_mbps','energy_mj_step','low_fraction','denial_rate'):
            row[field] = float(row[field])
        for field in ('n_seeds','n_episodes','seed_feasible','episode_violations'):
            row[field] = int(row[field])
    for row in paired:
        for field in ('diff_mbps', 'ci_low', 'ci_high'):
            row[field] = float(row[field])
    seeds = list(range(int(protocol['seed_start']), int(protocol['seed_start'])+int(protocol['n_seeds'])))
    make_plots(episodes, summary, paired, seeds, protocol['scenarios'], directory, float(protocol['energy_cap_mj_step']))


if __name__ == '__main__':
    main()
