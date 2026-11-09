"""Publication-style plots from completed frozen V21 simulation records."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

COLORS={'Guarded':'#2877ad','Fixed9dB':'#7f8791','Unguarded':'#d98c39'}
LABELS={'Guarded':'Guarded contextual','Fixed9dB':'Fixed 9-dB threshold','Unguarded':'Unguarded contextual'}
SHORT={'nominal':'Nominal','light_traffic':'Light traffic','heavy_traffic':'Heavy traffic','weaker_channel':'Weak channel','stronger_channel':'Strong channel','heavy_weak':'Heavy + weak'}

def make_plots(rows,summary,pairs,seeds,scenarios,out,cap):
    out=Path(out);fig=out/'figures';fig.mkdir(exist_ok=True)
    idx={(r['scenario'],r['method']):r for r in summary}; methods=list(COLORS); labels=[SHORT.get(s,s) for s in scenarios]
    def save(name):
        plt.tight_layout(pad=1.3);plt.savefig(fig/(name+'.pdf'),bbox_inches='tight');plt.savefig(fig/(name+'.png'),dpi=210,bbox_inches='tight');plt.close()
    def legend(ax,cols=3):
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=cols,frameon=False,fontsize=8)
    x=np.arange(len(scenarios));width=.25
    plt.figure(figsize=(10,3.6));ax=plt.gca()
    for j,m in enumerate(methods):ax.bar(x+(j-1)*width,[idx[s,m]['throughput_mbps'] for s in scenarios],width,color=COLORS[m],label=LABELS[m])
    ax.set_xticks(x,labels);ax.set_ylabel('Mean throughput (Mbps)');ax.set_ylim(bottom=0);legend(ax);save('Fig02_Throughput_Across_Scenarios')
    vals=[next(r for r in pairs if r['scenario']==s and r['comparison']=='Guarded_minus_Fixed9dB') for s in scenarios]
    means=np.array([v['diff_mbps'] for v in vals]);lower=np.array([v['ci_low'] for v in vals]);upper=np.array([v['ci_high'] for v in vals]);y=np.arange(len(scenarios))
    plt.figure(figsize=(8.3,3.5));ax=plt.gca();ax.errorbar(means,y,xerr=np.array([means-lower,upper-means]),fmt='o',capsize=3,color=COLORS['Guarded'],ms=5);ax.axvline(0,color='gray',linestyle='--',linewidth=1);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Guarded minus fixed-threshold throughput (Mbps)');save('Fig03_Paired_Throughput_Effects')
    preferred=[s for s in ('nominal','stronger_channel') if s in scenarios]
    if preferred:
        f,axes=plt.subplots(1,len(preferred),figsize=(8 if len(preferred)==2 else 5,3.6),squeeze=False)
        for ax,s in zip(axes[0],preferred):
            for m in methods:ax.scatter(idx[s,m]['energy_mj_step'],idx[s,m]['throughput_mbps'],s=65,color=COLORS[m],label=LABELS[m],zorder=3)
            ax.axvline(cap,color='#c33',linestyle='--',linewidth=1);ax.set_title(SHORT[s],fontsize=10);ax.set_xlabel('Energy (mJ per environment-step)');ax.set_ylabel('Throughput (Mbps)');ax.grid(alpha=.15)
        legend(axes[0][0]);save('Fig04_Throughput_Energy_Operating_Points')
    f,axes=plt.subplots(1,2,figsize=(10,3.6));width=.25
    for j,m in enumerate(methods):
        axes[0].bar(x+(j-1)*width,[100*idx[s,m]['episode_violations']/idx[s,m]['n_episodes'] for s in scenarios],width,label=LABELS[m],color=COLORS[m])
        axes[1].bar(x+(j-1)*width,[100*idx[s,m]['seed_feasible']/idx[s,m]['n_seeds'] for s in scenarios],width,color=COLORS[m])
    for ax in axes:
        ax.set_xticks(x,labels,rotation=20,ha='right',fontsize=8);ax.set_ylim(0,105)
    axes[0].set_ylabel('Episodes exceeding cap (%)');axes[1].set_ylabel('Feasible seed clusters (%)');legend(axes[0]);save('Fig05_Energy_Constraint_Compliance')
    if 'nominal' in scenarios:
        f,ax=plt.subplots(figsize=(8,3.8))
        for m in methods:
            byseed=[np.mean([r['throughput_mbps'] for r in rows if r['scenario']=='nominal' and r['method']==m and r['seed']==s]) for s in seeds]
            y=np.cumsum(byseed)/np.arange(1,len(byseed)+1);ax.plot(np.arange(1,len(y)+1),y,label=LABELS[m],color=COLORS[m],linewidth=1.7)
        ax.set_xlabel('Cumulative independent seed clusters');ax.set_ylabel('Running mean throughput (Mbps)');legend(ax);save('Fig06_MonteCarlo_Stabilization')
    panels=[('Mean throughput (Mbps)',lambda r:r['throughput_mbps'],'.2f'),('Energy budget used (%)',lambda r:100*r['energy_mj_step']/cap,'.1f'),('Episodes above cap (%)',lambda r:100*r['episode_violations']/r['n_episodes'],'.0f'),('Feasible seeds (%)',lambda r:100*r['seed_feasible']/r['n_seeds'],'.0f')]
    f,axes=plt.subplots(2,2,figsize=(9.2,6.4));axes=axes.flatten()
    for ax,(title,fn,fmt) in zip(axes,panels):
        values=np.array([[fn(idx[s,m]) for m in methods] for s in scenarios]);im=ax.imshow(values,cmap='YlGnBu',aspect='auto');ax.set_title(title,fontsize=10);ax.set_xticks(range(len(methods)),['Guarded','Fixed 9 dB','Unguarded'],rotation=22,ha='right',fontsize=8);ax.set_yticks(range(len(scenarios)),labels,fontsize=8)
        for i in range(len(scenarios)):
            for j in range(len(methods)):
                val=values[i,j];lo,hi=np.nanmin(values),np.nanmax(values);color='white' if hi>lo and (val-lo)/(hi-lo)>.65 else 'black'
                ax.text(j,i,format(val,fmt),ha='center',va='center',fontsize=8,color=color)
    save('Fig07_Scenario_Metric_Heatmap')
    print('FIGURES:',fig,flush=True)

if __name__=='__main__':
    import argparse,csv,json
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='outputs/final_long');a=parser.parse_args();folder=Path(a.out)
    with (folder/'protocol.json').open() as f:d=json.load(f)
    def read(name):
        with (folder/name).open(newline='') as f:return list(csv.DictReader(f))
    raw=read('episodes.csv');summary=read('summary.csv');paired=read('paired_ci.csv')
    for row in raw:
        for k in ('seed','episode'):row[k]=int(row[k])
        for k in ('throughput_mbps','energy_mj_step','low_fraction'):row[k]=float(row[k])
    for row in summary:
        for k in ('throughput_mbps','energy_mj_step','low_fraction','denial_rate'):row[k]=float(row[k])
        for k in ('n_seeds','n_episodes','seed_feasible','episode_violations'):row[k]=int(row[k])
    for row in paired:
        for k in ('diff_mbps','ci_low','ci_high'):row[k]=float(row[k])
    make_plots(raw,summary,paired,list(range(d['seed_start'],d['seed_start']+d['n_seeds'])),d['scenarios'],folder,d['energy_cap_mj_step'])
