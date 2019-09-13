import scanpy.api as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80)  # low dpi (dots per inch) yields small inline figures
sc.logging.print_version_and_date()
sc.logging.print_versions_dependencies_numerics()

color_dict = {'S': 'red', 'S-shift': 'red', 'G2/M': 'blue', 'G2/M-shift': 'blue', 
              'G1': 'green', 'G1-shift': 'green',
              'G2': 'orange', 'G2-shift': 'orange', 
              'G1/S': 'brown', 'G1/S-shift': 'brown', 
              'M/G1': 'purple', 'M/G1-shift': 'purple', 
              'not_assigned': 'black'}

def compute_angles(points):
    angles = np.arctan2(points[:,0], points[:,1])
    angles[angles < 0] += 2*np.pi
    return angles

def plot_phase_bar_dist(data,bin_size,return_data=False,plot_path=None):
    data_bb = data.copy()
    series_dict = {}
    for p in list(data_bb.obs['phase'].unique()):
        data_bb.obs['{}_count'.format(p)] = (data_bb.obs['phase'] == p).astype(int)
    data_bb.obs['phase_group'] = (np.arange(len(data_bb.obs)) // bin_size)*bin_size
    for p in list(data_bb.obs['phase'].unique()):
        series_dict['{}_counts'.format(p)] = data_bb.obs.groupby(['phase_group'])['{}_count'.format(p)].sum()
    count_df = pd.DataFrame({'G1': list(series_dict['G1_counts']), 
                             'S': list(series_dict['S_counts']), 
                             'G2M': list(series_dict['G2M_counts'])
                            }, 
                            index=np.arange(0,len(data_bb.obs),bin_size)
                    )
    
    if plot_path:
        if not os.path.exists(plot_path):
                os.makedirs(plot_path, exist_ok=True)
        #count_df.plot.bar(figsize=(30,10), xlabel='Pseudotime order', ylabel='Number of cells').get_figure().savefig(os.path.join(plot_path,"bar_dist.pdf"),bbox_inches = "tight")
        ax = count_df.plot.bar(figsize=(30,10))
        ax.set_xlabel('Pseudotime order')
        ax.set_ylabel('Number of cells')
        plt.savefig(os.path.join(plot_path,"bar_dist.pdf"),bbox_inches='tight')
    else:
        count_df.plot.bar(figsize=(30,10))
    
    values = count_df.as_matrix(columns=data_bb.obs['phase'].unique())
    for p in list(data_bb.obs['phase'].unique()):
        count_df['{}_ratio'.format(p)] = count_df[p]/values.sum(1)
    
    ratio_df = pd.DataFrame({'G1': list(count_df['G1_ratio']), 
                             'S': list(count_df['S_ratio']), 
                             'G2M': list(count_df['G2M_ratio'])
                            }, 
                            index=np.arange(0,len(data_bb.obs),bin_size)
                        )
    if plot_path:
        #count_df.plot.bar(figsize=(30,10)).get_figure().savefig(os.path.join(plot_path,"bar_dist.pdf"),bbox_inches = "tight")
        #ratio_df.plot(kind='line', figsize=(30,10), xlabel='Pseudotime order', ylabel='fractiosn', use_index=True).get_figure().savefig(os.path.join(plot_path,"line_dist.pdf"),bbox_inches = "tight")
        ax = ratio_df.plot(kind='line', figsize=(30,10), use_index=True)
        ax.set_xlabel('Pseudotime order')
        ax.set_ylabel('Fraction')
        plt.savefig(os.path.join(plot_path,"line_dist.pdf"),bbox_inches='tight')
    else:
        #count_df.plot.bar(figsize=(30,10))
        ratio_df.plot(kind='line', figsize=(30,10), use_index=True)
        
    if return_data:
        return count_df, ratio_df
    
def shift_data(data, n, direction = 'positive', reverse = False):
    if not direction in ['positive', 'negative']:
        raise ValueError('direction must be: positive,negative')
    
    order = data.obs['order'].copy()
    if direction == 'negative':
        data.obs['order'] -= n
        data.obs['order'][data.obs['order'] < 0] += len(data.obs)
    else:
        data.obs['order'] += n
        data.obs['order'][data.obs['order'] > len(data.obs)] -= len(data.obs)
    sort_order = data.obs['order'].argsort()[::-1] if reverse else data.obs['order'].argsort() 
    data = data[sort_order,:].copy()
    data.obs['order'] = np.arange(len(data.obs))
    return data


def find_phase_boundaries(data, ratio_df):
    #TODO: CHEKC IF ORIENTATION IS CORRECT
    #TODO: SHIFT DATA IF NECESSARY
    from scipy.interpolate import interp1d
    start_point = np.where(ratio_df['G1'] == np.max(ratio_df['G1']))[0][0]
    
    s_first = (np.where(ratio_df['S'][start_point+1:]>ratio_df['G1'][start_point+1:])[0] + start_point+1)[0]
    s_start = interp1d(ratio_df['G1'][start_point:s_first+1].values-ratio_df['S'][start_point:s_first+1].values,ratio_df[start_point:s_first+1].index)(0)
    
    
    g2m_first = (np.where(ratio_df['G2M'][s_first+1:]>ratio_df['S'][s_first+1:])[0] + s_first+1)[0]
    
    g2m_start = interp1d(ratio_df['S'][s_first+1:g2m_first+1].values-ratio_df['G2M'][s_first+1:g2m_first+1],ratio_df[s_first+1:g2m_first+1].index)(0)
    
    g1_first = (np.where(ratio_df['G1'][g2m_first+1:]>ratio_df['G2M'][g2m_first+1:])[0] + g2m_first+1)
    #if g1_first is empty, check if they cross in [0,s_first]
    if g1_first.size == 0:
        g1_first = np.where(ratio_df['G1'][:s_first]>ratio_df['G2M'][:s_first])[0][0]
        if g1_first == 0:
            g1_start = 0
        else:
            g1_start = interp1d(ratio_df['G2M'][:g1_first+1].values-ratio_df['G1'][:g1_first+1],ratio_df[:g1_first+1].index)(0)
    else:
        g1_start = interp1d(ratio_df['G2M'][g2m_first+1:].values-ratio_df['G1'][g2m_first+1:],ratio_df[g2m_first+1:].index)(0)
    
    return int(round(float(g1_start))), int(round(float(s_start))), int(round(float(g2m_start)))

def get_times(data):
    if not data.uns.get('phase_boundaries',None):
        raise ValueError("data.uns['phase_boundaries'] must exist and contain g1_start, s_start and g2m_start")
    
    N = data.uns['phase_boundaries']['s_start']
    g1_times = np.linspace(0,11.0-11.0/N,N)

    N = data.uns['phase_boundaries']['g2m_start']-data.uns['phase_boundaries']['s_start']
    s_times = np.linspace(11.0,19.0-8.0/N,N)
    
    N = len(data.obs)-data.uns['phase_boundaries']['g2m_start']
    g2m_times = np.linspace(19.0,24.0-5.0/N,N)
    
    return np.concatenate((g1_times,s_times,g2m_times),axis=None)

def get_times_alternative(data):
    if not data.uns.get('phase_boundaries',None):
        raise ValueError("data.uns['phase_boundaries'] must exist and contain g1_start, s_start and g2m_start")
    """
    g1_times = np.arange(0,11.0,11.0/data.uns['phase_boundaries']['s_start'])
    s_times = np.arange(11.0,19.0,8.0/(data.uns['phase_boundaries']['g2m_start']-data.uns['phase_boundaries']['s_start']))
    g2m_times = np.arange(19.0,24.0,5.0/(len(data.obs)-data.uns['phase_boundaries']['g2m_start']))
    """
    s_start = data.uns['phase_boundaries']['s_start']
    g2m_start = data.uns['phase_boundaries']['g2m_start']
    
    g1_angles = data.obs['order'][:s_start]
    g1_times = scale_range(g1_angles)*11.0
    
    s_angles = data.obs['order'][s_start:g2m_start]
    s_times = scale_range(s_angles)*8.0 + 11.0
    
    g2m_angles = data.obs['order'][g2m_start:]
    g2m_times = scale_range(g2m_angles)*5.0 + 19.0
    
    return np.concatenate((g1_times,s_times,g2m_times),axis=None)

def scale_range(my_range):
    return (my_range-np.min(my_range))/np.max(my_range-np.min(my_range))

def compare_marker_genes_per_phase(data,g1_start,s_start,g2m_start,skip_dropouts=False,plot_path="./figures"):
    s_genes = ['PCNA','NASP', 'RRM1', 'RRM2', 'TYMS']
    g2m_genes = ['TOP2A', 'PTTG1']
    g1_genes = ['CDC123', 'CDK6']
    
    known_df = pd.read_csv("./data/knownGenes_derived.txt",sep=" ").dropna()
    
    if skip_dropouts:
        data
    
    data_g1 = data[g1_start:s_start-1,:]
    data_s = data[s_start:g2m_start-1,:]
    data_g2m = data[g2m_start:,:]
    
    expression_df = pd.DataFrame()
    
    for p in ['G1','S','G2/M','G2']:
        p_df = known_df[known_df['phase'] == p]
        for i, r in p_df.iterrows():
            if not r['Ensembl'] in list(data.var.index):
                continue
            gene = r['Ensembl']
            
            tot_expr = data[:,gene].X.mean()
            
            if skip_dropouts:
                data_g1 = data[g1_start:s_start-1,:].copy()
                data_g1 = data_g1[data_g1[:,gene].X>0,:]
                data_s = data[s_start:g2m_start-1,:].copy()
                data_s = data_s[data_s[:,gene].X>0,:]
                data_g2m = data[g2m_start:,:].copy()
                data_g2m = data_g2m[data_g2m[:,gene].X>0,:]
                data_nonzero = data[data[:,gene].X>0,:].copy()
                tot_expr = data_nonzero[:,gene].X.mean()
            
            expression_df = expression_df.append({
                'known_phase': p,
                'Ensembl': gene,
                'gene_symbol': r['gene'],
                'G1_expr': data_g1[:,gene].X.mean(),
                'S_expr': data_s[:,gene].X.mean(),
                'G2M_expr': data_g2m[:,gene].X.mean(),
                'Total_expr': tot_expr
            }, 
                ignore_index=True)
    
    expression_df.index = expression_df['gene_symbol']
    #print(expression_df)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)
    for p in expression_df['known_phase'].unique():
        phase_df = expression_df[expression_df['known_phase'] == p].copy()
        if phase_df.empty:
            print("No {} genes detected".format(p))
            continue
        #print(phase_df)
        ax = phase_df.plot(kind='bar', title='Mean expression of {} marker genes by modelled phase'.format(p), figsize=(10,10))
        ax.set_xlabel('Gene symbols')
        ax.set_ylabel('Reads Per Million')
        plt.savefig(os.path.join(plot_path,"mean_expression_{}.pdf".format(p.replace('/','-'))))
        
    return expression_df


def score_ordering(expression_df):
    score_dict = {}
    ind_dict = {'G1': 0, 'S': 1, 'G2/M': 2, 'G2': 2}
    total_score = 0
    for p in expression_df['known_phase'].unique():
        print(p)
        p_df = expression_df[expression_df['known_phase'] == p]
        expr_mat = p_df[['G1_expr','S_expr','G2M_expr']].as_matrix()
        score_list = []
        for row in expr_mat:
            score_list.append(int(np.where(row == np.max(row))[0][0] == ind_dict[p]))    
        phase_score = sum(score_list)
        score_dict[p] = phase_score
        total_score += phase_score
        print("{}/{} genes classified correctly".format(phase_score,len(p_df)))
        
    score_dict['total'] = total_score
    score_dict['n_marker_genes'] = len(expression_df)
    print("Total score: {}/{} classified correctly".format(total_score,len(expression_df)))
    print("Total percent {} classified correctly".format(round(100*float(total_score)/float(len(expression_df)),2)))
    
    
def plot_gene(data, gene, color_phase=False):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    y = data[:,[gene]].X
    x = np.arange(data.shape[0])
    ys = lowess(y, x)[:,1]
    plt.figure()
    plt.plot(x,ys,linewidth=2, label='Lowess',color='orange')
    if not color_phase:
        plt.plot(x, y, 'o', color='gray', alpha=0.2, markersize=3)
    else:
        color_map = {'G1': 'green','S': 'red', 'G2M': 'blue'}
        for p in ['G1','S','G2M']:
            data_p = data[data.obs.phase == p,:]
            y = data_p[:,[gene]].X
            x = np.arange(data_p.shape[0]) + np.where(data.obs.index == (data.obs.phase == p).idxmax())[0][0]
            plt.plot(x, y, 'o', color=color_map[p], alpha=0.2, markersize=3, label=p)
        plt.legend()
    #plt.title(gene)
    plt.title(data.var.loc[gene]['gene_symbols'])
    
    
def plot_gene_new(data, gene, color_phase=False, skip_dropouts=False):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    if skip_dropouts:
        data = data[data[:,gene].X >0,:].copy()
    y = data[:,[gene]].X
    try:
        x = data.obs['times']
    except Exception as e:
        x = data.obs['order']
    ys = lowess(y, x)[:,1]
    plt.figure(figsize=(10,10))
    plt.plot(x,ys,linewidth=2, label='Lowess',color='orange')
    if not color_phase:
        plt.plot(x, y, 'o', color='gray', alpha=0.2, markersize=3)
    else:
        color_map = {'G1': 'green','S': 'red', 'G2M': 'blue'}
        for p in ['G1','S','G2M']:
            data_p = data[data.obs.phase == p,:]
            y = data_p[:,[gene]].X
            x = data_p.obs['order']
            plt.plot(x, y, 'o', color=color_map[p], alpha=0.2, markersize=3, label=p)
        plt.legend()
    #plt.title(gene)
    plt.title(data.var.loc[gene]['gene_symbols'])
    
    
def plot_known_genes(data, color_phase=False, skip_dropouts=False, plot_path='data/phase_plots', return_residuals=False, custom_genes=None):
    from math import ceil
    
    if not custom_genes:
        known_df = pd.read_csv("./data/knownGenes_derived.txt",sep=" ").dropna()
    else:
        known_df = pd.read_csv(custom_genes,sep=" ").dropna()
    
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    if return_residuals:
        res = {}
        
    for p in ['G1','S','G2/M','G2']:
        print("Plotting known {} genes".format(p))
        p_df = known_df[known_df['phase'] == p]
        ens_list = list(set(list(p_df['Ensembl'])).intersection(list(data.var.index)))
        col_size = 4
        n_cols = min(col_size, len(ens_list))
        n_rows = ceil(len(ens_list)/n_cols)
        row = 0 
        
        #plt.figure(figsize=(10,10))
        fig, ax = plt.subplots(n_rows,n_cols,squeeze=False,figsize=(3*n_cols,3*n_rows),sharey=True,sharex=True)

        #fig.suptitle("Known {} genes".format(p), size=14)

        for i, gene in enumerate(sorted(ens_list)):
            col = i%n_cols
            y = data[:,[gene]].X
            try:
                x = data.obs['time']
            except Exception as e:
                x = data.obs['order']
            if skip_dropouts:
                indexes = (y>0)
                y = y[indexes]
                x = x[indexes]
            ys = lowess(y, x,frac=0.45,it=0)[:,1]
            if return_residuals:
                res[gene] = {'phase': p, 'res_2-norm': np.linalg.norm(y-ys,axis=0), 'res_mean': np.mean(abs(y-ys)), 'res_var': np.var(abs(y-ys),ddof=1),'gene_symbol': data.var.loc[gene]['gene_symbols']}
            lines = []
            l = ax[row,col].plot(x,ys,linewidth=2, label='Lowess',color='orange',zorder=1)
            lines.append(l)
            if not color_phase:
                l = ax[row,col].plot(x, y, 'o', color='gray', alpha=0.2, markersize=3)
                lines.append(l)
            else:
                color_map = {'G1': 'green','S': 'red', 'G2M': 'blue'}
                for cp in ['G1','S','G2M']:
                    data_p = data[data.obs.phase == cp,:]
                    y = data_p[:,[gene]].X
                    try:
                        x = data_p.obs['time']
                    except Exception as e:
                        x = data_p.obs['order']
                    if skip_dropouts:
                        indexes = (y>0)
                        y = y[indexes]
                        x = x[indexes]
                    l = ax[row,col].plot(x, y, 'o', color=color_map[cp], alpha=0.2, markersize=3, label=cp, zorder=0)
                    lines.append(l)
            ax[row,col].set_title(data.var.loc[gene]['gene_symbols'])
            #ax[row,col].set_ylabel('log10(RPM+1)')
            #ax[row,col].set_xlabel('hours')
            #ax[row,col].tick_params(axis='x', labelcolor='none', top=False, bottom=False, labelbottom=False)
            if ((i+1)%n_cols == 0) and (i > 0):
                row += 1
        for i in range(n_rows*n_cols-len(ens_list)):
            fig.delaxes(ax[row,n_cols-1-i])
        #fig.legend(lines, labels=['Lowess', 'G1', 'S', 'G2M'], loc='center right')
        fig.add_subplot(111, frameon=False)
        plt.grid(False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.ylabel('Reads Per Million (n)')
        plt.xlabel('Pseudotime (hours)')
        plt.gcf().subplots_adjust(bottom=0.25)
        #plt.tight_layout()
        #fig.text(0.5, 0.04, 'hours', ha='center')
        #fig.text(0.04, 0.5, 'log10(RPM+1)', va='center', rotation='vertical')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        fig.savefig(os.path.join(plot_path,'known_{}.png'.format(p.replace('/','-'))),bbox_inches='tight')
    
    if return_residuals:
        return res
    
def plot_genes_from_list(data, ens_list, plot_lowess=False, color_phase=False, plot_filename='data/phase_plots/plot.png'):
    from math import ceil
    
    from statsmodels.nonparametric.smoothers_lowess import lowess
        
    ens_list = list(set(list(ens_list)).intersection(list(data.var.index)))
    col_size = 4
    n_cols = min(col_size, len(ens_list))
    n_rows = ceil(len(ens_list)/n_cols)
    row = 0 

    fig, ax = plt.subplots(n_rows,n_cols,squeeze=False,figsize=(3*n_cols,3*n_rows),sharey=True,sharex=True)

    for i, gene in enumerate(sorted(ens_list)):
        col = i%n_cols
        y = data[:,[gene]].X
        try:
            x = data.obs['time']
            plt.setp(ax, xticks = np.arange(0,25,6))
        except Exception as e:
            x = data.obs['order']
        lines = []
        if plot_lowess:
            ys = lowess(y, x,frac=0.45,it=0)[:,1]
            l = ax[row,col].plot(x,ys,linewidth=2, label='Lowess',color='orange',zorder=1)
            lines.append(l)
        if not color_phase:
            l = ax[row,col].plot(x, y, 'o', color='gray', alpha=0.2, markersize=3)
            lines.append(l)
        else:
            color_map = {'G1': 'green','S': 'red', 'G2M': 'blue'}
            for cp in ['G1','S','G2M']:
                data_p = data[data.obs.phase == cp,:]
                y = data_p[:,[gene]].X
                try:
                    x = data_p.obs['time']
                except Exception as e:
                    x = data_p.obs['order']
                l = ax[row,col].plot(x, y, 'o', color=color_map[cp], alpha=0.2, markersize=3, label=cp, zorder=0)
                lines.append(l)
        ax[row,col].set_title(data.var.loc[gene]['gene_symbols'])

        if ((i+1)%n_cols == 0) and (i > 0):
            row += 1
    for i in range(n_rows*n_cols-len(ens_list)):
        fig.delaxes(ax[row,n_cols-1-i])

    fig.add_subplot(111, frameon=False)
    plt.grid(False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Reads Per Million (n)')
    plt.xlabel('Pseudotime (hours)')
    plt.gcf().subplots_adjust(bottom=0.25)

    if not os.path.exists(os.path.dirname(plot_filename)):
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    fig.savefig(plot_filename,bbox_inches='tight')
    

def plot_known_genes_splines(data, color_phase=False, plot_path='data/spline_phase_plots', custom_genes=None, plot_ctrl_pts=False):
    from math import ceil
    
    if not custom_genes:
        known_df = pd.read_csv("./data/knownGenes_derived.txt",sep=" ").dropna()
    else:
        known_df = pd.read_csv(custom_genes,sep=" ").dropna()
        
    for p in ['G1','S','G2/M','G2']:
        print("Plotting known {} genes".format(p))
        p_df = known_df[known_df['phase'] == p]
        ens_list = list(set(list(p_df['Ensembl'])).intersection(list(data.var.index)))
        col_size = 4
        n_cols = min(col_size, len(ens_list))
        n_rows = ceil(len(ens_list)/n_cols)
        row = 0 
        
        fig, ax = plt.subplots(n_rows,n_cols,squeeze=False,figsize=(3*n_cols,3*n_rows),sharey=True,sharex=True)
        plt.setp(ax, xticks = np.arange(0,25,6))

        for i, gene in enumerate(sorted(ens_list)):
            col = i%n_cols

            curve, ctr_x = spline_model2_with_smoothing(data,gene,splipy_order=3,width=2,smooth=True,smooth_its=2,plot_figs=False)
            y_spline = plot_spline_curve(curve,0,24)
            
            lines = []

            if not color_phase:
                l = ax[row,col].plot(y_spline[:,0], y_spline[:,1])
                lines.append(l)
            else:
                color_map = {'G1': 'green','S': 'red', 'G2M': 'blue'}
                phase_boundaries = {'G1': 11, 'S': 19, 'G2M': 24}
                
                y_g1 = y_spline[y_spline[:,0] < 11,:]
                y_s = y_spline[(y_spline[:,0] >= 11) & (y_spline[:,0] < 19) ,:]
                y_g2m = y_spline[y_spline[:,0] >= 19,:]
                
                l = ax[row,col].plot(y_g1[:,0], y_g1[:,1], color=color_map['G1'],label='G1')
                lines.append(l)
                l = ax[row,col].plot(y_s[:,0], y_s[:,1], color=color_map['S'],label='S')
                lines.append(l)
                l = ax[row,col].plot(y_g2m[:,0], y_g2m[:,1], color=color_map['G2M'],label='G2M')
                lines.append(l)
                    
            ax[row,col].set_title(data.var.loc[gene]['gene_symbols'])
            
            if plot_ctrl_pts:
                l = ax[row,col].plot(ctr_x[:,0], ctr_x[:,1],'*',label='Control points')
                lines.append(l)
            
            if ((i+1)%n_cols == 0) and (i > 0):
                row += 1
        for i in range(n_rows*n_cols-len(ens_list)):
            fig.delaxes(ax[row,n_cols-1-i])
            
        fig.add_subplot(111, frameon=False)
        plt.grid(False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.ylabel('Reads Per Million (n)')
        plt.xlabel('Pseudotime (hours)')
        plt.gcf().subplots_adjust(bottom=0.25)
        
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        fig.savefig(os.path.join(plot_path,'known_{}.png'.format(p.replace('/','-'))),bbox_inches='tight')
        
        
def plot_splines_from_list(data, 
                              ens_list, 
                              color_phase=False, 
                              plot_filename='data/spline_phase_plots/spline_plot.png', 
                              subsample_width=2,
                              plot_ctrl_pts=False,
                              plt_col_size=4, 
                              spline_backend='csaps', 
                              csaps_smooth=0.3,
                              splipy_order=4,
                              splipy_smooth=True,
                              splipy_smooth_its=2):
    from math import ceil
        
    ens_list = list(set(list(ens_list)).intersection(list(data.var.index)))
    col_size = plt_col_size
    n_cols = min(col_size, len(ens_list))
    n_rows = ceil(len(ens_list)/n_cols)
    row = 0 

    fig, ax = plt.subplots(n_rows,n_cols,squeeze=False,figsize=(3*n_cols,3*n_rows),sharey=True,sharex=True)
    plt.setp(ax, xticks = np.arange(0,25,6))
    
    for i, gene in enumerate(sorted(ens_list)):
        col = i%n_cols
        """
        curve, ctr_x = spline_model2_with_smoothing(data,gene,splipy_order=3,width=2,smooth=True,smooth_its=2,plot_figs=False)
        y_spline = plot_spline_curve(curve,0,24)
        """
        
        if spline_backend == 'splipy':
            curve, ctr_x = spline_model2_with_smoothing(data,gene,splipy_order=3,width=subsample_width,splipy_smooth=splipy_smooth,splipy_smooth_its=splipy_smooth_its,plot_figs=False)
            y_spline = plot_spline_curve(curve,0,24)
        elif spline_backend == 'csaps':
            sp, ctr_x = spline_model2_csaps(data,gene,width=subsample_width,csaps_smooth=csaps_smooth,plot_figs=False)
            xs = np.linspace(0,24,10000)
            y_spline = sp(xs)
            y_spline = np.concatenate(([xs],[y_spline]),axis=0).T
        else:
            raise NotImplementedError("Only ('splipy', 'csaps') are available.")

        lines = []

        if not color_phase:
            l = ax[row,col].plot(y_spline[:,0], y_spline[:,1])
            lines.append(l)
        else:
            color_map = {'G1': 'green','S': 'red', 'G2M': 'blue'}
            phase_boundaries = {'G1': 11, 'S': 19, 'G2M': 24}

            y_g1 = y_spline[y_spline[:,0] < 11,:]
            y_s = y_spline[(y_spline[:,0] >= 11) & (y_spline[:,0] < 19) ,:]
            y_g2m = y_spline[y_spline[:,0] >= 19,:]

            l = ax[row,col].plot(y_g1[:,0], y_g1[:,1], color=color_map['G1'],label='G1')
            lines.append(l)
            l = ax[row,col].plot(y_s[:,0], y_s[:,1], color=color_map['S'],label='S')
            lines.append(l)
            l = ax[row,col].plot(y_g2m[:,0], y_g2m[:,1], color=color_map['G2M'],label='G2M')
            lines.append(l)

        ax[row,col].set_title(data.var.loc[gene]['gene_symbols'])

        if plot_ctrl_pts:
            l = ax[row,col].plot(ctr_x[:,0], ctr_x[:,1],'*',color='orange',label='Control points')
            lines.append(l)

        if ((i+1)%n_cols == 0) and (i > 0):
            row += 1
    for i in range(n_rows*n_cols-len(ens_list)):
        fig.delaxes(ax[row,n_cols-1-i])

    fig.add_subplot(111, frameon=False)
    plt.grid(False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Reads Per Million (n)')
    plt.xlabel('Pseudotime (hours)')
    plt.gcf().subplots_adjust(bottom=0.25)

    if not os.path.exists(os.path.dirname(plot_filename)):
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    fig.savefig(plot_filename,bbox_inches='tight')
    
def plot_splines_from_list_color_title(data, 
                              ens_list,
                              genes_df,
                              pls_clean,
                              color_phase=False, 
                              plot_filename='data/spline_phase_plots/spline_plot.png', 
                              subsample_width=2,
                              plot_ctrl_pts=False,
                              plt_col_size=4, 
                              spline_backend='csaps', 
                              csaps_smooth=0.3,
                              splipy_order=4,
                              splipy_smooth=True,
                              splipy_smooth_its=2):
    from math import ceil
        
    ens_list = list(set(list(ens_list)).intersection(list(data.var.index)))
    col_size = plt_col_size
    n_cols = min(col_size, len(ens_list))
    n_rows = ceil(len(ens_list)/n_cols)
    row = 0 

    fig, ax = plt.subplots(n_rows,n_cols,squeeze=False,figsize=(3*n_cols,3*n_rows),sharey=True,sharex=True)
    plt.setp(ax, xticks = np.arange(0,25,6))
    
    for i, gene in enumerate(sorted(ens_list)):
        col = i%n_cols
        """
        curve, ctr_x = spline_model2_with_smoothing(data,gene,splipy_order=3,width=2,smooth=True,smooth_its=2,plot_figs=False)
        y_spline = plot_spline_curve(curve,0,24)
        """
        
        if spline_backend == 'splipy':
            curve, ctr_x = spline_model2_with_smoothing(data,gene,splipy_order=3,width=subsample_width,splipy_smooth=splipy_smooth,splipy_smooth_its=splipy_smooth_its,plot_figs=False)
            y_spline = plot_spline_curve(curve,0,24)
        elif spline_backend == 'csaps':
            sp, ctr_x = spline_model2_csaps(data,gene,width=subsample_width,csaps_smooth=csaps_smooth,plot_figs=False)
            xs = np.linspace(0,24,10000)
            y_spline = sp(xs)
            y_spline = np.concatenate(([xs],[y_spline]),axis=0).T
        else:
            raise NotImplementedError("Only ('splipy', 'csaps') are available.")

        lines = []

        if not color_phase:
            l = ax[row,col].plot(y_spline[:,0], y_spline[:,1])
            lines.append(l)
        else:
            color_map = {'G1': 'green','S': 'red', 'G2M': 'blue'}
            phase_boundaries = {'G1': 11, 'S': 19, 'G2M': 24}

            y_g1 = y_spline[y_spline[:,0] < 11,:]
            y_s = y_spline[(y_spline[:,0] >= 11) & (y_spline[:,0] < 19) ,:]
            y_g2m = y_spline[y_spline[:,0] >= 19,:]

            l = ax[row,col].plot(y_g1[:,0], y_g1[:,1], color=color_map['G1'],label='G1')
            lines.append(l)
            l = ax[row,col].plot(y_s[:,0], y_s[:,1], color=color_map['S'],label='S')
            lines.append(l)
            l = ax[row,col].plot(y_g2m[:,0], y_g2m[:,1], color=color_map['G2M'],label='G2M')
            lines.append(l)
        
        if (gene in genes_df.index) and (gene in pls_clean.index):
            if genes_df.loc[gene,'phase'] == pls_clean.loc[gene,'assigned_phase']:
                ax[row,col].set_title('*{} [{}]'.format(data.var.loc[gene]['gene_symbols'],genes_df.loc[gene,'phase']),color='b')
            else:
                ax[row,col].set_title('~{} [{}]'.format(data.var.loc[gene]['gene_symbols'],genes_df.loc[gene,'phase']),color='r')
        else:
            ax[row,col].set_title(data.var.loc[gene]['gene_symbols'])

        if plot_ctrl_pts:
            l = ax[row,col].plot(ctr_x[:,0], ctr_x[:,1],'*',color='orange',label='Control points')
            lines.append(l)

        if ((i+1)%n_cols == 0) and (i > 0):
            row += 1
    for i in range(n_rows*n_cols-len(ens_list)):
        fig.delaxes(ax[row,n_cols-1-i])

    fig.add_subplot(111, frameon=False)
    plt.grid(False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Reads Per Million (n)')
    plt.xlabel('Pseudotime (hours)')
    plt.gcf().subplots_adjust(bottom=0.25)

    if not os.path.exists(os.path.dirname(plot_filename)):
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    fig.savefig(plot_filename,bbox_inches='tight')

    
def plot_scatter_with_splines(data, 
                              ens_list, 
                              color_phase=False, 
                              plot_filename='data/spline_phase_plots/spline_plot.png', 
                              subsample_width=2,
                              plot_ctrl_pts=False,
                              plt_col_size=4, 
                              spline_backend='csaps', 
                              csaps_smooth=0.3,
                              csaps_add_boundaries=True,
                              splipy_order=4,
                              splipy_smooth=True,
                              splipy_smooth_its=2):
    from math import ceil
        
    ens_list = list(set(list(ens_list)).intersection(list(data.var.index)))
    col_size = plt_col_size
    #n_cols = min(col_size, len(ens_list))
    n_cols = 2
    #n_rows = ceil(len(ens_list)/n_cols)
    n_rows = len(ens_list)
    row = 0 

    fig, ax = plt.subplots(n_rows,n_cols,squeeze=False,figsize=(3*n_cols,3*n_rows),sharey=True,sharex=True)
    plt.setp(ax, xticks = np.arange(0,25,6))

    for i, gene in enumerate(sorted(ens_list)):
        #col = i%n_cols
        if spline_backend == 'splipy':
            curve, ctr_x = spline_model2_with_smoothing(data,gene,splipy_order=3,width=subsample_width,splipy_smooth=splipy_smooth,splipy_smooth_its=splipy_smooth_its,plot_figs=False)
            y_spline = plot_spline_curve(curve,0,24)
        elif spline_backend == 'csaps':
            sp, ctr_x = spline_model2_csaps(data,gene,width=subsample_width,csaps_smooth=csaps_smooth,plot_figs=False,add_boundaries=csaps_add_boundaries)
            xs = np.linspace(0,24,10000)
            y_spline = sp(xs)
            y_spline = np.concatenate(([xs],[y_spline]),axis=0).T
        else:
            raise NotImplementedError("Only ('splipy', 'csaps') are available.")
            

        lines = []

        if not color_phase:
            y = data[:,[gene]].X
            try:
                x = data.obs['time']
                plt.setp(ax, xticks = np.arange(0,25,6))
            except Exception as e:
                x = data.obs['order']
            
            l = ax[i,0].plot(x, y, 'o', color='gray', alpha=0.2, markersize=3)
            lines.append(l)
            l = ax[i,1].plot(y_spline[:,0], y_spline[:,1])
            lines.append(l)
        else:
            color_map = {'G1': 'green','S': 'red', 'G2M': 'blue'}
            
            for cp in ['G1','S','G2M']:
                data_p = data[data.obs.phase == cp,:]
                y = data_p[:,[gene]].X
                try:
                    x = data_p.obs['time']
                except Exception as e:
                    x = data_p.obs['order']
                l = ax[i,0].plot(x, y, 'o', color=color_map[cp], alpha=0.2, markersize=3, label=cp, zorder=0)
                lines.append(l)
            
            phase_boundaries = {'G1': 11, 'S': 19, 'G2M': 24}

            y_g1 = y_spline[y_spline[:,0] < 11,:]
            y_s = y_spline[(y_spline[:,0] >= 11) & (y_spline[:,0] < 19) ,:]
            y_g2m = y_spline[y_spline[:,0] >= 19,:]

            l = ax[i,1].plot(y_g1[:,0], y_g1[:,1], color=color_map['G1'],label='G1')
            lines.append(l)
            l = ax[i,1].plot(y_s[:,0], y_s[:,1], color=color_map['S'],label='S')
            lines.append(l)
            l = ax[i,1].plot(y_g2m[:,0], y_g2m[:,1], color=color_map['G2M'],label='G2M')
            lines.append(l)

        ax[i,0].set_title(data.var.loc[gene]['gene_symbols'])
        ax[i,1].set_title(data.var.loc[gene]['gene_symbols'])

        if plot_ctrl_pts:
            l = ax[i,1].plot(ctr_x[:,0], ctr_x[:,1],'*',color='orange',label='Control points')
            lines.append(l)
        """
        if ((i+1)%n_cols == 0) and (i > 0):
            row += 1
        """
    """
    for i in range(n_rows*n_cols-len(ens_list)):
        fig.delaxes(ax[row,n_cols-1-i])
    """ 

    fig.add_subplot(111, frameon=False)
    plt.grid(False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Reads Per Million (n)')
    plt.xlabel('Pseudotime (hours)')
    plt.gcf().subplots_adjust(bottom=0.25)

    if not os.path.exists(os.path.dirname(plot_filename)):
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    fig.savefig(plot_filename,bbox_inches='tight')
    
        
def subsample_gene(data, gene_name, width=2):
    gene_cat = 'gene_ids' if gene_name.startswith("ENS") else 'gene_symbols'
    if not gene_name in list(data.var[gene_cat]):
        raise ValueError("Gene symbol {} not in data set!".format(gene_name))
    data_test = data[:,data.var[gene_cat] == gene_name]
    m_arr = []
    for i in range(0,24,width):
        m_arr.append(data_test[ (i <= data_test.obs['time']) & (data_test.obs['time'] < i+2),:].X.mean())
    return np.array(m_arr)

def make_knots(X,splipy_order=3):
    end = len(X[:,0]) + splipy_order+1 - 2*splipy_order
    return(list(np.zeros(splipy_order)) + list(range(1,end)) + list(np.ones(splipy_order)*end))

def spline_model(data,gene_name,splipy_order=3,width=2):

    import splipy as sp    
    
    subsample = subsample_gene(data,gene_name,width=width)
    boundary = (subsample[0]+subsample[-1])/2
    x = np.append([np.arange(width/2,24,width)], [subsample], axis=0).T
    
    x0 = x.copy()
    x1 = x.copy()
    
    x = np.vstack((np.array([0,boundary]), x,np.array([24,boundary])))

    #x0 = x.copy()
    x0[:,0] -= np.max(x[:,0])+1
    x01 = x0.copy()
    x01[:,0] -= np.max(x[:,0])+1
    
    #x1 = x.copy()
    x1[:,0] += np.max(x[:,0])+1
    x11 = x1.copy()
    x11[:,0] += np.max(x[:,0])+1

    X = np.vstack((x01,x0,x,x1,x11))

    knots = make_knots(X,splipy_order=splipy_order)

    basis = sp.BSplineBasis(order=splipy_order,knots=knots)
    
    start = np.min(X[:,0])
    end = np.max(X[:,0])

    basis.reparam(start=start,end=end)

    curve2 = sp.curve_factory.interpolate(X, basis)

    t2 = np.linspace(0,24,1000)
    y2 = curve2(t2)
    plt.figure()
    plt.plot(y2[:,0],y2[:,1])
    plt.plot(x[:,0],x[:,1],'*')
    plt.axhline(y=0, color='k')
    plt.xticks(np.arange(0,26,4))

    plt.show()


    plt.figure()
    t = np.linspace(start,end,1000)
    y = curve2(t)
    plt.plot(y[:,0],y[:,1])
    plt.plot(X[:,0],X[:,1],'*')
    plt.axhline(y=0, color='k')
    plt.show()
    
    
def spline_model2(data,gene_name,splipy_order=3,width=2):

    import splipy as sp    
    
    subsample = subsample_gene(data,gene_name,width=width)
    boundary = (subsample[0]+subsample[-1])/2
    x = np.append([np.arange(width/2,24,width)], [subsample], axis=0).T
    
    x0 = x.copy()
    x1 = x.copy()
    
    x = np.vstack((np.array([0,boundary]), x,np.array([24,boundary])))

    #x0 = x.copy()
    x0[:,0] -= np.max(x[:,0])+1
    #x01 = x0.copy()
    #x01[:,0] -= np.max(x[:,0])+1
    
    #x1 = x.copy()
    x1[:,0] += np.max(x[:,0])+1
    #x11 = x1.copy()
    #x11[:,0] += np.max(x[:,0])+1

    #X = np.vstack((x01,x0,x,x1,x11))
    #X = np.vstack((x0[-4:,:],x,x1[:4,:]))
    X = np.vstack((x0,x,x1))

    knots = make_knots(X,splipy_order=splipy_order)

    basis = sp.BSplineBasis(order=splipy_order,knots=knots)
    
    start = np.min(X[:,0])
    end = np.max(X[:,0])

    basis.reparam(start=start,end=end)

    curve2 = sp.curve_factory.interpolate(X, basis)

    t2 = np.linspace(-2,26,1000)
    y2 = curve2(t2)
    plt.figure()
    plt.plot(y2[:,0],y2[:,1])
    plt.plot(x[:,0],x[:,1],'*')
    plt.axhline(y=0, color='k')
    plt.xticks(np.arange(0,26,4))

    plt.show()


    plt.figure()
    t = np.linspace(start,end,1000)
    y = curve2(t)
    plt.plot(y[:,0],y[:,1])
    plt.plot(X[:,0],X[:,1],'*')
    plt.axhline(y=0, color='k')
    plt.show()
    
    
def spline_model2_with_smoothing(data,gene_name,splipy_order=3,width=2,smooth=False,smooth_its=1,plot_figs=False):

    import splipy as sp    
    
    subsample = subsample_gene(data,gene_name,width=width)
    boundary = (subsample[0]+subsample[-1])/2
    x = np.append([np.arange(width/2,24,width)], [subsample], axis=0).T
    
    x = np.vstack((np.array([0,boundary]), x,np.array([24,boundary])))
    
    if smooth:
        for k in range(smooth_its):
            xnew = x.copy()
            for i in range(1,len(x)-1):
                xnew[i] = (0.5*x[i-1] + x[i] + 0.5*x[i+1])*0.5
            x = xnew
    
    x0 = x[1:-1].copy()
    x1 = x[1:-1].copy()

    x0[:,0] -= np.max(x[:,0])
    
    x1[:,0] += np.max(x[:,0])
    
    X = np.vstack((x0,x,x1))

    knots = make_knots(X,splipy_order=splipy_order)

    basis = sp.BSplineBasis(order=splipy_order,knots=knots)
    
    start = np.min(X[:,0])
    end = np.max(X[:,0])

    basis.reparam(start=start,end=end)

    curve = sp.curve_factory.interpolate(X, basis)
    
    if plot_figs:
        y = plot_spline_curve(curve,0,24)
        plt.figure(figsize=(7,7))
        plt.plot(y[:,0],y[:,1])
        plt.plot(x[:,0],x[:,1],'*')
        plt.axhline(y=0, color='k')
        plt.xticks(np.arange(0,26,4))

        plt.show()

        """
        plt.figure(figsize=(7,7))
        t = np.linspace(start,end,1000)
        y = curve.evaluate(t)
        plt.plot(y[:,0],y[:,1])
        plt.plot(X[:,0],X[:,1],'*')
        plt.axhline(y=0, color='k')
        plt.show()
        """
    
    return curve, x


def spline_model2_csaps(data,gene_name,width=2,csaps_smooth=0.3,plot_figs=False,add_boundaries=True):

    import csaps    
    if add_boundaries:
        X, x = generate_cyclic_control_points(data, gene_name, width=width, smooth=False)
    else:
        X, x = generate_cyclic_control_points_no_boundaries(data, gene_name, width=width, smooth=False)

    sp = csaps.UnivariateCubicSmoothingSpline(X[:,0], X[:,1], smooth=csaps_smooth)

    xs = np.linspace(0,24,1000)

    if plot_figs:
        y = sp(xs)
        plt.figure(figsize=(7,7))
        plt.plot(xs,y)
        plt.plot(x[:,0],x[:,1],'*')
        plt.axhline(y=0, color='k')
        plt.xticks(np.arange(0,26,4))

        plt.show()

        """
        plt.figure(figsize=(7,7))
        t = np.linspace(start,end,1000)
        y = curve.evaluate(t)
        plt.plot(y[:,0],y[:,1])
        plt.plot(X[:,0],X[:,1],'*')
        plt.axhline(y=0, color='k')
        plt.show()
        """
    
    return sp, x

def generate_cyclic_control_points(data, gene_name, width=2, smooth=False, smooth_its=1):
    
    subsample = subsample_gene(data,gene_name,width=width)
    boundary = (subsample[0]+subsample[-1])/2
    x = np.append([np.arange(width/2,24,width)], [subsample], axis=0).T
    
    x = np.vstack((np.array([0,boundary]), x,np.array([24,boundary])))
    
    if smooth:
        for k in range(smooth_its):
            xnew = x.copy()
            for i in range(1,len(x)-1):
                xnew[i] = (0.5*x[i-1] + x[i] + 0.5*x[i+1])*0.5
            x = xnew
    
    x0 = x[1:-1].copy()
    x1 = x[1:-1].copy()

    x0[:,0] -= np.max(x[:,0])
    
    x1[:,0] += np.max(x[:,0])
    
    return np.vstack((x0,x,x1)), x

def generate_cyclic_control_points_no_boundaries(data, gene_name, width=2, smooth=False, smooth_its=1):
    
    subsample = subsample_gene(data,gene_name,width=width)
    #boundary = (subsample[0]+subsample[-1])/2
    x = np.append([np.arange(width/2,24,width)], [subsample], axis=0).T
    
    #x = np.vstack((np.array([0,boundary]), x,np.array([24,boundary])))
    
    if smooth:
        for k in range(smooth_its):
            xnew = x.copy()
            for i in range(1,len(x)-1):
                xnew[i] = (0.5*x[i-1] + x[i] + 0.5*x[i+1])*0.5
            x = xnew
    
    x0 = x.copy()
    x1 = x.copy()

    #x0[:,0] -= np.max(x[:,0])
    x0[:,0] -= 24.0
    
    #x1[:,0] += np.max(x[:,0])
    x1[:,0] += 24.0
    
    return np.vstack((x0,x,x1)), x


def sample_per_hour(curve):
    t = np.linspace(-2,26,28000)
    y = curve.evaluate(t)
    per_hour = []
    for i in range(25):
        below = np.where(y[:,0]<i)[0][-1]
        above = np.where(y[:,0]>i)[0][0]
        per_hour.append((y[below,1]+y[above,1])*0.5)
    return np.array(per_hour)

def sample_per_hour_csaps(sp):
    t = np.arange(0,25,1)
    return(sp(t))

def plot_spline_curve(curve,start,end,plot=False):
    t = np.linspace(curve.start()[0],curve.end()[0],10000)
    y = curve.evaluate(t)
    id_s = np.where(y[:,0]>start)[0][0]
    id_e = np.where(y[:,0]<end)[0][-1]
    if plot:
        plt.figure()
        plt.plot(y[id_s:id_e,0],y[id_s:id_e,1])
    return y[id_s:id_e,:]

def csaps_error(data,ens_list,csaps_smooth=0.3,subsample_width=2):
    ens_list = list(set(list(ens_list)).intersection(list(data.var.index)))
    res = {}
    for i, gene in enumerate(sorted(ens_list)):
        sp, ctr_x = spline_model2_csaps(data,gene,width=subsample_width,csaps_smooth=csaps_smooth,plot_figs=False)

        y_raw = data[:,[gene]].X
        y_est = sp(data.obs['time'])
        res[gene] = {'res_2-norm': np.linalg.norm(y_raw-y_est,axis=0), 'res_mean': np.mean(abs(y_est-y_raw)), 'res_var': np.var(abs(y_est-y_raw),ddof=1), 'sum_curvature': np.sum(csaps_curvature(data,sp=sp)),'gene_symbol': data.var.loc[gene]['gene_symbols']}
    
    return res

def csaps_curvature(data,sp=None,gene=None,width=2,csaps_smooth=0.3):
    if not sp:
        if not gene:
            raise ValueError("If not a csaps spline is defined (sp=) a gene must be specified with gene=")
        sp, x = spline_model2_csaps(data,gene=gene,widht=width,csaps_smooth=csaps_smooth)
    
    time = data.obs['time']
    ys = sp(time)
    grady = np.gradient(ys,time)
    grad2y = np.gradient(grady,time)
    denom = np.power(1+np.power(grady,2),1.5)
    
    return abs(grad2y)/denom

def csaps_demo_smoothing_param(data,gene,plt_col_size=3,plot_filename="./data/nice_plots/csaps/demo_smoothing_param.png"):
    from math import ceil
        
    #ens_list = list(set(list(ens_list)).intersection(list(data.var.index)))
    #col_size = plt_col_size
    n_cols = plt_col_size
    #n_cols = min(col_size, len(ens_list))
    #n_rows = ceil(len(ens_list)/n_cols)
    n_rows = 2
    row = 0 

    fig, ax = plt.subplots(n_rows,n_cols,squeeze=False,figsize=(3*n_cols,3*n_rows),sharey=True,sharex=True)
    plt.setp(ax, xticks = np.arange(0,25,6))
    
    for i, smooth in enumerate(np.arange(0,1.2,0.2)):
        col = i%n_cols
        
        sp, ctr_x = spline_model2_csaps(data,gene,width=2,csaps_smooth=smooth,plot_figs=False)
        xs = np.linspace(0,24,10000)
        y_spline = sp(xs)
        y_spline = np.concatenate(([xs],[y_spline]),axis=0).T

        #lines = []

        
        l = ax[row,col].plot(y_spline[:,0], y_spline[:,1])
        #lines.append(l)

        ax[row,col].set_title("p = {}".format(np.around(smooth,decimals=1)))

        
        l = ax[row,col].plot(ctr_x[:,0], ctr_x[:,1],'*',color='orange',label='Control points')
        #lines.append(l)

        if ((i+1)%n_cols == 0) and (i > 0):
            row += 1
    for i in range(n_rows*n_cols-len(np.arange(0,1.2,0.2))):
        fig.delaxes(ax[row,n_cols-1-i])

    fig.add_subplot(111, frameon=False)
    plt.grid(False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Reads Per Million (n)')
    plt.xlabel('Pseudotime (hours)')
    plt.gcf().subplots_adjust(bottom=0.25)

    if not os.path.exists(os.path.dirname(plot_filename)):
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    fig.savefig(plot_filename,bbox_inches='tight')
    
def demo_control_points(data,gene,width=2):
    import csaps    
    
    X, x = generate_cyclic_control_points(data, gene, width=width, smooth=False)
    
    plt.figure()
    plt.plot(x[1:-1,0],x[1:-1,1],'*',label='mean values')
    plt.xticks(np.arange(0,30,6))
    plt.legend()
    plt.ylabel('Reads Per Million (n)')
    plt.xlabel('Pseudotime (hours)')
    plt.savefig("./data/nice_plots/csaps/demo_control_points.png")
    
    plt.figure()
    plt.plot(x[1:-1,0],x[1:-1,1],'*',label='mean values')
    plt.plot([x[0,0],x[-1,0]],[x[0,1],x[-1,1]],'*',label='boundary values')
    plt.xticks(np.arange(0,30,6))
    plt.legend()
    plt.ylabel('Reads Per Million (n)')
    plt.xlabel('Pseudotime (hours)')

    plt.savefig("./data/nice_plots/csaps/demo_control_points_with_boundaries.png")
    
    plt.figure(figsize=(9,5))
    plt.plot(X[:,0],X[:,1],'*',label='cyclic control points')
    plt.xticks(np.arange(-24,54,6))
    plt.legend()
    plt.ylabel('Reads Per Million (n)')
    plt.xlabel('Pseudotime (hours)')
    plt.savefig("./data/nice_plots/csaps/demo_control_points_cyclic.png")
    
    
def subsample_from_splines(data,n_points,csaps_subsample_width=2,csaps_smooth=0.3,plot_figs=False,csaps_add_boundaries=True):
    
    t = np.linspace(0,24,n_points)
    df = pd.DataFrame({}, index=list(t))
    
    t_peak = np.linspace(0,24,10000)
    peak_times = []
    gene_list = []
    
    for gene in list(data.var['gene_ids']):
        sp, _ = spline_model2_csaps(data, gene, width=csaps_subsample_width, csaps_smooth=csaps_smooth, add_boundaries=csaps_add_boundaries)
        df[gene] = sp(t)
        #peak_df = peak_df.append('peak_time': pd.DataFrame('values': sp(t_peak), index=t_peak).idxmax(),index=g)
        #peak_df = peak_df.append({'peak_time': t_peak[np.argmax(sp(t_peak))]},index=gene)
        peak_times.append(t_peak[np.argmax(sp(t_peak))])
        gene_list.append(gene)
    
    peak_df = pd.DataFrame({'peak_times': peak_times}, index=gene_list)
    
    return df, peak_df

def subsample_genes_from_splines(data,genes,n_points,csaps_subsample_width=2,csaps_smooth=0.3,plot_figs=False,csaps_add_boundaries=True):
    
    t = np.linspace(0,24,n_points)
    df = pd.DataFrame({}, index=list(t))
    
    for gene in genes:
        sp, _ = spline_model2_csaps(data, gene, width=csaps_subsample_width, csaps_smooth=csaps_smooth, add_boundaries=csaps_add_boundaries)
        df[gene] = sp(t)
    
    return df
        
def my_score_genes_cell_cycle(
        adata,
        g1_genes,
        s_genes,
        g2m_genes,
        copy=False,
        **kwargs):
    """Score cell cycle genes [Satija15]_.
    Given two lists of genes associated to S phase and G2M phase, calculates
    scores and assigns a cell cycle phase (G1, S or G2M). See
    :func:`~scanpy.api.score_genes` for more explanation.
    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        The annotated data matrix.
    g1_genes : `list`
        List of genes associated with S phase.
    s_genes : `list`
        List of genes associated with S phase.
    g2m_genes : `list`
        List of genes associated with G2M phase.
    copy : `bool`, optional (default: `False`)
        Copy `adata` or modify it inplace.
    **kwargs : optional keyword arguments
        Are passed to :func:`~scanpy.api.score_genes`. `ctrl_size` is not
        possible, as it's set as `min(len(s_genes), len(g2m_genes))`.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **G1_score** : `adata.obs`, dtype `object`
        The score for G1 phase for each cell.
    **S_score** : `adata.obs`, dtype `object`
        The score for S phase for each cell.
    **G2M_score** : `adata.obs`, dtype `object`
        The score for G2M phase for each cell.
    **phase** : `adata.obs`, dtype `object`
        The cell cycle phase (`S`, `G2M` or `G1`) for each cell.
    See also
    --------
    score_genes
    Examples
    --------
    See this `notebook <https://github.com/theislab/scanpy_usage/tree/master/180209_cell_cycle>`__.
    """
    #logg.info('calculating cell cycle phase')
    import scanpy as sc

    adata = adata.copy() if copy else adata
    ctrl_size = min(len(s_genes), len(g2m_genes), len(g1_genes))
    # add s-score
    sc.tl.score_genes(adata, gene_list=s_genes, score_name='S_score', ctrl_size=ctrl_size, **kwargs)
    # add g2m-score
    sc.tl.score_genes(adata, gene_list=g2m_genes, score_name='G2M_score', ctrl_size=ctrl_size, **kwargs)
    # add g1-score
    sc.tl.score_genes(adata, gene_list=g1_genes, score_name='G1_score', ctrl_size=ctrl_size, **kwargs)
    scores = adata.obs[['S_score', 'G2M_score', 'G1_score']]

    # default phase is S
    phase = pd.Series('not_assigned', index=scores.index)

    # if G2M is higher than S and G1, it's G2M
    phase[(scores.G2M_score > scores.S_score) & (scores.G2M_score > scores.G1_score)] = 'G2M'
    
    # if S is higher than G2M and G1, it's S
    phase[(scores.S_score > scores.G2M_score) & (scores.S_score > scores.G1_score)] = 'S'
    
    # if G1 is higher than G2M and S, it's G1
    phase[(scores.G1_score > scores.G2M_score) & (scores.G1_score > scores.S_score)] = 'G1'

    

    adata.obs['phase'] = phase
    #logg.hint('    \'phase\', cell cycle phase (adata.obs)')
    return adata if copy else None

def my_score_genes_cell_cycle_improved(
        adata,
        g1_genes,
        s_genes,
        g2m_genes,
        copy=False,
        **kwargs):
    """Score cell cycle genes [Satija15]_.
    Given two lists of genes associated to S phase and G2M phase, calculates
    scores and assigns a cell cycle phase (G1, S or G2M). See
    :func:`~scanpy.api.score_genes` for more explanation.
    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        The annotated data matrix.
    g1_genes : `list`
        List of genes associated with S phase.
    s_genes : `list`
        List of genes associated with S phase.
    g2m_genes : `list`
        List of genes associated with G2M phase.
    copy : `bool`, optional (default: `False`)
        Copy `adata` or modify it inplace.
    **kwargs : optional keyword arguments
        Are passed to :func:`~scanpy.api.score_genes`. `ctrl_size` is not
        possible, as it's set as `min(len(s_genes), len(g2m_genes))`.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **G1_score** : `adata.obs`, dtype `object`
        The score for G1 phase for each cell.
    **S_score** : `adata.obs`, dtype `object`
        The score for S phase for each cell.
    **G2M_score** : `adata.obs`, dtype `object`
        The score for G2M phase for each cell.
    **phase** : `adata.obs`, dtype `object`
        The cell cycle phase (`S`, `G2M` or `G1`) for each cell.
    See also
    --------
    score_genes
    Examples
    --------
    See this `notebook <https://github.com/theislab/scanpy_usage/tree/master/180209_cell_cycle>`__.
    """
    #logg.info('calculating cell cycle phase')
    import scanpy as sc

    adata = adata.copy() if copy else adata
    ctrl_size = min(len(s_genes), len(g2m_genes), len(g1_genes))
    s_n_bins = max(round((len(g1_genes)+len(g2m_genes))/ctrl_size),3)
    g1_n_bins = max(round((len(s_genes)+len(g2m_genes))/ctrl_size),3)
    g2m_n_bins = max(round((len(g1_genes)+len(s_genes))/ctrl_size),3)
    
        
    # add s-score
    sc.tl.score_genes(adata, gene_list=s_genes, score_name='S_score', ctrl_size=ctrl_size, n_bins=s_n_bins, **kwargs)
    # add g2m-score
    sc.tl.score_genes(adata, gene_list=g2m_genes, score_name='G2M_score', ctrl_size=ctrl_size, n_bins=g2m_n_bins, **kwargs)
    # add g1-score
    sc.tl.score_genes(adata, gene_list=g1_genes, score_name='G1_score', ctrl_size=ctrl_size, n_bins=g1_n_bins, **kwargs)
    
    if not 'G1_score' in adata.obs.columns:
        print("WARNING: No G1-genes found in data set. Computing G1-score as -sum(S_score,G2M_score)")
        adata.obs['G1_score'] = -adata.obs[['S_score', 'G2M_score']].sum(1)
    
    scores = adata.obs[['S_score', 'G2M_score', 'G1_score']]

    # default phase is S
    phase = pd.Series('not_assigned', index=scores.index)

    # if G2M is higher than S and G1, it's G2M
    phase[(scores.G2M_score > scores.S_score) & (scores.G2M_score > scores.G1_score)] = 'G2M'
    
    # if S is higher than G2M and G1, it's S
    phase[(scores.S_score > scores.G2M_score) & (scores.S_score > scores.G1_score)] = 'S'
    
    # if G1 is higher than G2M and S, it's G1
    phase[(scores.G1_score > scores.G2M_score) & (scores.G1_score > scores.S_score)] = 'G1'

    

    adata.obs['phase'] = phase
    #logg.hint('    \'phase\', cell cycle phase (adata.obs)')
    return adata if copy else None


def plot_pls_loadings(pls_sigloadings,gene_list=[],plot_path=None):
    plt.figure(figsize=(7,7))
    plt.plot(pls_sigloadings['V1'],pls_sigloadings['V2'],'.')
    for g in gene_list:
        plt.plot(pls_sigloadings.loc[g]['V1'],pls_sigloadings.loc[g]['V2'],'*')
    plt.xlabel("Loading 1")
    plt.ylabel("Loading 2")
    if plot_path:
        plt.savefig(plot_path)
        
def compute_pls_loading_lengths(pls_sigloadings):
    return np.linalg.norm(np.append([pls_sigloadings['V1']],[pls_sigloadings['V2']],axis=0).T,axis=1)


def compute_angle_phase_distribution_with_resampling(pls_sigload_cutoff,clean=True):
    #known_df = pd.read_csv("./data/knownGenes_derived.txt",sep=" ").dropna()
    #known_df = pd.read_csv("./data/knownGenes_phase_dist.txt",sep=" ").dropna()
    known_df = pd.read_csv("./data/known_plus_regev.csv",index_col=0)
    z = np.zeros(len(known_df['phase'].unique()))
    angle_dist_df = pd.DataFrame({'mean': z, 'variance': z, 'n': z}, index=known_df['phase'].unique())
    for p in known_df['phase'].unique():
        p_df = known_df[known_df['phase'] == p].copy()
        p_genes = list(set(list(p_df['Ensembl'])).intersection(set(list(pls_sigload_cutoff.index))))
        angle_dist_df.loc[p,'n'] = len(p_genes)
        if len(p_genes) > 2:
            re_means = []
            re_vars = []
            #resampling with leave 10% out
            for i in range(500):
                re_genes = np.random.choice(p_genes, len(p_genes)-int(np.ceil(len(p_genes)*0.1)))
                re_angles = pls_sigload_cutoff.loc[re_genes, 'angles']
                if re_angles.max() - re_angles.min() > 1.5*np.pi:
                    re_angles[re_angles > np.pi] -= 2*np.pi
                m = np.mean(re_angles)
                v = np.var(re_angles,ddof=1)
                
                if v < 1.5:
                    re_means.append(m)
                    re_vars.append(v)
                """
                re_means.append(m)
                re_vars.append(v)
                """
            """
            mean = np.mean(re_means)
            variance = np.mean(re_vars)
            """
            m_v_df = pd.DataFrame({'mean': re_means, 'variance': re_vars})
            m_v_df = m_v_df.sort_values(by=['variance'])
            mean = m_v_df.loc[int(len(m_v_df)*0.5),'mean']
            variance = m_v_df.loc[int(len(m_v_df)*0.5),'variance']
        else:
            p_angles = pls_sigload_cutoff.loc[p_genes, 'angles']
            if p_angles.max() - p_angles.min() > 1.5*np.pi:
                p_angles[p_angles > np.pi] -= 2*np.pi
            mean = np.mean(p_angles)
            variance = np.var(p_angles,ddof=1) 

        angle_dist_df.loc[p,'mean'] = mean
        angle_dist_df.loc[p,'variance'] = variance
    
    if clean:
        angle_dist_df.dropna(inplace=True)
        angle_dist_df = angle_dist_df[angle_dist_df['variance'] != 0]
        angle_dist_df = angle_dist_df[angle_dist_df['variance'] < 1.5 ].copy()
        
    #make cyclic where needed
    for p in angle_dist_df.index.unique():
        if (angle_dist_df.loc[p,'mean'] + 3*np.sqrt(angle_dist_df.loc[p,'variance']) > 2*np.pi) and (np.sqrt(angle_dist_df.loc[p,'variance'])<1):
            shift_df = pd.DataFrame({
                'mean': angle_dist_df.loc[p,'mean']-2*np.pi, 
                'variance': angle_dist_df.loc[p,'variance'],
                'n': angle_dist_df.loc[p,'n']
            }, index=['{}-shift'.format(p)])
            angle_dist_df = angle_dist_df.append(shift_df)
        elif (angle_dist_df.loc[p,'mean'] - 3*np.sqrt(angle_dist_df.loc[p,'variance']) < 0) and (np.sqrt(angle_dist_df.loc[p,'variance'])<1):
            shift_df = pd.DataFrame({
                'mean': angle_dist_df.loc[p,'mean']+2*np.pi, 
                'variance': angle_dist_df.loc[p,'variance'], 
                'n': angle_dist_df.loc[p,'n']
            }, index=['{}-shift'.format(p)])
            angle_dist_df = angle_dist_df.append(shift_df)
    
    return angle_dist_df
    

def plot_phase_angle_distribution(angle_dist_df,plot_path=None):
    import scipy
    plt.figure(figsize=(7,7))
    t = np.linspace(0,2*np.pi,1000)
    for p in angle_dist_df.index.unique():
        mean = angle_dist_df.loc[p,'mean']
        scale = np.sqrt(angle_dist_df.loc[p,'variance'])
        plt.plot(t,scipy.stats.norm.pdf(t, mean, scale),'-',label=p, color=color_dict[p])
    plt.xlabel("Angle")
    plt.ylabel("Probability")
    plt.legend()
    if plot_path:
        plt.savefig(plot_path)
        
        
def assign_phase(pls_sigload_cutoff,angle_dist_df):
    import scipy
    assigned_phase = []
    for gene, row in pls_sigload_cutoff.iterrows():
        angle = row['angles']
        z_d = {}
        for p in angle_dist_df.index.unique():
            mean = angle_dist_df.loc[p,'mean']
            scale = np.sqrt(angle_dist_df.loc[p,'variance'])
            z_d[p] = scipy.stats.norm.pdf(angle, mean, scale)
        phase_max = max(z_d, key=z_d.get)
        assigned_phase.append(phase_max if z_d[phase_max]>0.05 else 'not_assigned')
    pls_sigload_cutoff['assigned_phase'] = assigned_phase
    pls_sigload_cutoff['assigned_phase'].value_counts()
    
    
def plot_pls_assigned_phases(pls_sigload_cutoff, plot_path=None):
    plt.figure(figsize=(9,9))
    plt.axis('equal')
    for p in pls_sigload_cutoff['assigned_phase'].unique():
        p_df = pls_sigload_cutoff[pls_sigload_cutoff['assigned_phase'] == p]
        if not p_df.empty:
            plt.plot(p_df['V1'],p_df['V2'],'.',color=color_dict[p],label=p)
    plt.xlabel("Loading 1")
    plt.ylabel("Loading 2")
    plt.legend()
    if plot_path:
        plt.savefig(plot_path)
        

def plot_pls_all_genes(data_bb,pls_sigload_cutoff,csaps_smooth,plot_base_path="data/nice_plots/PLS-genes"):
    for p in pls_sigload_cutoff['assigned_phase'].unique():
        genes = pls_sigload_cutoff[pls_sigload_cutoff['assigned_phase'] == p].index.values
        for i,j in enumerate(range(0,len(genes),25)):
            n = min(j+25,len(genes))
            ens_list = genes[j:n]
            plot_splines_from_list(data_bb, ens_list, 
                                        color_phase=True,
                                        csaps_smooth=csaps_smooth, 
                                        plot_ctrl_pts=False, 
                                        plt_col_size=5,
                                        plot_filename=os.path.join(plot_base_path,'{}_{}.png'.format(p.replace('/','-'),i)))

            
def plot_pls_biotypes(pls_sigload_cutoff, plot_path=None):
    plt.figure(figsize=(10,10))
    plt.axis('equal')
    for b in pls_sigload_cutoff['gene_biotype'].unique():
        b_df = pls_sigload_cutoff[pls_sigload_cutoff['gene_biotype'] == b]
        if not b_df.empty:
            plt.plot(b_df['V1'],b_df['V2'],'.',label=b)
    plt.legend()
    if plot_path:
        plt.savefig(plot_path)
        
def manually_curate_overlapped_genes(pls_overlap,opt_smooth_hacat=0.15,opt_smooth_293t=0.05,data_path_hacat=None,data_path_293t=None,output_path=None):
    if not (data_path_hacat or data_path_293t or output_path):
        raise ValueError("both data_path_hacat and data_path_293t must be set.")
    
    data_hacat = sc.read_h5ad(data_path_hacat)
    data_293t = sc.read_h5ad(data_path_293t)
    
    curated_index = []
    curated_phase = []
    curated_biotype = []
    curated_gene_symbol = []
    curated_status = []
    
    for g,row in pls_overlap.iterrows():
        if row['assigned_phase_hacat'] != row['assigned_phase_293t']:
            plot_splines_from_list(data_hacat, [g], 
                                        color_phase=True,
                                        csaps_smooth=opt_smooth_hacat, 
                                        plot_ctrl_pts=False, 
                                        plt_col_size=5,
                                        plot_filename="./data/strict_pca/manually_curated/hacat.png")
            plot_splines_from_list(data_293t, [g], 
                                        color_phase=True,
                                        csaps_smooth=opt_smooth_293t, 
                                        plot_ctrl_pts=False, 
                                        plt_col_size=5,
                                        plot_filename="./data/strict_pca/manually_curated/293t.png")
            plt.show()
            input_phase = input("Assign phase: ")
            if input_phase not in ["M/G1","G1","G1/S","G2","G2/M"]:
                continue
            else:
                curated_phase.append(input_phase)
                curated_status.append('manual')
        else:
            curated_phase.append(row['assigned_phase_hacat'])
            curated_status.append('automatic')
        curated_index.append(g)
        curated_biotype.append(row['gene_biotype_hacat'])
        curated_gene_symbol.append(row['gene_symbols_hacat'])
    
    curated_df = pd.DataFrame({
        'assigned_phase': curated_phase, 
        'gene_biotype': curated_biotype, 
        'gene_symbol': curated_gene_symbol, 
        'curated': curated_status
    }, 
        index=curated_index)
    
    curated_df.to_csv(output_path)
    return curated_df

def manually_curate_overlapped_genes2(pls_overlap,opt_smooth_hacat=0.15,opt_smooth_293t=0.05,opt_smooth_jurkat=0.2,data_path_hacat=None,data_path_293t=None,data_path_jurkat=None, output_path=None,):
    if not (data_path_hacat or data_path_293t or output_path):
        raise ValueError("both data_path_hacat and data_path_293t must be set.")
    
    data_hacat = sc.read_h5ad(data_path_hacat)
    data_293t = sc.read_h5ad(data_path_293t)
    data_jurkat = sc.read_h5ad(data_path_jurkat)
    
    curated_index = []
    curated_phase = []
    curated_biotype = []
    curated_gene_symbol = []
    curated_status = []
    curated_r = []
    
    for g,row in pls_overlap.iterrows():
        if row['assigned_phase_jurkat'] != row['assigned_phase_curated']:
            plot_splines_from_list(data_hacat, [g], 
                                        color_phase=True,
                                        csaps_smooth=opt_smooth_hacat, 
                                        plot_ctrl_pts=False, 
                                        plt_col_size=5,
                                        plot_filename="./data/strict_pca/manually_curated/hacat.png")
            plot_splines_from_list(data_293t, [g], 
                                        color_phase=True,
                                        csaps_smooth=opt_smooth_293t, 
                                        plot_ctrl_pts=False, 
                                        plt_col_size=5,
                                        plot_filename="./data/strict_pca/manually_curated/293t.png")
            plot_splines_from_list(data_jurkat, [g], 
                                        color_phase=True,
                                        csaps_smooth=opt_smooth_jurkat, 
                                        plot_ctrl_pts=False, 
                                        plt_col_size=5,
                                        plot_filename="./data/strict_pca/manually_curated/jurkat.png")
            plt.show()
            input_phase = input("Assign phase: ")
            if input_phase not in ["M/G1","G1","G1/S","S","G2","G2/M"]:
                continue
            else:
                curated_phase.append(input_phase)
                curated_status.append('manual')
        else:
            curated_phase.append(row['assigned_phase_curated'])
            curated_status.append('automatic')
        curated_index.append(g)
        curated_biotype.append(row['gene_biotype_curated'])
        curated_gene_symbol.append(row['gene_symbol'])
        curated_r.append(np.mean(row['r_hacat'],row['r_jurkat'],row['r_293t']))
    
    curated_df = pd.DataFrame({
        'assigned_phase': curated_phase, 
        'gene_biotype': curated_biotype, 
        'gene_symbol': curated_gene_symbol, 
        'curated': curated_status,
        'mean_loading': curated_r
    }, 
        index=curated_index)
    
    curated_df.to_csv(output_path)
    return curated_df

def manually_curate_overlapped_genes_with_correlation(pls_overlap,opt_smooth_hacat=0.2,opt_smooth_293t=0.15,opt_smooth_jurkat=0.15,data_path_hacat=None,data_path_293t=None,data_path_jurkat=None, output_path=None,):
    if not (data_path_hacat or data_path_293t or output_path or data_path_jurkat):
        raise ValueError("data_path_hacat, data_path_293t, data_path_jurkat and output_path must be set.")
    
    data_hacat = sc.read_h5ad(data_path_hacat)
    data_293t = sc.read_h5ad(data_path_293t)
    data_jurkat = sc.read_h5ad(data_path_jurkat)
    
    curated_index = []
    curated_phase = []
    curated_biotype = []
    curated_gene_symbol = []
    curated_status = []
    curated_r = []
    curated_mean_corr = []
    
    for g,row in pls_overlap.iterrows():
        plot_splines_from_list(data_hacat, [g], 
                                    color_phase=True,
                                    csaps_smooth=opt_smooth_hacat, 
                                    plot_ctrl_pts=False, 
                                    plt_col_size=5,
                                    plot_filename="./data/final_results/manually_curated/hacat.png")
        sample_hacat = subsample_gene_from_splines(data_hacat,[g],100,csaps_smooth=opt_smooth_hacat)
        plot_splines_from_list(data_293t, [g], 
                                    color_phase=True,
                                    csaps_smooth=opt_smooth_293t, 
                                    plot_ctrl_pts=False, 
                                    plt_col_size=5,
                                    plot_filename="./data/final_results/manually_curated/293t.png")
        sample_293t = subsample_gene_from_splines(data_293t,[g],100,csaps_smooth=opt_smooth_293t)
        plot_splines_from_list(data_jurkat, [g], 
                                    color_phase=True,
                                    csaps_smooth=opt_smooth_jurkat, 
                                    plot_ctrl_pts=False, 
                                    plt_col_size=5,
                                    plot_filename="./data/final_results/manually_curated/jurkat.png")
        sample_jurkat = subsample_gene_from_splines(data_jurkat,[g],100,csaps_smooth=opt_smooth_jurkat)
        corr = np.mean(np.corrcoef(np.vstack((sample_hacat[g], sample_293t[g], sample_jurkat[g])))[np.triu_indices(3)])
        print(corr)
        
        plt.show()
        
        input_phase = input("Assign phase: ")
        if input_phase == '':
            input_phase = 'not_assigned'

        curated_phase.append(input_phase)
        curated_status.append('manual')
        curated_mean_corr.append(corr)

        curated_index.append(g)
        curated_biotype.append(row['gene_biotype_hacat'])
        curated_gene_symbol.append(row['gene_symbols_hacat'])
        curated_r.append(np.mean([row['r_hacat'],row['r_jurkat'],row['r_293t']]))
    
    curated_df = pd.DataFrame({
        'assigned_phase': curated_phase, 
        'gene_biotype': curated_biotype, 
        'gene_symbol': curated_gene_symbol, 
        'curated': curated_status,
        'mean_loading': curated_r,
        'mean_correlation': curated_mean_corr
    }, 
        index=curated_index)
    try:
        curated_df.to_csv(output_path)
    except Exception as e:
        print(e)
        print("Could not save dataframe!")
    return curated_df


def check_phase_and_correlation_overlapped_genes(pls_overlap,opt_smooth_hacat=0.2,opt_smooth_293t=0.15,opt_smooth_jurkat=0.15,data_path_hacat=None,data_path_293t=None,data_path_jurkat=None, output_path=None):
    if not (data_path_hacat or data_path_293t or output_path or data_path_jurkat):
        raise ValueError("data_path_hacat, data_path_293t, data_path_jurkat and output_path must be set.")
    
    data_hacat = sc.read_h5ad(data_path_hacat)
    data_293t = sc.read_h5ad(data_path_293t)
    data_jurkat = sc.read_h5ad(data_path_jurkat)
    
    curated_index = []
    curated_phase = []
    curated_biotype = []
    curated_gene_symbol = []
    curated_status = []
    curated_r = []
    curated_mean_corr = []
    curated_mean_peak = []
    
    for g,row in pls_overlap.iterrows():
        sample_hacat = subsample_genes_from_splines(data_hacat,[g],1000,csaps_smooth=opt_smooth_hacat)
        sample_293t = subsample_genes_from_splines(data_293t,[g],1000,csaps_smooth=opt_smooth_293t)
        sample_jurkat = subsample_genes_from_splines(data_jurkat,[g],1000,csaps_smooth=opt_smooth_jurkat)
        
        #find peaktime and account for cyclic behaviour - first 2 hours should be counted towards M
        peak_hacat = sample_hacat[g].idxmax()
        peak_hacat = peak_hacat + 24.0 if peak_hacat < 2.0 else peak_hacat
        peak_293t = sample_293t[g].idxmax()
        peak_293t = peak_293t + 24.0 if peak_293t < 2.0 else peak_293t
        peak_jurkat = sample_jurkat[g].idxmax()
        peak_jurkat = peak_jurkat + 24.0 if peak_jurkat < 2.0 else peak_jurkat
        
        curated_mean_peak.append(np.mean([peak_hacat,peak_293t,peak_jurkat]))
        
        corr = np.mean(np.corrcoef(np.vstack((sample_hacat[g], sample_293t[g], sample_jurkat[g])))[np.triu_indices(3)])
        
        if len(set([row['assigned_phase_hacat'],row['assigned_phase_293t'],row['assigned_phase_jurkat']])) != 1:
            input_phase = 'not_assigned'
        else:
            input_phase = row['assigned_phase_hacat']

        curated_phase.append(input_phase)
        curated_status.append('automatic')
        curated_mean_corr.append(corr)

        curated_index.append(g)
        curated_biotype.append(row['gene_biotype_hacat'])
        curated_gene_symbol.append(row['gene_symbols_hacat'])
        curated_r.append(np.mean([row['r_hacat'],row['r_jurkat'],row['r_293t']]))
    
    curated_df = pd.DataFrame({
        'assigned_phase': curated_phase, 
        'gene_biotype': curated_biotype, 
        'gene_symbol': curated_gene_symbol, 
        'curated': curated_status,
        'mean_loading': curated_r,
        'mean_correlation': curated_mean_corr,
        'mean_peaktime': curated_mean_peak
    }, 
        index=curated_index)
    
    curated_df = curated_df.sort_values(by=['mean_correlation'],ascending=False)
    curated_df['rank'] = np.array(range(len(curated_df)))+1
    
    try:
        curated_df.to_csv(output_path)
    except Exception as e:
        print(e)
        print("Could not save dataframe!")
    return curated_df

def resample_corrcoefs(pls_overlap,n=500,alpha=0.05,opt_smooth_hacat=0.2,opt_smooth_293t=0.15,opt_smooth_jurkat=0.15,data_path_hacat=None,data_path_293t=None,data_path_jurkat=None):
    if not (data_path_hacat or data_path_293t or data_path_jurkat):
        raise ValueError("data_path_hacat, data_path_293t, data_path_jurkat and output_path must be set.")
    
    data_hacat = sc.read_h5ad(data_path_hacat)
    data_293t = sc.read_h5ad(data_path_293t)
    data_jurkat = sc.read_h5ad(data_path_jurkat)
    
    curated_mean_corr = []
    
    for i in range(n):
        
        rand_hacat = pls_overlap.iloc[np.random.randint(0,high=len(pls_overlap))].name
        rand_293t = pls_overlap.iloc[np.random.randint(0,high=len(pls_overlap))].name
        rand_jurkat = pls_overlap.iloc[np.random.randint(0,high=len(pls_overlap))].name
        
        
        sample_hacat = subsample_genes_from_splines(data_hacat,[rand_hacat],500,csaps_smooth=opt_smooth_hacat)
        sample_293t = subsample_genes_from_splines(data_293t,[rand_293t],500,csaps_smooth=opt_smooth_293t)
        sample_jurkat = subsample_genes_from_splines(data_jurkat,[rand_jurkat],500,csaps_smooth=opt_smooth_jurkat)
        
        corr = np.mean(np.corrcoef(np.vstack((sample_hacat[rand_hacat], sample_293t[rand_293t], sample_jurkat[rand_jurkat])))[np.triu_indices(3)])

        curated_mean_corr.append(corr)
    
    curated_df = pd.DataFrame({
        'mean_correlation': curated_mean_corr,
    }
    )
    curated_df.dropna(inplace=True)
    
    curated_df = curated_df.sort_values(by=['mean_correlation'],ascending=False)
    
    return curated_df.iloc[int(n*alpha)]['mean_correlation']


def compare_genes_by_datasets(pls_overlap, ens_list, 
                              color_phase=True, 
                              plot_basepath='figures/compare_by_datasets/', 
                              subsample_width=2,
                              plot_ctrl_pts=False,
                              plt_col_size=3, 
                              spline_backend='csaps', 
                              csaps_smooth=0.3,
                              csaps_add_boundaries=True,
                              splipy_order=4,
                              splipy_smooth=True,
                              splipy_smooth_its=2,
                              opt_smooth_hacat=0.2,
                              opt_smooth_293t=0.15,
                              opt_smooth_jurkat=0.15,
                              data_path_hacat=None,
                              data_path_293t=None,
                              data_path_jurkat=None 
                              ):
    from math import ceil
    
    if not (data_path_hacat or data_path_293t or output_path or data_path_jurkat):
        raise ValueError("data_path_hacat, data_path_293t, data_path_jurkat and output_path must be set.")
    
    data_hacat = sc.read_h5ad(data_path_hacat)
    data_293t = sc.read_h5ad(data_path_293t)
    data_jurkat = sc.read_h5ad(data_path_jurkat)
    
    data = {'hacat': data_hacat, '293t': data_293t, 'jurkat': data_jurkat}
    opt_smooth = {'hacat': opt_smooth_hacat, '293t': opt_smooth_293t, 'jurkat': opt_smooth_jurkat}
    
    n_cols = plt_col_size
    n_rows = len(ens_list)
    row = 0 
    
    if not os.path.exists(os.path.dirname(plot_basepath)):
        os.makedirs(os.path.dirname(plot_basepath), exist_ok=True)
    
    for i, gene in enumerate(sorted(ens_list)):
        fig, ax = plt.subplots(1,n_cols,squeeze=False,figsize=(3*n_cols,4),sharey=True,sharex=True)
        plt.setp(ax, xticks = np.arange(0,25,6))
        col = 0 
        for (k_d, data_set), (k_o,smooth) in zip(data.items(),opt_smooth.items()):
            if spline_backend == 'splipy':
                curve, ctr_x = spline_model2_with_smoothing(data,gene,splipy_order=3,width=subsample_width,splipy_smooth=splipy_smooth,splipy_smooth_its=splipy_smooth_its,plot_figs=False)
                y_spline = plot_spline_curve(curve,0,24)
            elif spline_backend == 'csaps':
                sp, ctr_x = spline_model2_csaps(data_set,gene,width=subsample_width,csaps_smooth=smooth,plot_figs=False)
                xs = np.linspace(0,24,10000)
                y_spline = sp(xs)
                y_spline = np.concatenate(([xs],[y_spline]),axis=0).T
            else:
                raise NotImplementedError("Only ('splipy', 'csaps') are available.")

            lines = []

            if not color_phase:
                l = ax[row,col].plot(y_spline[:,0], y_spline[:,1])
                lines.append(l)
            else:
                color_map = {'G1': 'green','S': 'red', 'G2M': 'blue'}
                phase_boundaries = {'G1': 11, 'S': 19, 'G2M': 24}

                y_g1 = y_spline[y_spline[:,0] < 11,:]
                y_s = y_spline[(y_spline[:,0] >= 11) & (y_spline[:,0] < 19) ,:]
                y_g2m = y_spline[y_spline[:,0] >= 19,:]

                ax[0,col].plot(y_g1[:,0], y_g1[:,1], color=color_map['G1'])
                ax[0,col].plot(y_s[:,0], y_s[:,1], color=color_map['S'])
                ax[0,col].plot(y_g2m[:,0], y_g2m[:,1], color=color_map['G2M'])

            ax[0,col].set_title('{}'.format(k_d))

            if plot_ctrl_pts:
                ax[0,col].plot(ctr_x[:,0], ctr_x[:,1],'*',color='orange',label='Control points')
            col += 1

        fig.add_subplot(111, frameon=False)
        plt.grid(False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.ylabel('RPM (n)')
        plt.xlabel('Pseudotime (hours)')
        fig.suptitle('{} - mean peaktime={}, mean correlation={}'.format(pls_overlap.loc[gene,'gene_symbol'], round(pls_overlap.loc[gene,'mean_peaktime'],2), round(pls_overlap.loc[gene, 'mean_correlation'],2)),fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.999])
        #plt.gcf().subplots_adjust(bottom=0.25)

        #fig.savefig(os.path.join(plot_basepath,'{}.png'.format(gene)),bbox_inches='tight')
        fig.savefig(os.path.join(plot_basepath,'{}.png'.format(pls_overlap.loc[gene,'gene_symbol'])))