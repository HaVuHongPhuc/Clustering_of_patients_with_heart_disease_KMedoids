from pathlib import Path
import sys, json, hashlib, time, platform, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score

PROJECT = Path(r'C:\Users\havuh\OneDrive\Desktop\HocTap\Clustering_ of patients_with heart_disease_KMedoids')
sys.path.insert(0, str(PROJECT))
from src.kmedoids import KMedoids
from src import data as data_module
from src.data import *
from src.preprocessing import preprocess_features, gower_distance_matrix
from src.library_model import fit_kmedoids

OUT = PROJECT/'outputs'/'report_completion_20261105'
OUT.mkdir(parents=True, exist_ok=True)
def js(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic): return obj.item()
    raise TypeError(str(type(obj)))
def save_json(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=js), encoding='utf-8')

raw = pd.read_csv(PROJECT/'data/raw/heart_disease_whole.csv')
harm = load_heart_data(PROJECT/'data/raw/heart_disease_whole.csv')
dedup = drop_duplicate_rows(harm)
saved = pd.read_csv(PROJECT/'data/processed/heart_dedup.csv')
Xsaved = pd.read_csv(PROJECT/'data/processed/X_manhattan.csv')
ysaved = pd.read_csv(PROJECT/'data/processed/target.csv').iloc[:,0]
Gsaved = np.load(PROJECT/'data/processed/gower_distance.npy')
sweep = pd.read_csv(PROJECT/'outputs/metrics/k_selection.csv')
saved_labels = pd.read_csv(PROJECT/'outputs/metrics/labels.csv')
old_config = json.loads((PROJECT/'outputs/metrics/run_config.json').read_text(encoding='utf-8'))
original_domains = {k:set(v) for k,v in data_module.EXPECTED_VALUES.items()}

summary = {'raw_shape':list(raw.shape), 'harm_shape':list(harm.shape), 'dedup_shape':list(dedup.shape),
 'sources':raw.source.value_counts().to_dict(), 'target_counts':raw.target.value_counts().to_dict(),
 'missing_raw':raw.isna().sum().to_dict(), 'missing_harmonized':harm.isna().sum().to_dict(),
 'numeric_describe':raw[NUMERIC_COLS].describe().round(3).to_dict(), 'old_config':old_config,
 'domains_baseline':{k:sorted(v) for k,v in original_domains.items()},
 'hashes':{str(q.relative_to(PROJECT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [PROJECT/'data/raw/heart_disease_whole.csv', PROJECT/'data/processed/X_manhattan.csv', PROJECT/'src/kmedoids.py',PROJECT/'src/data.py']}}
clean = replace_abnormal_with_nan(dedup)
summary['domain_replaced_counts']={c:int((dedup[c].notna() & clean[c].isna()).sum()) for c in FEATURE_COLS}
summary['missing_before_imputation']=clean.isna().sum().to_dict()
clean, mode = impute_mode(clean, CATEGORICAL_COLS+BINARY_COLS)
clean, median = impute_median(clean)
summary['baseline_recreated_matches_saved']=bool(np.allclose(clean[saved.columns].to_numpy(),saved.to_numpy(),equal_nan=True))
Xdf,y = split_features_target(clean)
X, names, pre = preprocess_features(Xdf)
summary['matrix_matches_saved']=bool(np.allclose(X,Xsaved.to_numpy()))
summary['scaler']={'mean':pre.named_transformers_['num'].named_steps['sc'].mean_, 'scale':pre.named_transformers_['num'].named_steps['sc'].scale_, 'columns':NUMERIC_COLS}
summary['feature_names']=names

def metrics(labels, inertia, Xmetric, metric):
    return {'silhouette':float(silhouette_score(Xmetric, labels, metric=metric)), 'inertia':float(inertia),
      'ARI_target':float(adjusted_rand_score(y,labels)), 'purity':float(pd.crosstab(labels,y).max(axis=1).sum()/len(y)),
      'sizes':np.bincount(np.asarray(labels)).tolist()}
def details(name,labels,medoids,df):
    labels=np.asarray(labels)
    q=df.copy();q['cluster']=labels
    profile=q.groupby('cluster')[NUMERIC_COLS+['sex','fbs','exang','target']].mean()
    profile.insert(0,'n',np.bincount(labels)); profile.to_csv(OUT/(name+'_profiles.csv'))
    pd.crosstab(labels,y).to_csv(OUT/(name+'_target.csv'))
    pd.crosstab(labels,raw['source']).to_csv(OUT/(name+'_sources.csv'))
    pd.DataFrame({'record_index':np.arange(len(labels)),'cluster':labels}).to_csv(OUT/(name+'_labels.csv'),index=False)
    z=df.iloc[list(medoids)].copy(); z.insert(0,'record_index',list(medoids));z.insert(0,'cluster',np.arange(len(medoids)));z.to_csv(OUT/(name+'_medoids.csv'),index=False)
    return profile.reset_index().to_dict(orient='records')

print('BASELINE custom full dataset', flush=True)
start=time.perf_counter(); custom=KMedoids(n_clusters=2,metric='manhattan',random_state=42,max_iter=100).fit(X)
summary['custom_baseline']=metrics(custom.labels_,custom.inertia_,X,'manhattan')
summary['custom_baseline'].update(medoids=custom.medoid_indices_,n_iter=custom.n_iter_,converged=custom.converged_,fit_seconds=time.perf_counter()-start,history=custom.history_)
details('baseline_custom_k2',custom.labels_,custom.medoid_indices_,clean)
comparison=[dict(model='custom_manhattan_k2',**summary['custom_baseline'])]
for metric_name,k,arr,metric in [('manhattan',2,X,'manhattan'),('gower',2,Gsaved,'precomputed'),('gower',3,Gsaved,'precomputed')]:
    print('BASELINE library',metric_name,k, flush=True)
    res=fit_kmedoids(arr,k,metric=metric,random_state=42)
    m=metrics(res.labels,res.inertia,arr,metric)
    m.update(medoids=res.medoid_indices,n_iter=res.n_iter,ARI_saved=float(adjusted_rand_score(res.labels,saved_labels[f'{metric_name}_k{k}'])))
    summary[f'library_baseline_{metric_name}_k{k}']=m
    comparison.append(dict(model=f'library_{metric_name}_k{k}',**m))
    details(f'baseline_library_{metric_name}_k{k}',res.labels,res.medoid_indices,clean)
    if metric_name=='manhattan': summary['ARI_custom_library']=float(adjusted_rand_score(custom.labels_,res.labels))
pd.DataFrame(comparison).to_csv(OUT/'baseline_comparison.csv',index=False)

print('MANUAL 30 records',flush=True)
Xm=X[:30]
manual=KMedoids(n_clusters=2,metric='manhattan',initial_medoids=[0,15],max_iter=100).fit(Xm)
Dm=np.abs(Xm[:,None,:]-Xm[None,:,:]).sum(axis=2)
manual_data={'records':clean.iloc[:30].to_dict(orient='records'),'history':manual.history_,'labels':manual.labels_,'medoids':manual.medoid_indices_,'inertia':manual.inertia_, 'n_iter':manual.n_iter_, 'scaler':summary['scaler']}
for step,meds in [('initial',[0,15]),('updated',list(manual.medoid_indices_))]:
    lab=np.argmin(Dm[:,meds],axis=1)
    rows=[]
    for i in range(30): rows.append({'record':f'R{i+1}', **{f'd_R{j+1}':Dm[i,j] for j in meds},'cluster':int(lab[i])})
    manual_data[step+'_assignment']=rows
    costs=[]
    for c in range(2):
        members=np.where(lab==c)[0]
        costs.extend({'cluster':c,'record':f'R{i+1}','cost':float(Dm[i,members].sum())} for i in members)
    manual_data[step+'_costs']=costs
manual_data['R2_R1_numeric_contributions']=np.abs(Xm[1,:5]-Xm[0,:5])
manual_data['R2_R1_categorical_contribution']=float(np.abs(Xm[1,5:22]-Xm[0,5:22]).sum())
manual_data['R2_R1_binary_contribution']=float(np.abs(Xm[1,22:]-Xm[0,22:]).sum())
libm=fit_kmedoids(Xm,2,metric='manhattan',random_state=42)
manual_data['library']={'medoids':libm.medoid_indices,'labels':libm.labels,'inertia':libm.inertia,'ARI_custom':adjusted_rand_score(manual.labels_,libm.labels)}
save_json('manual30.json',manual_data)

# Sensitivity analysis preserves the additional codes confirmed by the group.
# It is intentionally isolated from project source and existing processed artifacts.
print('SENSITIVITY expanded user-confirmed domains',flush=True)
data_module.EXPECTED_VALUES['cp'].add(5)
data_module.EXPECTED_VALUES['slope'].add(4)
data_module.EXPECTED_VALUES['ca'].add(4)
expanded=replace_abnormal_with_nan(dedup)
summary['sensitivity_missing_before_imputation']=expanded.isna().sum().to_dict()
expanded,smode=impute_mode(expanded,CATEGORICAL_COLS+BINARY_COLS)
expanded,smedian=impute_median(expanded)
Xs,ys=split_features_target(expanded)
Xsens,snames,sp=preprocess_features(Xs)
Gsens=gower_distance_matrix(Xs)
summary['sensitivity_features']=snames
summary['sensitivity_mode']=smode
summary['sensitivity_domains']={k:sorted(v) for k,v in data_module.EXPECTED_VALUES.items()}
sen_rows=[]
for tag,arr,metric in [('manhattan',Xsens,'manhattan'),('gower',Gsens,'precomputed')]:
    for k in range(2,9):
        print('SENSITIVITY',tag,k,flush=True)
        res=fit_kmedoids(arr,k,metric=metric,random_state=42)
        m=metrics(res.labels,res.inertia,arr,metric)
        m.update(medoids=res.medoid_indices,n_iter=res.n_iter)
        sen_rows.append(dict(distance=tag,k=k,**m))
        if k==2:
            summary[f'sensitivity_library_{tag}_k2']=m
            details(f'sensitivity_library_{tag}_k2',res.labels,res.medoid_indices,expanded)
senscustom=KMedoids(n_clusters=2,metric='manhattan',random_state=42,max_iter=100).fit(Xsens)
summary['custom_sensitivity']=metrics(senscustom.labels_,senscustom.inertia_,Xsens,'manhattan')
summary['custom_sensitivity'].update(medoids=senscustom.medoid_indices_,n_iter=senscustom.n_iter_,converged=senscustom.converged_)
details('sensitivity_custom_k2',senscustom.labels_,senscustom.medoid_indices_,expanded)
pd.DataFrame(sen_rows).to_csv(OUT/'sensitivity_k_selection.csv',index=False)
summary['sensitivity_best']={tag:int(max([r for r in sen_rows if r['distance']==tag],key=lambda r:r['silhouette'])['k']) for tag in ['manhattan','gower']}
summary['missing_total_raw']=int(raw.isna().sum().sum())
summary['missing_total_baseline']=int(sum(summary['missing_before_imputation'].values()))
summary['numeric_stats']=raw[NUMERIC_COLS].describe().to_dict()
save_json('report_analysis.json',summary)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':160})
fig,ax=plt.subplots(figsize=(8,3.4));miss=raw[FEATURE_COLS].isna().mean().sort_values(ascending=False)*100
ax.bar(miss.index,miss.values,color='#537999');ax.set_ylabel('Missing values (%)');ax.set_ylim(0,max(miss)*1.17);ax.tick_params(axis='x',rotation=40);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'eda_missing.png');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(8,3.2));vc=raw.target.value_counts().sort_index();axs[0].bar(['target=0','target=1'],vc.values,color=['#537999','#b47c56']);axs[0].set_ylabel('Records');sc=raw.source.value_counts();axs[1].barh([x.replace('uci_multi_','uci_') for x in sc.index],sc.values,color='#537999');axs[1].invert_yaxis();axs[1].set_xlabel('Records');fig.tight_layout();fig.savefig(OUT/'eda_sources_target.png');plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(8,5));
for ax,col in zip(axs.flat,NUMERIC_COLS): ax.hist(raw[col].dropna(),bins=22,color='#537999',edgecolor='white');ax.set_title(col);ax.set_ylabel('Records')
axs.flat[-1].axis('off');fig.tight_layout();fig.savefig(OUT/'eda_numeric.png');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(8,3.2));
for ax,tag in zip(axs,['manhattan','gower']):
    b=sweep[sweep.distance==tag];s=pd.DataFrame(sen_rows);s=s[s.distance==tag];ax.plot(b.k,b.silhouette,'o-',label='Baseline domains');ax.plot(s.k,s.silhouette,'s--',label='Expanded domains');ax.set_title(tag.capitalize());ax.set_xlabel('k');ax.set_ylabel('Silhouette');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'sensitivity_silhouette.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(7.8,3.3));pro=pd.read_csv(OUT/'baseline_custom_k2_sources.csv',index_col=0);pro.div(pro.sum(axis=1),axis=0).plot.bar(stacked=True,ax=ax,colormap='tab20');ax.set_ylabel('Share within cluster');ax.set_xlabel('Custom cluster');ax.legend(bbox_to_anchor=(1.01,1),fontsize=7);fig.tight_layout();fig.savefig(OUT/'custom_source_composition.png');plt.close(fig)
print('COMPLETE',OUT,flush=True)
print(json.dumps({k:summary[k] for k in ['custom_baseline','ARI_custom_library','sensitivity_best','custom_sensitivity']},default=js),flush=True)
