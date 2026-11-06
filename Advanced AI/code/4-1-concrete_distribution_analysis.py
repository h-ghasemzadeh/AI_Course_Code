# -*- coding: utf-8 -*-
"""Comprehensive statistical analysis of the UCI Concrete Compressive Strength data."""
import argparse, json, math, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
TARGET = "Concrete compressive strength (MPa)"
COLS = [
"Cement (component 1)(kg in a m^3 mixture)",
"Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
"Fly Ash (component 3)(kg in a m^3 mixture)",
"Water (component 4)(kg in a m^3 mixture)",
"Superplasticizer (component 5)(kg in a m^3 mixture)",
"Coarse Aggregate (component 6)(kg in a m^3 mixture)",
"Fine Aggregate (component 7)(kg in a m^3 mixture)",
"Age (day)", TARGET]
SHORT = dict(zip(COLS, ["Cement","Blast furnace slag","Fly ash","Water","Superplasticizer",
                         "Coarse aggregate","Fine aggregate","Age","Compressive strength"]))
DISTS = {"normal":stats.norm, "lognormal":stats.lognorm, "gamma":stats.gamma,
         "weibull_min":stats.weibull_min, "exponential":stats.expon,
         "chi_square":stats.chi2, "beta_scaled":stats.beta}

def load_data(path):
    if path:
        p=Path(path)
        if not p.exists(): raise FileNotFoundError(p)
        df=pd.read_excel(p) if p.suffix.lower() in (".xls",".xlsx") else pd.read_csv(p)
        source=str(p.resolve())
    else:
        try:
            df=pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls")
            source="UCI Machine Learning Repository (Concrete Compressive Strength)"
        except Exception as e:
            raise RuntimeError("دانلود خودکار ناموفق بود. فایل را از https://archive.ics.uci.edu/dataset/165/concrete-compressive-strength دریافت و با --data معرفی کنید. "+str(e))
    df.columns=[str(c).strip() for c in df.columns]
    if len(df.columns)==9 and TARGET not in df.columns: df.columns=COLS
    if TARGET not in df.columns:
        cand=[c for c in df.columns if "strength" in c.lower() or "compressive" in c.lower()]
        if cand: df=df.rename(columns={cand[-1]:TARGET})
        elif df.shape[1]==9: df.columns=COLS
        else: raise ValueError("Target column not found; expected 9 columns.")
    for c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(how="all").dropna(subset=[TARGET]).reset_index(drop=True)
    return df,source

def slug(s):
    import re
    return re.sub(r"[^A-Za-z0-9]+","_",str(s)).strip("_").lower()[:60] or "feature"

def savefig(path):
    plt.tight_layout(); plt.savefig(path,dpi=170,bbox_inches="tight",facecolor="white"); plt.close()

def descriptive(df):
    rows=[]
    for c in df:
        x=df[c].dropna().to_numpy(float); n=len(x); mean=np.mean(x) if n else np.nan
        sd=np.std(x,ddof=1) if n>1 else np.nan
        rows.append({"feature":c,"short_name":SHORT.get(c,c),"n":n,"missing_n":int(df[c].isna().sum()),
          "mean":mean,"median":np.median(x) if n else np.nan,"variance_sample":np.var(x,ddof=1) if n>1 else np.nan,
          "std_sample":sd,"min":np.min(x) if n else np.nan,"q1":np.quantile(x,.25) if n else np.nan,
          "q3":np.quantile(x,.75) if n else np.nan,"max":np.max(x) if n else np.nan,
          "skewness":stats.skew(x,bias=False) if n>2 else np.nan,
          "excess_kurtosis":stats.kurtosis(x,bias=False) if n>3 else np.nan,
          "coefficient_of_variation":sd/mean if n>1 and mean!=0 else np.nan})
    return pd.DataFrame(rows)

def fit_dist(x,name,dist,bootstrap,rng):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    r={"distribution":name,"n":n,"parameters":"","log_likelihood":np.nan,"AIC":np.nan,"BIC":np.nan,
       "KS_statistic":np.nan,"KS_pvalue_naive":np.nan,"KS_pvalue_bootstrap":np.nan,"AD_statistic":np.nan,"fit_status":"failed"}
    if n<8 or np.ptp(x)<=0: r["fit_status"]="insufficient data/variation"; return r
    try:
        if name=="beta_scaled":
            lo,hi=float(x.min()),float(x.max()); z=np.clip((x-lo)/(hi-lo),1e-6,1-1e-6)
            par=dist.fit(z,floc=0,fscale=1)
            logpdf=dist.logpdf(z,*par)
            cdf=lambda v:dist.cdf(np.clip((np.asarray(v)-lo)/(hi-lo),1e-6,1-1e-6),*par)
            partext=json.dumps({"shape_loc_scale":[float(v) for v in par],"original_min":lo,"original_max":hi})
        else:
            if name in ("lognormal","gamma","weibull_min","exponential","chi_square") and x.min()>=0:
                par=dist.fit(x,floc=0)
            else: par=dist.fit(x)
            logpdf=dist.logpdf(x,*par); cdf=lambda v:dist.cdf(v,*par)
            partext=json.dumps([float(v) for v in par])
        if not np.all(np.isfinite(logpdf)): raise ValueError("nonfinite logpdf")
        ll=float(logpdf.sum()); k=len(par); ks=stats.kstest(x,cdf)
        r.update(parameters=partext,log_likelihood=ll,AIC=2*k-2*ll,BIC=np.log(n)*k-2*ll,
                 KS_statistic=float(ks.statistic),KS_pvalue_naive=float(ks.pvalue),fit_status="ok")
        if name=="normal":
            try:r["AD_statistic"]=float(stats.anderson(x,dist="norm").statistic)
            except Exception:pass
        if bootstrap>0:
            try:
                exceed=0
                for _ in range(bootstrap):
                    sim=dist.rvs(*par,size=n,random_state=rng)
                    if name=="beta_scaled":
                        sim=lo+sim*(hi-lo); a,b=sim.min(),sim.max()
                        z2=np.clip((sim-a)/(b-a+1e-15),1e-6,1-1e-6); p2=dist.fit(z2,floc=0,fscale=1)
                        c2=lambda v:dist.cdf(np.clip((np.asarray(v)-a)/(b-a+1e-15),1e-6,1-1e-6),*p2)
                    else:
                        p2=dist.fit(sim,floc=0) if name in ("lognormal","gamma","weibull_min","exponential","chi_square") and sim.min()>=0 else dist.fit(sim)
                        c2=lambda v:dist.cdf(v,*p2)
                    exceed += stats.kstest(sim,c2).statistic >= ks.statistic
                r["KS_pvalue_bootstrap"]=(exceed+1)/(bootstrap+1)
            except Exception: pass
    except Exception as e:r["fit_status"]="failed: "+type(e).__name__
    return r

def distribution_analysis(df,bootstrap,seed):
    rng=np.random.default_rng(seed); rows=[]
    for c in df:
        x=df[c].dropna().to_numpy(float)
        for name,d in DISTS.items():
            a=fit_dist(x,name,d,bootstrap,rng); a.update(feature=c,short_name=SHORT.get(c,c)); rows.append(a)
    fits=pd.DataFrame(rows)
    fits["AIC_rank"]=fits.groupby("feature")["AIC"].rank(method="min")
    fits["BIC_rank"]=fits.groupby("feature")["BIC"].rank(method="min")
    ok=fits[(fits.fit_status=="ok") & fits.distribution.isin(DISTS)]
    best=ok.sort_values(["feature","BIC","AIC"]).groupby("feature",as_index=False).first()
    return fits,best[["feature","short_name","distribution","AIC","BIC","KS_statistic","KS_pvalue_naive","KS_pvalue_bootstrap","AD_statistic","parameters"]]

def normality(df):
    rows=[]
    for c in df:
        x=df[c].dropna().to_numpy(float); r={"feature":c,"short_name":SHORT.get(c,c),"n":len(x)}
        for label,func in [("Shapiro",stats.shapiro),("D'Agostino",stats.normaltest),("Jarque_Bera",stats.jarque_bera)]:
            try:
                if (label=="Shapiro" and 3<=len(x)<=5000) or (label!="Shapiro" and len(x)>=8):
                    z=func(x); r[label+"_statistic"]=float(z.statistic); r[label+"_pvalue"]=float(z.pvalue)
                else:r[label+"_statistic"]=np.nan;r[label+"_pvalue"]=np.nan
            except Exception:r[label+"_statistic"]=np.nan;r[label+"_pvalue"]=np.nan
        rows.append(r)
    return pd.DataFrame(rows)

def relationships(df,seed):
    X=df.drop(columns=[TARGET]); y=df[TARGET]; rows=[]
    for c in X:
        d=pd.DataFrame({"x":X[c],"y":y}).dropna(); x=d.x.to_numpy(); yy=d.y.to_numpy()
        if len(d)<3:continue
        p=stats.pearsonr(x,yy); s=stats.spearmanr(x,yy)
        mi=mutual_info_regression(x.reshape(-1,1),yy,random_state=seed)[0]
        rows.append({"feature":c,"short_name":SHORT.get(c,c),"n":len(d),"pearson_r":p.statistic,
                     "pearson_pvalue":p.pvalue,"spearman_rho":s.statistic,"spearman_pvalue":s.pvalue,
                     "mutual_information":mi})
    out=pd.DataFrame(rows)
    for col in ["pearson_pvalue","spearman_pvalue"]:
        if len(out):
            p=out[col].to_numpy(); order=np.argsort(p); q=np.empty(len(p)); adj=p[order]*len(p)/(np.arange(len(p))+1)
            adj=np.minimum.accumulate(adj[::-1])[::-1]; q[order]=np.clip(adj,0,1)
            out[col.replace("pvalue","BH_qvalue")]=q
    return out.sort_values("mutual_information",ascending=False)

def make_plots(df,fits,best,rels,pca_var,out):
    plots=[]
    for c in df:
        x=df[c].dropna().to_numpy(float)
        if len(x)<3:continue
        fig,ax=plt.subplots(figsize=(7.5,4.7)); ax.hist(x,bins="auto",density=True,alpha=.45,edgecolor="white",label="Observed")
        chosen=best[best.feature==c]
        if len(chosen):
            name=chosen.iloc[0].distribution; row=fits[(fits.feature==c)&(fits.distribution==name)&(fits.fit_status=="ok")]
            try:
                pars=json.loads(row.iloc[0].parameters); dist=DISTS[name]; xx=np.linspace(x.min(),x.max(),400)
                if name=="beta_scaled":
                    lo,hi=pars["original_min"],pars["original_max"]; pp=pars["shape_loc_scale"]
                    yy=dist.pdf(np.clip((xx-lo)/(hi-lo),1e-6,1-1e-6),*pp)/(hi-lo)
                else:yy=dist.pdf(xx,*pars)
                ax.plot(xx,yy,lw=2,label="Selected: "+name); ax.legend()
                # Q-Q plot
                if name=="beta_scaled":
                    lo,hi=pars["original_min"],pars["original_max"]; xxq=dist.ppf(np.linspace(.01,.99,len(x)),*pars["shape_loc_scale"]); yyq=np.sort(np.clip((x-lo)/(hi-lo),1e-6,1-1e-6))
                else:xxq=dist.ppf(np.linspace(.01,.99,len(x)),*pars); yyq=np.sort(x)
                f2,a2=plt.subplots(figsize=(5.5,5.5)); a2.scatter(xxq,yyq,s=12,alpha=.6)
                lim=[min(np.nanmin(xxq),np.nanmin(yyq)),max(np.nanmax(xxq),np.nanmax(yyq))]
                a2.plot(lim,lim,"--");a2.set(xlabel="Theoretical quantiles",ylabel="Observed quantiles",title=f"Q-Q: {SHORT.get(c,c)}")
                fn="qq_"+slug(SHORT.get(c,c))+".png";savefig(out/fn);plots.append((fn,"Q-Q plot: "+SHORT.get(c,c)))
            except Exception:pass
        ax.set(xlabel=c,ylabel="Density",title="Histogram and fitted distribution: "+SHORT.get(c,c))
        fn="hist_fit_"+slug(SHORT.get(c,c))+".png";savefig(out/fn);plots.append((fn,"Histogram and fitted distribution: "+SHORT.get(c,c)))
    for metric in ["AIC","BIC"]:
        d=fits[(fits.fit_status=="ok") & fits.distribution.isin(DISTS)]
        piv=d.pivot_table(index="distribution",columns="short_name",values=metric,aggfunc="min")
        fig,ax=plt.subplots(figsize=(11,5));piv.plot(kind="bar",ax=ax);ax.set(title=metric+" comparison",ylabel=metric,xlabel="Distribution");ax.legend(fontsize=7,ncol=2)
        fn=metric.lower()+"_comparison.png";savefig(out/fn);plots.append((fn,metric+" comparison"))
    corr=df.corr(numeric_only=True)
    fig,ax=plt.subplots(figsize=(10,8));im=ax.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1);fig.colorbar(im,ax=ax)
    ax.set_xticks(range(len(corr)),[SHORT.get(c,c) for c in corr.columns],rotation=65,ha="right")
    ax.set_yticks(range(len(corr)),[SHORT.get(c,c) for c in corr.index])
    for i in range(len(corr)):
        for j in range(len(corr)):ax.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center",fontsize=7)
    ax.set_title("Correlation matrix");fn="correlation_heatmap.png";savefig(out/fn);plots.append((fn,"Correlation heatmap"))
    # Pairwise plots (full 9-variable matrix can be large; use all variables).
    try:
        import seaborn as sns
        pp=sns.pairplot(df.rename(columns={c:SHORT.get(c,c) for c in df.columns}),corner=True,diag_kind="hist",plot_kws={"s":8,"alpha":.4})
        pp.fig.suptitle("Pairwise plots of inputs and strength",y=1.02);fn="pairplot_all_variables.png";pp.savefig(out/fn,dpi=130,bbox_inches="tight");plt.close(pp.fig);plots.append((fn,"Pairwise scatter plots and histograms"))
    except Exception:
        pass
    fig,ax=plt.subplots(figsize=(7,4.5));ix=np.arange(1,len(pca_var)+1)
    ax.bar(ix,pca_var.explained_variance_ratio);ax.plot(ix,pca_var.cumulative_explained_variance,"o-")
    ax.set(xlabel="Principal component",ylabel="Explained variance ratio",title="PCA explained variance",ylim=(0,1.05))
    fn="pca_explained_variance.png";savefig(out/fn);plots.append((fn,"PCA explained variance"))
    for c in df.drop(columns=[TARGET]):
        d=df[[c,TARGET]].dropna();fig,ax=plt.subplots(figsize=(6.5,4.5));ax.scatter(d[c],d[TARGET],s=12,alpha=.5)
        if d[c].nunique()>1:
            sl,inter,*_=stats.linregress(d[c],d[TARGET]);xx=np.linspace(d[c].min(),d[c].max(),100);ax.plot(xx,inter+sl*xx)
        rr=rels[rels.feature==c]
        title=f"{SHORT.get(c,c)} vs strength"
        if len(rr):title+=f" (Pearson r={rr.iloc[0].pearson_r:.3f})"
        ax.set(xlabel=SHORT.get(c,c),ylabel="Strength (MPa)",title=title)
        fn="relationship_"+slug(SHORT.get(c,c))+".png";savefig(out/fn);plots.append((fn,"Relationship: "+SHORT.get(c,c)))
    return plots

def model_analysis(df,folds,seed,out):
    X=df.drop(columns=[TARGET]);y=df[TARGET]; nfold=min(max(2,folds),len(y))
    cv=KFold(n_splits=nfold,shuffle=True,random_state=seed)
    model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=1.0))
    pred=cross_val_predict(model,X,y,cv=cv);res=y.to_numpy()-pred
    metrics=pd.DataFrame([{"model":"StandardScaler + Ridge(alpha=1)","n":len(y),"cv_folds":nfold,
      "R2_CV":r2_score(y,pred),"RMSE_CV_MPa":np.sqrt(mean_squared_error(y,pred)),
      "MAE_CV_MPa":mean_absolute_error(y,pred),"residual_mean_MPa":res.mean(),"residual_std_MPa":res.std(ddof=1)}])
    metrics.to_csv(out/"model_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame({"actual_strength_MPa":y.to_numpy(),"cross_validated_prediction_MPa":pred,"residual_MPa":res}).to_csv(out/"cross_validated_predictions.csv",index=False,encoding="utf-8-sig")
    fig,ax=plt.subplots(figsize=(6,5));ax.scatter(y,pred,s=14,alpha=.5);lo=min(y.min(),pred.min());hi=max(y.max(),pred.max());ax.plot([lo,hi],[lo,hi],"--")
    ax.set(xlabel="Observed strength (MPa)",ylabel="CV prediction (MPa)",title=f"Observed vs predicted, R²={metrics.iloc[0].R2_CV:.3f}");savefig(out/"actual_vs_predicted.png")
    fig,ax=plt.subplots(figsize=(6,4.5));ax.scatter(pred,res,s=14,alpha=.5);ax.axhline(0,ls="--");ax.set(xlabel="CV prediction",ylabel="Residual",title="Residuals vs predictions");savefig(out/"residuals_vs_predicted.png")
    fig,ax=plt.subplots(figsize=(6,4.5));ax.hist(res,bins="auto",edgecolor="white");ax.set(xlabel="Residual (MPa)",ylabel="Frequency",title="Residual distribution");savefig(out/"residual_histogram.png")
    fig,ax=plt.subplots(figsize=(5.5,5.5));stats.probplot(res,dist="norm",plot=ax);ax.set_title("Residual Q-Q plot");savefig(out/"residual_qq.png")
    return metrics

def html_table(df,rows=100):
    if df is None or df.empty:return "<p>No results.</p>"
    d=df.head(rows).copy()
    for c in d.select_dtypes(include=[np.number]).columns:d[c]=d[c].map(lambda v:"" if pd.isna(v) else f"{v:.4g}")
    return d.to_html(index=False,escape=True,border=0,classes="tbl")

def report(df,source,desc,fits,best,norm,corr,pca_var,rels,metrics,plots,out):
    s=desc[desc.feature==TARGET].iloc[0];m=metrics.iloc[0]
    top=rels.sort_values("mutual_information",ascending=False).iloc[0] if len(rels) else None
    css="""body{font:15px Arial,sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:#202a35;line-height:1.5}h1,h2{color:#164e7a}.tbl{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0 24px}.tbl td,.tbl th{border:1px solid #d8e0e8;padding:5px}.tbl th{background:#edf3f8}figure{border:1px solid #ddd;padding:10px;margin:20px 0}img{max-width:100%;height:auto}.note{background:#fff8e6;padding:12px}"""
    parts=[f'<!doctype html><html><head><meta charset="utf-8"><title>Concrete statistical report</title><style>{css}</style></head><body>',
      '<h1>Concrete Compressive Strength: Statistical Analysis Report</h1>',
      f'<p><b>Source:</b> {source}<br><b>Rows:</b> {len(df)}; <b>variables:</b> {df.shape[1]} (8 inputs + target, when canonical dataset used).</p>',
      f'<h2>1. Executive summary</h2><p>Mean target = {s["mean"]:.3f} MPa; standard deviation = {s["std_sample"]:.3f} MPa; observed range = {s["min"]:.3f} to {s["max"]:.3f} MPa.</p>',
      f'<p>Ridge cross-validation: R²={m["R2_CV"]:.3f}, RMSE={m["RMSE_CV_MPa"]:.3f} MPa, MAE={m["MAE_CV_MPa"]:.3f} MPa.</p>']
    if top is not None:parts.append(f'<p>Highest estimated mutual information with target: <b>{top["short_name"]}</b> (MI={top["mutual_information"]:.4g}); this is not evidence of causality.</p>')
    parts += ['<h2>2. Descriptive statistics</h2>',html_table(desc),
      '<h2>3. Best initial distribution candidates</h2><p>Candidate selected by minimum BIC (AIC is also reported). Use plots and goodness-of-fit results before judging adequacy.</p>',
      html_table(best[['short_name','distribution','AIC','BIC','KS_statistic','KS_pvalue_naive','KS_pvalue_bootstrap','AD_statistic']]),
      '<h3>All candidate fits</h3>',html_table(fits[['short_name','distribution','parameters','log_likelihood','AIC','BIC','KS_statistic','KS_pvalue_naive','KS_pvalue_bootstrap','AD_statistic','fit_status']]),
      '<h3>Normality tests</h3>',html_table(norm),
      '<h2>4. Dependence, correlation, and PCA</h2><h3>Correlation matrix</h3>',html_table(corr.reset_index()),
      '<h3>PCA explained variance</h3>',html_table(pca_var),
      '<h2>5. Input–target relationships</h2>',html_table(rels),
      '<h2>6. Prediction and residual analysis</h2>',html_table(metrics),
      '<p>Predictions are out-of-fold predictions from shuffled K-fold cross-validation.</p><h2>7. Figures</h2>']
    for fn,caption in plots:parts.append(f'<figure><img src="{fn}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
    parts.append("""<h2>8. Statistical cautions</h2><ul>
<li>AIC/BIC rank only the candidate distributions considered; they do not prove a true generating distribution.</li>
<li>The ordinary KS p-value is not calibrated for parameters estimated from the same sample. Parametric-bootstrap KS p-values are included where possible; results depend on bootstrap replicate count.</li>
<li>Anderson–Darling statistic is shown for the normal candidate only and is not a calibrated p-value for all fitted families.</li>
<li>Beta is fitted after scaling each variable to its observed minimum and maximum; this is a descriptive candidate and can be sensitive to sample bounds.</li>
<li>Poisson is not appropriate for continuous mix quantities or compressive strength, so discrete distributions are not mixed into the continuous AIC/BIC ranking.</li>
<li>Correlation and PCA do not fully specify a joint probability distribution; association does not establish causation.</li>
<li>Ridge is a baseline model only; compare with other models and external validation before engineering use.</li>
</ul></body></html>""")
    (out/"final_report.html").write_text("\n".join(parts),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default=None,help="Optional local CSV/XLS/XLSX")
    ap.add_argument("--output-dir",default="concrete_analysis_output")
    ap.add_argument("--cv-folds",type=int,default=5)
    ap.add_argument("--bootstrap",type=int,default=99,help="Parametric bootstrap repetitions for KS (0 disables)")
    ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args();out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    df,source=load_data(a.data)
    df=df.select_dtypes(include=[np.number]).copy()
    if df.shape[1]!=9:print(f"WARNING: expected 9 numeric columns; found {df.shape[1]}")
    desc=descriptive(df);desc.to_csv(out/"descriptive_statistics.csv",index=False,encoding="utf-8-sig")
    fits,best=distribution_analysis(df,a.bootstrap,a.seed)
    fits.to_csv(out/"distribution_fits.csv",index=False,encoding="utf-8-sig")
    best.to_csv(out/"best_initial_distributions.csv",index=False,encoding="utf-8-sig")
    norm=normality(df);norm.to_csv(out/"normality_tests.csv",index=False,encoding="utf-8-sig")
    rels=relationships(df,a.seed);rels.to_csv(out/"feature_target_relationships.csv",index=False,encoding="utf-8-sig")
    corr=df.corr(numeric_only=True);cov=df.cov(numeric_only=True)
    corr.to_csv(out/"correlation_matrix.csv",encoding="utf-8-sig");cov.to_csv(out/"covariance_matrix.csv",encoding="utf-8-sig")
    X=df.drop(columns=[TARGET]);Xi=SimpleImputer(strategy="median").fit_transform(X);Xs=StandardScaler().fit_transform(Xi)
    pca=PCA().fit(Xs);scores=pca.transform(Xs)
    pv=pd.DataFrame({"component":[f"PC{i+1}" for i in range(len(X.columns))],"explained_variance_ratio":pca.explained_variance_ratio_,"cumulative_explained_variance":np.cumsum(pca.explained_variance_ratio_)})
    pv.to_csv(out/"pca_explained_variance.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(pca.components_.T,index=X.columns,columns=[f"PC{i+1}" for i in range(len(X.columns))]).rename_axis("feature").to_csv(out/"pca_loadings.csv",encoding="utf-8-sig")
    pd.DataFrame(scores[:,:min(3,scores.shape[1])],columns=[f"PC{i+1}" for i in range(min(3,scores.shape[1]))]).to_csv(out/"pca_scores.csv",index=False,encoding="utf-8-sig")
    metrics=model_analysis(df,a.cv_folds,a.seed,out)
    plots=make_plots(df,fits,best,rels,pv,out)
    # Include model plots in the report.
    for fn,cap in [("actual_vs_predicted.png","Observed vs cross-validated predicted strength"),("residuals_vs_predicted.png","Residuals vs predictions"),("residual_histogram.png","Residual histogram"),("residual_qq.png","Residual Q-Q plot")]:
        plots.append((fn,cap))
    meta={"source":source,"n_rows":len(df),"columns":list(df.columns),"target":TARGET,"seed":a.seed,"cv_folds":a.cv_folds,"bootstrap_replicates":a.bootstrap,"python":sys.version,"pandas":pd.__version__,"numpy":np.__version__}
    (out/"run_metadata.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    report(df,source,desc,fits,best,norm,corr,pv,rels,metrics,plots,out)
    print("Analysis complete.")
    print("Output folder:",out)
    print("Final HTML report:",out/"final_report.html")
    print(best[["short_name","distribution","AIC","BIC"]].to_string(index=False))
    print(metrics.to_string(index=False))

if __name__=="__main__": main()
