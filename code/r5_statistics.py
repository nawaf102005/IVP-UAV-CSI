"""Paired effect intervals, equivalence tests and explicitly scoped correction."""
import numpy as np
from scipy import stats

def mean_ci(x,alpha=.05):
    x=np.asarray(x,float);m=float(x.mean());se=float(stats.sem(x));h=stats.t.ppf(1-alpha/2,len(x)-1)*se
    return dict(n=len(x),mean=m,lo=float(m-h),hi=float(m+h),se=se)

def wilson(k,n,alpha=.05):
    z=stats.norm.isf(alpha/2);p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(n=int(n),count=int(k),mean=p,lo=mid-h,hi=mid+h)

def tost(x,margin=.5):
    r=mean_ci(x,.1)
    if r['se']==0:return dict(**r,margin=margin,p=1.,status='zero observed variance; t inference not used')
    df=len(x)-1;p=max(stats.t.sf((r['mean']+margin)/r['se'],df),stats.t.cdf((r['mean']-margin)/r['se'],df))
    return dict(**r,margin=margin,p=float(p),status='paired t-TOST')

def holm(p):
    p=np.asarray(p,float);order=np.argsort(p);adj=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]);ans=np.empty(len(p));ans[order]=np.minimum(adj,1);return ans
