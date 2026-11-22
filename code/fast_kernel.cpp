// Portable C++17 numerical kernel. No external libraries or hidden training.
// Positive minimum-power solves replace unnecessary eigenvalue computations.
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>
#include <set>
static constexpr int MAXN=24;
static thread_local unsigned long long work_counts[4]{};
static thread_local bool count_work=false;
extern "C" void ivp_work_reset(int enabled){std::fill(work_counts,work_counts+4,0);count_work=enabled!=0;}
extern "C" void ivp_work_get(unsigned long long* out){std::copy(work_counts,work_counts+4,out);}
struct Action {
 int n,s,ids[MAXN]; double value,cap[2],duration[2],B,noise;
 double lf[2][MAXN][MAXN],sf[2][MAXN][MAXN],lu[2][MAXN],su[2][MAXN];
 double ls[2][MAXN],ss[2][MAXN],bits[2][MAXN];
};
static double ex(double x){return std::exp(std::clamp(x,-700.,600.));}
static double softplus(double x){return x>30?x:std::log1p(ex(x));}
static void prepare(Action& a,int n,int mask,const double* mu,const double* cov,
 const double* direction,const double* payload,const double* par,const double* kr,const double* ko){
 if(count_work)++work_counts[2];
 a.n=n;a.s=0;a.value=0;a.B=par[1];a.noise=par[3];int D=2*n*n;
 a.cap[0]=par[4];a.cap[1]=par[5];a.duration[0]=par[0]*par[2];a.duration[1]=par[0]*(1-par[2]);
 for(int i=0;i<n;i++)if(mask&(1<<i)){a.ids[a.s++]=i;a.value+=(payload[i]+payload[n+i])/1000.;}
 double guard=par[6]*std::log(10.)/10.;
 for(int d=0;d<2;d++)for(int ii=0;ii<a.s;ii++){
  int i=a.ids[ii],di=d*n*n+i*(n+1);double vi=std::max(cov[di*D+di],0.);
  a.bits[d][ii]=payload[d*n+i];double theta=std::expm1(std::log(2.)*a.bits[d][ii]/(a.B*a.duration[d]));
  a.lu[d][ii]=std::log(theta*a.noise)-mu[di]+kr[a.s]*std::sqrt(vi)+guard;
  a.su[d][ii]=direction?-direction[di]:0.;
  a.ls[d][ii]=std::log(a.cap[d]/a.noise)+mu[di]-ko[a.s]*std::sqrt(vi)-guard;
  a.ss[d][ii]=direction?direction[di]:0.;
  for(int jj=0;jj<a.s;jj++){
   int j=a.ids[jj],ij=d*n*n+i*n+j;
   double v=std::max(0.,cov[ij*D+ij]+vi-2*cov[ij*D+di]);
   a.lf[d][ii][jj]=ii==jj?-INFINITY:std::log(theta)+mu[ij]-mu[di]+kr[a.s]*std::sqrt(v)+guard;
   a.sf[d][ii][jj]=direction?direction[ij]-direction[di]:0.;
  }
 }
}
static bool linear(double M[MAXN][MAXN+1],int s,double* p){
 if(count_work)++work_counts[1];
 for(int k=0;k<s;k++){
  int pivot=k;for(int i=k+1;i<s;i++)if(std::abs(M[i][k])>std::abs(M[pivot][k]))pivot=i;
  if(std::abs(M[pivot][k])<1e-14)return false;
  if(pivot!=k)for(int j=k;j<=s;j++)std::swap(M[pivot][j],M[k][j]);
  for(int i=k+1;i<s;i++){
   double r=M[i][k]/M[k][k];for(int j=k+1;j<=s;j++)M[i][j]-=r*M[k][j];M[i][k]=0;
  }
 }
 for(int i=s-1;i>=0;i--){double v=M[i][s];for(int j=i+1;j<s;j++)v-=M[i][j]*p[j];p[i]=v/M[i][i];}
 return true;
}
static bool feasible(const Action& a,double z,int mode,double* powers=nullptr,double* times=nullptr,double* margins=nullptr){
 if(count_work)++work_counts[0];
 if(a.duration[0]<=0||a.duration[1]<=0)return false;
 for(int d=0;d<2;d++){
  if(mode==1){
   if(count_work)++work_counts[3];
   double sum=0;for(int i=0;i<a.s;i++){
    double t=a.bits[d][i]*std::log(2.)/(a.B*softplus(a.ls[d][i]+a.ss[d][i]*z));
    sum+=t;if(times)times[d*a.n+a.ids[i]]=t;if(powers)powers[d*a.n+a.ids[i]]=a.cap[d];
   }
   if(margins)margins[d]=sum/a.duration[d];
   if(sum>a.duration[d]*(1+1e-12))return false;
  }else{
   double M[MAXN][MAXN+1]{},F[MAXN][MAXN]{},u[MAXN]{},p[MAXN]{};
   for(int i=0;i<a.s;i++){
    u[i]=ex(a.lu[d][i]+a.su[d][i]*z);M[i][a.s]=u[i]/a.cap[d];
    for(int j=0;j<a.s;j++){F[i][j]=i==j?0:ex(a.lf[d][i][j]+a.sf[d][i][j]*z);M[i][j]=(i==j?1.:0.)-F[i][j];}
   }
   if(!linear(M,a.s,p))return false;
   double mx=0;for(int i=0;i<a.s;i++){
    // p is normalized by its directional power cap. A positive solution to
    // (I-F)p=u establishes rho(F)<1 by Collatz-Wielandt, since u>0.
    if(!std::isfinite(p[i])||p[i]<=0||p[i]>1.+1e-12)return false;
    double rhs=u[i]/a.cap[d];for(int j=0;j<a.s;j++)rhs+=F[i][j]*p[j];
    if(std::abs(p[i]-rhs)>1e-9*std::max(p[i],1e-8))return false;
    mx=std::max(mx,p[i]);if(powers)powers[d*a.n+a.ids[i]]=p[i]*a.cap[d];
   }
   if(margins)margins[d]=mx;
  }
 }
 return true;
}
extern "C" void ivp_schedule(int n,const double* mu,const double* cov,const double* payload,
 const double* par,const double* kr,const double* ko,const int* masks,int count,int only_oma,double* out){
 std::fill(out,out+5+4*n,0.);out[1]=-1;
 if(n>MAXN||par[0]<=0)return;
 std::vector<Action> acts(count);for(int k=0;k<count;k++)prepare(acts[k],n,masks[k],mu,cov,nullptr,payload,par,kr,ko);
 std::vector<int> order(count);for(int k=0;k<count;k++)order[k]=k;
 std::stable_sort(order.begin(),order.end(),[&](int i,int j){return acts[i].value>acts[j].value;});
 for(int k:order)for(int mode=only_oma?1:0;mode<2;mode++){
  double p[2*MAXN]{},t[2*MAXN]{},margin[2]{};
  if(feasible(acts[k],0,mode,p,t,margin)){
   out[0]=masks[k];out[1]=mode;out[2]=acts[k].value;out[3]=margin[0];out[4]=margin[1];
   for(int i=0;i<2*n;i++){out[5+i]=p[i];out[5+2*n+i]=t[i];}return;
  }
 }
}
extern "C" void ivp_intervals(int n,const double* mu,const double* cov,const double* direction,const double* payload,
 const double* par,const double* kr,const double* ko,const int* masks,int count,const double* grid,int ng,int refinements,double* out){
 // The hull of tested feasible points is an inner interval by convexity.
 // Bisection always retains the feasible side; unobserved narrow intervals
 // are omitted, never assigned an optimistic value.
 std::fill(out,out+count*6,0.);if(n>MAXN||par[0]<=0)return;
 for(int k=0;k<count;k++){
  Action a;prepare(a,n,masks[k],mu,cov,direction,payload,par,kr,ko);
  for(int mode=0;mode<2;mode++){
   int first=-1,last=-1;
   for(int j=0;j<ng;j++)if(feasible(a,grid[j],mode)){if(first<0)first=j;last=j;}
   if(first<0)continue;
   double left=grid[first],right=grid[last];
   if(first>0){double bad=grid[first-1];for(int it=0;it<refinements;it++){double mid=(bad+left)/2;if(feasible(a,mid,mode))left=mid;else bad=mid;}}
   if(last<ng-1){double bad=grid[last+1];for(int it=0;it<refinements;it++){double mid=(bad+right)/2;if(feasible(a,mid,mode))right=mid;else bad=mid;}}
   int ix=k*6+mode*3;out[ix]=left;out[ix+1]=right;out[ix+2]=a.value;
  }
 }
}
extern "C" void ivp_scores(int n,const double* mu,const double* cov,const double* payload,
 const double* par,const double* kr,const double* ko,const int* masks,int count,double* out){
 for(int k=0;k<count;k++){
  Action a;prepare(a,n,masks[k],mu,cov,nullptr,payload,par,kr,ko);double stress=0.;
  for(int d=0;d<2;d++)for(int i=0;i<a.s;i++){
   double row=ex(a.lu[d][i])/a.cap[d];for(int j=0;j<a.s;j++)if(i!=j)row+=ex(a.lf[d][i][j]);
   stress=std::max(stress,row);
  }
  out[k]=a.value/std::max(1.,stress);
 }
}
extern "C" double ivp_integrate(const double* intervals,int count){
 struct Event {double x,value;int kind;};std::vector<Event> events;
 for(int j=0;j<count;j++){
  double l=intervals[3*j],r=intervals[3*j+1],w=intervals[3*j+2];
  if(w>0&&r>l){events.push_back({l,w,1});events.push_back({r,w,-1});}
 }
 if(events.empty())return 0.;
 std::sort(events.begin(),events.end(),[](const Event& a,const Event& b){return a.x<b.x;});
 std::multiset<double> active{0.};double result=0,prev=events[0].x;
 for(const auto& e:events){
  result+=*active.rbegin()*(.5*std::erfc(-e.x/std::sqrt(2.))-.5*std::erfc(-prev/std::sqrt(2.)));
  if(e.kind>0)active.insert(e.value);else active.erase(active.find(e.value));prev=e.x;
 }
 return result;
}

// Finite frontier candidate construction and explicit sampled validation.
// No innovation-interval claim is made for this empirical acceptance predicate.
static bool sampled_pass(const Action& a,const double* gains,const double* scale,
 int samples,int max_bad,double z,int mode,const double* p,const double* t,int& bad){
 bad=0;int D=2*a.n*a.n;
 double theta[2][MAXN]{};
 for(int d=0;d<2;d++)for(int i=0;i<a.s;i++)theta[d][i]=std::expm1(std::log(2.)*a.bits[d][i]/(a.B*a.duration[d]));
 for(int k=0;k<samples;k++){
  bool failed=false;const double* g=gains+k*D;
  for(int d=0;d<2&&!failed;d++)for(int ii=0;ii<a.s&&!failed;ii++){
   int i=a.ids[ii],ix=d*a.n*a.n+i*a.n+i;double desired=g[ix]*scale[ix]*p[d*a.n+i];
   if(mode==1){double bits=a.B*t[d*a.n+i]*std::log1p(desired/a.noise)/std::log(2.);failed=bits<a.bits[d][ii]*(1-1e-7);}
   else{
    double interference=a.noise;
    for(int jj=0;jj<a.s;jj++)if(jj!=ii){int j=a.ids[jj],ij=d*a.n*a.n+i*a.n+j;interference+=g[ij]*scale[ij]*p[d*a.n+j];}
    failed=desired/interference<theta[d][ii]*(1-1e-7);
   }
  }
  if(failed&&++bad>max_bad)return false;
 }
 return true;
}
extern "C" void ivp_frontier_values(int n,const double* mu,const double* cov,const double* direction,const double* payload,
 const double* par,const double* kr,const double* ko,int knobs,const int* masks,int count,
 const double* gains,int samples,int max_bad,const double* grid,int ng,double* values,double* outputs){
 int stride=5+4*n;std::fill(values,values+ng,0.);std::fill(outputs,outputs+ng*stride,0.);
 for(int j=0;j<ng;j++)outputs[j*stride+1]=-1;
 if(n>MAXN||par[0]<=0||max_bad<0)return;
 std::vector<Action> acts(count*knobs);
 for(int k=0;k<count;k++)for(int h=0;h<knobs;h++)prepare(acts[k*knobs+h],n,masks[k],mu,cov,direction,payload,par,kr+h*(n+1),ko+h*(n+1));
 std::vector<int> order(count);for(int k=0;k<count;k++)order[k]=k;
 std::stable_sort(order.begin(),order.end(),[&](int i,int j){return acts[i*knobs].value>acts[j*knobs].value;});
 for(int j=0;j<ng;j++){
  double z=grid[j];std::vector<double> scale(2*n*n);for(int i=0;i<2*n*n;i++)scale[i]=ex(direction?direction[i]*z:0.);
  bool found=false;
  for(int k:order){
   if(found)break;
   for(int h=0;h<knobs&&!found;h++)for(int mode=0;mode<2&&!found;mode++){
    const Action& a=acts[k*knobs+h];double p[2*MAXN]{},t[2*MAXN]{};
    if(!feasible(a,z,mode,p,t))continue;
    for(int d=0;d<2;d++){
     if(mode==0){double factor=INFINITY;for(int i=0;i<a.s;i++)factor=std::min(factor,a.cap[d]/p[d*n+a.ids[i]]);for(int i=0;i<a.s;i++)p[d*n+a.ids[i]]*=factor;}
     else{double sum=0.;for(int i=0;i<a.s;i++)sum+=t[d*n+a.ids[i]];for(int i=0;i<a.s;i++)t[d*n+a.ids[i]]*=a.duration[d]/sum;}
    }
    int bad=0;if(!sampled_pass(a,gains,scale.data(),samples,max_bad,z,mode,p,t,bad))continue;
    double* out=outputs+j*stride;out[0]=masks[k];out[1]=mode;out[2]=values[j]=a.value;out[3]=bad;out[4]=h;
    std::copy(p,p+2*n,out+5);std::copy(t,t+2*n,out+5+2*n);found=true;
   }
  }
 }
}
