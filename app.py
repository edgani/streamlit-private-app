"""
QuantFinal V10.9 — Complete build
15 spillover chains, intermarket confirm, scenarios,
position sizing, hedge recs, historical stats, data freshness.
Engine: V10.4 audited. All UI fixes from V10.5-V10.8.
"""

import math,json,hashlib
from datetime import datetime,timedelta,timezone
from dataclasses import dataclass,field,asdict
from typing import Dict,List,Tuple,Optional
import numpy as np,pandas as pd,requests,streamlit as st
try:
    import yfinance as yf
except: yf=None

st.set_page_config(page_title="QuantV10.9",layout="wide")

# ---- CONFIG ----
@dataclass
class Cfg:
    gw:List[float]=field(default_factory=lambda:[.22,.22,.22,.18,.16])
    iow:List[float]=field(default_factory=lambda:[.32,.30,.24])
    inw:List[float]=field(default_factory=lambda:[.26,.28,.24,.14,.08])
    sg_go:float=1.50;sg_io:float=1.50;sg_gn:float=1.65;sg_in:float=1.60;sg_gl:float=1.55;sg_il:float=1.60
    gi_go:float=0.0;gi_gn:float=0.0;gi_gl:float=0.0;gi_io:float=0.0;gi_in:float=0.0;gi_il:float=0.0
    bo:float=.30;bd:float=.35;bll:float=.35;mob:float=.10
    gcw:List[float]=field(default_factory=lambda:[.22,.22,.18,.16,.22])
    icw:List[float]=field(default_factory=lambda:[.26,.24,.22,.14,.14])
    glm:float=.52;glc:float=.48;ilm:float=.52;ilc:float=.48
    r2lb:int=4;cdm:int=3;cdp:float=.25;pb:float=.20
    sv:float=.34;q3t:float=.30;eo:bool=True;mt:float=.06
    lb:int=36;clb:int=52;dw:float=.60;lw2:float=.40;roc2_thresh:float=.02
    def h(self): return hashlib.md5(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()[:12]
CF=Cfg()

# ---- STYLE ----
st.markdown("""<style>
:root{--bg:#07101f;--card:#0c1526;--ln:#263246;--mt:#9fb0c8;--tx:#f3f6fb;--rd:#ff4d4d;--gn:#2ecc71;--am:#f5b041}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--tx)}
.block-container{padding-top:1rem;max-width:1700px}*{color:var(--tx)}
.card{background:linear-gradient(180deg,rgba(16,24,41,.98),rgba(10,17,30,.98));border:1px solid var(--ln);border-radius:16px;padding:12px 14px}
.hero{background:linear-gradient(180deg,rgba(19,29,50,1),rgba(12,20,35,1));border:1px solid var(--ln);border-radius:14px;padding:10px 12px;min-height:88px}
.st{font-weight:800;margin-bottom:8px;font-size:.95rem}
.mt2{font-size:.72rem;color:var(--mt);text-transform:uppercase;letter-spacing:.05em}
.mv{font-size:1.6rem;font-weight:800;line-height:1.1}
.ms{font-size:.82rem;color:#c4d2e6;margin-top:3px}
.p{display:inline-block;border:1px solid #33435d;border-radius:999px;padding:2px 9px;font-size:.78rem;color:#dbe7ff;background:rgba(30,42,63,.82);margin-right:5px;margin-top:3px}
.pr{display:inline-block;border:1px solid var(--rd);border-radius:999px;padding:2px 9px;font-size:.78rem;color:#ffd7d7;background:rgba(120,20,20,.22);margin-right:5px;margin-top:3px}
.pg{display:inline-block;border:1px solid var(--gn);border-radius:999px;padding:2px 9px;font-size:.78rem;color:#d4fce4;background:rgba(20,80,30,.22);margin-right:5px;margin-top:3px}
.pa{display:inline-block;border:1px solid var(--am);border-radius:999px;padding:2px 9px;font-size:.78rem;color:#fef3c7;background:rgba(100,60,10,.22);margin-right:5px;margin-top:3px}
.sm{color:var(--mt);font-size:.88rem}
.tt table{width:100%;border-collapse:collapse;table-layout:fixed}
.tt th,.tt td{border:1px solid #243147;padding:7px 8px;font-size:.82rem;text-align:left;vertical-align:top;word-break:break-word}
.tt th{color:#9fb0c8;font-weight:700;background:rgba(20,29,44,.85)}.tt td{color:#f5f8fd}
.nb{border:1px solid #27476e;background:rgba(18,46,79,.65);border-radius:10px;padding:10px 12px}
.ar{border:1px solid var(--rd);background:rgba(120,20,20,.15);border-radius:10px;padding:10px 12px}
.aa{border:1px solid var(--am);background:rgba(100,60,10,.15);border-radius:10px;padding:10px 12px}
.mc{color:var(--mt);font-size:.78rem;margin-top:2px;margin-bottom:6px}
.evt-box{border:1px solid #2a4a6e;background:rgba(15,35,60,.70);border-radius:10px;padding:10px 12px;margin-top:6px}
</style>""",unsafe_allow_html=True)

# ---- UTILS ----
def c01(x): return float(max(0,min(1,x)))
def cl(x,a,b): return float(max(a,min(b,x)))
def sg(x): return 1/(1+math.exp(-x))
def pc(x): return f"{100*x:.1f}%" if np.isfinite(x) else "n/a"
def pl(t,c=""):
    m={"r":"pr","g":"pg","a":"pa"};return f"<span class='{m.get(c,'p')}'>{t}</span>"
def tb(h,r):
    th="".join(f"<th>{x}</th>" for x in h)
    tr="".join("<tr>"+"".join(f"<td>{c}</td>" for c in row)+"</tr>" for row in r)
    return f"<div class='tt'><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>"
def rz(s,lb=36):
    s=s.dropna()
    if len(s)<max(12,lb//3): return 0.0
    h=s.iloc[-lb:];med=float(h.median());mad=float(np.median(np.abs(h-med)));x=float(h.iloc[-1]);sc=1.4826*mad
    if not np.isfinite(sc) or sc<1e-9:
        sd=float(h.std(ddof=0));return float((x-float(h.mean()))/sd) if np.isfinite(sd) and sd>1e-9 else 0.0
    return float((x-med)/sc)
def wm(v,w):
    a,ww=np.array(v,float),np.array(w,float);m=np.isfinite(a)&np.isfinite(ww)
    if not m.any(): return 0.0
    return float(np.sum(a[m]*ww[m])/np.sum(ww[m])) if ww[m].sum()>0 else float(np.nanmean(a[m]))
def np2(p):
    t=sum(p.values());return{k:v/t for k,v in p.items()} if t>0 else{k:.25 for k in p}
def qp(gu,iu): return np2({"Q1":gu*(1-iu),"Q2":gu*iu,"Q3":(1-gu)*iu,"Q4":(1-gu)*(1-iu)})
def rn(s,n=21):
    s=s.dropna();return float(s.iloc[-n:].pct_change().add(1).prod()-1) if len(s)>=n+1 else 0.0
def sn(s,n=63):
    s=s.dropna()
    if len(s)<n: return 0.0
    m=float(s.iloc[-n:].mean());return float(s.iloc[-1]/m-1) if np.isfinite(m) and abs(m)>1e-12 else 0.0
def tsf(s):
    s=s.dropna()
    if len(s)<30: return 0.0
    px=float(s.iloc[-1]);s20=float(s.rolling(20).mean().iloc[-1]) if len(s)>=20 else px
    s50=float(s.rolling(50).mean().iloc[-1]) if len(s)>=50 else s20
    ln=min(200,len(s));sl=float(s.rolling(ln).mean().iloc[-1]) if ln>=20 else s50
    return float(np.mean([px>s20,px>s50,px>sl,s20>s50,s50>sl]))
def nt(x,sc): return c01(.5+.5*math.tanh(x/max(sc,1e-9))) if np.isfinite(x) else .5
def mll(s):
    s=s.dropna();return s.resample("M").last().dropna() if not s.empty else s
def an(s,n=3):
    s=s.dropna()
    if len(s)<n+1: return pd.Series(dtype=float)
    return((s/s.shift(n))**(12.0/n)-1).replace([np.inf,-np.inf],np.nan)
def lv(s):
    s=s.dropna();return float(s.iloc[-1]) if not s.empty else float("nan")
def ld(s,n=21):
    s=s.dropna();return float(s.iloc[-1]-s.iloc[-n-1]) if len(s)>=n+1 else float("nan")

# ---- F&G ----
@st.cache_data(ttl=1800,show_spinner=False)
def fetch_fg():
    hdrs={"User-Agent":"Mozilla/5.0"}
    for url in["https://production.dataviz.cnn.io/index/fearandgreed/graphdata","https://production.dataviz.cnn.io/index/fearandgreed/current"]:
        try:
            r=requests.get(url,timeout=10,headers=hdrs)
            if r.ok:
                j=r.json()
                if "fear_and_greed" in j: return int(j["fear_and_greed"]["score"]),"CNN"
                if "score" in j: return int(j["score"]),"CNN"
        except: pass
    try:
        r=requests.get("https://api.alternative.me/fng/?limit=1",timeout=8);v=r.json().get("data",[{}])[0];sc=int(v.get("value",0))
        if sc>0: return sc,"Alt.me"
    except: pass
    try:
        import re;r=requests.get("https://edition.cnn.com/markets/fear-and-greed",timeout=12,headers=hdrs)
        if r.ok:
            m2=re.search(r'"score"\s*:\s*(\d+\.?\d*)',r.text)
            if m2: return int(float(m2.group(1))),"CNN scrape"
    except: pass
    return 0,"Failed"
def fgll(sc):
    if sc<=0: return "No data",""
    if sc<25: return "Extreme Fear","r"
    if sc<45: return "Fear","a"
    if sc<56: return "Neutral",""
    if sc<76: return "Greed","g"
    return "Extreme Greed","g"

# ---- DATA ----
@st.cache_data(ttl=3600*6,show_spinner=False)
def fred(sid):
    try:
        df=pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}");d2,v=df.columns[0],df.columns[1]
        df[d2]=pd.to_datetime(df[d2],errors="coerce");df[v]=pd.to_numeric(df[v],errors="coerce")
        s=df.dropna().set_index(d2)[v];s.name=sid;return s
    except: return pd.Series(dtype=float,name=sid)
@st.cache_data(ttl=3600*4,show_spinner=False)
def ycc(t,p="1y"):
    if yf is None: return pd.Series(dtype=float,name=t)
    try:
        data=yf.download(t,period=p,interval="1d",auto_adjust=True,progress=False,threads=False)
        if data is None or len(data)==0: return pd.Series(dtype=float,name=t)
        if isinstance(data.columns,pd.MultiIndex):
            if("Close",t) in data.columns: s=data[("Close",t)]
            elif "Close" in data.columns.get_level_values(0): s=data["Close"].iloc[:,0]
            else: s=data.iloc[:,0]
        else: s=data["Close"] if "Close" in data.columns else data.iloc[:,0]
        return pd.to_numeric(s,errors="coerce").dropna()
    except: return pd.Series(dtype=float,name=t)
@st.cache_data(ttl=3600*4,show_spinner=False)
def yb(tickers,p="1y"):
    tickers=[t for t in tickers if t]
    if yf is None or not tickers: return pd.DataFrame()
    for cs in(min(40,len(tickers)),15,8,1):
        frames,ok=[],True
        for i in range(0,len(tickers),cs):
            ch=tickers[i:i+cs]
            try: data=yf.download(ch,period=p,interval="1d",auto_adjust=True,progress=False,threads=False)
            except: ok=False;break
            if data is None or len(data)==0: continue
            try:
                if isinstance(data.columns,pd.MultiIndex): cl2=data["Close"].copy() if "Close" in data.columns.get_level_values(0) else data.iloc[:,:len(ch)].copy()
                else: c2="Close" if "Close" in data.columns else data.columns[0];cl2=data[[c2]].copy();cl2.columns=[ch[0]]
                frames.append(cl2.apply(pd.to_numeric,errors="coerce"))
            except: ok=False;break
        if ok and frames:
            m=pd.concat(frames,axis=1);return m.loc[:,~m.columns.duplicated()].sort_index()
    return pd.DataFrame()
S={k:fred(v) for k,v in{"INDPRO":"INDPRO","RSAFS":"RSAFS","PAYEMS":"PAYEMS","UNRATE":"UNRATE","ICSA":"ICSA","CPI":"CPIAUCSL","CCPI":"CPILFESL","PPI":"PPIACO","SAHM":"SAHMCURRENT","NFCI":"NFCI","HY":"BAMLH0A0HYM2","WTI":"DCOILWTICO","DGS2":"DGS2","DGS10":"DGS10","DFII10":"DFII10","T10Y2Y":"T10Y2Y"}.items()}

# ---- ENGINE V10.4 ----
@dataclass
class QR:
    op:Dict[str,float];dp:Dict[str,float];lp:Dict[str,float];mb:Dict[str,float];mq:str;mp:float;mnq:str;mnp:float
    go:float;gn:float;gl:float;io:float;ino:float;il:float;gcx:float;icx:float;sm:float;lw:float;stag:float;q3p:float;q2v:bool
    gnb:float;gpb:float;ipb:float;cgn:float;cip:float;lc:float;od:str
    g_roc2:float;i_roc2:float;g_cd:int;i_cr:int;gdir:str;idir:str;ppb:float;mmc:float
    gz:Dict[str,float]=field(default_factory=dict);iz:Dict[str,float]=field(default_factory=dict);sz:Dict[str,float]=field(default_factory=dict)

class Eng:
    def __init__(s,c):s.c=c
    def _cut(s,sr,dt):sr=sr.dropna();return sr.loc[:dt].dropna() if dt and not sr.empty else sr
    def _cd(s,keys):
        dd=[]
        for k in keys:
            sr=S.get(k,pd.Series(dtype=float)).dropna()
            if sr.empty: return None
            dd.append(sr.index.max())
        return min(dd) if dd else None
    def _pair(s,a,b="SPY",p="1y"):
        sa,sb=ycc(a,p),ycc(b,p)
        if sa.empty or sb.empty: return np.nan
        df=pd.concat([sa.rename("a"),sb.rename("b")],axis=1).sort_index().ffill().dropna()
        if len(df)<50: return np.nan
        rel=(df["a"]/df["b"]).replace([np.inf,-np.inf],np.nan).dropna()
        if len(rel)<50: return np.nan
        s2=.50*rn(rel,21)+.35*rn(rel,63)+.15*sn(rel,63);v=rel.pct_change().dropna().tail(63).std(ddof=0);sc=.05+2.5*(v if np.isfinite(v) else .02)
        return float(math.tanh(s2/max(sc,.04)))
    def _ab(s,t,p="1y"):
        sr=ycc(t,p)
        if sr.empty or len(sr)<50: return np.nan
        s2=.50*rn(sr,21)+.35*rn(sr,63)+.15*sn(sr,63);v=sr.pct_change().dropna().tail(63).std(ddof=0);sc=.05+2.5*(v if np.isfinite(v) else .02)
        return float(math.tanh(s2/max(sc,.04)))
    def _roc2(s,zs,lb=4):
        sr=zs.dropna()
        if len(sr)<lb+2: return 0.0,0,"flat"
        rc=sr.iloc[-(lb+1):];diffs=rc.diff().dropna()
        if len(diffs)<2: return 0.0,0,"flat"
        vals=diffs.iloc[-min(lb,len(diffs)):].values
        if len(vals)>=3: trimmed=np.sort(vals)[1:-1];avg=float(np.mean(trimmed)) if len(trimmed)>0 else float(np.mean(vals))
        else: avg=float(np.mean(vals))
        last=float(diffs.iloc[-1]);consec=0
        for v in reversed(diffs.values):
            if v<-s.c.roc2_thresh: consec+=1
            else: break
        r2=.55*avg+.45*last
        if r2<-.025 or consec>=2: return float(r2),int(consec),"decelerating"
        elif r2>.025 or(consec==0 and avg>.015): return float(r2),int(consec),"accelerating"
        return float(r2),int(consec),"flat"
    def _zs(s,raw,lb=36,n=12):
        sr=raw.dropna()
        if len(sr)<lb+n: return pd.Series(dtype=float)
        res,idx=[],[]
        for i in range(n):
            end=len(sr)-i
            if end<lb: break
            ch=sr.iloc[end-lb:end];med=float(ch.median());mad=float(np.median(np.abs(ch-med)));x=float(ch.iloc[-1]);sc=1.4826*mad
            if np.isfinite(sc) and sc>1e-9: z=(x-med)/sc
            else: sd=float(ch.std(ddof=0));z=(x-float(ch.mean()))/sd if np.isfinite(sd) and sd>1e-9 else 0.0
            res.append(z);idx.append(sr.index[end-1])
        return pd.Series(list(reversed(res)),index=list(reversed(idx))) if res else pd.Series(dtype=float)
    def _pipe(s,iz):
        p=iz.get("ppi",0);o3=iz.get("oil3m",0);o1=iz.get("oil1m",0);c2=iz.get("cpi",0);co=iz.get("core",0)
        lead=max(0,p-c2)+max(0,o3-c2)*.5+max(0,o1-co)*.3;return c01(lead*.45) if lead>.12 and(p>0 or o3>0) else 0.0
    def compute(s):
        c=s.c;dt=s._cd(["INDPRO","RSAFS","PAYEMS","UNRATE","ICSA","CPI","CCPI","PPI"])
        ind=s._cut(S["INDPRO"],dt);rs=s._cut(S["RSAFS"],dt);pay=s._cut(S["PAYEMS"],dt);un=s._cut(S["UNRATE"],dt);ic=s._cut(S["ICSA"],dt)
        cpi=s._cut(S["CPI"],dt);ccpi=s._cut(S["CCPI"],dt);ppi=s._cut(S["PPI"],dt)
        g_ia=an(ind,3)-ind.pct_change(12);g_sa=an(rs,3)-rs.pct_change(12);g_ja=(pay.diff(3)/3)-(pay.diff(12)/12)
        gzs=[x for x in[s._zs(g_ia,c.lb,12),s._zs(g_sa,c.lb,12),s._zs(g_ja,c.lb,12)] if not x.empty]
        gc=pd.concat(gzs,axis=1).mean(axis=1).dropna() if gzs else pd.Series(dtype=float)
        gr2,gcd,gdir=s._roc2(gc,c.r2lb);wti_m=mll(S["WTI"])
        i_ca=an(cpi,3)-cpi.pct_change(12);i_co=an(ccpi,3)-ccpi.pct_change(12);i_pa=an(ppi,3)-ppi.pct_change(12)
        izs=[x for x in[s._zs(i_ca,c.lb,12),s._zs(i_co,c.lb,12),s._zs(i_pa,c.lb,12)] if not x.empty]
        ic2=pd.concat(izs,axis=1).mean(axis=1).dropna() if izs else pd.Series(dtype=float)
        ir2,_,idir=s._roc2(ic2,c.r2lb);icr=0
        if not ic2.empty:
            dfs=ic2.diff().dropna()
            for v in reversed(dfs.values):
                if v>c.roc2_thresh: icr+=1
                else: break
        gz={"indpro":rz(g_ia,c.lb),"sales":rz(g_sa,c.lb),"jobs":rz(g_ja,c.lb),"unrate":-rz(un.diff(3),c.lb),"claims":-rz(ic.rolling(4).mean()-ic.rolling(26).mean(),c.clb)}
        gi=[gz[k] for k in gz];g_raw=wm(gi,c.gw)
        iz2={"cpi":rz(i_ca,c.lb),"core":rz(i_co,c.lb),"ppi":rz(i_pa,c.lb),"oil3m":rz(an(wti_m,3),c.lb),"oil1m":rz(an(wti_m,1),c.lb)}
        i_off_r=wm([iz2["cpi"],iz2["core"],iz2["ppi"]],c.iow);i_now_r=wm([iz2[k] for k in iz2],c.inw);ppb=s._pipe(iz2)
        sz={"sahm":rz(S["SAHM"].dropna(),c.lb),"nfci":rz(S["NFCI"].dropna(),c.clb),"hy":rz(S["HY"].dropna(),c.clb)};sm=float(np.nanmean(list(sz.values())))
        gnb=float(np.mean([x<0 for x in gi]));gpb=float(np.mean([x>0 for x in gi]));ipb=float(np.mean([x>0 for x in iz2.values()]))
        lw=wm([max(0,-gz["jobs"]),max(0,-gz["unrate"]),max(0,-gz["claims"])],[.45,.30,.25]);gd=.22*max(0,gnb-.35)+.24*max(0,lw-.06)
        gds=gr2;ids=ir2
        if gcd>=c.cdm: gds-=c.cdp*(gcd-c.cdm+1)
        if icr>=2: ids+=.08*icr
        g_off=c.lw2*(g_raw-.10*gd)+c.dw*gds;g_now=c.lw2*(g_raw-gd)+c.dw*gds
        i_comp=c.lw2*i_off_r+c.dw*ids+ppb*c.pb;i_off=i_comp+.06*max(0,float(np.mean([iz2[k]>0 for k in["cpi","core","ppi"]]))-.50)
        i_now=c.lw2*(i_now_r+.12*max(0,ipb-.50)+.04*max(0,iz2["oil1m"]))+c.dw*ids+ppb*c.pb
        op=qp(sg(c.sg_go*(g_off-c.gi_go)-.06*max(0,sm)),sg(c.sg_io*(i_off-c.gi_io)+.05*max(0,sm)))
        dp=qp(sg(c.sg_gn*(g_now-c.gi_gn)-.10*max(0,sm)-.08*gnb),sg(c.sg_in*(i_now-c.gi_in)+.06*max(0,sm)+ppb*.15))
        gcx=[s._pair("IWM","SPY"),s._pair("XLY","XLP"),s._pair("XLI","XLU"),s._pair("XLF","XLU"),s._pair("HYG","IEF")]
        icx=[s._pair("XLE","SPY"),s._pair("GLD","SPY"),s._pair("DBC","SPY"),s._ab("UUP"),-s._ab("TLT")]
        gcxs=wm(gcx,c.gcw);icxs=wm(icx,c.icw)
        vg=[x for x in gcx if np.isfinite(x)];vi=[x for x in icx if np.isfinite(x)]
        cgn=float(np.mean([x<-.02 for x in vg])) if vg else 0;cip=float(np.mean([x>.02 for x in vi])) if vi else 0
        mmc=1.0 if(gdir=="decelerating" and gcxs>.08) else 0.0
        em=c.glm+(c.mob if mmc>.5 else 0);ec=c.glc-(c.mob if mmc>.5 else 0)
        g_live=em*g_now+ec*gcxs-.06*gd;i_live=c.ilm*i_now+c.ilc*icxs+.03*max(0,iz2["oil1m"])
        lp=qp(sg(c.sg_gl*(g_live-c.gi_gl)-.06*max(0,sm)-.08*cgn),sg(c.sg_il*(i_live-c.gi_il)+.04*max(0,sm)+.08*cip))
        inf_push=max(0,i_live);stag=c01(.42*sg(2*(lw-.08))+.34*sg(2*(inf_push-.05))+.24*sg(3.6*(gnb-.52)))
        q2v=(stag>c.sv) or(gdir=="decelerating" and idir=="accelerating" and gnb>=.40)
        if q2v and dp["Q2"]>=dp["Q3"]:
            sh=min(dp["Q2"]*.55,dp["Q2"]*(.22+.28*stag));dp["Q2"]-=sh;dp["Q3"]+=sh*.88;dp["Q4"]+=sh*.12*float(inf_push<.08);dp=np2(dp)
        q3lp=c01(.26*sg(2.2*(stag-.28))+.24*sg(2.4*(-gcxs-.02))+.22*sg(2.1*(icxs-.00))+.14*sg(3.8*(cgn-.38))+.14*float(gdir=="decelerating"))
        if q3lp>c.q3t and lp["Q2"]>=lp["Q3"]:
            sh=min(lp["Q2"]*.50,lp["Q2"]*(.16+.30*q3lp));lp["Q2"]-=sh;lp["Q3"]+=sh*.88;lp["Q4"]+=sh*.12*float(icxs<0);lp=np2(lp)
        eo2=c.bo+(.06 if mmc>.5 else 0);ed=c.bd+(.04 if mmc>.5 else 0);el=c.bll-(.10 if mmc>.5 else 0);tw=eo2+ed+el;eo2/=tw;ed/=tw;el/=tw
        mb=np2({k:eo2*op[k]+ed*dp[k]+el*lp[k] for k in["Q1","Q2","Q3","Q4"]})
        if c.eo:
            ok=(op["Q3"]>.15 or dp["Q3"]>.20 or lp["Q3"]>.20 or q3lp>.32 or stag>.35 or gdir=="decelerating")
            if ok:
                tilt=min(.03+.04*q3lp+.03*stag+.02*float(gdir=="decelerating"),c.mt)
                take={"Q2":min(mb["Q2"]*.20,tilt*.55),"Q4":min(mb["Q4"]*.12,tilt*.30),"Q1":min(mb["Q1"]*.08,tilt*.15)}
                for k,v in take.items(): mb[k]-=v
                mb["Q3"]+=sum(take.values());mb=np2(mb)
        rk=sorted(mb.items(),key=lambda x:x[1],reverse=True);lvc=sum(np.isfinite(x) for x in gcx+icx)
        return QR(op=op,dp=dp,lp=lp,mb=mb,mq=rk[0][0],mp=rk[0][1],mnq=rk[1][0],mnp=rk[1][1],go=g_off,gn=g_now,gl=g_live,io=i_off,ino=i_now,il=i_live,gcx=gcxs,icx=icxs,sm=sm,lw=lw,stag=stag,q3p=q3lp,q2v=q2v,gnb=gnb,gpb=gpb,ipb=ipb,cgn=cgn,cip=cip,lc=lvc/len(gcx+icx) if len(gcx+icx) else 0,od=dt.strftime("%Y-%m-%d") if dt else "n/a",g_roc2=gr2,i_roc2=ir2,g_cd=gcd,i_cr=icr,gdir=gdir,idir=idir,ppb=ppb,mmc=mmc,gz=gz,iz=iz2,sz=sz)

# ---- DIAGNOSTICS ----
@dataclass
class D:
    cq:str;cp:float;nq:str;np_:float;ag:float;cn:float;fg:float;br:float;ps:float;mg:float
    tsc2:float;bsc:float;tp:float;tc:float;sp:float;pa:str;sub:str;sq:str;ssg:float;si_:float;sl:float
def dgf(qr,fb):
    r=sorted(fb.items(),key=lambda x:x[1],reverse=True);cq,cp=r[0];nq,np3=r[1]
    o,d2,l=qr.op,qr.dp,qr.lp;ds=[sum(abs(o[k]-d2[k]) for k in o),sum(abs(o[k]-l[k]) for k in o),sum(abs(d2[k]-l[k]) for k in o)]
    ag=c01(1-.18*ds[0]-.18*ds[1]-.24*ds[2]);cn=c01(.50*cp+.24*ag+.26*max(l.values()));rd=abs(qr.gl-qr.go)+abs(qr.il-qr.io)
    fg=c01(.34*(1-cp)+.24*(1-ag)+.16*max(0,qr.sm)+.14*min(1,rd/2.5)+.12*abs(cp-qr.mp))
    br=.34*qr.gpb+.26*float(np.mean([qr.iz.get(k,0)>0 for k in["cpi","core","ppi"]]))+.20*max(0,1-qr.cgn)+.20*qr.cip
    ps=c01(.34*abs(qr.gl)+.34*abs(qr.il)+.16*max(0,cp-.25)+.16*qr.q3p);mg=cp-np3
    if mg<.03: fg=c01(fg+.08)
    tsc2=c01(.28*max(0,qr.il)+.18*ps+.18*fg+.18*float(cq in["Q2","Q3"])+.18*qr.q3p)
    bsc=c01(.34*float(cq=="Q4")+.18*max(0,-qr.gl)+.18*(1-fg)+.15*float(qr.mq=="Q4")+.15*float(max(l,key=l.get)=="Q4"))
    tp=c01(.30*fg+.24*(1-cp)+.18*min(1,rd/2.5)+.14*abs(qr.gl-qr.il)/3+.14*qr.q3p);tc=c01(.52*tp+.48*np3);sp=c01(cp*(1-fg*.45))
    if mg>.05: pa="Valid" if tc>.70 else("Building" if tc>.48 else("Starting" if tc>.28 else "Stable"))
    else: pa="Confirmed" if tc>.78 and np3>cp-.01 else("Valid" if tc>.60 else("Building" if tc>.42 else("Starting" if tc>.26 else "Stable")))
    su={"Q1":"Goldilocks","Q2":"Reflation" if qr.q3p<.40 else "Hot","Q3":"Stagflation" if qr.q3p>.55 else "Stagfl building","Q4":"Bottoming" if qr.gl>-.20 else "Growth scare"}.get(cq,"Base")
    return D(cq=cq,cp=cp,nq=nq,np_=np3,ag=ag,cn=cn,fg=fg,br=br,ps=ps,mg=mg,tsc2=tsc2,bsc=bsc,tp=tp,tc=tc,sp=sp,pa=pa,sub=su,
        sq="High" if .5*cn+.3*ag+.2*(1-fg)>.72 else("Medium" if .5*cn+.3*ag+.2*(1-fg)>.52 else "Low"),
        ssg=c01(sg(1.25*(-qr.gl+.05))),si_=c01(sg(1.25*(qr.il+.02))),sl=c01(sg(1.25*qr.sm)))

# ---- 15 SPILLOVER CHAINS ----
CHAINS={
    "Oil shock":{"trigger":"CL=F","trigger_name":"WTI Oil","beneficiaries":[("XOM","Energy majors",.95),("SLB","Oil services",.90),("COP","E&P",.88),("MPC","Refining",.85),("STNG","Tankers",.82),("BTU","Coal",.75),("UNG","NatGas",.65),("UUP","USD",.60),("GLD","Gold",.55),("DBA","Agriculture",.45)],"victims":[("JETS","Airlines",-.90),("XLY","Cons disc",-.70),("EEM","EM equities",-.60),("FXA","AUD",-.55),("TLT","Duration",-.50)]},
    "USD strength":{"trigger":"UUP","trigger_name":"USD Index","beneficiaries":[("IWM","US small caps",.65),("XLF","Financials",.55)],"victims":[("EEM","EM equities",-.85),("FXA","AUD",-.80),("GLD","Gold",-.70),("DBC","Commodities",-.65),("EFA","Intl dev",-.55),("FXE","EUR",-.50)]},
    "Rates up":{"trigger":"^TNX","trigger_name":"10Y Yield","beneficiaries":[("XLF","Banks",.80),("KRE","Regionals",.75),("VLUE","Value",.40)],"victims":[("TLT","Duration",-.95),("ARKK","Spec growth",-.85),("XLRE","REITs",-.80),("XLU","Utilities",-.75),("XLK","Tech",-.55)]},
    "Growth scare":{"trigger":"IWM","trigger_name":"Small caps (inv)","beneficiaries":[("TLT","Duration",.90),("GLD","Gold",.80),("FXY","JPY",.75),("XLP","Staples",.65),("XLU","Utilities",.60),("XLV","Healthcare",.55)],"victims":[("IWM","Small caps",-.90),("ARKK","Speculative",-.85),("HYG","High yield",-.80),("XLI","Industrials",-.70),("XLY","Cons disc",-.65),("EEM","EM",-.60)]},
    "China stimulus":{"trigger":"FXI","trigger_name":"China large cap","beneficiaries":[("COPX","Copper miners",.85),("BHP","Iron ore",.80),("FXA","AUD",.75),("EEM","EM",.70),("PICK","Metals",.65),("LVMUY","Luxury",.55)],"victims":[("UUP","USD",-.40),("TLT","Duration",-.35)]},
    "Inflation surprise":{"trigger":"TIP","trigger_name":"TIPS","beneficiaries":[("GLD","Gold",.85),("DBC","Commodities",.80),("XLE","Energy",.75),("TIP","TIPS",.70),("PDBC","Broad cmdty",.65),("SLV","Silver",.60)],"victims":[("TLT","Nominal dur",-.85),("XLRE","REITs",-.60),("XLK","Tech",-.50),("IWM","Small caps",-.45)]},
    "Volatility spike":{"trigger":"^VIX","trigger_name":"VIX","beneficiaries":[("VIXY","Long vol",.95),("TLT","Duration",.80),("UUP","USD",.70),("FXY","JPY",.70),("FXF","CHF",.65),("GLD","Gold",.60)],"victims":[("SPY","US equities",-.90),("IWM","Small caps",-.90),("HYG","High yield",-.80),("EEM","EM",-.75),("BTC-USD","Bitcoin",-.70)]},
    "Credit stress":{"trigger":"HYG","trigger_name":"HY bonds (inv)","beneficiaries":[("TLT","Duration",.85),("LQD","IG credit",.75),("XLP","Staples",.60),("XLU","Utilities",.60)],"victims":[("KRE","Regionals",-.85),("XLF","Financials",-.80),("IWM","Small caps",-.75),("EEM","EM",-.70),("XLI","Industrials",-.65)]},
    "AI/Tech melt-up":{"trigger":"SMH","trigger_name":"Semiconductors","beneficiaries":[("NVDA","AI leader",.95),("AVGO","AI infra",.90),("AMD","AI GPU",.85),("MSFT","AI platform",.80),("META","AI/ads",.75),("GOOGL","AI/search",.70),("QQQ","Nasdaq 100",.70),("ARKK","Spec growth",.60)],"victims":[("XLU","Utilities",-.60),("XLP","Staples",-.55),("XLE","Energy",-.50),("VLUE","Value",-.45)]},
    "Yen carry unwind":{"trigger":"FXY","trigger_name":"JPY strength","beneficiaries":[("FXY","JPY",.95),("FXF","CHF",.75),("TLT","Duration",.70),("GLD","Gold",.65),("XLU","Utilities",.45),("XLP","Staples",.40)],"victims":[("EWJ","Japan eq",-.85),("EEM","EM",-.80),("ARKK","Spec/leverage",-.80),("IWM","Small caps",-.75),("HYG","HY",-.70),("SPY","US broad",-.60),("BTC-USD","Bitcoin",-.60),("FXA","AUD",-.55),("FXB","GBP",-.50)]},
    "Housing/RE":{"trigger":"XHB","trigger_name":"Homebuilders","beneficiaries":[("XHB","Homebuilders",.95),("ITB","iShares HB",.90),("HD","Home Depot",.80),("LOW","Lowe's",.75),("XLF","Banks (mortgage)",.55),("XLY","Cons disc",.50)],"victims":[("XLRE","REITs (rate-sens)",-.50),("XLU","Utilities (rel)",-.30)]},
    "Geopolitical":{"trigger":"ITA","trigger_name":"Defense ETF","beneficiaries":[("LMT","Lockheed",.90),("RTX","RTX",.88),("GD","Gen Dynamics",.85),("NOC","Northrop",.82),("LHX","L3Harris",.78),("CL=F","Oil",.70),("GLD","Gold",.65),("UUP","USD",.55),("BDRY","Shipping",.50)],"victims":[("JETS","Airlines",-.80),("EXPE","Travel",-.65),("EEM","EM",-.55),("FXE","EUR",-.45),("EWG","Germany",-.40)]},
    "Liquidity drain":{"trigger":"BNKU","trigger_name":"Banks (proxy)","beneficiaries":[("BIL","T-bills",.70),("UUP","USD",.55),("TLT","Duration",.45)],"victims":[("IWM","Small caps",-.85),("ARKK","Speculative",-.80),("HYG","HY",-.75),("KRE","Regionals",-.70),("BTC-USD","Crypto",-.65),("EEM","EM",-.60),("XLRE","REITs",-.55)]},
    "Commodity supercycle":{"trigger":"DBC","trigger_name":"Broad commodities","beneficiaries":[("XLE","Energy",.90),("COPX","Copper",.85),("XME","Metals",.80),("SLX","Steel",.75),("CAT","Caterpillar",.70),("DE","Deere",.65),("BHP","BHP",.60),("FXA","AUD",.55),("FXC","CAD",.50),("EEM","EM producers",.45)],"victims":[("TLT","Duration",-.70),("XLK","Tech",-.45),("XLP","Staples",-.40),("XLY","Cons disc",-.35)]},
    "Weak dollar":{"trigger":"UDN","trigger_name":"USD bearish","beneficiaries":[("EEM","EM equities",.90),("GLD","Gold",.85),("DBC","Commodities",.80),("FXA","AUD",.75),("FXE","EUR",.70),("EFA","Intl dev",.65),("SLV","Silver",.60),("COPX","Copper",.55),("XLE","Energy",.50)],"victims":[("UUP","USD",-.95),("IWM","US small (rel)",-.30)]},
}

# ---- CHAIN SCORER ----
@st.cache_data(ttl=3600*3,show_spinner=False)
def score_chain(ck):
    ch=CHAINS[ck];trig=ycc(ch["trigger"],"6mo")
    if trig.empty or len(trig)<30: return None,[]
    t_mom=rn(trig,21);t_trend=tsf(trig);t_score=.60*t_mom/.05+.40*(t_trend-.5)*2
    is_inv="inv" in ch["trigger_name"].lower()
    if is_inv: t_score=-t_score
    active=abs(t_score)>.3
    rows=[]
    for ticker,name,sens in ch["beneficiaries"]:
        s=ycc(ticker,"6mo")
        if s.empty or len(s)<25: rows.append((name,ticker,sens,0,"No data","ben"));continue
        mom=rn(s,21);trend=tsf(s);live=.60*mom/.05+.40*(trend-.5)*2
        conf=(t_score>0 and live>0) or(t_score<0 and live<0)
        rows.append((name,ticker,sens,live,"Confirming" if conf else "Diverging","ben"))
    for ticker,name,sens in ch["victims"]:
        s=ycc(ticker,"6mo")
        if s.empty or len(s)<25: rows.append((name,ticker,sens,0,"No data","vic"));continue
        mom=rn(s,21);trend=tsf(s);live=.60*mom/.05+.40*(trend-.5)*2
        conf=(t_score>0 and live<0) or(t_score<0 and live>0)
        rows.append((name,ticker,sens,live,"Confirming" if conf else "Diverging","vic"))
    bens=sorted([r for r in rows if r[5]=="ben"],key=lambda x:x[3],reverse=True)
    vics=sorted([r for r in rows if r[5]=="vic"],key=lambda x:x[3])
    return{"trigger":ch["trigger_name"],"t_score":t_score,"active":active},bens+vics

# ---- INTERMARKET ----
def intermarket_confirm(qr,d2):
    checks=[];q=d2.cq
    gc=qr.gcx;checks.append(("IWM/SPY","Growth cross",gc>0,"↑" if gc>0 else "↓",(q in["Q1","Q2"] and gc>0) or(q in["Q3","Q4"] and gc<0)))
    ic=qr.icx;checks.append(("Infl cross","XLE/GLD/DBC",ic>0,"↑" if ic>0 else "↓",(q in["Q2","Q3"] and ic>0) or(q in["Q1","Q4"] and ic<0)))
    hy=qr.sz.get("hy",0);checks.append(("Credit","HY spread",hy>0,"wide" if hy>0 else "calm",(q in["Q3","Q4"] and hy>0) or(q in["Q1","Q2"] and hy<0)))
    curve=lv(S["T10Y2Y"]);checks.append(("Curve","2s10s",curve>0 if np.isfinite(curve) else False,"steep" if (np.isfinite(curve) and curve>0) else "inv",(q in["Q1","Q2"] and np.isfinite(curve) and curve>0) or(q=="Q4" and np.isfinite(curve) and curve<0)))
    gld_m=rn(ycc("GLD","3mo"),42) if len(ycc("GLD","3mo"))>45 else 0
    usd_m=rn(ycc("UUP","3mo"),42) if len(ycc("UUP","3mo"))>45 else 0
    both=gld_m>0 and usd_m>0;checks.append(("Gold+USD","Both rising",both,"both ↑" if both else "no",q=="Q3" and both))
    n_c=sum(1 for c in checks if c[4]);return checks,n_c,len(checks)

# ---- SCENARIOS ----
def build_scenarios(d2,qr):
    q=d2.cq
    return[
        {"ev":"CPI HOT","g":"Neutral","i":"↑ Accel","qs":f"{'Stays '+q if q=='Q3' else 'Toward Q3'}","tr":"Add gold, energy. Cut duration, tech.","pr":"High" if qr.idir=="accelerating" else "Medium"},
        {"ev":"CPI COOL","g":"Neutral","i":"↓ Decel","qs":f"{'Q1 goldilocks' if q in['Q1','Q4'] else 'Q4 if G down' if q=='Q3' else 'Stays'}","tr":"Add duration, tech. Cut energy, gold.","pr":"Low" if qr.idir=="accelerating" else "Medium"},
        {"ev":"Payrolls WEAK","g":"↓ Decel","i":"Neutral","qs":f"{'Q3→Q4' if q=='Q3' else 'Confirms Q4' if q=='Q4' else 'Q3/Q4 risk'}","tr":"Add duration, defensives. Cut cyclicals.","pr":"Medium" if qr.gdir=="decelerating" else "Low"},
        {"ev":"Payrolls STRONG","g":"↑ Accel","i":"Slight ↑","qs":f"{'Q3→Q2' if q=='Q3' else 'Toward Q2'}","tr":"Add cyclicals, small caps. Cut duration.","pr":"Low" if qr.gdir=="decelerating" else "Medium"},
        {"ev":"FOMC HAWKISH","g":"↓ Tightens","i":"Cred ↑","qs":"Q3 sticky short-term","tr":"Add USD, short-dur. Cut EM, risk.","pr":"Medium"},
        {"ev":"FOMC DOVISH","g":"↑ Supports","i":"Risk ↑","qs":f"{'Q3→Q2' if q=='Q3' else 'Q1/Q2'}","tr":"Add risk, EM, small caps. Cut USD.","pr":"Low" if q=="Q3" else "Medium"},
    ]

# ---- POSITION SIZING ----
def pos_guide(d2,cp_val):
    sc=.35*d2.cn+.25*(1-d2.fg)+.20*d2.br+.20*(1-cp_val)
    if sc>.72: return "Aggressive","High confidence, low fragility"
    if sc>.58: return "Normal","Adequate confidence"
    if sc>.42: return "Reduced","Mixed signals, elevated risk"
    return "Minimal","Low confidence / high crash risk"

# ---- HEDGE RECS ----
def hedge_recs(d2,qr,cp_val):
    h=[];q=d2.cq
    if cp_val>=.45: h.append(("SPY puts (2-3mo 5% OTM)","Tail protection","High"))
    if cp_val>=.35: h.append(("Long TLT","Flight to safety","Medium" if q!="Q3" else "Low"))
    if q=="Q3":
        h.append(("Long GLD / miners","Stagflation hedge","High"))
        h.append(("Long UUP","USD strength Q3","High"))
        h.append(("Short IWM","Most vulnerable Q3","Medium"))
    elif q=="Q2":
        h.append(("VIX calls","Q2 top risk","Medium"))
        h.append(("Long GLD","Insurance if Q2→Q3","Medium"))
    elif q=="Q4":
        h.append(("Long TLT aggressive","Primary Q4 play","High"))
        h.append(("Long GLD","Second haven","Medium"))
        h.append(("Short HYG","Credit deterioration","Medium"))
    elif q=="Q1":
        h.append(("Tight stops","Q1 ride, low hedge need","Low"))
    pri={"High":0,"Medium":1,"Low":2};h.sort(key=lambda x:pri.get(x[2],9));return h

# ---- HISTORICAL ----
QUAD_HIST={"Q1":{"l":"Goldilocks","spx":"+12-18% ann","dur":"3-6mo","best":"Growth, semis","worst":"Commodities","wr":"~75%"},
    "Q2":{"l":"Reflation","spx":"+8-15% ann","dur":"2-5mo","best":"Small caps, cyclicals","worst":"Duration, defensives","wr":"~65%"},
    "Q3":{"l":"Stagflation","spx":"-5 to +3% ann","dur":"2-4mo","best":"Gold, energy, USD","worst":"Small caps, growth, EM","wr":"~30%"},
    "Q4":{"l":"Deflation","spx":"-8 to +5% ann","dur":"1-3mo","best":"Duration, defensives","worst":"Cyclicals, small caps","wr":"~35%"}}

# ---- DATA FRESHNESS ----
def data_fresh():
    sn2={"INDPRO":"Ind Prod","RSAFS":"Retail","PAYEMS":"Payrolls","UNRATE":"Unemp","CPI":"CPI","CCPI":"Core CPI","PPI":"PPI","ICSA":"Claims","SAHM":"Sahm"}
    now=datetime.now();rows=[]
    for k,l in sn2.items():
        s=S.get(k,pd.Series(dtype=float)).dropna()
        if s.empty: rows.append((l,"No data",999,"r"));continue
        last=s.index[-1];days=(now-last).days
        c="r" if days>60 else("a" if days>35 else "g");rows.append((l,last.strftime("%Y-%m-%d"),days,c))
    rows.sort(key=lambda x:x[2],reverse=True);return rows

# ---- MAPS ----
AS={"US cyclicals":{"Q1":.55,"Q2":.85,"Q3":-.55,"Q4":-.45},"US defensives":{"Q1":-.25,"Q2":-.45,"Q3":.70,"Q4":.75},"US small caps":{"Q1":.50,"Q2":.95,"Q3":-.75,"Q4":-.65},"EM equities":{"Q1":.45,"Q2":.75,"Q3":-.40,"Q4":-.25},"IHSG cyclicals":{"Q1":.35,"Q2":.70,"Q3":-.20,"Q4":-.20},"Gold/miners":{"Q1":-.10,"Q2":.00,"Q3":.90,"Q4":.35},"Oil/energy":{"Q1":.05,"Q2":.55,"Q3":.80,"Q4":-.15},"USD":{"Q1":-.40,"Q2":.05,"Q3":.80,"Q4":.55},"Duration/bonds":{"Q1":-.30,"Q2":-.85,"Q3":.10,"Q4":.90},"BTC/crypto":{"Q1":.60,"Q2":.75,"Q3":-.35,"Q4":-.10}}
FXM={"USD":{"Q1":-.35,"Q2":.05,"Q3":.85,"Q4":.55},"JPY":{"Q1":-.15,"Q2":-.45,"Q3":.55,"Q4":.70},"CHF":{"Q1":-.10,"Q2":-.35,"Q3":.45,"Q4":.55},"AUD":{"Q1":.35,"Q2":.75,"Q3":-.60,"Q4":-.35},"NZD":{"Q1":.35,"Q2":.70,"Q3":-.65,"Q4":-.35},"CAD":{"Q1":.10,"Q2":.45,"Q3":-.10,"Q4":-.10},"EMFX":{"Q1":.30,"Q2":.65,"Q3":-.55,"Q4":-.30}}
FPM={("USD","AUD"):"SHORT AUD/USD",("USD","NZD"):"SHORT NZD/USD",("USD","EMFX"):"LONG USD/EM",("USD","CAD"):"LONG USD/CAD",("JPY","AUD"):"SHORT AUD/JPY",("JPY","NZD"):"SHORT NZD/JPY",("CHF","AUD"):"SHORT AUD/CHF",("CHF","NZD"):"SHORT NZD/CHF"}
SGD={"Q1":{"Early":("quality growth, semis","staples, utilities","breadth widens"),"Mid":("broad equities","bond proxies","breadth clean"),"Late":("cyclicals frothy","defensives lag","Q2 heat")},"Q2":{"Early":("small caps, cyclicals","duration, staples","rates orderly"),"Mid":("financials, materials","REITs","credit stable"),"Late":("energy, value","duration beta","top risk")},"Q3":{"Early":("gold, miners, energy, USD","small caps, EMFX","infl re-accel"),"Mid":("gold, defensives","broad beta","breadth narrow"),"Late":("gold+dur barbell","crowded beta","Q2 or Q4")},"Q4":{"Early":("duration, defensives","cyclicals","growth scare"),"Mid":("duration, quality","broad beta","bottoming unclear"),"Late":("duration, gold","junky beta","true vs false")}}
PICKS={"Q1":{"UL":["NVDA","AVGO","MSFT","AMZN","GOOGL","PLTR"],"US":["XOM","CVX","KO","PG"],"IL":["BBCA.JK","TLKM.JK","ICBP.JK"],"IS":["ADRO.JK","PTBA.JK"]},"Q2":{"UL":["CAT","ETN","DE","GE","FCX","XOM","CVX","JPM","GS"],"US":["KO","PG","PFE","MRK"],"IL":["ADRO.JK","PTBA.JK","UNTR.JK","ANTM.JK"],"IS":["TLKM.JK","ICBP.JK"]},"Q3":{"UL":["XOM","CVX","COP","SLB","MPC","GLD","NEM","WMT","COST","LLY"],"US":["SNOW","MDB","RBLX","SMCI","DDOG","AMD"],"IL":["ADRO.JK","PTBA.JK","MEDC.JK","ANTM.JK"],"IS":["CTRA.JK","BSDE.JK","EXCL.JK"]},"Q4":{"UL":["TLT","GLD","WMT","COST","PG","KO","JNJ","UNH"],"US":["CAT","DE","FCX","XOM"],"IL":["BBCA.JK","TLKM.JK","ICBP.JK"],"IS":["ADRO.JK","PTBA.JK","CTRA.JK"]}}
US_TH={"Semis/AI":{"NVDA","AVGO","AMD","QCOM"},"Software":{"MSFT","PLTR","CRWD"},"Energy":{"XOM","CVX","COP","SLB","MPC"},"Defense/Ind":{"RTX","LMT","CAT","GE","DE","ETN","AXON"},"Financials":{"JPM","GS","BAC","V"},"Gold/Metals":{"GLD","NEM","FCX"},"Consumer":{"WMT","COST","PG","KO","HD"},"Health":{"LLY","ABBV","UNH"},"Mega":{"AAPL","META","AMZN","GOOGL"}}

def vr(d2):
    if d2.cq=="Q3" and d2.nq=="Q2": return "Bad reflation" if d2.si_>.58 and d2.sl>.45 else "Good reflation"
    if d2.cq=="Q2": return "Crash-prone Q2" if d2.tsc2>.56 and d2.si_>.56 else "Healthy Q2"
    if d2.cq=="Q4" and d2.nq=="Q1": return "False dawn" if d2.ag<.55 else "True bottoming"
    return "Base path"
def stgf(d2):
    q=d2.cq
    if q=="Q1": m=.45*d2.ps+.30*d2.tsc2+.25*d2.tp
    elif q=="Q2": m=.42*d2.tsc2+.28*d2.ps+.20*d2.tp+.10*d2.fg
    elif q=="Q3": m=.35*d2.tsc2+.25*d2.ps+.20*d2.tp+.20*d2.fg
    else: m=.40*d2.bsc+.22*d2.tp+.18*d2.ps+.20*d2.fg
    return("Early" if m<.36 else("Mid" if m<.67 else "Late")),float(m)
def adj(a,nx,v):
    if v=="Good reflation":
        if a in["US small caps","EM equities","BTC/crypto"]: nx+=.18
        if a in["USD","Duration/bonds"]: nx-=.10
    elif v=="Bad reflation":
        if a in["Oil/energy","Gold/miners","USD"]: nx+=.18
        if a in["US small caps","EM equities","BTC/crypto"]: nx-=.18
    return cl(nx,-1,1)
def asc2(d2,a):
    bk=AS[a];cu=float(bk[d2.cq]);nx=adj(a,float(bk[d2.nq]),vr(d2));return(1-d2.tc*.35)*cu+(d2.tc*.35)*nx,cu,nx
def fxsc(d2,ccy):
    bk=FXM[ccy];cu=float(bk[d2.cq]);nx=float(bk[d2.nq]);return cl((1-d2.tc*.35)*cu+(d2.tc*.35)*nx,-1,1)
def crash2(d2):
    base={"Q1":.18,"Q2":.32,"Q3":.45,"Q4":.55}.get(d2.cq,.40);v=vr(d2);a=.06 if v in["Bad reflation","Crash-prone Q2"] else(-.05 if v in["Good reflation","Healthy Q2","True bottoming"] else 0)
    sm=.24*d2.sl+.20*d2.si_+.16*(1-d2.br)+.16*d2.tsc2+.12*d2.tp+.12*d2.fg;return c01(base+(sm-.45)*.50+a)
bl=lambda x:"Strong" if x>=.55 else("Above avg" if x>=.18 else("Mixed" if x>-.18 else("Below avg" if x>-.55 else "Weak")))
br3=lambda x:"Long" if x>=.55 else("Lean L" if x>=.18 else("Neutral" if x>-.18 else("Lean S" if x>-.55 else "Short")))
cml=lambda s:"Very high" if s>=.80 else("High" if s>=.65 else("Elevated" if s>=.45 else("Watch" if s>=.25 else "Low")))
epl=lambda c2,f:"Size up" if .6*c2+.4*(1-f)>.72 else("Normal" if .6*c2+.4*(1-f)>.54 else "Small")

def build_events():
    now=datetime.now(timezone.utc)
    def _dt(y,m,dd,h,mi=0): return datetime(y,m,dd,h,mi,tzinfo=timezone.utc)
    def _cd2(t):
        s=int((t-now).total_seconds())
        if s<=0: return "Released",0
        dd,r=divmod(s,86400);h,r=divmod(r,3600);mi,_=divmod(r,60)
        return(f"{dd}d {h}h" if dd else(f"{h}h {mi}m" if h else f"{mi}m")),s
    events=[(_dt(2026,4,1,12,30),"Retail Sales","Growth","Weak→Q3. Strong→Q2.","High"),(_dt(2026,4,3,12,30),"Payrolls","Growth","Weak=Q3/Q4. Strong=Q2.","High"),(_dt(2026,4,10,12,30),"CPI","Inflation","Hot=Q3. Cool=Q4/Q1.","Critical"),(_dt(2026,4,14,12,30),"PPI","Inflation","Pipeline.","Medium"),(_dt(2026,4,29,18),"FOMC","Policy","Hawkish→Q3. Dovish→Q4.","Critical"),(_dt(2026,5,2,12,30),"PCE","Inflation","Hot=Q3.","High"),(_dt(2026,5,8,12,30),"Payrolls","Growth","G confirm.","High"),(_dt(2026,5,12,12,30),"CPI","Inflation","I trend.","Critical"),(_dt(2026,6,17,18),"FOMC+SEP","Policy","Projections.","Critical")]
    rows=[]
    for dt,name,axis,impact,imp in events:
        cd,secs=_cd2(dt)
        if secs<=0: continue
        rows.append({"name":name,"axis":axis,"cd":cd,"secs":secs,"impact":impact,"imp":imp})
    rows.sort(key=lambda x:x["secs"]);return rows

# ---- LEADERSHIP ----
US_U=["AAPL","MSFT","NVDA","AVGO","AMD","META","AMZN","GOOGL","PLTR","CRWD","JPM","GS","BAC","V","XOM","CVX","COP","SLB","MPC","LLY","ABBV","UNH","CAT","GE","RTX","LMT","DE","ETN","WMT","COST","PG","KO","HD","GLD","NEM","FCX","AXON","QCOM"]
IH_U=["BBCA.JK","BBRI.JK","BMRI.JK","BBNI.JK","TLKM.JK","ASII.JK","ADRO.JK","PTBA.JK","MEDC.JK","CTRA.JK","BSDE.JK","EXCL.JK","ANTM.JK","UNTR.JK"]
CM={"Gold":"GC=F","Silver":"SI=F","Copper":"HG=F","WTI":"CL=F","NatGas":"NG=F"}
CR={"BTC":"BTC-USD","ETH":"ETH-USD","SOL":"SOL-USD","XRP":"XRP-USD","LINK":"LINK-USD"}
def _th(t,m):
    for th,s in m.items():
        if t in s: return th
    return "Other"
@st.cache_data(ttl=3600*3,show_spinner=False)
def scn(tickers,bench,period="1y",fb2=None):
    if not tickers: return pd.DataFrame()
    fetch=[bench]+tickers+([fb2] if fb2 else []);px=yb(fetch,period)
    if px.empty or bench not in px.columns:
        if fb2 and not px.empty and fb2 in px.columns: bench=fb2
        else: return pd.DataFrame()
    bm=px[bench].dropna()
    if len(bm)<25: return pd.DataFrame()
    rows=[]
    for t in tickers:
        if t not in px.columns: continue
        s=px[t].dropna()
        if s.empty: continue
        df=pd.concat([s.rename("a"),bm.rename("b")],axis=1).sort_index().ffill().dropna()
        if len(df)<40: continue
        a2,b2=df["a"],df["b"];r21,r63=rn(a2,21),rn(a2,63);b21,b63=rn(b2,21),rn(b2,63)
        al21,al63=r21-b21,r63-b63;tr=tsf(a2);rs=c01(.35*nt(al21,.08)+.30*nt(al63,.15)+.15*nt(al21,.25)+.20*tr)
        if al21>.02 and al63>.04 and tr>=.60: st2,cm2="Leader","RS leader"
        elif al21>.015 and tr>=.45: st2,cm2="Emerging","improving"
        elif al21<-.03 and al63<-.03: st2,cm2="Weak","underperforming"
        else: st2,cm2="Neutral","mixed"
        rows.append({"T":t,"S":st2,"RS":rs,"A21":al21,"A63":al63,"C":cm2})
    return pd.DataFrame(rows).sort_values(["RS","A21"],ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
def tkr(df,mode,n=6):
    if df is None or df.empty: return [["No data","-","-","-"]]
    if mode=="long": w=df.sort_values(["RS","A21"],ascending=False).head(n)
    else: w=df.sort_values(["RS","A21"],ascending=True).head(n)
    return [[str(r["T"]).replace(".JK",""),r["S"],f"{100*r['A21']:+.1f}%/{100*r['A63']:+.1f}%",r["C"]] for _,r in w.iterrows()] or [["No data","-","-","-"]]
def rks(uni,bench=None,side="long",n=5):
    tickers=list(uni.values())+([bench] if bench else []);px=yb(tickers,"6mo");rows=[]
    if px.empty: return [["No data","-","-"]]
    bm=px[bench].dropna() if bench and bench in px.columns else None
    for label,t in uni.items():
        if t not in px.columns: continue
        s=px[t].dropna()
        if len(s)<35: continue
        sc=.55*rn(s,21)+.30*rn(s,63)+.15*(2*tsf(s)-1)*.10;c2="Abs"
        if bm is not None and not bm.empty:
            d2=pd.concat([s.rename("a"),bm.rename("b")],axis=1).sort_index().ffill().dropna()
            if len(d2)>=35: sc=.65*(rn(d2["a"],21)-rn(d2["b"],21))+.35*sc;c2=f"vs {bench}"
        rows.append((label,float(sc),c2))
    rows.sort(key=lambda x:x[1],reverse=True)
    if not rows: return [["No data","-","-"]]
    if side=="long": return [[l,f"{100*s:+.1f}",c2] for l,s,c2 in rows[:n]]
    else: return [[l,f"{100*s:+.1f}",c2] for l,s,c2 in reversed(rows[-n:])]

IH_TH={"Banks":{"BBCA.JK","BBRI.JK","BMRI.JK","BBNI.JK"},"Telco":{"TLKM.JK","EXCL.JK"},"Autos/Consumer":{"ASII.JK"},"Property":{"CTRA.JK","BSDE.JK"},"Coal/Energy":{"ADRO.JK","PTBA.JK","MEDC.JK"},"Metals/Resources":{"ANTM.JK","UNTR.JK"}}
ROLE_W={
    "MSFT":1.00,"AAPL":1.00,"NVDA":1.00,"AMZN":0.95,"META":0.95,"GOOGL":0.95,
    "AVGO":0.90,"JPM":0.85,"XOM":0.85,"LLY":0.85,"V":0.82,"CVX":0.82,
    "AMD":0.78,"PLTR":0.70,"CRWD":0.68,"GS":0.72,"BAC":0.72,"COP":0.70,
    "SLB":0.62,"MPC":0.62,"ABBV":0.70,"UNH":0.72,"CAT":0.70,"GE":0.68,
    "RTX":0.68,"LMT":0.68,"DE":0.62,"ETN":0.68,"WMT":0.72,"COST":0.72,
    "PG":0.68,"KO":0.62,"HD":0.70,"GLD":0.64,"NEM":0.56,"FCX":0.58,"AXON":0.52,"QCOM":0.62,
    "BBCA.JK":1.00,"BBRI.JK":0.96,"BMRI.JK":0.92,"BBNI.JK":0.82,"TLKM.JK":0.78,"ASII.JK":0.72,
    "ADRO.JK":0.74,"PTBA.JK":0.64,"MEDC.JK":0.62,"CTRA.JK":0.56,"BSDE.JK":0.54,"EXCL.JK":0.52,
    "ANTM.JK":0.60,"UNTR.JK":0.62,
}

def _rolew(t):
    return ROLE_W.get(str(t).upper(),0.50)

def prep_lead(df,market="US"):
    if df is None or df.empty: return pd.DataFrame()
    d2=df.copy()
    if "Theme" not in d2.columns:
        d2["Theme"]=d2["T"].apply(lambda x:_th(x,US_TH if market=="US" else IH_TH))
    d2["RoleW"]=d2["T"].apply(_rolew)
    d2["ImpactProxy"]=d2["RS"]*d2["RoleW"]
    tot=float(d2["ImpactProxy"].sum())
    d2["ImpactShare"]=d2["ImpactProxy"]/tot if tot>1e-9 else 0.0
    d2["Up"]=d2["A21"]>0
    return d2.sort_values(["ImpactProxy","RS","A21"],ascending=False).reset_index(drop=True)

def lead_snap(df):
    if df is None or df.empty:
        return {"verdict":"No data","breadth":0.0,"conc":0.0,"n_pos":0,"themes":0,"line":"Belum ada coverage yang valid.","why":"No data","capeq":"n/a","russell":"n/a"}
    n=len(df); n_pos=int((df["A21"]>0).sum()); breadth=float((df["A21"]>0).mean())
    top5=float(df["ImpactShare"].head(min(5,n)).sum())
    top7=float(df["ImpactShare"].head(min(7,n)).sum())
    pos_themes=int(df.loc[df["A21"]>0,"Theme"].nunique())
    if breadth<0.30 and top5>=0.60:
        verdict="Breaking"
        line="Kalau cuma 5–7 saham yang gerakin indeks dan sisanya tidak ikut, market rapuh dan gampang gagal confirm."
    elif breadth<0.45 and top5>=0.52:
        verdict="Fragile"
        line="Leadership sempit: beberapa nama besar masih menahan indeks, tapi participation tipis dan rally gampang goyah."
    elif breadth>=0.55 and top5<0.42 and pos_themes>=4:
        verdict="Broad"
        line="Banyak sektor ikut bantu. Rally lebih sehat karena tidak cuma ditopang 5–7 saham besar."
    else:
        verdict="Narrow"
        line="Masih jalan, tapi leadership belum merata. Perlu lebih banyak sektor bantu supaya naiknya lebih sehat."
    if top5>=0.55:
        why=f"Top-5 proxy impact {pc(top5)} / top-7 {pc(top7)} — sempit"
    elif pos_themes>=4 and breadth>=0.55:
        why=f"{pos_themes} tema ikut bantu / breadth {pc(breadth)} — sehat"
    else:
        why=f"Breadth {pc(breadth)} / top-5 proxy impact {pc(top5)}"
    capeq="Cap-weight > equal-weight" if top5>=0.52 else ("Cap-weight ~ equal-weight" if top5>=0.40 else "Equal-weight ikut confirm")
    russell="Russell confirm" if verdict=="Broad" else ("Russell mixed" if verdict=="Narrow" else "Russell / breadth belum confirm")
    return {"verdict":verdict,"breadth":breadth,"conc":top5,"n_pos":n_pos,"themes":pos_themes,"line":line,"why":why,"capeq":capeq,"russell":russell}

def impact_rows(df,side="top",n=7):
    if df is None or df.empty: return [["No data","-","-","-","-"]]
    w=df.sort_values(["ImpactProxy","A21"],ascending=[False,False]).head(n) if side=="top" else df.sort_values(["ImpactProxy","A21"],ascending=[False,True]).head(n)
    out=[]
    for _,r in w.iterrows():
        read="Matter buat tape" if r["ImpactShare"]>=0.12 else ("Matter" if r["ImpactShare"]>=0.08 else "Support")
        out.append([str(r["T"]).replace('.JK',''),r["Theme"],pc(r["ImpactShare"]),r["S"],read])
    return out or [["No data","-","-","-","-"]]

def rotation_rows(df,n=6):
    if df is None or df.empty or "Theme" not in df.columns: return [["No data","-","-","-"]]
    g=df.groupby("Theme").agg(RS=("RS","mean"),A21=("A21","mean"),N=("T","count")).sort_values(["RS","A21"],ascending=False)
    hi=g.head(min(n//2,len(g))); lo=g.tail(min(n//2,max(0,len(g)-len(hi)))) if len(g)>1 else g.iloc[0:0]
    rows=[]
    for idx,r in hi.iterrows(): rows.append([idx,"Absorbed / bid",f"{100*r['A21']:+.1f}%",f"{r['N']} names"])
    for idx,r in lo.sort_values(["RS","A21"]).iterrows(): rows.append([idx,"Dibuang / under pressure",f"{100*r['A21']:+.1f}%",f"{r['N']} names"])
    return rows or [["No data","-","-","-"]]

def rs_rows(df,n=7):
    if df is None or df.empty: return [["No data","-","-","-"]]
    w=df.sort_values(["RS","A21"],ascending=False).head(n)
    out=[]
    for _,r in w.iterrows():
        read="Kandidat continuation" if r["A21"]>0 and r["RS"]>=0.62 else ("Masih kuat" if r["RS"]>=0.55 else "Watch")
        out.append([str(r["T"]).replace('.JK',''),r["Theme"],r["S"],read])
    return out or [["No data","-","-","-"]]

# ==== COMPUTE ====
eng=Eng(CF);qr=eng.compute();d=dgf(qr,qr.mb);s2,m2=stgf(d);v2=vr(d);cp=crash2(d)
ro=c01(.32*d.cn+.28*d.br+.18*(1-d.sl)+.12*(1-d.si_)+.10*(1-d.fg))
rf=c01(.34*d.sl+.24*d.si_+.20*(1-d.br)+.12*d.fg+.10*d.tsc2)
evts=build_events();fg_sc,fg_src=fetch_fg();fg_vb,fg_cl=fgll(fg_sc)
udf=scn(US_U,"SPY","1y","QQQ")
if not udf.empty: udf["Theme"]=udf["T"].apply(lambda x:_th(x,US_TH))
idf=scn(IH_U,"^JKSE","1y","EIDO")
if idf is None or idf.empty: idf=scn(IH_U,"EIDO","6mo")
if not idf.empty: idf["Theme"]=idf["T"].apply(lambda x:_th(x,IH_TH))
lead_us=prep_lead(udf,"US")
lead_ih=prep_lead(idf,"IHSG")
ls_us=lead_snap(lead_us)
ls_ih=lead_snap(lead_ih)
if ls_us["verdict"] in ["Breaking","Fragile"] or ls_ih["verdict"] in ["Breaking","Fragile"]:
    lq_verdict="Fragile" if (ls_us["verdict"]!="Breaking" and ls_ih["verdict"]!="Breaking") else "Breaking"
elif ls_us["verdict"]=="Broad" and ls_ih["verdict"]=="Broad":
    lq_verdict="Broad"
else:
    lq_verdict="Narrow"
lq_text=ls_us["line"] if lead_us is not None and not lead_us.empty else ls_ih["line"]
im_checks,im_n,im_t=intermarket_confirm(qr,d)
scenarios=build_scenarios(d,qr)
psz,psr=pos_guide(d,cp)
hedges=hedge_recs(d,qr,cp)

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown(f"**F&G:** {fg_sc} ({fg_vb}) via {fg_src}")
    mfg=st.number_input("Override (0=auto)",0,100,0,key="fgo")
    if mfg>0: fg_sc=mfg;fg_vb,fg_cl=fgll(mfg);fg_src="Manual"

# ==== RENDER ====
st.title("QuantFinal V10.9")
st.markdown(f"<div class='mc'>Config {CF.h()} · F&G {fg_src} · Intermarket {im_n}/{im_t}</div>",unsafe_allow_html=True)

# Alerts (crash only here, not in hero)
if cp>=.55: st.markdown(f"<div class='ar'><b>CRASH HIGH {pc(cp)}</b> — Reduce size, hedge tail.</div>",unsafe_allow_html=True)
elif cp>=.35: st.markdown(f"<div class='aa'><b>CRASH ELEVATED {pc(cp)}</b></div>",unsafe_allow_html=True)
if qr.gdir=="decelerating" and qr.idir=="accelerating":
    st.markdown(f"<div class='ar'><b>Q3 ALIGNED</b> — G↓ ({qr.g_cd}mo) + I↑ ({qr.i_cr}mo)</div>",unsafe_allow_html=True)

if evts:
    ne=evts[0];ic="r" if ne["imp"]=="Critical" else("a" if ne["imp"]=="High" else "")
    st.markdown(f"<div class='evt-box'>⏱ <b>Next:</b> {ne['name']} in <b>{ne['cd']}</b> {pl(ne['imp'],ic)} — {ne['impact']}</div>",unsafe_allow_html=True)

# Hero (7 cards, no crash dupe)
st.write("")
hc=st.columns(7)
for col,(t,v,sub) in zip(hc,[
    ("Regime",f"{s2} {d.cq}",pl(f"→ {d.nq} ({d.pa})")),
    ("Growth",qr.gdir[:5].upper(),pl(f"{qr.g_cd}mo · {qr.g_roc2:+.2f}","r" if qr.gdir=="decelerating" else("g" if qr.gdir=="accelerating" else ""))),
    ("Inflation",qr.idir[:5].upper(),pl(f"{qr.i_cr}mo · {qr.i_roc2:+.2f}","r" if qr.idir=="accelerating" else("g" if qr.idir=="decelerating" else ""))),
    ("Variant",v2,pl(d.sub)),
    ("Confidence",pc(d.cn),pl(epl(d.cn,d.fg))),
    ("Leadership",lq_verdict,pl(ls_us["why"] if lead_us is not None and not lead_us.empty else ls_ih["why"],"g" if lq_verdict=="Broad" else("a" if lq_verdict=="Narrow" else "r"))),
    ("F&G",f"{fg_sc}",pl(fg_vb,fg_cl)),
]):
    with col:
        st.markdown(f"<div class='hero'><div class='mt2'>{t}</div><div class='mv'>{v}</div><div class='ms'>{sub}</div></div>",unsafe_allow_html=True)

# Quick read
st.write("");gn=SGD[d.cq][s2];gnx=SGD[d.nq]["Early"];pk=PICKS.get(d.cq,{})
st.markdown(f"""<div class='nb'>
<b>NOW: {s2} {d.cq}</b> · {d.sub} · {v2} · intermarket {im_n}/{im_t}<br>
<b>Strong:</b> {gn[0]} · <b>Weak:</b> {gn[1]} · <b>Confirms:</b> {gn[2]}<br>
<b>Longs:</b> {', '.join(pk.get('UL',[]))} · <b>Shorts:</b> {', '.join(pk.get('US',[]))}<br>
<b>IHSG L:</b> {', '.join([t.replace('.JK','') for t in pk.get('IL',[])])} · <b>S:</b> {', '.join([t.replace('.JK','') for t in pk.get('IS',[])])}<br>
<b>Position:</b> {psz} — {psr}<br>
<b>IF {d.nq} →</b> {gnx[0]} strong · {gnx[1]} weak<br>
<span class='sm'>Transition {d.pa} ({pc(d.tc)}) · Stay {pc(d.sp)} · Signal {d.sq}</span>
</div>""",unsafe_allow_html=True)

st.markdown(f"<div class='nb'><b>Leadership quality:</b> {lq_verdict} — {lq_text} <span class='sm'>US: {ls_us['why']} · IHSG: {ls_ih['why']}</span></div>",unsafe_allow_html=True)
# ==== TABS ====
tabs=st.tabs(["Playbook","Spillovers","Leadership","Risk & Events","Audit"])

# ---- PLAYBOOK ----
with tabs[0]:
    st.markdown("<div class='card'><div class='st'>CROSS-ASSET PLAYBOOK</div>",unsafe_allow_html=True)
    c1,c2=st.columns([1.15,.85])
    with c1:
        scored=[]
        for a in AS:
            now,cur,nxt=asc2(d,a);delta=nxt-cur;sh="↑" if delta>.25 else("↓" if delta<-.25 else "→")
            scored.append((a,now,bl(now),bl(nxt),br3(now),sh))
        scored.sort(key=lambda x:x[1],reverse=True)
        st.markdown(tb(["Asset","Now","If "+d.nq,"Action","Δ"],[[a,b,bn,act,sh] for a,_,b,bn,act,sh in scored]),unsafe_allow_html=True)
    with c2:
        fxs=sorted([(cy,fxsc(d,cy)) for cy in FXM],key=lambda x:x[1],reverse=True)
        fxr=[]
        for ccy,sc in fxs:
            if sc>=.15:
                wk=[w for w,ws in fxs if ws<=-.15 and w!=ccy];expr=FPM.get((ccy,wk[0]),"") if wk else ""
            elif sc<=-.15:
                st3=[s3 for s3,ss in fxs if ss>=.15 and s3!=ccy];expr=FPM.get((st3[0],ccy),"") if st3 else ""
            else: expr="—"
            fxr.append([ccy,bl(sc),br3(sc),expr])
        st.markdown(tb(["FX","Bias","Use","Trade"],fxr),unsafe_allow_html=True)
    if d.nq!=d.cq:
        npk=PICKS.get(d.nq,{})
        with st.expander(f"If {d.nq} takes over →"):
            st.markdown(f"**US Long:** {', '.join(npk.get('UL',[]))} · **Short:** {', '.join(npk.get('US',[]))}")
            st.markdown(f"**IHSG Long:** {', '.join([t.replace('.JK','') for t in npk.get('IL',[])])} · **Short:** {', '.join([t.replace('.JK','') for t in npk.get('IS',[])])}")
    st.write("")
    cm1,cm2=st.columns(2)
    with cm1: st.markdown("**Commodities**");st.markdown(tb(["Name","Score","Read"],rks(CM,"DBC","long")+[["---","---","---"]]+rks(CM,"DBC","short")),unsafe_allow_html=True)
    with cm2: st.markdown("**Crypto**");st.markdown(tb(["Name","Score","Read"],rks(CR,"BTC-USD","long")+[["---","---","---"]]+rks(CR,"BTC-USD","short")),unsafe_allow_html=True)
    st.write("")
    st.markdown("**Hedging**")
    hr=[[h[0],h[1],pl(h[2],"r" if h[2]=="High" else("a" if h[2]=="Medium" else ""))] for h in hedges]
    st.markdown(tb(["Hedge","Rationale","Priority"],hr),unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)

# ---- SPILLOVERS ----
with tabs[1]:
    st.markdown("<div class='card'><div class='st'>SPILLOVERS · INTERMARKET · SCENARIOS</div>",unsafe_allow_html=True)
    st.markdown(f"**Intermarket: {im_n}/{im_t} confirm {d.cq}**")
    imr=[]
    for name,desc,val,direction,conf in im_checks:
        imr.append([f"{'✓' if conf else '✗'} {name}",desc,direction,"Confirms" if conf else "Diverges"])
    st.markdown(tb(["Signal","What","Dir","vs "+d.cq],imr),unsafe_allow_html=True)
    st.write("")
    st.markdown("**Active spillover chains** (sorted by trigger strength)")
    ac,iac=[],[]
    for key in CHAINS:
        res,det=score_chain(key)
        if res is None: continue
        if res["active"]: ac.append((key,res,det))
        else: iac.append((key,res,det))
    ac.sort(key=lambda x:abs(x[1]["t_score"]),reverse=True)
    if not ac: st.markdown("*No chains active*")
    for key,res,det in ac:
        dr="↑" if res["t_score"]>0 else "↓"
        st.markdown(f"**{key}** — {res['trigger']} {dr} ({res['t_score']:+.1f})")
        rows=[]
        for name,ticker,sens,live,status,role in det:
            icon="🟢" if role=="ben" else "🔴"
            rows.append([f"{icon} {name}",ticker,f"{sens:+.0%}",f"{live:+.1f}",f"{'✓' if status=='Confirming' else '✗'} {status}"])
        st.markdown(tb(["Asset","Ticker","Sens","Live","Status"],rows),unsafe_allow_html=True)
        st.write("")
    if iac:
        with st.expander(f"Inactive ({len(iac)})"):
            for key,res,det in iac:
                st.markdown(f"**{key}** — {res['trigger']} ({res['t_score']:+.1f})")
    st.write("")
    st.markdown("**Scenarios — what if?**")
    scr=[]
    for sc in scenarios:
        pc2="r" if sc["pr"]=="High" else("a" if sc["pr"]=="Medium" else "")
        scr.append([sc["ev"],sc["g"],sc["i"],sc["qs"],sc["tr"],pl(sc["pr"],pc2)])
    st.markdown(tb(["Event","G","I","Quad shift","Trades","Prob"],scr),unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)

# ---- LEADERSHIP ----
with tabs[2]:
    st.markdown("<div class='card'><div class='st'>LEADERSHIP · FLOW CONCENTRATION · ROTATION · RELATIVE STRENGTH</div>",unsafe_allow_html=True)
    st.markdown("<div class='mc'>Yang gue merge di sini: impact leaders, flow concentration, rotation tape, dan relative strength. Jadi yang saling berhubungan kebaca dalam satu blok, bukan panel kecil yang kepisah-pisah.</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='nb'><b>Leadership</b> = saham yang bukan cuma hijau, tapi impact proxy-nya besar, jadi benar-benar matter buat market. <b>Flow concentration</b> = kalau cuma 5–7 saham yang gerakin indeks, market rapuh; kalau banyak sektor bantu, rally lebih sehat. <b>Rotation</b> = lihat tema yang lagi dibuang vs diserap. <b>Relative strength</b> = saham yang tetap kuat saat market goyah sering jadi kandidat continuation.<br><span class='sm'>Catatan: impact di build ini pakai <b>proxy impact</b> (RS × role weight di watched universe), bukan cap-weight indeks resmi. Gunanya untuk tape-read cepat, bukan menggantikan data index constituent resmi.</span></div>",unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)
    for col,(ttl,sn) in zip([c1,c2,c3,c4],[("US leadership",ls_us),("IHSG leadership",ls_ih)]):
        pass
    with c1:
        st.markdown(f"<div class='hero'><div class='mt2'>US leadership</div><div class='mv'>{ls_us['verdict']}</div><div class='ms'>{ls_us['why']}</div></div>",unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='hero'><div class='mt2'>IHSG leadership</div><div class='mv'>{ls_ih['verdict']}</div><div class='ms'>{ls_ih['why']}</div></div>",unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='hero'><div class='mt2'>Flow concentration</div><div class='mv'>{pc(ls_us['conc'])}</div><div class='ms'>US top-5 proxy impact share · {ls_us['russell']}</div></div>",unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='hero'><div class='mt2'>Rotation state</div><div class='mv'>{'Absorption' if ls_us['breadth']>=.5 else 'Mixed' if ls_us['breadth']>=.38 else 'Defensive'}</div><div class='ms'>{'Banyak sektor bantu' if ls_us['breadth']>=.55 else 'Masih sempit / pilih-pilih' if ls_us['breadth']>=.38 else 'Sedang buang beta / cari aman'}</div></div>",unsafe_allow_html=True)

    st.write("")
    st.markdown(f"<div class='aa'><b>Skenario konsentrasi:</b> {ls_us['line']} <span class='sm'>Kalau breadth membaik <b>dan</b> top-5 concentration turun, rally makin sehat. Kalau indeks tetap naik tapi concentration tetap tinggi dan tema pendukung sedikit, itu cenderung <b>narrow fake strength</b>. Kalau concentration rendah tapi breadth juga jelek, itu bukan broad rally — bisa cuma market tanpa pemimpin jelas atau distribusi yang merata.</span></div>",unsafe_allow_html=True)

    u1,u2=st.columns(2)
    with u1:
        st.markdown("**US — top impact leaders**")
        st.markdown(tb(["Ticker","Theme","Impact","State","Read"],impact_rows(lead_us,"top",7)),unsafe_allow_html=True)
        st.write("")
        st.markdown("**US — relative strength continuation**")
        st.markdown(tb(["Ticker","Theme","State","Read"],rs_rows(lead_us,7)),unsafe_allow_html=True)
    with u2:
        st.markdown("**IHSG — top impact leaders**")
        st.markdown(tb(["Ticker","Theme","Impact","State","Read"],impact_rows(lead_ih,"top",7)),unsafe_allow_html=True)
        st.write("")
        st.markdown("**IHSG — relative strength continuation**")
        st.markdown(tb(["Ticker","Theme","State","Read"],rs_rows(lead_ih,7)),unsafe_allow_html=True)

    st.write("")
    r1,r2=st.columns(2)
    with r1:
        st.markdown("**US rotation tape**")
        st.markdown(tb(["Theme","Flow","21d α","Coverage"],rotation_rows(lead_us,6)),unsafe_allow_html=True)
    with r2:
        st.markdown("**IHSG rotation tape**")
        st.markdown(tb(["Theme","Flow","21d α","Coverage"],rotation_rows(lead_ih,6)),unsafe_allow_html=True)

    with st.expander("Impact board detail", expanded=False):
        x1,x2=st.columns(2)
        with x1:
            st.markdown("**Top contributors (proxy)**")
            st.markdown(tb(["Ticker","Theme","Impact","State","Read"],impact_rows(lead_us,"top",10)+[["---","---","---","---","---"]]+impact_rows(lead_ih,"top",6)),unsafe_allow_html=True)
        with x2:
            st.markdown("**Top detractors (proxy)**")
            st.markdown(tb(["Ticker","Theme","Impact","State","Read"],impact_rows(lead_us,"bottom",10)+[["---","---","---","---","---"]]+impact_rows(lead_ih,"bottom",6)),unsafe_allow_html=True)

    wl=st.text_input("Custom relative strength watchlist",value="",key="wl11",placeholder="TSLA, COIN")
    eu=[t.strip().upper() for t in wl.split(",") if t.strip()];ued=scn(eu,"SPY","6mo") if eu else pd.DataFrame()
    if not ued.empty:
        ued["Theme"]="Custom"; ued=prep_lead(ued,"US")
        st.write("")
        st.markdown("**Custom watchlist — relative strength**")
        st.markdown(tb(["Ticker","Theme","State","Read"],rs_rows(ued,10)),unsafe_allow_html=True)

    with st.expander("Reference charts", expanded=False):
        ch1,ch2=st.columns(2)
        with ch1:
            spy=ycc("SPY","1y");iwm=ycc("IWM","1y")
            if not spy.empty and not iwm.empty:
                df=pd.concat([spy.rename("SPY"),iwm.rename("IWM")],axis=1).ffill().dropna();df=df/df.iloc[0]*100
                st.markdown("**SPY vs IWM**");st.line_chart(df)
        with ch2:
            gld=ycc("GLD","1y");uup=ycc("UUP","1y")
            if not gld.empty and not uup.empty:
                df=pd.concat([gld.rename("GLD"),uup.rename("UUP")],axis=1).ffill().dropna();df=df/df.iloc[0]*100
                st.markdown("**Gold vs USD**");st.line_chart(df)
    st.markdown("</div>",unsafe_allow_html=True)

# ---- RISK & EVENTS ----
with tabs[3]:
    st.markdown("<div class='card'><div class='st'>RISK & EVENTS</div>",unsafe_allow_html=True)
    r1,r2=st.columns(2)
    with r1:
        st.markdown(tb(["Metric","Now","Read"],[["Risk-on",pc(ro),epl(d.cn,d.fg)],["Risk-off",pc(rf),cml(rf)],["Crash",pc(cp),cml(cp)],["Stress G/I/L",f"{pc(d.ssg)}/{pc(d.si_)}/{pc(d.sl)}",""],["Top risk",pc(d.tsc2),"Extended" if d.tsc2>.55 else("Building" if d.tsc2>.35 else "Low")],["Transition",pc(d.tc),d.pa],["F&G",f"{fg_sc} ({fg_src})",fg_vb]]),unsafe_allow_html=True)
        yr=[]
        for k,l in[("DGS2","2Y"),("DGS10","10Y"),("DFII10","10Yr"),("T10Y2Y","Curve")]:
            v3=lv(S[k]);dl=ld(S[k],21)
            if not np.isfinite(v3): yr.append([l,"n/a","-"]);continue
            yr.append([l,f"{v3:.2f}%",f"{dl:+.2f}" if np.isfinite(dl) else "-"])
        st.markdown(tb(["Rate","Now","1m Δ"],yr),unsafe_allow_html=True)
    with r2:
        st.markdown("**Events**")
        evr=[]
        for e in evts[:8]:
            tag="🔴" if e["imp"]=="Critical" else("🟡" if e["imp"]=="High" else "⚪")
            evr.append([f"{tag} {e['name']}",e["cd"],e["axis"],e["impact"]])
        st.markdown(tb(["Event","In","Axis","Impact"],evr),unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)

# ---- AUDIT ----
with tabs[4]:
    st.markdown("<div class='card'><div class='st'>AUDIT</div>",unsafe_allow_html=True)
    qn={"Q1":("G↑","I↓"),"Q2":("G↑","I↑"),"Q3":("G↓","I↑"),"Q4":("G↓","I↓")}
    alr=[]
    for q,(gn2,in2) in qn.items():
        gm="✓" if(("↑" in gn2 and qr.gdir=="accelerating") or("↓" in gn2 and qr.gdir=="decelerating")) else("?" if qr.gdir=="flat" else "✗")
        im="✓" if(("↑" in in2 and qr.idir=="accelerating") or("↓" in in2 and qr.idir=="decelerating")) else("?" if qr.idir=="flat" else "✗")
        mk="◀" if gm=="✓" and im=="✓" else "";alr.append([q,gn2,gm,in2,im,pc(qr.mb[q]),mk])
    st.markdown(tb(["Q","G","G?","I","I?","Prob",""],alr),unsafe_allow_html=True)
    st.write("")
    st.markdown(tb(["Q","Off","Dir","Live","Final"],[[q,pc(qr.op[q]),pc(qr.dp[q]),pc(qr.lp[q]),pc(qr.mb[q])] for q in["Q1","Q2","Q3","Q4"]]),unsafe_allow_html=True)
    st.write("")
    c1,c2,c3=st.columns(3)
    with c1: st.markdown(tb(["G","Z"],[[k,f"{v:+.2f}"] for k,v in qr.gz.items()]),unsafe_allow_html=True)
    with c2: st.markdown(tb(["I","Z"],[[k,f"{v:+.2f}"] for k,v in qr.iz.items()]),unsafe_allow_html=True)
    with c3: st.markdown(tb(["S","Z"],[[k,f"{v:+.2f}"] for k,v in qr.sz.items()]),unsafe_allow_html=True)
    st.write("")
    # Historical
    qh=QUAD_HIST[d.cq]
    st.markdown("**Historical quad behavior**")
    st.markdown(tb(["","Value"],[["Quad",f"{d.cq} — {qh['l']}"],["Avg SPX",qh["spx"]],["Duration",qh["dur"]],["Best",qh["best"]],["Worst",qh["worst"]],["Win rate",qh["wr"]]]),unsafe_allow_html=True)
    st.write("")
    # Data freshness
    st.markdown("**Data freshness**")
    fresh=data_fresh()
    fr=[[l,dt,f"{days}d",pl("Fresh" if c=="g" else("Stale" if c=="r" else "Aging"),c)] for l,dt,days,c in fresh]
    st.markdown(tb(["Series","Last","Age","Status"],fr),unsafe_allow_html=True)
    st.write("")
    st.download_button("Export",json.dumps({"cq":d.cq,"cp":d.cp,"nq":d.nq,"gdir":qr.gdir,"idir":qr.idir,"variant":v2,"crash":cp,"fg":fg_sc,"position":psz,"intermarket":f"{im_n}/{im_t}","config":CF.h()},indent=2),file_name="v109.json")
    st.markdown("</div>",unsafe_allow_html=True)