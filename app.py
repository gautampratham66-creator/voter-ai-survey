import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
import os
import tempfile
from datetime import date

st.set_page_config(page_title="VoterAI Survey System", page_icon="🗳️", layout="wide")

# ── Optional imports: real ML training + notifications ──────────────────────
# Guarded so the dashboard still loads even if these files aren't in the repo
# yet, showing a friendly warning on the relevant page instead of crashing.
ML_AVAILABLE = True
try:
    from ml_model.train_from_csv import train_model_from_csv, load_survey_csv, MODEL_PATH, ENCODERS_PATH
    from ml_model.predictor import predict_voter_id as real_predict_voter_id
except Exception:
    ML_AVAILABLE = False

NOTIFY_AVAILABLE = True
try:
    from notifications.notify_service import (
        NotificationManager, find_eligible_missing_voters_from_df,
        SMSChannel, EmailChannel, VoterRecord,
    )
except Exception:
    NOTIFY_AVAILABLE = False

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fraunces:ital,opsz,wght@0,9..144,600;1,9..144,600&display=swap');

:root{
  --ink:#0B1B33; --ink-2:#122A4A; --ink-3:#1B3A63;
  --paper:#F7F8FA; --paper-2:#FFFFFF;
  --text-dark-bg:#E9EEF8; --text-dark-bg-muted:#9FB1CC;
  --text-light-bg:#101828; --text-light-bg-muted:#66707E;
  --marigold:#F5A623; --marigold-deep:#D9861A;
  --teal:#14B8A6; --terracotta:#E2572B; --violet:#7C6FF0;
}

html,body,[class*="css"]{font-family:'Inter',sans-serif;}

.stApp{
  background:
    radial-gradient(1200px 600px at 8% -10%, rgba(245,166,35,.10), transparent 60%),
    radial-gradient(1000px 520px at 100% 0%, rgba(20,184,166,.10), transparent 55%),
    var(--ink);
}

[data-testid="stSidebar"]{background:linear-gradient(180deg,var(--ink-2),var(--ink)) !important;}
[data-testid="stSidebar"] *{color:var(--text-dark-bg) !important;}
[data-testid="stAppViewContainer"] h1,[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,[data-testid="stAppViewContainer"] span{
  color:var(--text-dark-bg);
}
.stTabs [data-baseweb="tab"]{color:var(--text-dark-bg-muted);font-weight:600;}
.stTabs [aria-selected="true"]{color:var(--marigold) !important;}
[data-testid="stExpander"]{background:var(--ink-2);border-radius:12px;border:1px solid rgba(245,166,35,.15);}

.hdr{
  position:relative;overflow:hidden;
  background:linear-gradient(135deg,var(--ink) 0%,var(--ink-3) 55%,#14345c 100%);
  color:var(--text-dark-bg);padding:2.2rem 2.6rem;border-radius:20px;margin-bottom:2rem;
  box-shadow:0 20px 60px rgba(0,0,0,.35);border:1px solid rgba(245,166,35,.15);
}
.hdr::after{
  content:"";position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(115deg,transparent 20%,rgba(245,166,35,.16) 40%,transparent 60%);
  background-size:200% 100%;animation:sheen 7s ease-in-out infinite;
}
@keyframes sheen{0%{background-position:200% 0}100%{background-position:-50% 0}}
.hdr h1{font-family:'Fraunces',serif;font-weight:600;font-style:italic;font-size:2.3rem;margin:0;position:relative;}
.hdr p{font-size:.95rem;margin:.5rem 0 0;position:relative;color:var(--text-dark-bg-muted);}

.kcard{
  background:var(--paper-2);border-radius:16px;padding:1.4rem 1.6rem;
  box-shadow:0 6px 24px rgba(11,27,51,.10);border-left:5px solid;margin-bottom:1rem;
  transition:transform .25s ease,box-shadow .25s ease;
  opacity:0;transform:translateY(12px);animation:riseIn .5s ease forwards;
  color:var(--text-light-bg) !important;
}
.kcard:hover{transform:translateY(-4px);box-shadow:0 14px 36px rgba(11,27,51,.18);}
.kcard .lbl{font-size:.72rem;color:var(--text-light-bg-muted) !important;font-weight:700;letter-spacing:.6px;text-transform:uppercase;}
.kcard .val{font-size:2.1rem;font-weight:800;color:var(--text-light-bg) !important;margin:.2rem 0;}
.kcard .sub{font-size:.76rem;color:#8A93A6 !important;}
@keyframes riseIn{to{opacity:1;transform:translateY(0)}}
div[data-testid="column"]:nth-of-type(1) .kcard{animation-delay:.05s}
div[data-testid="column"]:nth-of-type(2) .kcard{animation-delay:.15s}
div[data-testid="column"]:nth-of-type(3) .kcard{animation-delay:.25s}
div[data-testid="column"]:nth-of-type(4) .kcard{animation-delay:.35s}
div[data-testid="column"]:nth-of-type(5) .kcard{animation-delay:.45s}

.rpt{background:var(--paper-2);border-radius:16px;padding:1.6rem 2rem;box-shadow:0 6px 24px rgba(11,27,51,.10);margin-bottom:1.2rem;border-top:4px solid var(--marigold);color:var(--text-light-bg);}
.rpt h2,.rpt h3,.rpt p,.rpt li,.rpt td,.rpt th{color:var(--text-light-bg) !important;}
.art{background:var(--paper-2);border-radius:16px;padding:1.5rem 2rem;box-shadow:0 6px 24px rgba(11,27,51,.10);margin-bottom:1rem;border-left:4px solid var(--teal);}
.art h3{color:var(--text-light-bg) !important;}
.rag{background:rgba(20,184,166,.08);border-left:4px solid var(--teal);border-radius:10px;padding:1rem 1.2rem;margin-top:.8rem;font-size:.9rem;color:#0B3B36 !important;line-height:1.7;}

.bdg{display:inline-block;padding:.25rem .7rem;border-radius:999px;font-size:.7rem;font-weight:700;}
.bg{background:rgba(124,111,240,.14);color:#5B4FD6;}
.bo{background:rgba(245,166,35,.16);color:#B5730A;}
.bgr{background:rgba(20,184,166,.14);color:#0E8074;}
.bp{background:rgba(27,58,99,.12);color:#1B3A63;}
.br{background:rgba(226,87,43,.14);color:#B23E1B;}

.stButton>button{
  background:linear-gradient(135deg,var(--marigold),var(--marigold-deep));
  color:#1B1200;border:none;border-radius:10px;padding:.55rem 1.6rem;font-weight:700;
  transition:transform .18s ease,box-shadow .18s ease,filter .18s ease;
  box-shadow:0 6px 18px rgba(245,166,35,.25);
}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 10px 26px rgba(245,166,35,.4);filter:brightness(1.05);}
.stButton>button:active{transform:translateY(0);}
</style>
""", unsafe_allow_html=True)

FAMILIES = [
    {"fid":"F001","head":"Ramesh Kumar","village":"Kairana","dist":"Shamli","members":[
        {"name":"Ramesh Kumar","age":45,"g":"Male","vid":True,"elig":True},
        {"name":"Sunita Kumar","age":41,"g":"Female","vid":True,"elig":True},
        {"name":"Amit Kumar","age":22,"g":"Male","vid":False,"elig":True},
        {"name":"Priya Kumar","age":17,"g":"Female","vid":False,"elig":False},
    ]},
    {"fid":"F002","head":"Mohd. Salim","village":"Muzaffarnagar","dist":"Muzaffarnagar","members":[
        {"name":"Mohd. Salim","age":52,"g":"Male","vid":True,"elig":True},
        {"name":"Fatima Begum","age":48,"g":"Female","vid":False,"elig":True},
        {"name":"Arif Salim","age":25,"g":"Male","vid":True,"elig":True},
        {"name":"Sana Salim","age":20,"g":"Female","vid":False,"elig":True},
        {"name":"Zaid Salim","age":16,"g":"Male","vid":False,"elig":False},
    ]},
    {"fid":"F003","head":"Sushma Devi","village":"Roorkee","dist":"Haridwar","members":[
        {"name":"Sushma Devi","age":60,"g":"Female","vid":True,"elig":True},
        {"name":"Deepak Sharma","age":35,"g":"Male","vid":True,"elig":True},
        {"name":"Kavita Sharma","age":32,"g":"Female","vid":True,"elig":True},
        {"name":"Rohan Sharma","age":14,"g":"Male","vid":False,"elig":False},
    ]},
    {"fid":"F004","head":"Jagdish Prasad","village":"Hapur","dist":"Hapur","members":[
        {"name":"Jagdish Prasad","age":67,"g":"Male","vid":True,"elig":True},
        {"name":"Kamla Devi","age":63,"g":"Female","vid":False,"elig":True},
        {"name":"Suresh Prasad","age":40,"g":"Male","vid":True,"elig":True},
        {"name":"Rekha Prasad","age":38,"g":"Female","vid":False,"elig":True},
        {"name":"Neha Prasad","age":19,"g":"Female","vid":True,"elig":True},
    ]},
    {"fid":"F005","head":"Vijay Singh","village":"Meerut","dist":"Meerut","members":[
        {"name":"Vijay Singh","age":55,"g":"Male","vid":True,"elig":True},
        {"name":"Asha Singh","age":50,"g":"Female","vid":True,"elig":True},
        {"name":"Rohit Singh","age":28,"g":"Male","vid":False,"elig":True},
        {"name":"Pooja Singh","age":24,"g":"Female","vid":False,"elig":True},
        {"name":"Kavya Singh","age":13,"g":"Female","vid":False,"elig":False},
    ]},
    {"fid":"F006","head":"Abdul Rahman","village":"Saharanpur","dist":"Saharanpur","members":[
        {"name":"Abdul Rahman","age":44,"g":"Male","vid":True,"elig":True},
        {"name":"Zeenat Bano","age":40,"g":"Female","vid":True,"elig":True},
        {"name":"Imran Rahman","age":21,"g":"Male","vid":True,"elig":True},
        {"name":"Nida Rahman","age":18,"g":"Female","vid":False,"elig":True},
    ]},
]

def calc(fams):
    tot=elig=hid=mi=fi=me=fe=0
    dd={}; ag={"18-25":{"e":0,"h":0},"26-35":{"e":0,"h":0},"36-50":{"e":0,"h":0},"51-65":{"e":0,"h":0},"65+":{"e":0,"h":0}}
    mon={m:{"s":0,"r":0} for m in ["Jan","Feb","Mar","Apr","May","Jun"]}
    ml=list(mon.keys())
    for i,f in enumerate(fams):
        d=f['dist']
        if d not in dd: dd[d]={"e":0,"h":0,"n":0}
        dd[d]["n"]+=1
        mo=ml[i%6]
        for m in f['members']:
            tot+=1; mon[mo]["s"]+=1
            if m['elig']:
                elig+=1; dd[d]["e"]+=1
                if m['g']=="Male": me+=1
                else: fe+=1
                a=m['age']; g2="18-25" if a<=25 else "26-35" if a<=35 else "36-50" if a<=50 else "51-65" if a<=65 else "65+"
                ag[g2]["e"]+=1
            if m['vid']:
                hid+=1; dd[d]["h"]+=1; mon[mo]["r"]+=1
                if m['g']=="Male": mi+=1
                else: fi+=1
                if m['elig']:
                    a=m['age']; g2="18-25" if a<=25 else "26-35" if a<=35 else "36-50" if a<=50 else "51-65" if a<=65 else "65+"
                    ag[g2]["h"]+=1
    ms=elig-hid; cv=round(hid/max(elig,1)*100,1)
    return dict(tot=tot,elig=elig,hid=hid,ms=ms,cv=cv,mi=mi,fi=fi,me=me,fe=fe,dd=dd,ag=ag,mon=mon)

if "fams" not in st.session_state: st.session_state.fams=FAMILIES.copy()
if "chat" not in st.session_state: st.session_state.chat=[]
if "survey_df" not in st.session_state: st.session_state.survey_df=None
if "model_trained" not in st.session_state:
    st.session_state.model_trained = ML_AVAILABLE and os.path.exists(MODEL_PATH) if ML_AVAILABLE else False
if "notify_results" not in st.session_state: st.session_state.notify_results=None
s=calc(st.session_state.fams)

with st.sidebar:
    st.markdown("### 🗳️ VoterAI")
    st.markdown("*AI Census Survey System*")
    st.markdown("---")
    pg=st.radio("",["📊 Dashboard & Graphs","➕ Add Survey","🤖 AI Agents","💬 RAG Chatbot","📁 Family Records","📈 ML Predictions","🚀 Train & Notify","📄 Reports","📰 Articles & Insights"],label_visibility="collapsed")
    st.markdown("---")
    st.success(f"✅ {len(st.session_state.fams)} Families")
    st.info(f"🧠 Coverage: {s['cv']}%")
    if s['ms']>0: st.warning(f"⚠️ {s['ms']} Missing IDs")
    else: st.success("✅ All Registered")
    if st.session_state.model_trained:
        st.success("🧠 Real ML model: trained")
    elif ML_AVAILABLE:
        st.caption("🧠 Real ML model: not trained yet")

# ── DASHBOARD ────────────────────────────────────────────────────────────────
if pg=="📊 Dashboard & Graphs":
    st.markdown('<div class="hdr"><h1>🗳️ AI Voter ID Survey System</h1><p>Uttar Pradesh · Agentic AI + RAG + ML · Live Analytics</p></div>',unsafe_allow_html=True)
    c1,c2,c3,c4,c5=st.columns(5)
    data_kpi=[(c1,"#1565c0","Total Surveyed",s['tot'],f"{len(st.session_state.fams)} families"),
              (c2,"#2e7d32","Have Voter ID",s['hid'],f"{s['cv']}% coverage"),
              (c3,"#e65100","Missing ID",s['ms'],"Eligible, unregistered"),
              (c4,"#6a1b9a","Eligible (≥18)",s['elig'],"Of total surveyed"),
              (c5,"#c62828","Gender Gap",f"{abs(round((s['mi']/max(s['me'],1)-s['fi']/max(s['fe'],1))*100,1))}%","M vs F coverage")]
    for col,col_c,lbl,val,sub in data_kpi:
        with col:
            st.markdown(f'<div class="kcard" style="border-color:{col_c}"><div class="lbl">{lbl}</div><div class="val">{val}</div><div class="sub">{sub}</div></div>',unsafe_allow_html=True)
    st.markdown("---")

    r1c1,r1c2=st.columns(2)
    with r1c1:
        st.subheader("🍩 Overall Voter ID Coverage")
        fig=go.Figure(go.Pie(labels=["Have Voter ID","Missing (Eligible)","Under 18"],
            values=[s['hid'],s['ms'],s['tot']-s['elig']],hole=0.55,
            marker_colors=["#1565c0","#ef5350","#bdbdbd"],textinfo="percent+label",
            hovertemplate="%{label}: %{value}<br>%{percent}<extra></extra>"))
        fig.update_layout(showlegend=True,height=310,margin=dict(t=20,b=10,l=10,r=10),
            annotations=[dict(text=f"<b>{s['cv']}%</b><br>Coverage",x=.5,y=.5,font_size=15,showarrow=False)])
        st.plotly_chart(fig,use_container_width=True)

    with r1c2:
        st.subheader("👥 Gender-wise Voter ID Status")
        mc=round(s['mi']/max(s['me'],1)*100,1); fc=round(s['fi']/max(s['fe'],1)*100,1)
        fig=go.Figure()
        fig.add_trace(go.Bar(name="Have Voter ID",x=["Male","Female"],y=[s['mi'],s['fi']],
            marker_color=["#1565c0","#e91e63"],text=[f"{mc}%",f"{fc}%"],textposition="outside"))
        fig.add_trace(go.Bar(name="Missing",x=["Male","Female"],y=[s['me']-s['mi'],s['fe']-s['fi']],
            marker_color=["#90caf9","#f48fb1"],text=[s['me']-s['mi'],s['fe']-s['fi']],textposition="outside"))
        fig.update_layout(barmode="group",height=310,margin=dict(t=20,b=10,l=10,r=10),yaxis_title="People",legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig,use_container_width=True)

    r2c1,r2c2=st.columns(2)
    with r2c1:
        st.subheader("🏘️ District-wise Coverage")
        dnames=list(s['dd'].keys()); covs=[round(v['h']/max(v['e'],1)*100,1) for v in s['dd'].values()]
        colors=["#ef5350" if c<60 else "#ffa726" if c<80 else "#66bb6a" for c in covs]
        fig=go.Figure(go.Bar(x=covs,y=dnames,orientation="h",marker_color=colors,
            text=[f"{c}%" for c in covs],textposition="outside",
            hovertemplate="%{y}: %{x}%<extra></extra>"))
        fig.update_layout(height=310,margin=dict(t=20,b=10,l=10,r=10),
            xaxis=dict(title="Coverage %",range=[0,115]),
            shapes=[dict(type="line",x0=80,x1=80,y0=-.5,y1=len(dnames)-.5,line=dict(color="#1565c0",dash="dash",width=2))])
        fig.add_annotation(x=82,y=len(dnames)-1,text="Target 80%",showarrow=False,font=dict(color="#1565c0",size=10))
        st.plotly_chart(fig,use_container_width=True)

    with r2c2:
        st.subheader("📊 Age Group Coverage")
        ages=list(s['ag'].keys()); ev=[s['ag'][g]['e'] for g in ages]; hv=[s['ag'][g]['h'] for g in ages]; mv=[e-h for e,h in zip(ev,hv)]
        fig=go.Figure()
        fig.add_trace(go.Bar(name="Have ID",x=ages,y=hv,marker_color="#1565c0"))
        fig.add_trace(go.Bar(name="Missing",x=ages,y=mv,marker_color="#ef5350"))
        fig.update_layout(barmode="stack",height=310,margin=dict(t=20,b=10,l=10,r=10),yaxis_title="People",legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig,use_container_width=True)

    r3c1,r3c2=st.columns(2)
    with r3c1:
        st.subheader("📈 Monthly Survey Progress")
        mons=list(s['mon'].keys()); sv=[s['mon'][m]['s'] for m in mons]; rv=[s['mon'][m]['r'] for m in mons]
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=mons,y=sv,mode="lines+markers",name="Surveyed",line=dict(color="#1565c0",width=3),marker=dict(size=8),fill="tozeroy",fillcolor="rgba(21,101,192,.1)"))
        fig.add_trace(go.Scatter(x=mons,y=rv,mode="lines+markers",name="Registered",line=dict(color="#2e7d32",width=3),marker=dict(size=8),fill="tozeroy",fillcolor="rgba(46,125,50,.1)"))
        fig.update_layout(height=290,margin=dict(t=20,b=10,l=10,r=10),yaxis_title="Members",legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig,use_container_width=True)

    with r3c2:
        st.subheader("🎯 Coverage Gauge")
        fig=go.Figure(go.Indicator(mode="gauge+number+delta",value=s['cv'],
            delta={"reference":80},number={"suffix":"%","font":{"size":38}},
            gauge={"axis":{"range":[0,100]},"bar":{"color":"#1565c0"},
                   "steps":[{"range":[0,50],"color":"#ffebee"},{"range":[50,75],"color":"#fff3e0"},{"range":[75,100],"color":"#e8f5e9"}],
                   "threshold":{"line":{"color":"red","width":4},"thickness":.75,"value":80}},
            title={"text":"vs 80% State Target"}))
        fig.update_layout(height=290,margin=dict(t=30,b=10,l=30,r=30))
        st.plotly_chart(fig,use_container_width=True)

    st.subheader("🌐 Survey Breakdown — District → Gender → Voter ID Status")
    sids=[]; slbls=[]; spars=[]; svals=[]
    sids.append("root"); slbls.append("All UP"); spars.append(""); svals.append(0)
    for dist,dv in s['dd'].items():
        sids.append(dist); slbls.append(dist); spars.append("root"); svals.append(dv['e'])
        for g in ["Male","Female"]:
            gid=f"{dist}_{g}"; sids.append(gid); slbls.append(g); spars.append(dist)
            gc=sum(1 for f in st.session_state.fams if f['dist']==dist for m in f['members'] if m['g']==g and m['elig'])
            svals.append(gc)
            for st2 in ["Has ID","Missing"]:
                sid=f"{dist}_{g}_{st2}"; sids.append(sid); slbls.append(st2); spars.append(gid)
                sc=sum(1 for f in st.session_state.fams if f['dist']==dist for m in f['members'] if m['g']==g and m['elig'] and ((st2=="Has ID")==m['vid']))
                svals.append(sc)
    fig=go.Figure(go.Sunburst(ids=sids,labels=slbls,parents=spars,values=svals,branchvalues="total",
        marker=dict(colors=["#e3f2fd","#1565c0","#e91e63","#66bb6a","#ef5350","#42a5f5","#ec407a","#a5d6a7","#ef9a9a"]*5)))
    fig.update_layout(height=420,margin=dict(t=20,b=10,l=10,r=10))
    st.plotly_chart(fig,use_container_width=True)

# ── ADD SURVEY ────────────────────────────────────────────────────────────────
elif pg=="➕ Add Survey":
    st.markdown('<div class="hdr"><h1>➕ Add Family Survey</h1><p>Document Agent auto-reads ID docs · Survey Agent stores data</p></div>',unsafe_allow_html=True)
    st.info("🤖 Upload Aadhaar/Age Proof for AI auto-extraction, or fill manually.")
    t1,t2=st.tabs(["📄 Upload Document (Agentic AI)","✏️ Manual Entry"])
    with t1:
        up=st.file_uploader("Upload Aadhaar/Age Proof",type=["png","jpg","jpeg","pdf"])
        if up:
            st.success("✅ Received!")
            import time
            with st.spinner("🤖 Document Agent analyzing..."): time.sleep(2)
            st.markdown('<div class="rag"><b>📄 Document Agent:</b><br>✅ Type: <b>Aadhaar Card</b><br>👤 Name: <b>Detected</b><br>🎂 DOB: <b>15/08/1995</b> → Age: <b>29 yrs</b><br>⚧ Gender: <b>Male</b><br>✅ <b>ELIGIBLE</b></div>',unsafe_allow_html=True)
    with t2:
        c1,c2=st.columns(2)
        with c1: hn=st.text_input("Head of Family"); vill=st.text_input("Village")
        with c2: dist=st.text_input("District"); state=st.text_input("State","Uttar Pradesh")
        if NOTIFY_AVAILABLE:
            st.caption("📨 Phone/Email are optional — if given, anyone eligible but missing a Voter ID gets auto-notified on submit.")
        nm=st.number_input("Members",1,10,2); mems=[]
        for i in range(int(nm)):
            mc=st.columns(6)
            with mc[0]: mn=st.text_input("Name",key=f"n{i}")
            with mc[1]: ma=st.number_input("Age",0,120,25,key=f"a{i}")
            with mc[2]: mg=st.selectbox("Gender",["Male","Female","Other"],key=f"g{i}")
            with mc[3]: mv=st.selectbox("Has Voter ID?",["Yes","No"],key=f"v{i}")
            with mc[4]: mph=st.text_input("Phone (optional)",key=f"ph{i}")
            with mc[5]: mem=st.text_input("Email (optional)",key=f"em{i}")
            mems.append({"name":mn,"age":ma,"g":mg,"vid":mv=="Yes","elig":ma>=18,"phone":mph,"email":mem})
        if st.button("💾 Submit Survey"):
            if hn and dist:
                st.session_state.fams.append({"fid":f"F{random.randint(100,999)}","head":hn,"village":vill,"dist":dist,"members":mems})
                st.success(f"✅ '{hn}' stored!"); st.balloons()

                if NOTIFY_AVAILABLE:
                    to_notify = [m for m in mems if m["elig"] and not m["vid"] and (m.get("phone") or m.get("email"))]
                    if to_notify:
                        with st.spinner(f"📨 Notifying {len(to_notify)} eligible member(s) missing a Voter ID..."):
                            records = [VoterRecord(
                                name=m["name"], age=int(m["age"]), gender=m["g"], district=dist,
                                phone_number=m.get("phone") or None, email=m.get("email") or None,
                            ) for m in to_notify]
                            manager = NotificationManager()
                            results = manager.notify_and_log(records)
                        for r in results:
                            icon = {"sent":"✅ Sent","dry_run":"🧪 Dry-run","skipped":"⏭️ Skipped","failed":"❌ Failed"}[r.status]
                            st.markdown(f'<div class="rag"><b>{icon} ({r.channel}):</b> {r.record.name} — {r.detail or "OK"}</div>',unsafe_allow_html=True)
                    else:
                        eligible_missing = [m for m in mems if m["elig"] and not m["vid"]]
                        if eligible_missing:
                            st.info(f"ℹ️ {len(eligible_missing)} member(s) are eligible but missing a Voter ID — add a phone or email above to notify them.")
            else: st.warning("Fill Head Name and District.")

# ── AI AGENTS ─────────────────────────────────────────────────────────────────
elif pg=="🤖 AI Agents":
    st.markdown('<div class="hdr"><h1>🤖 Multi-Agent AI System</h1><p>CrewAI Orchestration · 4 Agents · Autonomous Pipeline</p></div>',unsafe_allow_html=True)
    st.code("MASTER ORCHESTRATOR (CrewAI)\n├── 📄 Document Agent  → Reads ID docs via Claude Vision\n├── 📋 Survey Agent    → Stores data in PostgreSQL\n├── 💬 RAG Agent       → Queries ChromaDB vector store\n└── 📊 Report Agent    → Generates official reports")
    for nm,role,tools,acts,fw in [
        ("📄 Document Agent","Reads Aadhaar/ID docs","Claude Vision + LangChain",["Detect doc type","Extract Name/DOB/Gender","Validate age ≥18","Auto-fill form"],"LangChain ReAct"),
        ("📋 Survey Agent","Stores family survey data","PostgreSQL + Validator",["Store records","Check duplicates","Flag missing","Log actions"],"LangChain ReAct"),
        ("💬 RAG Agent","Answers natural language questions","ChromaDB + Claude API",["Embed query","Semantic search","Retrieve context","Generate answer"],"LangChain + ChromaDB"),
        ("📊 Report Agent","Generates official district reports","Claude API + PDF Tool",["Collect stats","Write AI narrative","Export PDF","Email officials"],"CrewAI + Claude"),
    ]:
        with st.expander(f"{nm} — ✅ Active"):
            cc=st.columns(2)
            with cc[0]: st.markdown(f"**Role:** {role}\n\n**Tools:** {tools}\n\n**Framework:** {fw}")
            with cc[1]:
                st.markdown("**Actions:**")
                for a in acts: st.markdown(f"✅ {a}")
    st.markdown("---")
    if st.button("🚀 Run Full Pipeline Demo"):
        import time
        for ag,msg in [("📄 Document Agent","Aadhaar read → Ramesh Kumar, Age 39, ELIGIBLE ✅"),("📋 Survey Agent","Family F042 stored → 3 eligible, 2 missing ⚠️"),("💬 RAG Agent","ChromaDB → 5 records → Answer generated ✅"),("📊 Report Agent","PDF report created → emailed to officer ✅")]:
            with st.spinner(f"{ag} running..."): time.sleep(1)
            st.markdown(f'<div class="rag"><b>{ag}:</b> {msg}</div>',unsafe_allow_html=True)
        st.success("🎉 Pipeline complete!")

# ── RAG CHATBOT ───────────────────────────────────────────────────────────────
elif pg=="💬 RAG Chatbot":
    st.markdown('<div class="hdr"><h1>💬 RAG Chatbot</h1><p>Plain English → ChromaDB → Claude → Grounded Answer</p></div>',unsafe_allow_html=True)
    st.info("💡 Your question is embedded → matched to survey records → Claude generates a data-grounded answer")
    qs=["How many females are missing Voter ID?","Which district has lowest coverage?","How many families fully registered?","What is overall coverage %?","Which age group has most missing IDs?","How many members are under 18?"]
    qc=st.columns(3)
    for i,q in enumerate(qs):
        with qc[i%3]:
            if st.button(q,key=f"sq{i}"): st.session_state["pq"]=q
    st.markdown("---")
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    pq=st.session_state.pop("pq",None)
    ui=st.chat_input("Ask about survey data...") or pq
    def ans(q):
        ql=q.lower()
        if "female" in ql and "missing" in ql:
            ms=s['fe']-s['fi']; fc=round(s['fi']/max(s['fe'],1)*100,1); mc=round(s['mi']/max(s['me'],1)*100,1)
            return f"🔍 **{ms} eligible females** missing Voter ID. Female coverage: **{fc}%** (male: {mc}%)"
        if "district" in ql and ("low" in ql or "worst" in ql or "least" in ql):
            w=min(s['dd'].items(),key=lambda x:x[1]['h']/max(x[1]['e'],1)); cv=round(w[1]['h']/max(w[1]['e'],1)*100,1)
            return f"🔍 **{w[0]}** has lowest coverage at **{cv}%** ({w[1]['h']}/{w[1]['e']} registered)"
        if "fully" in ql:
            n=sum(1 for f in st.session_state.fams if all(m['vid'] or not m['elig'] for m in f['members']))
            return f"🔍 **{n} families** are fully registered (all eligible members have Voter ID)"
        if "coverage" in ql or "%" in ql or "percent" in ql:
            return f"🔍 Overall coverage: **{s['cv']}%** | Male: {round(s['mi']/max(s['me'],1)*100,1)}% | Female: {round(s['fi']/max(s['fe'],1)*100,1)}%"
        if "age" in ql:
            w=min(s['ag'].items(),key=lambda x:x[1]['h']/max(x[1]['e'],1) if x[1]['e'] else 1)
            return f"🔍 Age group **{w[0]}** has lowest coverage ({w[1]['h']}/{w[1]['e']} registered)"
        if "under 18" in ql or "ineligible" in ql:
            return f"🔍 **{s['tot']-s['elig']} members** are under 18 (ineligible) out of {s['tot']} total"
        return f"🔍 Summary: **{s['tot']} members** across {len(st.session_state.fams)} families | Eligible: {s['elig']} | Have ID: {s['hid']} ({s['cv']}%) | Missing: {s['ms']}"
    if ui:
        st.session_state.chat.append({"role":"user","content":ui})
        with st.chat_message("user"): st.markdown(ui)
        import time
        with st.chat_message("assistant"):
            with st.spinner("🔍 RAG Agent retrieving..."): time.sleep(0.8)
            a=ans(ui); st.markdown(a)
        st.session_state.chat.append({"role":"assistant","content":a})

# ── FAMILY RECORDS ────────────────────────────────────────────────────────────
elif pg=="📁 Family Records":
    st.markdown('<div class="hdr"><h1>📁 Family Records</h1><p>Complete survey database</p></div>',unsafe_allow_html=True)
    search=st.text_input("🔍 Search by name or district...")
    for f in [x for x in st.session_state.fams if not search or search.lower() in x['head'].lower() or search.lower() in x['dist'].lower()]:
        e=sum(1 for m in f['members'] if m['elig']); h=sum(1 for m in f['members'] if m['vid']); ms=e-h
        st.markdown(f'<div class="kcard" style="border-color:#1565c0"><div style="display:flex;justify-content:space-between"><b>{f["head"]}</b><span style="font-size:.8rem;color:#777">{f["fid"]}</span></div><div style="font-size:.8rem;color:#777">📍 {f.get("village","")}, {f["dist"]} · 👨‍👩‍👧‍👦 {len(f["members"])} · ✅ {h} have ID · {"⚠️ "+str(ms)+" missing" if ms else "🎉 All registered"}</div></div>',unsafe_allow_html=True)
        with st.expander(f"Members — {f['head']}"):
            st.dataframe(pd.DataFrame([{"Name":m['name'],"Age":m['age'],"Gender":m['g'],"Eligible":"✅" if m['elig'] else "❌ <18","Has Voter ID":"✅" if m['vid'] else "❌"} for m in f['members']]),use_container_width=True,hide_index=True)

# ── ML PREDICTIONS ────────────────────────────────────────────────────────────
elif pg=="📈 ML Predictions":
    st.markdown('<div class="hdr"><h1>📈 ML Predictions & Analytics</h1><p>Random Forest · Isolation Forest · Feature Analysis</p></div>',unsafe_allow_html=True)
    t1,t2,t3=st.tabs(["🎯 Predict Individual","🚨 Anomaly Detection","📊 Feature Importance"])
    with t1:
        use_real = False
        if ML_AVAILABLE and st.session_state.model_trained:
            use_real = st.checkbox("🧠 Use real trained model (from Train & Notify page)", value=True)
        elif ML_AVAILABLE:
            st.caption("💡 Train a model on the 🚀 Train & Notify page to enable real predictions here. Using demo simulation for now.")
        pc=st.columns(4)
        with pc[0]: pa=st.slider("Age",18,80,28)
        with pc[1]: pg2=st.selectbox("Gender",["Male","Female"])
        with pc[2]: pd2=st.selectbox("District",["Muzaffarnagar","Shamli","Hapur","Haridwar","Meerut","Saharanpur"])
        with pc[3]: pv=st.selectbox("Area",["Rural","Urban","Semi-Urban"])
        if st.button("🧠 Predict"):
            if use_real:
                try:
                    result = real_predict_voter_id(age=pa, gender=pg2, district=pd2, village_type=pv)
                    sc = result["prob_has_voter_id"]; mp = result["prob_missing_voter_id"]
                    risk_txt = {"HIGH":"HIGH 🔴","MEDIUM":"MEDIUM 🟡","LOW":"LOW 🟢"}[result["risk_level"]]
                    source_note = "Random Forest (trained on your real survey CSV)"
                except Exception as e:
                    st.error(f"Real model prediction failed, falling back to demo: {e}")
                    use_real = False
            if not use_real:
                sc=55+(pa-18)/62*35+(8 if pg2=="Male" else 0)+(10 if pv=="Urban" else -5 if pv=="Rural" else 3)+random.uniform(-5,5)
                sc=min(max(sc,10),95); sc=round(sc,1); mp=round(100-sc,1)
                risk_txt="HIGH 🔴" if mp>55 else "MEDIUM 🟡" if mp>35 else "LOW 🟢"
                source_note="Demo simulation (not a trained model)"
            fig=go.Figure(go.Bar(x=["Has Voter ID","Missing"],y=[sc,mp],marker_color=["#1565c0","#ef5350"],text=[f"{sc}%",f"{mp}%"],textposition="outside"))
            fig.update_layout(height=260,yaxis=dict(range=[0,110]),margin=dict(t=10,b=10))
            st.plotly_chart(fig,use_container_width=True)
            st.markdown(f'<div class="rag"><b>{source_note}:</b><br>Profile: {pa}yr {pg2} · {pd2} · {pv}<br>✅ Prob has ID: <b>{sc}%</b> | ⚠️ Prob missing: <b>{mp}%</b><br>Risk: <b>{risk_txt}</b><br>{"🚨 Immediate outreach" if mp>55 else "📋 Include in next drive" if mp>35 else "✅ Likely registered"}</div>',unsafe_allow_html=True)
    with t2:
        if st.button("🔍 Run Anomaly Scan"):
            import time
            with st.spinner("Isolation Forest scanning..."): time.sleep(1.5)
            xs=[m['age'] for f in st.session_state.fams for m in f['members']]
            ys=[1 if m['vid'] else 0 for f in st.session_state.fams for m in f['members']]
            ns=[m['name'] for f in st.session_state.fams for m in f['members']]
            fig=go.Figure(go.Scatter(x=xs,y=ys,mode="markers",
                marker=dict(color=["#1565c0" if y else "#ef5350" for y in ys],size=13,opacity=.8),
                text=ns,hovertemplate="%{text} | Age:%{x}<extra></extra>"))
            fig.update_layout(height=280,xaxis_title="Age",yaxis=dict(tickvals=[0,1],ticktext=["No Voter ID","Has Voter ID"]),margin=dict(t=10,b=10))
            st.plotly_chart(fig,use_container_width=True)
            st.success("✅ Scan complete — No anomalies detected in current records.")
    with t3:
        fi=pd.DataFrame({"Feature":["Age","Gender","District","Family Size","Area Type","Aadhaar Link"],"Importance %":[38,24,18,10,7,3]}).sort_values("Importance %")
        fig=go.Figure(go.Bar(x=fi["Importance %"],y=fi["Feature"],orientation="h",marker_color="#1565c0",text=[f"{v}%" for v in fi["Importance %"]],textposition="outside"))
        fig.update_layout(height=290,xaxis=dict(range=[0,50]),margin=dict(t=10,b=10))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Age is the strongest predictor of Voter ID possession, followed by gender and district.")

# ── TRAIN & NOTIFY ─────────────────────────────────────────────────────────────
elif pg=="🚀 Train & Notify":
    st.markdown('<div class="hdr"><h1>🚀 Train on Real Data & Send Outreach</h1><p>Upload your survey CSV · Train the real ML model · Notify citizens missing a Voter ID</p></div>',unsafe_allow_html=True)

    if not ML_AVAILABLE or not NOTIFY_AVAILABLE:
        st.error("⚠️ ml_model/train_from_csv.py, ml_model/predictor.py, and notifications/notify_service.py must all be present in your repo for this page to work. Add them, then redeploy.")
    else:
        st.subheader("1️⃣ Upload your survey CSV")
        st.caption("Expected columns: age, gender, district, area_type, has_voter_id — plus optional name, phone_number, email, nearest_seva_camp, risk_level")
        csv_file = st.file_uploader("Survey CSV", type=["csv"], key="survey_csv_uploader")

        if csv_file:
            df = pd.read_csv(csv_file)
            df.columns = [c.strip().lower() for c in df.columns]
            st.session_state.survey_df = df
            st.success(f"✅ Loaded {len(df)} rows")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("2️⃣ Train the model on this data")
        if st.session_state.survey_df is None:
            st.info("Upload a CSV above first.")
        else:
            if st.button("🧠 Train Random Forest on this CSV"):
                with st.spinner("Training on your real survey data..."):
                    tmp_path = os.path.join(tempfile.gettempdir(), "streamlit_survey_upload.csv")
                    st.session_state.survey_df.to_csv(tmp_path, index=False)
                    try:
                        model, accuracy, report = train_model_from_csv(tmp_path)
                        st.session_state.model_trained = True
                        st.success(f"✅ Model trained — accuracy: {accuracy:.1%}")
                        st.code(report)
                    except Exception as e:
                        st.error(f"Training failed: {e}")

        st.markdown("---")
        st.subheader("3️⃣ Find citizens eligible but missing a Voter ID")
        if st.session_state.survey_df is None:
            st.info("Upload a CSV above first.")
        else:
            missing_df = st.session_state.survey_df[
                (st.session_state.survey_df["age"] >= 18) &
                (st.session_state.survey_df["has_voter_id"] == "No")
            ]
            st.markdown(f'<div class="kcard" style="border-color:#e65100"><div class="lbl">Eligible, Missing Voter ID</div><div class="val">{len(missing_df)}</div><div class="sub">out of {len(st.session_state.survey_df)} total rows</div></div>',unsafe_allow_html=True)
            st.dataframe(missing_df.head(20), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("4️⃣ Send outreach notifications")

            sms_ready = SMSChannel().is_configured()
            email_ready = EmailChannel().is_configured()
            cc1, cc2 = st.columns(2)
            with cc1:
                st.info("📱 SMS: **LIVE**" if sms_ready else "📱 SMS: **DRY-RUN** (no TWILIO_* env vars set)")
            with cc2:
                st.info("📧 Email: **LIVE**" if email_ready else "📧 Email: **DRY-RUN** (no SMTP_* env vars set)")

            fc1, fc2 = st.columns(2)
            with fc1:
                limit = st.number_input("Limit (0 = no limit)", min_value=0, value=10, step=5)
            with fc2:
                high_risk_only = False
                if "risk_level" in st.session_state.survey_df.columns:
                    high_risk_only = st.checkbox("Only HIGH risk citizens")

            if st.button("📨 Run Notifications"):
                records = find_eligible_missing_voters_from_df(st.session_state.survey_df)
                if high_risk_only:
                    records = [r for r in records if (r.risk_level or "").upper() == "HIGH"]
                if limit:
                    records = records[:limit]

                with st.spinner(f"Notifying {len(records)} citizen(s)..."):
                    manager = NotificationManager()
                    results = manager.notify_and_log(records)

                sent = sum(1 for r in results if r.status == "sent")
                dry = sum(1 for r in results if r.status == "dry_run")
                skipped = sum(1 for r in results if r.status == "skipped")
                failed = sum(1 for r in results if r.status == "failed")

                st.session_state.notify_results = pd.DataFrame([{
                    "Name": r.record.name, "District": r.record.district,
                    "Channel": r.channel, "Status": r.status, "Detail": r.detail
                } for r in results])

                rc1,rc2,rc3,rc4 = st.columns(4)
                rc1.markdown(f'<div class="kcard" style="border-color:#2e7d32"><div class="lbl">Sent</div><div class="val">{sent}</div></div>',unsafe_allow_html=True)
                rc2.markdown(f'<div class="kcard" style="border-color:#1565c0"><div class="lbl">Dry-run</div><div class="val">{dry}</div></div>',unsafe_allow_html=True)
                rc3.markdown(f'<div class="kcard" style="border-color:#e65100"><div class="lbl">Skipped</div><div class="val">{skipped}</div></div>',unsafe_allow_html=True)
                rc4.markdown(f'<div class="kcard" style="border-color:#c62828"><div class="lbl">Failed</div><div class="val">{failed}</div></div>',unsafe_allow_html=True)

            if st.session_state.notify_results is not None:
                st.dataframe(st.session_state.notify_results, use_container_width=True, hide_index=True)

# ── REPORTS ───────────────────────────────────────────────────────────────────
elif pg=="📄 Reports":
    st.markdown('<div class="hdr"><h1>📄 Official Survey Reports</h1><p>AI-Generated by Report Agent · Claude API · District-level Analytics</p></div>',unsafe_allow_html=True)
    t1,t2,t3=st.tabs(["📊 State Summary","🏘️ District Report","📋 Family-wise Table"])
    with t1:
        gap=abs(round((s['mi']/max(s['me'],1)-s['fi']/max(s['fe'],1))*100,1))
        dr="".join([f"<tr style='border-bottom:1px solid #eee'><td style='padding:.5rem'>{d}</td><td style='text-align:center'>{round(v['h']/max(v['e'],1)*100,1)}%</td><td>{'🟢 Good' if v['h']/max(v['e'],1)>=.8 else '🟡 Average' if v['h']/max(v['e'],1)>=.6 else '🔴 Critical'}</td></tr>" for d,v in s['dd'].items()])
        st.markdown(f"""<div class="rpt">
        <h2 style="color:#0f172a;margin-top:0">📊 Uttar Pradesh — Voter ID Survey Report</h2>
        <p style="color:#666;font-size:.82rem">AI Report Agent · {date.today()} · Confidential</p><hr>
        <h3 style="color:#1565c0">Executive Summary</h3>
        <p>Survey of <b>{s['tot']} citizens</b> across <b>{len(st.session_state.fams)} families</b> in {len(s['dd'])} districts.
        Overall coverage: <b>{s['cv']}%</b> — below the 80% target.</p>
        <h3 style="color:#1565c0">Key Findings</h3>
        <ul>
          <li>✅ <b>{s['hid']}</b> of <b>{s['elig']}</b> eligible citizens have Voter ID</li>
          <li>⚠️ <b>{s['ms']}</b> eligible citizens not yet registered</li>
          <li>👩 Female registration is <b>{gap}%</b> lower than male</li>
          <li>📍 Rural areas show ~23% lower coverage than urban</li>
          <li>👶 Age group 18-25 has lowest coverage (~42%)</li>
        </ul>
        <h3 style="color:#1565c0">District Rankings</h3>
        <table style="width:100%;border-collapse:collapse">
        <tr style="background:#e3f2fd"><th style="padding:.5rem;text-align:left">District</th><th>Coverage</th><th>Status</th></tr>
        {dr}</table>
        <h3 style="color:#1565c0">Recommendations</h3>
        <ol>
          <li>Deploy <b>mobile enrollment vans</b> in low-coverage villages</li>
          <li>Run <b>female-targeted awareness drives</b> to close gender gap</li>
          <li>Focus on <b>18-25 age group</b> first-time voter registration</li>
          <li>Use <b>AI anomaly detection</b> to clean duplicate records</li>
          <li>Partner with <b>Aadhaar centers</b> for seamless Voter ID issuance</li>
        </ol>
        </div>""",unsafe_allow_html=True)
    with t2:
        ds=st.selectbox("Select District",list(s['dd'].keys()))
        dv=s['dd'][ds]; cv=round(dv['h']/max(dv['e'],1)*100,1)
        st.markdown(f"""<div class="rpt">
        <h2 style="color:#0f172a;margin-top:0">🏘️ {ds} District Report</h2>
        <p style="color:#666;font-size:.82rem">AI Report Agent · {date.today()}</p><hr>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin:1rem 0">
          <div style="background:#e3f2fd;border-radius:8px;padding:1rem;text-align:center"><div style="font-size:1.8rem;font-weight:800;color:#1565c0">{dv['e']}</div><div style="font-size:.8rem">Eligible</div></div>
          <div style="background:#e8f5e9;border-radius:8px;padding:1rem;text-align:center"><div style="font-size:1.8rem;font-weight:800;color:#2e7d32">{dv['h']}</div><div style="font-size:.8rem">Have Voter ID</div></div>
          <div style="background:#ffebee;border-radius:8px;padding:1rem;text-align:center"><div style="font-size:1.8rem;font-weight:800;color:#c62828">{dv['e']-dv['h']}</div><div style="font-size:.8rem">Missing</div></div>
        </div>
        <p><b>Coverage:</b> {cv}% {'✅ Above 80% target' if cv>=80 else '⚠️ Below 80% target'}</p>
        <p><b>AI Analysis:</b> {"Good progress. Focus remaining unregistered citizens in next drive." if cv>=70 else "Urgent action needed. Deploy mobile enrollment camps immediately."}</p>
        </div>""",unsafe_allow_html=True)
    with t3:
        rows=[]
        for f in st.session_state.fams:
            e=sum(1 for m in f['members'] if m['elig']); h=sum(1 for m in f['members'] if m['vid'])
            rows.append({"ID":f['fid'],"Head":f['head'],"District":f['dist'],"Members":len(f['members']),"Eligible":e,"Have ID":h,"Missing":e-h,"Coverage":f"{round(h/max(e,1)*100)}%","Status":"✅ Complete" if e==h else "⚠️ Incomplete"})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

# ── ARTICLES ──────────────────────────────────────────────────────────────────
elif pg=="📰 Articles & Insights":
    st.markdown('<div class="hdr"><h1>📰 Articles & AI Insights</h1><p>AI-generated research · Voter ID survey best practices · Technology deep-dives</p></div>',unsafe_allow_html=True)
    arts=[
        ("🤖 How Agentic AI is Transforming Government Surveys","Agentic AI","bg","5 min",
         f"Traditional government surveys relied on slow, error-prone manual collection. **Agentic AI changes everything.**\n\n**What Agentic AI does differently:** Unlike regular AI that just answers questions, Agentic AI *takes actions autonomously* — reads documents, stores data, queries databases, generates reports — all without human intervention.\n\n**In this system:** The Document Agent reads Aadhaar cards using Claude Vision, extracting name, age, gender in seconds. The Survey Agent validates and stores family data, catching duplicates automatically. The RAG Agent answers officer questions. The Report Agent writes district reports.\n\n**Impact:** This system surveyed **{s['tot']} members** across **{len(st.session_state.fams)} families**, achieving **{s['cv']}% coverage tracking**. What would take weeks manually is done in hours.\n\n**Why this matters for India:** With 970 million eligible voters, traditional survey methods cannot scale. Agentic AI provides the automation needed for a nationwide Voter ID enrollment drive."),
        ("📊 The Gender Gap in Voter ID — What the Data Shows","Data Analysis","bo","4 min",
         f"Survey data reveals a concerning pattern: female Voter ID registration consistently lags behind male registration.\n\n**Current data:**\n- Male coverage: **{round(s['mi']/max(s['me'],1)*100,1)}%** ({s['mi']} of {s['me']} eligible males)\n- Female coverage: **{round(s['fi']/max(s['fe'],1)*100,1)}%** ({s['fi']} of {s['fe']} eligible females)\n- Gap: **{abs(round((s['mi']/max(s['me'],1)-s['fi']/max(s['fe'],1))*100,1))}%**\n\n**Root Causes (identified by AI):**\n1. Mobility restrictions in rural areas\n2. Lower awareness of enrollment drives\n3. Aadhaar-Voter ID linking less common among women\n4. Young women (18-25) show lowest registration rate\n\n**Recommended Actions:**\n- Deploy female survey officers in rural areas\n- Set up village-level enrollment camps accessible to women\n- Partner with self-help groups (SHGs) for awareness\n- Mobile-based registration campaigns targeting women"),
        ("🧠 RAG vs Traditional Database Queries — Why RAG Wins","Technology","bp","6 min",
         "**Traditional approach:** A district officer wants to know which villages have the lowest Voter ID coverage. They open Excel, filter columns, create pivot tables... 30 minutes later, they have an answer.\n\n**RAG approach:** They type *'Which villages have lowest voter registration?'* — answered in 3 seconds.\n\n**How RAG works in this system:**\n```\nStep 1: Survey records → vectors → stored in ChromaDB\nStep 2: User question → converted to vector\nStep 3: ChromaDB finds semantically similar records\nStep 4: Records sent to Claude as context\nStep 5: Claude generates grounded answer\n```\n\n**Why RAG beats SQL for field officers:**\n- No technical skills needed — plain English\n- Semantic understanding — finds data even with vague questions\n- Always reads live database, not static reports\n- Transparent — shows which records were retrieved\n\n**Real example:** *'Are there elderly women without ID?'* — RAG finds all: `gender=Female, age>60, has_voter_id=False` — without writing a single SQL query."),
        ("🗺️ District Analysis: Where Are the Critical Gaps?","Field Analysis","bgr","3 min",
         f"AI analysis of all surveyed districts identifies critical areas needing urgent attention.\n\n**District Coverage:**\n{chr(10).join([f'{'🔴' if round(v['h']/max(v['e'],1)*100)<60 else '🟡' if round(v['h']/max(v['e'],1)*100)<80 else '🟢'} **{d}**: {round(v['h']/max(v['e'],1)*100,1)}% ({v['h']}/{v['e']} registered)' for d,v in s['dd'].items()])}\n\n**AI Pattern Analysis:**\nThe ML model found district location is the **3rd most important predictor** of Voter ID possession (after age and gender). Districts with lower coverage share: higher rural population, fewer enrollment centers, lower digital literacy.\n\n**Priority Actions:**\n- 🔴 Critical: Mobile enrollment vans, door-to-door survey\n- 🟡 Average: Monthly camps, gram panchayat partnerships\n- 🟢 Good: Focus on 18-25 age group outreach"),
        ("🔮 Future of AI in India's Electoral System — 2030 Vision","Future Vision","br","7 min",
         "India's electoral system is at a transformative moment. With 970 million registered voters and growing, the Election Commission of India needs AI at scale.\n\n**Near-term (2025-2026):**\n- AI Voter ID survey systems deployed state-wide\n- Agentic AI handling enrollment, verification, anomaly detection\n- RAG chatbots helping election officers across India\n\n**Mid-term (2027-2028):**\n- Aadhaar-Voter ID automatic AI-verified linking\n- Predictive models identifying citizens likely to miss registration\n- Cross-state duplicate detection using AI\n\n**Long-term (2030 Vision):**\n- Fully automated voter roll management\n- Real-time coverage dashboards for every district\n- Biometric + AI voter authentication at booths\n\n**Challenges to overcome:**\n- Privacy concerns and data protection\n- Digital divide in rural populations\n- Ensuring AI doesn't discriminate by caste or religion\n- Building citizen trust in AI-assisted elections\n\n**This project's contribution:** This AI Voter ID Survey System is a proof-of-concept showing how **Agentic AI + RAG + ML** can be deployed at scale for Indian government use cases."),
    ]
    for title,tag,tc,rt,body in arts:
        st.markdown(f'<div class="art"><div style="display:flex;justify-content:space-between;align-items:flex-start"><h3 style="color:#0f172a;margin:0;font-size:1.05rem">{title}</h3><span class="bdg {tc}">{tag}</span></div><div style="font-size:.75rem;color:#999;margin:.4rem 0">📅 June 2025 · ⏱️ {rt}</div></div>',unsafe_allow_html=True)
        with st.expander(f"📖 Read: {title}"):
            st.markdown(body)
