import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
from datetime import datetime
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
import tempfile
import shutil

#Page setup
st.set_page_config(page_title="Portfolio Manager", page_icon="💼", layout="centered")
st.title("Portfolio Manager")
st.markdown("Personal investment recommendation, SIP planning and PDF export.")

#Load model artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_investment_model.pkl")
    scaler = joblib.load("rf_scaler.pkl")
    pca = joblib.load("rf_pca.pkl")
    return model, scaler, pca

try:
    model, scaler, pca = load_artifacts()
except Exception as e:
    st.error("Model artifacts not found. Run backend_with_pca.py first. Error: " + str(e))
    st.stop()

#Database (profiles)
DB = "profiles.db"

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY,
            name TEXT,
            created_at TEXT,
            data TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_profile(name, data_dict):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute("INSERT INTO profiles (name, created_at, data) VALUES (?,?,?)", (name, now, str(data_dict)))
    conn.commit()
    conn.close()

def list_profiles():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id, name, created_at FROM profiles ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def load_profile(pid):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT data FROM profiles WHERE id=?", (pid,))
    row = c.fetchone()
    conn.close()
    if row:
        return eval(row[0])
    return None

#Helpers
def fv_sip(monthly, r_annual, years):
    """Future value of SIP (monthly contribution)."""
    if monthly <= 0 or years <= 0:
        return 0.0
    r_month = r_annual / 12
    n = int(years * 12)
    if r_month == 0:
        return monthly * n
    return monthly * (((1 + r_month)**n - 1) / r_month) * (1 + r_month)

def format_currency(x):
    try:
        return f"₹{int(x):,}"
    except:
        return str(x)

#Pretty profile display (no action buttons)
def pretty_display_profile(pdata):
    """Display profile nicely; NO apply/copy/download buttons."""
    import json
    if not pdata:
        st.info("No profile data.")
        return

    label_map = {
        "age":"Age", "salary":"Monthly Salary", "city_type":"City Type",
        "employment_type":"Employment Type", "marital_status":"Marital Status",
        "risk_level":"Risk Level", "financial_knowledge":"Financial Knowledge",
        "dependents":"Dependents", "current_savings":"Current Savings",
        "monthly_expenses":"Monthly Expenses", "loan_amount":"Outstanding Loan",
        "investment_amount":"Investment Amount", "investment_goal":"Investment Goal"
    }

    colA, colB = st.columns([2,3])
    with colA:
        st.markdown("**Profile snapshot**")
        st.metric(label="Age", value=pdata.get("age", "—"))
        st.metric(label="City", value=pdata.get("city_type", "—"))
        st.metric(label="Risk", value=pdata.get("risk_level", "—"))
        st.markdown("---")
        st.markdown("**Income & Cash Flow**")
        st.write(f"• Salary: **{format_currency(pdata.get('salary',0))}/mo**")
        st.write(f"• Savings: **{format_currency(pdata.get('current_savings',0))}**")
        st.write(f"• Monthly expenses: **{format_currency(pdata.get('monthly_expenses',0))}**")

    with colB:
        st.markdown("**Investment Intent**")
        st.write(f"• Investment amount: **{format_currency(pdata.get('investment_amount',0))}**")
        st.write(f"• Goal: **{pdata.get('investment_goal','—')}**")
        st.write(f"• Dependents: **{pdata.get('dependents',0)}**")
        st.markdown("---")
        try:
            efr = float(pdata.get("current_savings",0)) / max(1, float(pdata.get("monthly_expenses",1)))
            st.write("Emergency fund ratio (months):", f"**{efr:.1f}**")
            st.progress(min(1.0, efr/12.0))
        except Exception:
            pass

    st.markdown("**Full profile details**")
    for k in pdata.keys():
        label = label_map.get(k, k.replace("_", " ").title())
        val = pdata[k]
        if k in ("salary","current_savings","monthly_expenses","loan_amount","investment_amount"):
            try:
                val = f"₹{int(val):,}"
            except Exception:
                pass
        st.write(f"**{label}:** {val}")

# session_state init
if 'alloc' not in st.session_state: st.session_state['alloc'] = None
if 'amounts' not in st.session_state: st.session_state['amounts'] = None
if 'categories' not in st.session_state: st.session_state['categories'] = ['Stocks','Mutual Funds','Gold/Silver','Bonds']
if 'user_df' not in st.session_state: st.session_state['user_df'] = None
if 'last_generated_time' not in st.session_state: st.session_state['last_generated_time'] = None

# Input form (with keys)
with st.form("profile_form"):
    st.subheader("Enter your investment profile")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Name (optional)", key="profile_name")
        st.slider("Age", 18, 65, 30, key="age")
        st.number_input("Monthly Salary (₹)", 10000, 500000, 80000, key="salary")
        st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"], key="city_type")
        st.selectbox("Employment Type", ["Salaried","Self-employed","Student","Retired"], key="employment_type")
        st.selectbox("Marital Status", ["Single","Married","Divorced"], key="marital_status")
    with col2:
        st.selectbox("Risk Level", ["Low","Medium","High"], key="risk_level")
        st.selectbox("Financial Knowledge", ["Low","Medium","High"], key="financial_knowledge")
        st.number_input("Dependents", 0, 6, 0, key="dependents")
        st.number_input("Current Savings (₹)", 0, 5000000, 50000, key="current_savings")
        st.number_input("Monthly Expenses (₹)", 1000, 500000, 20000, key="monthly_expenses")
        st.number_input("Outstanding Loan Amount (₹)", 0, 2000000, 0, key="loan_amount")
    st.number_input("Amount to Invest Now (₹)", 1000, 2000000, 50000, key="investment_amount")
    st.selectbox("Investment Goal", ["Short-term","Medium-term","Long-term"], key="investment_goal")
    submitted = st.form_submit_button("Save profile locally")

if submitted:
    pdata = {
        "age": st.session_state['age'],
        "salary": st.session_state['salary'],
        "city_type": st.session_state['city_type'],
        "employment_type": st.session_state['employment_type'],
        "marital_status": st.session_state['marital_status'],
        "risk_level": st.session_state['risk_level'],
        "financial_knowledge": st.session_state['financial_knowledge'],
        "dependents": st.session_state['dependents'],
        "current_savings": st.session_state['current_savings'],
        "monthly_expenses": st.session_state['monthly_expenses'],
        "loan_amount": st.session_state['loan_amount'],
        "investment_amount": st.session_state['investment_amount'],
        "investment_goal": st.session_state['investment_goal']
    }
    save_profile(st.session_state.get('profile_name', 'unnamed'), pdata)
    st.success("Profile saved to local DB.")

#Show saved profiles
st.markdown("---")
st.subheader("Saved profiles")
rows = list_profiles()
if rows:
    selection = st.selectbox("Select profile", [""] + [f"{r[0]} - {r[1]} ({r[2][:19]})" for r in rows], key="profile_select")
    if selection:
        pid = int(selection.split(" - ")[0])
        pdata = load_profile(pid)
        pretty_display_profile(pdata)
else:
    st.info("No saved profiles yet. Use the form above to create one.")

#Generate allocation and persist in session_state
st.markdown("---")
st.subheader("Generate portfolio allocation")
if st.button("Generate Allocation"):
    user_df = pd.DataFrame([{
        'age': st.session_state['age'],
        'salary': int(st.session_state['salary']),
        'city_type': {"Tier 1":1,"Tier 2":2,"Tier 3":3}[st.session_state['city_type']],
        'education_level': 2,
        'employment_type': {"Salaried":1,"Self-employed":2,"Student":3,"Retired":4}[st.session_state['employment_type']],
        'marital_status': {"Single":0,"Married":1,"Divorced":2}[st.session_state['marital_status']],
        'dependents': int(st.session_state['dependents']),
        'risk_level': {"Low":1,"Medium":2,"High":3}[st.session_state['risk_level']],
        'financial_knowledge': {"Low":1,"Medium":2,"High":3}[st.session_state['financial_knowledge']],
        'monthly_expenses': int(st.session_state['monthly_expenses']),
        'current_savings': int(st.session_state['current_savings']),
        'loan_amount': int(st.session_state['loan_amount']),
        'has_loans': 1 if st.session_state['loan_amount']>0 else 0,
        'investment_goal': {"Short-term":1,"Medium-term":2,"Long-term":3}[st.session_state['investment_goal']],
        'emergency_fund_ratio': (st.session_state['current_savings'] / st.session_state['monthly_expenses']) if st.session_state['monthly_expenses']>0 else 0
    }])

    Xs = scaler.transform(user_df)
    Xp = pca.transform(Xs)
    alloc = model.predict(Xp)[0]
    amounts = [st.session_state['investment_amount'] * a for a in alloc]

    st.session_state['alloc'] = alloc
    st.session_state['amounts'] = amounts
    st.session_state['user_df'] = user_df
    st.session_state['last_generated_time'] = datetime.now().isoformat()
    st.success("✅ Allocation generated.")

# If no allocation stored, prompt user
if st.session_state['alloc'] is None:
    st.info("No allocation stored. Click 'Generate Allocation' to compute your portfolio.")
    st.stop()

#Display stored allocation
st.markdown("---")
st.subheader("Stored allocation (locked until regenerate)")
st.write(f"Generated at: {st.session_state.get('last_generated_time')}")
categories = st.session_state['categories']
amounts = st.session_state['amounts']
fig = go.Figure(data=[go.Pie(labels=categories, values=amounts, hole=0.45,
                             marker=dict(colors=['#00B8A9','#3A0CA3','#06D6A0','#FFD166'],
                                         line=dict(color='white', width=2)),
                             textinfo='label+percent')])
fig.update_layout(title_text="Overall Portfolio Allocation", title_x=0.5)
st.plotly_chart(fig, use_container_width=True)

#Detailed sub-allocations
def detailed_allocation(category, total_amt, age_local, risk_local, goal_local, fk_local):
    r = {"Low":1,"Medium":2,"High":3}[risk_local]
    g = {"Short-term":1,"Medium-term":2,"Long-term":3}[goal_local]
    fk = {"Low":1,"Medium":2,"High":3}[fk_local]
    sub = {}
    if category=="Stocks":
        sub = {"Large Cap":0.6,"Mid Cap":0.25,"Small Cap":0.15}
        if r==3: sub={"Large Cap":0.5,"Mid Cap":0.3,"Small Cap":0.2}
        if r==1: sub={"Large Cap":0.75,"Mid Cap":0.2,"Small Cap":0.05}
        if age_local<30: sub["Small Cap"]+=0.05; sub["Large Cap"]-=0.05
        if age_local>50: sub["Large Cap"]+=0.05; sub["Small Cap"]-=0.05
        if g==1: sub["Large Cap"]+=0.05; sub["Small Cap"]-=0.05
        if g==3: sub["Small Cap"]+=0.05; sub["Large Cap"]-=0.05
        if fk==3: sub["Mid Cap"]+=0.03; sub["Small Cap"]+=0.02; sub["Large Cap"]-=0.05
        s=sum(sub.values()); sub={k:v/s for k,v in sub.items()}
    elif category=="Mutual Funds":
        sub={"Equity MF":0.5,"Hybrid MF":0.3,"Debt MF":0.2}
        if g==1: sub["Debt MF"]+=0.1; sub["Equity MF"]-=0.1
        if r==3: sub["Equity MF"]+=0.05; sub["Debt MF"]-=0.05
    elif category=="Gold/Silver":
        sub={"Gold":0.8,"Silver":0.2}
        if g==3: sub["Silver"]+=0.05; sub["Gold"]-=0.05
    elif category=="Bonds":
        sub={"Government Bonds":0.7,"Corporate Bonds":0.3}
        if r==3: sub["Corporate Bonds"]+=0.1; sub["Government Bonds"]-=0.1
        if g==1: sub["Government Bonds"]+=0.1; sub["Corporate Bonds"]-=0.1
    return {k: round(v*total_amt,2) for k,v in sub.items()}

st.subheader("Detailed Sub-Allocations")
for cat, amt in zip(categories, amounts):
    st.write(f"**{cat}: ₹{amt:,.2f} ({(amt/sum(amounts))*100:.1f}%)**")
    sub = detailed_allocation(cat, amt,
                              st.session_state['age'],
                              st.session_state['risk_level'],
                              st.session_state['investment_goal'],
                              st.session_state['financial_knowledge'])
    for k,v in sub.items():
        st.write(f"  - {k}: ₹{v:,.2f} ({(v/amt)*100:.1f}%)")
    fig_sub = go.Figure(data=[go.Pie(labels=list(sub.keys()), values=list(sub.values()), hole=0.45)])
    fig_sub.update_layout(title_text=f"{cat} breakdown")
    st.plotly_chart(fig_sub, use_container_width=True)

#SIP & Optimization
st.markdown("---")
st.header("SIP & Optimization (uses stored allocation)")
RETURNS = {
    "Conservative": {"Stocks":0.08,"Mutual Funds":0.07,"Gold/Silver":0.05,"Bonds":0.04},
    "Moderate": {"Stocks":0.12,"Mutual Funds":0.10,"Gold/Silver":0.07,"Bonds":0.05},
    "Aggressive": {"Stocks":0.15,"Mutual Funds":0.12,"Gold/Silver":0.08,"Bonds":0.06}
}

sip_monthly = st.number_input("Planned monthly SIP (₹)", 0, 1000000, 0, key="sip_monthly")
sip_years = st.number_input("SIP horizon (years)", 1, 30, 10, key="sip_years")
sip_scenario = st.selectbox("SIP return scenario", ["Conservative","Moderate","Aggressive"], index=1, key="sip_scenario")

# --- ADDED: Custom expected annual SIP return slider ---
st.markdown("**Optional:** set a custom expected annual SIP return to override scenario defaults.")
sip_expected_return = st.slider(
    "Custom expected annual SIP return (%) — set to 0 to use scenario defaults",
    min_value=0.0, max_value=30.0, value=0.0, step=0.1, key="sip_expected_return"
)
# If sip_expected_return > 0, we'll use this as r for all categories (simple override).
# Otherwise we'll use RETURNS[sip_scenario][cat] as before.

if sip_monthly > 0:
    alloc_arr = np.array(st.session_state['alloc'])
    alloc_sum = alloc_arr.sum() if alloc_arr.sum()!=0 else 1.0
    sip_alloc = {cat: sip_monthly * (a/alloc_sum) for cat,a in zip(categories, alloc_arr)}
    sip_fvs = {}
    for cat in categories:
        if sip_expected_return and float(sip_expected_return) > 0.0:
            r = float(sip_expected_return) / 100.0
        else:
            r = RETURNS[sip_scenario][cat]
        fv = fv_sip(sip_alloc[cat], r, sip_years)
        sip_fvs[cat] = fv
    st.write("Estimated SIP corpus after", sip_years, "years:")
    for cat, fv in sip_fvs.items():
        st.write(f"  - {cat}: ₹{fv:,.2f}")
    st.write("Total SIP corpus:", f"₹{sum(sip_fvs.values()):,.2f}")

recommend_monthly_total = st.number_input("If you can invest monthly, how much can you commit? (₹)", 0, 200000, 10000, key="recommend_monthly_total")
if recommend_monthly_total > 0:
    weights = np.array(st.session_state['alloc']) / np.sum(st.session_state['alloc']) if np.sum(st.session_state['alloc'])!=0 else np.array([0.25,0.25,0.25,0.25])
    rec_sip = {cat: round(recommend_monthly_total * w, 2) for cat,w in zip(categories, weights)}
    st.write("Recommended monthly SIP split (based on stored allocation):")
    for cat,amt in rec_sip.items():
        st.write(f"  - {cat}: ₹{amt:,.2f}")

# -------------------------
# PDF export (matplotlib + embedded font fixes + KeepTogether)
# -------------------------
st.markdown("---")
st.header("Export / Report")
if st.button("Export PDF report (uses stored allocation)"):
    try:
        # Validate and prepare
        if not st.session_state.get('amounts'):
            st.error("No allocation amounts found in session_state. Generate allocation first.")
            st.st

        amounts = [float(a) for a in st.session_state['amounts']]
        categories = st.session_state.get('categories', ['Stocks','Mutual Funds','Gold/Silver','Bonds'])
        colors = ['#00B8A9','#3A0CA3','#06D6A0','#FFD166']

        # Ensure a display name that is not blank
        given_name = st.session_state.get('profile_name')
        if given_name and str(given_name).strip():
            display_name = str(given_name).strip()
        else:
            display_name = f"Customer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        safe_name = display_name.replace(" ", "_")

        # Prepare tmp directory for images
        tmpdir = tempfile.mkdtemp(prefix="report_imgs_")

        # Helper to draw donut chart and save
        def save_donut(labels, values, out_path, title=None):
            fig, ax = plt.subplots(figsize=(6,4), dpi=150)
            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                autopct=lambda p: ('%1.1f%%' % p) if p > 0 else '',
                pctdistance=0.75,
                startangle=90,
                colors=[colors[i % len(colors)] for i in range(len(labels))],
                wedgeprops=dict(width=0.45, edgecolor='white')
            )
            ax.axis('equal')
            if title:
                ax.set_title(title, fontsize=12)
            plt.tight_layout()
            fig.savefig(out_path, bbox_inches='tight', transparent=False)
            plt.close(fig)

        # Save main donut
        main_img = os.path.join(tmpdir, "main.png")
        save_donut(categories, amounts, main_img, title="Overall Portfolio Allocation")

        # Save sub-allocation images
        sub_imgs = []
        for cat, amt in zip(categories, amounts):
            sub = detailed_allocation(cat, amt,
                                      st.session_state['age'],
                                      st.session_state['risk_level'],
                                      st.session_state['investment_goal'],
                                      st.session_state['financial_knowledge'])
            labels = list(sub.keys())
            values = list(sub.values())
            if sum(values) == 0:
                values = [1 for _ in values]
            fname = f"{cat.replace('/','_')}.png"
            sub_path = os.path.join(tmpdir, fname)
            save_donut(labels, values, sub_path, title=f"{cat} Breakdown")
            sub_imgs.append((cat, sub_path))

        # Register a Unicode TTF font so rupee symbol and other glyphs render correctly
        try:
            dejavu_path = mfm.findfont("DejaVu Sans")
            pdfmetrics.registerFont(TTFont('DejaVuSans', dejavu_path))
            font_name_for_pdf = 'DejaVuSans'
        except Exception:
            # fallback if matplotlib/DejaVu not available; use default
            font_name_for_pdf = None

        # Build PDF with ReportLab and ensure headings + images stay together
        pdf_file = f"Portfolio_Report_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc = SimpleDocTemplate(pdf_file, pagesize=A4)
        styles = getSampleStyleSheet()

        # If we registered a font, set it on styles so Paragraph uses it (fixes ₹ rendering)
        if font_name_for_pdf:
            for sname in ['Normal','Title','Heading1','Heading2','Heading3']:
                if sname in styles:
                    styles[sname].fontName = font_name_for_pdf

        story = []
        # Title and name (keep title separate but the name and generated timestamp should be together to avoid blank name)
        story.append(Paragraph("Investment Portfolio Report", styles['Title']))
        story.append(Spacer(1,12))
        story.append(Paragraph(f"Name: {display_name}", styles['Normal']))
        story.append(Paragraph(f"Generated: {st.session_state.get('last_generated_time','-')}", styles['Normal']))
        story.append(Spacer(1,12))

        # Overall allocation - keep heading and image together
        story.append(KeepTogether([
            Paragraph("Overall Allocation:", styles['Heading2']),
            RLImage(main_img, width=420, height=260)
        ]))
        story.append(Spacer(1,12))

        # Sub allocation sections, each kept together
        for cat, imgp in sub_imgs:
            story.append(KeepTogether([
                Paragraph(f"{cat} Breakdown:", styles['Heading3']),
                RLImage(imgp, width=330, height=220)
            ]))
            story.append(Spacer(1,8))

        # Detailed allocations (removed the unwanted "(text)")
        story.append(Paragraph("Detailed allocations:", styles['Heading2']))
        total_invest = float(st.session_state.get('investment_amount', sum(amounts) if amounts else 0))
        for cat, amt in zip(categories, amounts):
            pct = (amt/total_invest*100) if total_invest>0 else 0.0
            # Use the rupee symbol; with embedded font it will render correctly
            story.append(Paragraph(f"{cat}: ₹{amt:,.2f} ({pct:.1f}%)", styles['Normal']))

        doc.build(story)

        # Serve PDF as download
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()

        st.success(f"PDF created: {pdf_file}")
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=pdf_file,
            mime="application/pdf"
        )

    except Exception as e:
        import traceback
        st.error("Failed to create or serve PDF. See error details below.")
        st.code(str(e))
        st.code(traceback.format_exc())
    finally:
        # cleanup temp images
        try:
            if 'tmpdir' in locals() and os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir)
        except Exception:
            pass

st.caption("Tip: allocation stays stored until you press 'Generate Allocation' again. You can change SIP inputs freely.")
