import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Config & estilo
# -----------------------------
st.set_page_config(
    page_title="Cartera & Cobranza | Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
/* Ocultar sidebar completo */
section[data-testid="stSidebar"] { display: none !important; }
button[kind="header"] { display: none !important; }

/* Fondo y espaciado */
.main { background: #0b1220; }
.block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; }

/* Cards */
.etd-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.04));
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 18px 12px 18px;
  box-shadow: 0 14px 38px rgba(0,0,0,0.28);
}
.etd-title {
  font-size: 1.20rem;
  font-weight: 750;
  letter-spacing: 0.2px;
  color: rgba(255,255,255,0.94);
  margin-bottom: 0.35rem;
}
.etd-subtitle {
  color: rgba(255,255,255,0.72);
  font-size: 0.95rem;
  margin-top: -2px;
  line-height: 1.35rem;
}

/* Insight box */
.insight {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 12px 14px;
  margin-top: 10px;
}
.insight b { color: rgba(255,255,255,0.92); }
.insight span { color: rgba(255,255,255,0.80); }

/* Metric cards */
div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  padding: 12px 14px;
  border-radius: 16px;
}

/* Text */
h1, h2, h3, h4 { color: rgba(255,255,255,0.92); }
p, li, div { color: rgba(255,255,255,0.84); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Helpers formato
# -----------------------------
def money(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"${x:,.0f}".replace(",", ".")

def pct01(x: float) -> str:
    """x en [0,1] -> %"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:,.1f}%".replace(",", ".")

def pct100(x: float) -> str:
    """x ya en % (0-100)"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.1f}%".replace(",", ".")

def safe_div(a, b):
    return (a / b) if (b is not None and b != 0 and not (isinstance(b, float) and np.isnan(b))) else np.nan

def quantile_label(x, q1, q3):
    if np.isnan(x):
        return "Sin dato"
    if x <= q1:
        return "Bajo"
    if x >= q3:
        return "Alto"
    return "Medio"

# -----------------------------
# Carga y limpieza
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Data base")
    df.columns = [c.strip() for c in df.columns]

    # Renombrar DPD (en tu archivo viene como Unnamed: 15)
    if "Unnamed: 15" in df.columns:
        df = df.rename(columns={"Unnamed: 15": "DPD"})

    # Tipos fecha
    date_cols = ["Fecha de plano", "Fecha de vencimiento", "Fecha de pago"]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Normalizar cliente
    if "Cliente " in df.columns and "Cliente" not in df.columns:
        df = df.rename(columns={"Cliente ": "Cliente"})
    if "Cliente" in df.columns:
        df["Cliente"] = df["Cliente"].astype(str).str.strip()

    # Cohorte / cosecha
    if "Cosecha / Batch" in df.columns:
        raw = df["Cosecha / Batch"].astype(str).str.strip()

        def parse_cohort(x: str):
            # Ej: "2024-012" -> 2024-12, "2024-01" -> 2024-01
            try:
                parts = x.replace("_", "-").split("-")
                y = int(parts[0])
                m_str = parts[1]
                m = int(m_str)
                if m > 12 and len(m_str) == 3:
                    m = int(m_str[-2:])  # 012 -> 12
                m = max(1, min(12, m))
                return pd.Period(f"{y}-{m:02d}", freq="M")
            except:
                return pd.NaT

        df["Cohorte"] = raw.apply(parse_cohort)
    else:
        df["Cohorte"] = pd.NaT

    # LoanID: si ID está vacío, usamos Cliente como proxy
    if "ID" in df.columns and df["ID"].notna().any():
        df["LoanID"] = df["ID"].astype(str)
    else:
        df["LoanID"] = df["Cliente"].astype(str) if "Cliente" in df.columns else df.index.astype(str)

    # Numéricos
    for c in ["Valor credito", "Saldo inicial", "Cuota", "Saldo", "DPD"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Bucket mora (para PAR y Roll rates)
    def bucket(dpd):
        if pd.isna(dpd):
            return "Sin dato"
        if dpd <= 1:
            return "Corriente"
        if 2 <= dpd <= 29:
            return "2-29"
        if 30 <= dpd <= 59:
            return "30-59"
        if 60 <= dpd <= 89:
            return "60-89"
        return "90+"

    df["Bucket"] = df["DPD"].apply(bucket)

    # Mes de corte
    df["PlanoMes"] = df["Fecha de plano"].dt.to_period("M")

    return df

# -----------------------------
# Métricas
# -----------------------------
def portfolio_kpis(snapshot: pd.DataFrame) -> dict:
    snap = snapshot.copy().dropna(subset=["Saldo"])
    total_saldo = snap["Saldo"].sum()

    npl_mask = snap["DPD"] > 1
    npl_saldo = snap.loc[npl_mask, "Saldo"].sum()
    npl_num = snap.loc[npl_mask, "LoanID"].nunique()
    total_num = snap["LoanID"].nunique()

    par30 = snap.loc[snap["DPD"] > 30, "Saldo"].sum()
    par60 = snap.loc[snap["DPD"] > 60, "Saldo"].sum()
    par90 = snap.loc[snap["DPD"] > 90, "Saldo"].sum()

    return {
        "Saldo total": total_saldo,
        "NPL saldo": safe_div(npl_saldo, total_saldo),
        "NPL #": safe_div(npl_num, total_num),
        "PAR30": safe_div(par30, total_saldo),
        "PAR60": safe_div(par60, total_saldo),
        "PAR90": safe_div(par90, total_saldo),
        "Créditos": total_num,
        "Créditos en mora (>1)": npl_num,
        "_npl_saldo_abs": npl_saldo,
        "_par30_abs": par30,
        "_par60_abs": par60,
        "_par90_abs": par90,
    }

def cohort_kpis(df: pd.DataFrame, plano_mes: pd.Period) -> pd.DataFrame:
    snap = df[df["PlanoMes"] == plano_mes].copy()
    snap = snap.dropna(subset=["Cohorte"])

    rows = []
    for coh, g in snap.groupby("Cohorte"):
        k = portfolio_kpis(g)
        k["Cohorte"] = str(coh)
        rows.append(k)

    return pd.DataFrame(rows).sort_values("Cohorte") if rows else pd.DataFrame()

def roll_rate_matrix(df: pd.DataFrame, mes_t: pd.Period, mes_t1: pd.Period) -> pd.DataFrame:
    a = df[df["PlanoMes"] == mes_t][["LoanID", "Bucket"]].dropna()
    b = df[df["PlanoMes"] == mes_t1][["LoanID", "Bucket"]].dropna()

    m = a.merge(b, on="LoanID", how="inner", suffixes=("_t", "_t1"))
    if m.empty:
        return pd.DataFrame()

    order = ["Corriente", "2-29", "30-59", "60-89", "90+"]
    m["Bucket_t"] = pd.Categorical(m["Bucket_t"], categories=order, ordered=True)
    m["Bucket_t1"] = pd.Categorical(m["Bucket_t1"], categories=order, ordered=True)

    pivot = pd.crosstab(m["Bucket_t"], m["Bucket_t1"], normalize="index") * 100
    pivot = pivot.reindex(index=order, columns=order)
    return pivot.round(1)

def cei_by_month(df: pd.DataFrame) -> pd.DataFrame:
    if "Fecha de pago" not in df.columns or df["Fecha de pago"].isna().all():
        return pd.DataFrame()

    months = sorted([m for m in df["PlanoMes"].dropna().unique()])
    if len(months) < 2:
        return pd.DataFrame()

    dfx = df.copy()
    dfx["PagoMes"] = dfx["Fecha de pago"].dt.to_period("M")
    rec_month = dfx.groupby("PagoMes", dropna=True)["Cuota"].sum().rename("Recuperaciones")

    rows = []
    for i in range(1, len(months)):
        m_prev = months[i - 1]
        m_now = months[i]

        snap_prev = dfx[dfx["PlanoMes"] == m_prev]
        mora_inicial = snap_prev.loc[snap_prev["DPD"] > 1, "Saldo"].sum()

        rec = float(rec_month.get(m_now, 0.0))
        cei = (rec / mora_inicial * 100) if mora_inicial else np.nan

        rows.append({
            "Mes": str(m_now),
            "Mora inicial": mora_inicial,
            "Recuperaciones": rec,
            "CEI": cei
        })

    return pd.DataFrame(rows)

def vintage_curves(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = df.dropna(subset=["Cohorte", "PlanoMes"]).copy()
    base["Age"] = (base["PlanoMes"].dt.to_timestamp() - base["Cohorte"].dt.to_timestamp()).dt.days // 30
    base["Age"] = base["Age"].clip(lower=0)

    par = (base.assign(PAR30_saldo=lambda x: np.where(x["DPD"] > 30, x["Saldo"], 0.0))
              .groupby(["Cohorte", "Age"], as_index=False)
              .agg(Saldo=("Saldo", "sum"), PAR30_saldo=("PAR30_saldo", "sum")))
    par["PAR30"] = np.where(par["Saldo"] > 0, par["PAR30_saldo"] / par["Saldo"], np.nan)

    rec = (base.groupby(["Cohorte", "PlanoMes"], as_index=False)["Cuota"].sum()
              .assign(Age=lambda x: (x["PlanoMes"].dt.to_timestamp() - x["Cohorte"].dt.to_timestamp()).dt.days // 30)
              .sort_values(["Cohorte", "Age"]))
    rec["Recuperado_acum"] = rec.groupby("Cohorte")["Cuota"].cumsum()

    first_plane = (base.sort_values("Fecha de plano")
                      .groupby("Cohorte", as_index=False)
                      .first()[["Cohorte", "PlanoMes"]])
    init = base.merge(first_plane, on=["Cohorte", "PlanoMes"], how="inner")
    saldo_vencido_inicial = init.loc[init["DPD"] > 1].groupby("Cohorte")["Saldo"].sum()

    rec["Saldo_vencido_inicial"] = rec["Cohorte"].map(saldo_vencido_inicial).astype(float)
    rec["RecoveryRate_acum"] = np.where(
        rec["Saldo_vencido_inicial"] > 0,
        rec["Recuperado_acum"] / rec["Saldo_vencido_inicial"],
        np.nan
    )

    return par, rec

# -----------------------------
# Componentes de interpretación (FORMATO EJECUTIVO)
# -----------------------------
def insight_box(hallazgo: str, implicacion: str, accion: str):
    st.markdown(
        f"""
<div class='insight'>
  <span><b>Hallazgo:</b> {hallazgo}</span><br>
  <span><b>Implicación:</b> {implicacion}</span><br>
  <span><b>Acción sugerida:</b> {accion}</span>
</div>
""",
        unsafe_allow_html=True
    )

def latex_block():
    with st.expander("Definiciones y fórmulas"):
        st.markdown("#### NPL (saldo)")
        st.latex(r"\mathrm{NPL}_{\$}=\frac{\sum_i \mathrm{Saldo}_i \cdot \mathbb{1}(\mathrm{DPD}_i>1)}{\sum_i \mathrm{Saldo}_i}")
        st.markdown("**Interpretación:** proporción del saldo en mora (DPD > 1).")

        st.markdown("#### NPL (número)")
        st.latex(r"\mathrm{NPL}_{\#}=\frac{\#\{i:\mathrm{DPD}_i>1\}}{\#\{i\}}")
        st.markdown("**Interpretación:** porcentaje de créditos en mora.")

        st.markdown("#### PAR30 / PAR60 / PAR90")
        st.latex(r"\mathrm{PAR}_x=\frac{\sum_i \mathrm{Saldo}_i \cdot \mathbb{1}(\mathrm{DPD}_i>x)}{\sum_i \mathrm{Saldo}_i}")
        st.markdown("**Interpretación:** severidad de mora a partir del umbral x.")

        st.markdown("#### Roll Rate (rodamiento)")
        st.latex(r"RR_{A\to B}=\frac{\#\{i:\mathrm{Bucket}_t(i)=A \land \mathrm{Bucket}_{t+1}(i)=B\}}{\#\{i:\mathrm{Bucket}_t(i)=A\}}")
        st.markdown("**Interpretación:** probabilidad de migración entre buckets de mora entre cortes.")

        st.markdown("#### CEI")
        st.latex(r"\mathrm{CEI}_t=\frac{\mathrm{Recuperaciones}_t}{\mathrm{MoraInicial}_t}\times 100")
        st.markdown("**Interpretación:** efectividad relativa de cobranza sobre la mora base del periodo.")

        st.markdown("#### Costo por $ recuperado")
        st.latex(r"\mathrm{CostoPor\$}_t=\frac{\mathrm{GastoCobranza}_t}{\mathrm{Recuperaciones}_t}")
        st.markdown("**Interpretación:** eficiencia operativa (menor es mejor).")

# -----------------------------
# Cargar base (mismo folder)
# -----------------------------
XLSX_FILE = "Prueba - Data base.xlsx"
df = load_data(XLSX_FILE)

available_months = sorted([m for m in df["PlanoMes"].dropna().unique()])
if not available_months:
    st.error("No se encontraron meses válidos en 'Fecha de plano'. Revisa el Excel.")
    st.stop()

# -----------------------------
# Header (más sobrio)
# -----------------------------
st.markdown(
    """
<div class="etd-card">
  <div class="etd-title">Dashboard de Cartera y Cobranza</div>
  <div class="etd-subtitle">
    Indicadores de mora, análisis por cohortes, migración entre buckets, efectividad de cobranza (CEI) y productividad.
  </div>
</div>
""",
    unsafe_allow_html=True
)
st.write("")

# -----------------------------
# Controles superiores (sin sidebar)
# -----------------------------
top1, top2, top3 = st.columns([1.2, 1.0, 1.0])
with top1:
    selected_month = st.selectbox("Mes de análisis (corte)", available_months, index=len(available_months)-1)
with top2:
    monthly_collection_cost = st.number_input(
        "Gasto mensual de cobranza (supuesto)",
        min_value=0.0,
        value=3_000_000.0,
        step=100_000.0
    )
with top3:
    thr_pack = st.selectbox("Perfil de umbrales", ["Conservador", "Base", "Agresivo"], index=1)

if thr_pack == "Conservador":
    thr_npl, thr_par30, thr_roll_c_to_30 = 0.08, 0.06, 0.04
elif thr_pack == "Agresivo":
    thr_npl, thr_par30, thr_roll_c_to_30 = 0.15, 0.10, 0.07
else:
    thr_npl, thr_par30, thr_roll_c_to_30 = 0.10, 0.08, 0.05

latex_block()

# -----------------------------
# Tabs (más sobrios)
# -----------------------------
tabs = st.tabs([
    "Resumen",
    "Cohortes y Vintage",
    "Roll Rate",
    "Cobranza (CEI)",
    "Costos",
    "Alertas",
    "Plan 90 días"
])

# -----------------------------
# TAB 1: Resumen KPIs
# -----------------------------
with tabs[0]:
    snap = df[df["PlanoMes"] == selected_month]
    k = portfolio_kpis(snap)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saldo total", money(k["Saldo total"]))
    c2.metric("NPL (saldo)", pct01(k["NPL saldo"]))
    c3.metric("PAR30", pct01(k["PAR30"]))
    c4.metric("Créditos (únicos)", f"{int(k['Créditos']):,}".replace(",", "."))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("NPL (número)", pct01(k["NPL #"]))
    c6.metric("PAR60", pct01(k["PAR60"]))
    c7.metric("PAR90", pct01(k["PAR90"]))
    c8.metric("Créditos en mora (>1)", f"{int(k['Créditos en mora (>1)']):,}".replace(",", "."))

    dist = (snap.groupby("Bucket")["Saldo"].sum().reset_index().sort_values("Saldo", ascending=False))
    fig = px.bar(dist, x="Bucket", y="Saldo", title=f"Distribución de saldo por bucket — Corte {selected_month}")
    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

    total = k["Saldo total"]
    par90_abs = k["_par90_abs"]
    npl_abs = k["_npl_saldo_abs"]
    dist_top = dist.iloc[0]["Bucket"] if len(dist) else "—"
    dist_top_amt = dist.iloc[0]["Saldo"] if len(dist) else np.nan

    insight_box(
        hallazgo=(
            f"Saldo total {money(total)}. NPL (saldo) {pct01(k['NPL saldo'])} ({money(npl_abs)}). "
            f"PAR90 {pct01(k['PAR90'])} ({money(par90_abs)}). Bucket con mayor concentración: {dist_top}."
        ),
        implicacion="El crecimiento de PAR30/PAR60 anticipa migración hacia mora severa si no se actúa en mora temprana.",
        accion="Reforzar prevención en Corriente/2-29 y focalizar gestión en 30-59 por saldo y probabilidad de cura."
    )

    with st.expander("Detalle del corte (tabla)"):
        st.dataframe(snap.sort_values(["Cohorte", "LoanID"]), use_container_width=True)

# -----------------------------
# TAB 2: Cohortes & Vintage
# -----------------------------
with tabs[1]:
    st.markdown("### Indicadores por cohorte (cosecha) — corte seleccionado")
    ck = cohort_kpis(df, selected_month)

    if not ck.empty:
        show = ck[["Cohorte", "Saldo total", "NPL saldo", "PAR30", "PAR60", "PAR90", "Créditos"]].copy()
        show["NPL saldo"] = (show["NPL saldo"] * 100).round(2)
        show["PAR30"] = (show["PAR30"] * 100).round(2)
        show["PAR60"] = (show["PAR60"] * 100).round(2)
        show["PAR90"] = (show["PAR90"] * 100).round(2)
        st.dataframe(show, use_container_width=True)

        ck2 = ck.replace([np.inf, -np.inf], np.nan).dropna(subset=["PAR30"])
        if not ck2.empty:
            worst = ck2.sort_values("PAR30", ascending=False).head(1).iloc[0]
            best = ck2.sort_values("PAR30", ascending=True).head(1).iloc[0]

            insight_box(
                hallazgo=(
                    f"Cohorte con mayor PAR30: {worst['Cohorte']} ({pct01(worst['PAR30'])}). "
                    f"Cohorte con menor PAR30: {best['Cohorte']} ({pct01(best['PAR30'])})."
                ),
                implicacion="Cohortes con PAR30 alto tienden a alimentar PAR60/90 si no se interviene en los primeros 30 días.",
                accion="Priorizar campañas por cohorte con foco en curas tempranas (2-29 y 30-59) y revisar canal/originación si aplica."
            )
    else:
        st.info("No hay cohortes válidas para el corte seleccionado.")

    st.write("")
    st.markdown("### Vintage: morosidad (PAR30) y recuperación acumulada")

    par, rec = vintage_curves(df)

    if not par.empty:
        par_plot = par.copy()
        par_plot["PAR30_pct"] = par_plot["PAR30"] * 100
        fig1 = px.line(
            par_plot, x="Age", y="PAR30_pct", color=par_plot["Cohorte"].astype(str),
            markers=True, title="Vintage PAR30 (%) por antigüedad"
        )
        fig1.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=60, b=10))
        fig1.update_xaxes(title="Antigüedad (meses)")
        fig1.update_yaxes(title="PAR30 (%)")
        st.plotly_chart(fig1, use_container_width=True)

        early = par_plot[par_plot["Age"].between(0, 2)].copy()
        if not early.empty:
            slope = (early.groupby("Cohorte")
                          .apply(lambda g: (g.sort_values("Age")["PAR30_pct"].iloc[-1] - g.sort_values("Age")["PAR30_pct"].iloc[0])
                               if len(g) >= 2 else np.nan)).dropna()
            if len(slope):
                coh_name = str(slope.sort_values(ascending=False).index[0])
                insight_box(
                    hallazgo=f"Mayor deterioro temprano de PAR30 en cohorte {coh_name} (0–2 meses).",
                    implicacion="Deterioro temprano suele indicar fricción de recaudo inicial o perfil de originación más riesgoso.",
                    accion="Refuerzo de prevención y contacto temprano (2-29) en esa cohorte; monitoreo de roll rates Corriente→30-59."
                )

    if not rec.empty:
        rec_plot = rec.copy()
        rec_plot["Recovery_pct"] = rec_plot["RecoveryRate_acum"] * 100
        fig2 = px.line(
            rec_plot, x="Age", y="Recovery_pct", color=rec_plot["Cohorte"].astype(str),
            markers=True, title="Recuperación acumulada (%)"
        )
        fig2.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=60, b=10))
        fig2.update_xaxes(title="Antigüedad (meses)")
        fig2.update_yaxes(title="Recuperación acumulada (%)")
        st.plotly_chart(fig2, use_container_width=True)

        last_age = rec_plot["Age"].max()
        tail = rec_plot[rec_plot["Age"] == last_age].dropna(subset=["Recovery_pct"])
        if not tail.empty:
            best = tail.sort_values("Recovery_pct", ascending=False).iloc[0]
            worst = tail.sort_values("Recovery_pct", ascending=True).iloc[0]
            insight_box(
                hallazgo=f"A edad {int(last_age)} meses: mejor cohorte {best['Cohorte']} ({pct100(best['Recovery_pct'])}), peor {worst['Cohorte']} ({pct100(worst['Recovery_pct'])}).",
                implicacion="Recuperación baja con morosidad alta sugiere consolidación de mora severa.",
                accion="Ajustar estrategia de escalamiento para cohortes con baja recuperación (reestructuración/tercerización/legal según ROI)."
            )

# -----------------------------
# TAB 3: Roll Rates
# -----------------------------
with tabs[2]:
    st.markdown("### Matriz de Roll Rate entre cortes")

    if len(available_months) < 2:
        st.info("Se requieren al menos 2 meses para calcular roll rates.")
    else:
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            mes_t = st.selectbox("Mes T", available_months, index=max(0, len(available_months) - 2))
        with mcol2:
            mes_t1 = st.selectbox("Mes T+1", available_months, index=len(available_months) - 1)

        mat = roll_rate_matrix(df, mes_t, mes_t1)
        if mat.empty:
            st.info("No hay suficientes créditos comunes entre ambos meses.")
        else:
            fig = go.Figure(data=go.Heatmap(
                z=mat.values,
                x=mat.columns.astype(str),
                y=mat.index.astype(str),
                hoverongaps=False,
                text=mat.values,
                texttemplate="%{text:.1f}%",
                showscale=True
            ))
            fig.update_layout(
                template="plotly_dark",
                title=f"Roll Rate (%) — {mes_t} → {mes_t1}",
                height=520,
                margin=dict(l=10, r=10, t=60, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c_to_30 = float(mat.loc["Corriente", "30-59"]) if ("Corriente" in mat.index and "30-59" in mat.columns) else np.nan
            d2_to_30 = float(mat.loc["2-29", "30-59"]) if ("2-29" in mat.index and "30-59" in mat.columns) else np.nan
            d30_to_60 = float(mat.loc["30-59", "60-89"]) if ("30-59" in mat.index and "60-89" in mat.columns) else np.nan
            d60_to_90 = float(mat.loc["60-89", "90+"]) if ("60-89" in mat.index and "90+" in mat.columns) else np.nan

            c1.metric("Corriente → 30-59", pct100(c_to_30))
            c2.metric("2-29 → 30-59", pct100(d2_to_30))
            c3.metric("30-59 → 60-89", pct100(d30_to_60))
            c4.metric("60-89 → 90+", pct100(d60_to_90))

            risk_flag = (not np.isnan(c_to_30)) and (c_to_30/100.0 >= thr_roll_c_to_30)
            insight_box(
                hallazgo=f"Corriente→30-59 = {pct100(c_to_30)} bajo el perfil '{thr_pack}'.",
                implicacion="Un valor alto suele anticipar deterioro estructural si no se refuerza prevención y gestión temprana.",
                accion="Aumentar recordatorios pre-vencimiento y contacto <48h en 2-29; priorizar 30-59 por saldo."
            )

# -----------------------------
# TAB 4: Recuperaciones & CEI
# -----------------------------
with tabs[3]:
    st.markdown("### Recuperaciones y CEI")
    cei = cei_by_month(df)

    if cei.empty:
        st.info("No se pudo calcular CEI (revisa 'Fecha de pago' y 'Cuota').")
    else:
        colA, colB = st.columns([2, 1])

        with colA:
            fig = px.bar(cei, x="Mes", y="Recuperaciones", title="Recuperaciones por mes")
            fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with colB:
            fig2 = px.line(cei, x="Mes", y="CEI", markers=True, title="CEI (%)")
            fig2.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(cei, use_container_width=True)

        cei_clean = cei.dropna(subset=["CEI"]).copy()
        if len(cei_clean) >= 2:
            last = cei_clean.iloc[-1]
            prev = cei_clean.iloc[-2]
            delta = last["CEI"] - prev["CEI"]
            insight_box(
                hallazgo=f"Último CEI: {last['CEI']:.1f}% (variación vs mes anterior: {delta:+.1f} pp).",
                implicacion="CEI cayendo sugiere menor efectividad relativa frente a la mora base del periodo.",
                accion="Revisar estrategia en mora temprana (2-29) y tasas de cura; ajustar ofertas/canales."
            )
        else:
            insight_box(
                hallazgo="CEI disponible con pocos puntos históricos para tendencia.",
                implicacion="Aún no es posible evaluar estabilidad; sirve como línea base.",
                accion="Monitorear mensualmente y comparar contra PAR30/roll rates para detectar deterioro temprano."
            )

# -----------------------------
# TAB 5: Costos & Productividad
# -----------------------------
with tabs[4]:
    st.markdown("### Costo por $ recuperado")
    cei = cei_by_month(df)

    if cei.empty:
        st.info("No se puede calcular sin recuperaciones mensuales.")
    else:
        costs = cei.copy()
        costs["Gasto cobranza"] = monthly_collection_cost
        costs["Costo por $ recuperado"] = np.where(costs["Recuperaciones"] > 0,
                                                   costs["Gasto cobranza"] / costs["Recuperaciones"],
                                                   np.nan)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.line(costs, x="Mes", y="Costo por $ recuperado", markers=True,
                          title="Costo por $ recuperado (menor es mejor)")
            fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            last = costs.dropna(subset=["Costo por $ recuperado"]).tail(1)
            if not last.empty:
                st.metric("Último mes", last["Mes"].iloc[0])
                st.metric("Costo por $", f"{last['Costo por $ recuperado'].iloc[0]:.3f}")

        series = costs["Costo por $ recuperado"].dropna()
        if len(series) >= 3:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            last_val = series.iloc[-1]
            insight_box(
                hallazgo=f"Costo actual {last_val:.3f} (referencia: {quantile_label(last_val, q1, q3)} vs histórico).",
                implicacion="Costo alto implica baja eficiencia (gasto alto o baja recuperación).",
                accion="Automatizar mora temprana, optimizar priorización y evaluar tercerización selectiva para 60+."
            )
        else:
            insight_box(
                hallazgo="Histórico insuficiente para benchmark robusto.",
                implicacion="La métrica sirve como control mensual inicial.",
                accion="Definir meta de reducción (10–15%) y construir baseline con más meses."
            )

# -----------------------------
# TAB 6: Alertas & Seguimiento
# -----------------------------
with tabs[5]:
    st.markdown("### Alertas operativas (perfil: " + thr_pack + ")")
    snap = df[df["PlanoMes"] == selected_month]
    k = portfolio_kpis(snap)

    rr_val = np.nan
    if len(available_months) >= 2:
        m_prev = available_months[-2]
        m_now = available_months[-1]
        mat_last = roll_rate_matrix(df, m_prev, m_now)
        if not mat_last.empty and "Corriente" in mat_last.index and "30-59" in mat_last.columns:
            rr_val = float(mat_last.loc["Corriente", "30-59"]) / 100.0

    alerts = [
        ("NPL (saldo)", k["NPL saldo"], thr_npl),
        ("PAR30", k["PAR30"], thr_par30),
        ("Corriente→30-59", rr_val, thr_roll_c_to_30),
    ]

    def status(val, thr):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "Sin dato"
        return "Riesgo alto" if val >= thr else "En rango"

    a1, a2, a3 = st.columns(3)
    for (name, val, thr), col in zip(alerts, [a1, a2, a3]):
        vtxt = pct01(val) if name != "Corriente→30-59" else ("—" if np.isnan(val) else f"{val*100:.1f}%")
        col.markdown(f"**{name}**")
        col.markdown(f"<div class='insight'><span><b>Valor:</b> {vtxt}<br><b>Umbral:</b> {thr*100:.1f}%<br><b>Estado:</b> {status(val, thr)}</span></div>", unsafe_allow_html=True)

    insight_box(
        hallazgo="Alertas enfocadas en indicadores líderes (migración) y severidad (PAR).",
        implicacion="Cuando se activa un líder (Corriente→30-59), los rezagados (PAR60/PAR90) suelen empeorar en los próximos cortes.",
        accion="Ajustar capacidad a mora temprana y reforzar prevención; monitorear semanalmente buckets y curas."
    )

    st.markdown("### Seguimiento recomendado")
    st.markdown("""
- Cadencia semanal para mora temprana y migración.  
- Cadencia mensual para CEI y costos.  
- Enriquecer con contactabilidad y promesas si se dispone de esa data.
""")

# -----------------------------
# TAB 7: Estrategia (90 días)
# -----------------------------
with tabs[6]:
    st.markdown("### Plan de acción (90 días)")
    st.markdown("""
**Foco operativo**
- Prevención y normalización en Corriente / 2-29.
- Curar 30-59 antes de que escale a 60+.
- Definir escalamiento económico para 60+ y 90+.

**Metas sugeridas**
- Reducir PAR30 en 1–2 pp.
- Mejorar CEI en 5–10 pp.
- Reducir costo por $ recuperado en 10–15%.
""")

    insight_box(
        hallazgo="La mejora de resultados viene de intervenir antes de que la mora se vuelva severa.",
        implicacion="Si el roll rate hacia 30+ se mantiene alto, la cartera tenderá a deteriorarse aunque la cobranza sea reactiva.",
        accion="Medir y optimizar prevención + curas tempranas; ajustar oferta y canal de contacto por cohorte/bucket."
    )
