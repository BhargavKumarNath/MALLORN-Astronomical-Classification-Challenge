import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="MALLORN | Astronomical Classification",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #4C1D95; /* Deep Purple */
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #6D28D9;
        margin-top: 1.5rem;
    }
    .card {
        background-color: #F5F3FF; /* Light Purple Bg */
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .metric-box {
        text-align: center;
        padding: 10px;
        background: white;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }
    .highlight {
        color: #D97706; /* Amber */
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/telescope.png", width=80)
    st.title("Project MALLORN")
    st.caption("Classifying Tidal Disruption Events (TDEs) from LSST Lightcurves")
    
    nav = st.radio("Navigation", [
        "1. Mission Briefing",
        "2. Data Exploration (EDA)",
        "3. The Strategic Pivot",
        "4. Champion vs Challenger",
        "5. Inference Lab"
    ])
    
    st.divider()
    st.info("👨‍🔬 **Coder:** Bhargav Kumar Nath\n\n🏆 **Rank:** 7th Place (Leaderboard)\n\n⚡ **Key Insight:** Feature Eng > Deep Learning for sparse data.")

# Helper Functions (Simulation)

def generate_lightcurve(is_tde=False):
    """Simulates a multi-band astronomical lightcurve."""
    time = np.sort(np.random.uniform(59000, 60000, 50))
    filters = np.random.choice(['u', 'g', 'r', 'i', 'z', 'y'], 50)
    
    # Base flux
    flux = np.random.normal(10, 2, 50)
    
    if is_tde:
        # TDEs have a massive flare
        peak_time = 59500
        flare = 500 * np.exp(-0.5 * ((time - peak_time) / 50)**2) * (np.random.rand(50) * 0.5 + 0.5)
        # Blue filters (u, g) are brighter
        filter_boost = np.array([1.5 if f in ['u', 'g'] else 0.8 for f in filters])
        flux += flare * filter_boost
        
    flux_err = np.abs(flux * 0.1) + np.random.normal(0, 1, 50)
    
    return pd.DataFrame({'Time (MJD)': time, 'Flux': flux, 'Flux Error': flux_err, 'Filter': filters})

# Pages

if nav == "1. Mission Briefing":
    st.markdown('<div class="main-header">The Hunt for Tidal Disruption Events</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **The Objective:** Build a machine learning system to identify rare **Tidal Disruption Events (TDEs)**—stars being torn apart by black holes—using simulated data from the Vera C. Rubin Observatory (LSST).
        
        **The Core Challenge:**
        * **Extreme Rarity:** TDEs are only **4.86%** of the universe (dataset).
        * **Data Sparsity:** Lightcurves are irregular, noisy, and missing data points.
        * **The "Pivot":** We initially bet on Deep Learning (RNNs) but pivoted to Gradient Boosting (LightGBM) with automated feature engineering when DL failed to capture sparse signals.
        """)
        
        st.markdown("### 🏗 System Architecture")
        st.graphviz_chart("""
        digraph G {
            rankdir=LR;
            node [shape=box, style="filled,rounded", fontname="Arial"];
            
            subgraph cluster_0 {
                label = "Ingestion";
                style=filled; color="#F3F4F6";
                Raw [label="Raw Lightcurves\n(6 Filters)", fillcolor="#E0E7FF"];
                Meta [label="Metadata\n(Redshift Z)", fillcolor="#E0E7FF"];
            }
            
            subgraph cluster_1 {
                label = "The Fork (Experimentation)";
                style=filled; color="#FEF3C7";
                
                DL [label="path B: Challenger\n(Bi-Directional GRU)", fillcolor="#FECACA"];
                FE [label="path A: Champion\n(tsfresh Feature Eng)", fillcolor="#D1FAE5"];
            }
            
            subgraph cluster_2 {
                label = "Deployment";
                style=filled; color="#E0F2FE";
                LGBM [label="LightGBM\n(Gradient Boosting)", fillcolor="#10B981", fontcolor="white"];
                Thresh [label="Threshold Optimization\n(P > 0.35)", shape=diamond];
                Pred [label="TDE / Non-TDE"];
            }
            
            Raw -> DL;
            Raw -> FE;
            Meta -> DL;
            Meta -> FE;
            
            DL -> LGBM [label="Failed (F1 0.18)", style=dotted, color="red"];
            FE -> LGBM [label="Success (F1 0.53)", style=bold, color="green"];
            
            LGBM -> Thresh -> Pred;
        }
        """, width='stretch')

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Final F1 Score", "0.5353", "7th Place")
        st.markdown("---")
        st.metric("Class Imbalance", "1 : 20", "Rare Event")
        st.markdown("---")
        st.caption("Tech Stack")
        st.code("Python, LightGBM\ntsfresh, PyTorch\nOptuna", language="text")
        st.markdown('</div>', unsafe_allow_html=True)

elif nav == "2. Data Exploration (EDA)":
    st.markdown('<div class="main-header">Understanding the Signal</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 1. The Class Imbalance")
        st.markdown("The dataset mimics the real universe: interesting events are needles in a haystack.")
        
        # Pie chart
        df_class = pd.DataFrame({'Label': ['Non-TDE', 'TDE'], 'Count': [2895, 148]})
        fig_pie = px.pie(df_class, values='Count', names='Label', color='Label', 
                         color_discrete_map={'Non-TDE': 'lightgrey', 'TDE': '#8B5CF6'},
                         hole=0.4)
        st.plotly_chart(fig_pie, width='stretch')
        
    with col2:
        st.markdown("### 2. Physical Invariants")
        st.markdown("Metadata like **Redshift (Z)** and **Galactic Extinction (EBV)** provide context, but their distributions overlap heavily between classes.")
        
        # Simulated overlap histograms
        z_tde = np.random.beta(2, 5, 1000)
        z_non = np.random.beta(2, 5, 1000) # Similar distribution
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=z_non, name='Non-TDE', opacity=0.75, marker_color='grey'))
        fig_hist.add_trace(go.Histogram(x=z_tde, name='TDE', opacity=0.75, marker_color='#8B5CF6'))
        fig_hist.update_layout(barmode='overlay', title="Redshift (Z) Distribution Overlap", xaxis_title="Redshift")
        st.plotly_chart(fig_hist, width='stretch')

    st.markdown("### 3. Visualizing Lightcurves")
    st.markdown("Toggle between a standard object and a violent TDE to see the difference in signal.")
    
    obj_type = st.radio("Select Object Type:", ["Background Noise (Non-TDE)", "Tidal Disruption Event (TDE)"], horizontal=True)
    is_tde_sim = "TDE" in obj_type
    
    df_lc = generate_lightcurve(is_tde=is_tde_sim)
    
    fig_lc = px.scatter(df_lc, x="Time (MJD)", y="Flux", error_y="Flux Error", color="Filter",
                        color_discrete_map={'u': 'blue', 'g': 'green', 'r': 'red', 'i': 'orange', 'z': 'black', 'y': 'brown'},
                        title=f"Simulated Lightcurve: {obj_type}")
    fig_lc.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
    st.plotly_chart(fig_lc, width='stretch')
    
    if is_tde_sim:
        st.success("Note the massive flare around MJD 59500, especially in the blue ('u', 'g') bands. This is the signature we need to capture.")

elif nav == "3. The Strategic Pivot":
    st.markdown('<div class="main-header">The Pivot: Feature Engineering vs. Deep Learning</div>', unsafe_allow_html=True)
    
    st.markdown("""
    We hypothesized that Deep Learning (RNNs) would learn the temporal patterns automatically. **We were wrong.**
    The sparsity of the data meant the RNNs mostly learned noise. We pivoted to **Automated Feature Engineering** using `tsfresh`.
    """)
    
    st.markdown("#### The Architecture Fork")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.error("❌ Path B: Deep Sequence Modeling")
        st.markdown("**Method:** Bi-Directional GRU + Attention.")
        st.markdown("**Outcome:** F1 Score ~0.18")
        st.markdown("**Failure Mode:** Overfitting to gaps in data. The model couldn't handle the irregular sampling.")
        
    with col2:
        st.success("✅ Path A: Feature Extraction (tsfresh)")
        st.markdown("**Method:** Extract 1000+ stats per filter -> Select top 198.")
        st.markdown("**Outcome:** F1 Score ~0.53")
        st.markdown("**Why it worked:** Aggregated stats (slope, skew, mean) are robust to missing data points.")

    st.markdown("### Feature Importance (The 'Why')")
    st.markdown("What features did LightGBM find most useful?")
    
    # Mock feature importance data based on readme
    features = ['g_Flux_min', 'Flux_Ratio_skew', 'Redshift_Z', 'u_Flux_max', 'r_Flux_kurtosis', 'i_Flux_mean']
    importance = [85, 78, 72, 65, 50, 45]
    
    df_imp = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance')
    
    fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                     color='Importance', color_continuous_scale='Purples',
                     title="Top Predictive Features (LightGBM)")
    st.plotly_chart(fig_imp, width='stretch')
    
    st.info("**Interpretation:** The model relies heavily on the **minimum brightness in the green band ('g')** and the **skew of the flux ratio**. This aligns with the physics of TDEs, which are bright and blue.")

elif nav == "4. Champion vs Challenger":
    st.markdown('<div class="main-header">Model Showdown</div>', unsafe_allow_html=True)
    
    # Data from Readme
    results = pd.DataFrame({
        'Model': ['LightGBM (Basic Stats)', 'LightGBM (Interp. Colors)', 'RNN (Single Channel)', 'LightGBM (tsfresh)'],
        'F1 Score': [0.4281, 0.4974, 0.1800, 0.5225],
        'Type': ['Baseline', 'Experiment', 'Deep Learning', 'Champion']
    })
    
    fig_res = px.bar(results, x='Model', y='F1 Score', color='Type', 
                     text='F1 Score', title="Cross-Validation Performance Comparison",
                     color_discrete_map={'Champion': '#10B981', 'Deep Learning': '#EF4444', 'Baseline': 'grey', 'Experiment': 'blue'})
    fig_res.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_res.update_layout(yaxis_range=[0, 0.6])
    st.plotly_chart(fig_res, width='stretch')
    
    st.markdown("""
    ### Key Takeaways
    1.  **Complexity $\\neq$ Performance:** The complex RNN performed worse than the simplest baseline.
    2.  **Domain Knowledge Matters:** `tsfresh` captures characteristics (like flares) better than raw point-clouds for decision trees.
    3.  **Color Interpolation Failed:** Attempting to manually interpolate color between filters (to fix sparsity) added noise, lowering the score from 0.52 to 0.49.
    """)

elif nav == "5. Inference Lab":
    st.markdown('<div class="main-header">Inference & Thresholding</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Since the data is imbalanced, the default decision threshold of `0.5` is suboptimal. 
    We optimized the threshold to maximize F1 (balancing Precision and Recall).
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Settings")
        threshold = st.slider("Decision Threshold (Probability)", 0.0, 1.0, 0.35, 0.01)
        st.caption("Lower threshold = More Recall (Catch more TDEs, but more false alarms).")
        
        # Simulated probability
        true_prob = 0.42
        st.markdown(f"**Sample Object Prediction:**")
        st.markdown(f"Model Confidence: `{true_prob:.2f}`")
        
        if true_prob >= threshold:
            st.error(f"**Classified as TDE** (Confidence > {threshold})")
        else:
            st.success(f"**Classified as Non-TDE** (Confidence < {threshold})")
            
    with col2:
        # Simulate F1 Curve
        thresh_x = np.linspace(0.01, 0.99, 100)
        # F1 typically peaks around the ratio of positives, or slightly higher for strong models
        # Simulating a peak at 0.35
        f1_y = 2 * (thresh_x * (1-thresh_x)) + 0.1 # Dummy curve shape
        f1_y = 0.5 * np.exp(-10 * (thresh_x - 0.35)**2)
        
        fig_thresh = px.line(x=thresh_x, y=f1_y, labels={'x': 'Threshold', 'y': 'F1 Score'}, title="Threshold Optimization Curve")
        fig_thresh.add_vline(x=0.35, line_dash="dash", line_color="green", annotation_text="Optimal (0.35)")
        fig_thresh.add_vline(x=threshold, line_dash="dot", line_color="red", annotation_text="Current")
        st.plotly_chart(fig_thresh, width='stretch')

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data Source: MALLORN Challenge")