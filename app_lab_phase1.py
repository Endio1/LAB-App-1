import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---------- Processing Function ----------
def process_file(file):
    df = pd.read_excel(file)
    #df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 1. Anomaly Score
    df['anomaly_score'] = np.where(df['ace_before'] != df['ace_after'], 1, -1)

    # 2. is_anomaly
    df['is_anomaly'] = df['anomaly_score'] == 1

    # 3. More or Less
    df['more_or_less'] = df.apply(
        lambda row: 'more' if row['ace_after'] > row['ace_before']
        else 'less' if row['ace_after'] < row['ace_before']
        else 'equal',
        axis=1
    )

    # 4. Error (absolute difference)
    df['error'] = df.apply(
        lambda row: abs(row['ace_after'] - row['ace_before']) if row['ace_before'] != row['ace_after'] else 0,
        axis=1
    )

    # Save Table 1
    table1 = df.copy()

    # 5. Summary stats
    avg_error = table1['error'].mean()
    avg_error_pct = (avg_error / table1['ace_before'].mean()) * 100
    total_error = table1['error'].sum()
    capped_pct = min(avg_error_pct, 5.0)

    # 6. Estimated values (for Table 2)
    table2 = table1.copy()
    table2['estimated'] = table2.apply(
        lambda row: row['ace_before'] if row['anomaly_score'] == -1
        else round(row['ace_before'] * (1 + capped_pct / 100), 4),
        axis=1
    )

    # Save both tables
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    time_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    file1 = os.path.join(output_dir, f"table1_raw_{time_id}.xlsx")
    file2 = os.path.join(output_dir, f"table2_corrected_{time_id}.xlsx")

    table1.to_excel(file1, index=False)
    table2.to_excel(file2, index=False)

    return table1, table2, avg_error, avg_error_pct, total_error, capped_pct, file1, file2


# ---------- Streamlit UI ----------
st.set_page_config(page_title="ACE Anomaly Detection Lab", layout="wide")
st.title("🔬 ACE Signal Lab App - Phase 1")

uploaded_file = st.file_uploader("📁 Upload Excel File (timestamp, ace_before, ace_after)", type="xlsx")

if uploaded_file:
    with st.spinner("Processing..."):
        table1, table2, avg_error, avg_error_pct, total_error, capped_pct, file1, file2 = process_file(uploaded_file)

    st.success("✅ File processed successfully!")

    # ------------------- TABLE 1 -------------------
    st.subheader("📄 Table 1 - Raw Detection and Error Summary")
    st.dataframe(table1)

    st.subheader("📊 Error Statistics (Table 1)")
    st.info(f"📉 Total Error: `{total_error:.4f}`")
    st.info(f"📉 Average Error: `{avg_error:.4f}`")
    st.info(f"📉 Average Error Percentage: `{avg_error_pct:.4f}%` (capped at 5%)")

    with open(file1, "rb") as f:
        st.download_button("⬇️ Download Table 1 (Raw)", f, file_name=os.path.basename(file1))

   # ------------------- TABLE 2 -------------------
    st.subheader("📄 Table 2 - Estimated ACE After Correction")

   # Debugging info
    st.write("✅ Table 2 Shape:", table2.shape)
    #st.write("🔍 Table 2 Preview:")
    #st.write(table2.head())  # Show first few rows

   # Try showing dataframe
    try:
     st.dataframe(table2)
    except Exception as e:
     st.error(f"❌ Error displaying Table 2: {e}")


    with open(file2, "rb") as f:
        st.download_button("⬇️ Download Table 2 (Corrected)", f, file_name=os.path.basename(file2))

    # ------------------- STATS -------------------
    st.subheader("📈 Statistical Summary (Table 2)")
    stat_cols = ['ace_before', 'ace_after', 'estimated']
    stats = {
        "Mean": table2[stat_cols].mean(),
        "Variance": table2[stat_cols].var(),
        "Standard Deviation": table2[stat_cols].std(),
        "Min": table2[stat_cols].min(),
        "Max": table2[stat_cols].max(),
    }
    st.dataframe(pd.DataFrame(stats).T.style.format("{:.4f}"))

    # ------------------- CORRELATION GRAPH -------------------
    st.subheader("🔗 Correlation Heatmap")
    corr = table2[stat_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

    # ------------------- LINE PLOTS -------------------
    st.subheader("📊 ACE Time Series (Line Plots)")

    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(x=table2['timestamp'], y=table2['ace_before'], mode='lines', name='ACE Before'))
    fig_all.add_trace(go.Scatter(x=table2['timestamp'], y=table2['ace_after'], mode='lines', name='ACE After'))
    fig_all.add_trace(go.Scatter(x=table2['timestamp'], y=table2['estimated'], mode='lines', name='Estimated'))
    fig_all.update_layout(title="All ACE Signals Over Time", xaxis_title="Timestamp", yaxis_title="Value")
    st.plotly_chart(fig_all, use_container_width=True)

    fig_before_after = px.line(table2, x='timestamp', y=['ace_before', 'ace_after'],
                               title="ACE Before vs After Over Time")
    st.plotly_chart(fig_before_after, use_container_width=True)

    fig_before_estimated = px.line(table2, x='timestamp', y=['ace_before', 'estimated'],
                                   title="ACE Before vs Estimated Over Time")
    st.plotly_chart(fig_before_estimated, use_container_width=True)

    fig_after_estimated = px.line(table2, x='timestamp', y=['ace_after', 'estimated'],
                                  title="ACE After vs Estimated Over Time")
    st.plotly_chart(fig_after_estimated, use_container_width=True)

    # Additional: Dual trace line chart using go
    fig_dual = go.Figure()
    fig_dual.add_trace(go.Scatter(x=table2['timestamp'], y=table2['ace_before'], mode='lines', name='ACE Before', line=dict(color='blue')))
    fig_dual.add_trace(go.Scatter(x=table2['timestamp'], y=table2['estimated'], mode='lines', name='Estimated', line=dict(color='orange')))
    fig_dual.update_layout(title="ACE Before vs Estimated (Bright Colors)", xaxis_title="Timestamp", yaxis_title="Value")
    st.plotly_chart(fig_dual, use_container_width=True)

else:
    st.info("Please upload a valid Excel file with columns: `timestamp`, `ace_before`, `ace_after`.")
    # ------------------- ADDITIONAL PLOTS -------------------

     st.subheader("📦 Boxplot of ACE Signals")
     fig_box = px.box(table2, y=["ace_before", "ace_after", "estimated"], title="Boxplot of ACE Signals")
     st.plotly_chart(fig_box, use_container_width=True)

     st.subheader("🔴 Anomalies Over Time")
     anomalies = table1[table1['is_anomaly']]
     fig_anomaly = px.scatter(anomalies, x='timestamp', y='error',
                              color_discrete_sequence=["red"],
                              title="Anomalies Over Time")
     st.plotly_chart(fig_anomaly, use_container_width=True)

    #python -m streamlit run app.py (to run, write or copy this in the terminal)
