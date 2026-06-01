import os
import subprocess
import time
import streamlit as st
import pandas as pd
import ollama
from profiler import (
    CHECKS, ask_ollama_streaming, ask_ollama_batched,
    ask_ollama_rules, validate_rules, quick_audit, build_report
)

st.set_page_config(
    page_title="DQ Profiler",
    page_icon="🔍",
    layout="wide"
)

# start ollama once per session
if "ollama_started" not in st.session_state:
    subprocess.run(["pkill", "ollama"], capture_output=True)
    time.sleep(1)
    subprocess.Popen(
        ["ollama", "serve"],
        env={**os.environ, "OLLAMA_NUM_PARALLEL": "8"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)
    st.session_state.ollama_started = True

# -------------------------------------------------------------------
# Sidebar: chat
# -------------------------------------------------------------------
with st.sidebar:
    st.title("💬 Follow-up Chat")
    st.caption("Ask questions about the analysis after running it.")

    if "messages" in st.session_state and len(st.session_state.messages) > 1:
        for msg in st.session_state.messages[1:]:
            st.chat_message(msg["role"]).markdown(msg["content"])

        user_input = st.chat_input("Ask a follow-up...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").markdown(user_input)
            with st.chat_message("assistant"):
                reply = st.write_stream(
                    chunk["message"]["content"]
                    for chunk in ollama.chat(
                        model="gemma3:4b",
                        messages=st.session_state.messages,
                        stream=True
                    )
                )
            st.session_state.messages.append({"role": "assistant", "content": reply})
    elif "pitfalls" in st.session_state:
        st.chat_input("Ask a follow-up...")
        st.caption("No follow-up questions yet.")
    else:
        st.info("Run an analysis first to enable the chat.")

# -------------------------------------------------------------------
# Main content
# -------------------------------------------------------------------
st.title("🔍 Local GenAI Data Quality Profiler")
st.caption("Fully offline — powered by Ollama + gemma3:4b")

st.divider()

# step 1: upload
st.subheader("Step 1 — Upload a dataset")
uploaded_file = st.file_uploader("CSV files only", type="csv", label_visibility="collapsed")

if not uploaded_file:
    st.info("Upload a CSV file to get started.")
    st.stop()

df = pd.read_csv(uploaded_file)

# reset on new file
if st.session_state.get("last_file") != uploaded_file.name:
    for key in ["profile", "pitfalls", "messages", "rules", "results", "failing_rows"]:
        st.session_state.pop(key, None)
    st.session_state.last_file = uploaded_file.name

st.success(f"{uploaded_file.name} — {len(df):,} rows, {len(df.columns)} columns")

with st.expander("Preview dataset"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Full statistical profile"):
    if "profile" not in st.session_state:
        profile = {"row_count": len(df), "column_count": len(df.columns)}
        for name, check in CHECKS:
            profile[name] = check(df)
        st.session_state.profile = profile
    st.json(st.session_state.profile)

st.divider()

# step 2: instant audit
st.subheader("Step 2 — Instant Audit")
st.caption("Runs immediately on the full dataset, no AI needed.")

audit_df = quick_audit(df)

if audit_df.empty:
    st.success("No obvious issues found programmatically.")
else:
    critical = len(audit_df[audit_df["severity"] == "critical"])
    warnings = len(audit_df[audit_df["severity"] == "warning"])
    info = len(audit_df[audit_df["severity"] == "info"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Critical", critical, delta=None)
    c2.metric("Warnings", warnings, delta=None)
    c3.metric("Info", info, delta=None)

    def color_severity(val):
        colors = {
            "critical": "background-color: #ff4b4b; color: white",
            "warning":  "background-color: #ffa500; color: white",
            "info":     "background-color: #4b8bff; color: white"
        }
        return colors.get(val, "")

    st.dataframe(
        audit_df.style.applymap(color_severity, subset=["severity"]),
        use_container_width=True,
        hide_index=True
    )

st.divider()

# step 3: ai analysis
st.subheader("Step 3 — AI Analysis")
st.caption("Sends a sample of suspicious rows to Ollama for interpretation.")

col_mode, col_batch, col_btn = st.columns([2, 2, 1])
with col_mode:
    mode = st.radio("Mode", ["Sample (fast)", "Full dataset (batched)"], horizontal=True)
with col_batch:
    batch_size = st.slider("Batch size", 50, 200, 100, 50) if "Full" in mode else None
with col_btn:
    st.write("")
    st.write("")
    run_analysis = st.button("Run AI analysis", type="primary", use_container_width=True)

if run_analysis:
    if "profile" not in st.session_state:
        profile = {"row_count": len(df), "column_count": len(df.columns)}
        for name, check in CHECKS:
            profile[name] = check(df)
        st.session_state.profile = profile

    if "Full" in mode:
        progress_bar = st.progress(0, text="Analyzing batches...")
        def update_progress(current, total):
            progress_bar.progress(
                current / total,
                text=f"Analyzing batch {current} of {total}..."
            )
        result = ask_ollama_batched(
            st.session_state.profile, df,
            batch_size=batch_size,
            progress_callback=update_progress
        )
        progress_bar.empty()
    else:
        with st.spinner("Asking Ollama..."):
            result = "".join(ask_ollama_streaming(st.session_state.profile, df))

    st.session_state.pitfalls = result
    st.session_state.messages = [{"role": "assistant", "content": result}]

if "pitfalls" in st.session_state:
    st.markdown(st.session_state.pitfalls)
else:
    st.info("Click 'Run AI analysis' to generate findings.")

st.divider()

# step 4: rule validation
st.subheader("Step 4 — Rule Validation")
st.caption("Ollama generates testable rules, then Python runs them on every row.")

if st.button("Generate and validate rules", type="primary"):
    if "profile" not in st.session_state:
        profile = {"row_count": len(df), "column_count": len(df.columns)}
        for name, check in CHECKS:
            profile[name] = check(df)
        st.session_state.profile = profile

    with st.spinner("Generating rules..."):
        rules = ask_ollama_rules(st.session_state.profile, df)

    if not rules:
        st.warning("Could not parse rules from model output. Try again.")
    else:
        with st.spinner("Validating against full dataset..."):
            results_df = validate_rules(df, rules)

        failing_rows = {}
        for _, row in results_df.iterrows():
            if row["failing_rows"] > 0 and not str(row["filter"]).startswith("ERROR"):
                try:
                    mask = eval(row["filter"], {"df": df})
                    failing_rows[row["description"]] = df[mask].head(50)
                except Exception:
                    pass

        st.session_state.rules = rules
        st.session_state.results = results_df
        st.session_state.failing_rows = failing_rows

if "results" in st.session_state:
    results_df = st.session_state.results

    passed = int(results_df["passed"].sum())
    failed = int((~results_df["passed"]).sum())
    r1, r2 = st.columns(2)
    r1.metric("Rules passed", passed)
    r2.metric("Rules failed", failed)

    def color_passed(val):
        return ("background-color: #2ecc71; color: white" if val
                else "background-color: #ff4b4b; color: white")

    st.dataframe(
        results_df[["column", "description", "failing_rows", "fail_pct", "passed"]]
        .style.applymap(color_passed, subset=["passed"]),
        use_container_width=True,
        hide_index=True
    )

    failed_rules = results_df[results_df["failing_rows"] > 0]
    if not failed_rules.empty:
        st.subheader("Inspect failing rows")
        selected = st.selectbox(
            "Select a rule",
            failed_rules["description"].tolist(),
            label_visibility="collapsed"
        )
        selected_filter = failed_rules[
            failed_rules["description"] == selected
        ]["filter"].values[0]
        if not str(selected_filter).startswith("ERROR"):
            mask = eval(selected_filter, {"df": df})
            st.dataframe(df[mask].head(50), use_container_width=True, hide_index=True)
else:
    st.info("Click 'Generate and validate rules' to run validation.")

st.divider()

# step 5: export
st.subheader("Step 5 — Export Report")
st.caption("Generates a full Excel report with all findings, even sections not yet run.")

if st.button("Generate and download report", type="primary"):
    if "profile" not in st.session_state:
        profile = {"row_count": len(df), "column_count": len(df.columns)}
        for name, check in CHECKS:
            profile[name] = check(df)
        st.session_state.profile = profile

    if "pitfalls" not in st.session_state:
        with st.spinner("Running AI analysis..."):
            result = "".join(ask_ollama_streaming(st.session_state.profile, df))
        st.session_state.pitfalls = result
        st.session_state.messages = [{"role": "assistant", "content": result}]

    if "results" not in st.session_state:
        with st.spinner("Generating and validating rules..."):
            rules = ask_ollama_rules(st.session_state.profile, df)
        if rules:
            results_df = validate_rules(df, rules)
            st.session_state.rules = rules
            st.session_state.results = results_df
            failing_rows = {}
            for _, row in results_df.iterrows():
                if row["failing_rows"] > 0 and not str(row["filter"]).startswith("ERROR"):
                    try:
                        mask = eval(row["filter"], {"df": df})
                        failing_rows[row["description"]] = df[mask].head(50)
                    except Exception:
                        pass
            st.session_state.failing_rows = failing_rows

    with st.spinner("Building report..."):
        report_bytes = build_report(
            filename=uploaded_file.name,
            df=df,
            audit_df=audit_df,
            ai_analysis=st.session_state.get("pitfalls"),
            rules_df=st.session_state.get("results"),
            failing_rows=st.session_state.get("failing_rows"),
        )

    st.download_button(
        label="📥 Download Excel report",
        data=report_bytes,
        file_name=f"dq_report_{uploaded_file.name.replace('.csv', '')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )