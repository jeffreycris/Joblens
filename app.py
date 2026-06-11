"""
app.py

JobLens Streamlit UI.
Run with: streamlit run app.py
"""

import streamlit as st
from rag import load_qa_chain, ask

# --- Page config ---
st.set_page_config(
    page_title="JobLens",
    page_icon="🔍",
    layout="centered"
)

# --- Load the RAG chain once and cache it ---
# @st.cache_resource means this only runs once when the app starts,
# not on every question. Saves ~10 seconds per query.
@st.cache_resource
def get_chain():
    return load_qa_chain()

# --- Header ---
st.title("🔍 JobLens")
st.markdown("Ask questions about your job market using your real job tracker data.")
st.divider()

# --- Example questions to help users get started ---
st.markdown("**Try asking:**")
col1, col2 = st.columns(2)
with col1:
    st.markdown("- What skills show up most in data scientist roles?")
    st.markdown("- What do fintech companies look for?")
with col2:
    st.markdown("- Which roles mention Python and SQL together?")
    st.markdown("- What locations come up most often?")

st.divider()

# --- Question input ---
question = st.text_input(
    "Your question",
    placeholder="e.g. What machine learning frameworks are most in demand?"
)

# --- Answer ---
if question:
    chain = get_chain()

    with st.spinner("Searching your job data..."):
        answer, sources = ask(chain, question)

    st.markdown("### Answer")
    st.write(answer)

    # Show retrieved source chunks so user can see what the LLM read
    with st.expander(f"📄 Sources ({len(sources)} JD chunks retrieved)"):
        for i, doc in enumerate(sources):
            st.markdown(f"**Chunk {i+1}**")
            st.text(doc.page_content[:400])
            st.divider()
