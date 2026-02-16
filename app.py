import streamlit as st
from ipl_agent import IPLAgent

st.set_page_config(page_title="Startup Failure RAG Chatbot", layout="wide")
st.title("🚀 Startup Failure Intelligence (RAG)")
st.markdown("*Powered by Local LLM (Ollama) + Vector Search (FAISS)*")

if "agent" not in st.session_state:
    with st.spinner("⏳ Loading RAG system..."):
        st.session_state.agent = IPLAgent()

# Create columns for input
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input("Ask about startup failures, sectors, or funding:", placeholder="e.g., Why did Theranos fail?")

with col2:
    search_btn = st.button("🔍 Search", use_container_width=True)

if search_btn and query:
    with st.spinner("🔄 Searching startup data..."):
        response = st.session_state.agent.query(query)
        
        # Extract answer and sources
        answer = response.get("result", "No answer found.")
        sources = response.get("source_documents", [])
        
        # Display answer
        st.markdown("### 📊 Answer")
        st.write(answer)
        
        # Display sources if available
        if sources:
            st.markdown("### 📚 Sources")
            with st.expander(f"View {len(sources)} source documents", expanded=False):
                for idx, doc in enumerate(sources, 1):
                    with st.container(border=True):
                        st.markdown(f"**Source {idx}**: {doc.metadata.get('file', 'Unknown')}")
                        st.caption(f"Startup: {doc.metadata.get('name', 'N/A')} | Sector: {doc.metadata.get('sector', 'N/A')}")
                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

