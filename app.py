from embeddings import search, rag_answer, build_context
import streamlit as st

st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("📄 Research Assistant")

query = st.text_input("Ask a research question:")

if st.button("Search") and query:
    with st.spinner("Searching papers..."):
        answer = rag_answer(query)
    st.markdown("### Answer")
    st.write(answer)
