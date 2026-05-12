import streamlit as st

from rag_qa import answer_question

st.set_page_config(page_title="CAAD Legal RAG", page_icon=":scales:")

st.title("CAAD Legal RAG")

question = st.text_input("Ask a question about the cases")

top_k = st.slider("Number of cases to retrieve", min_value=1, max_value=5, value=3)

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving and answering..."):
        answer, case_ids = answer_question(question, top_k=top_k)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved case IDs")
    st.write(", ".join(case_ids) if case_ids else "None")
