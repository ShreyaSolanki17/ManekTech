import streamlit as st

from rag_qa import answer_question

CAAD_BASE_URL = "https://caad.org.pt/tributario/decisoes/"


def build_caad_url(case_id: str) -> str:
    if case_id.startswith("http"):
        return case_id
    return f"{CAAD_BASE_URL}{case_id}"


def extract_summary(document: str) -> str:
    summary_marker = "Summary: "
    if summary_marker in document:
        return document.split(summary_marker, 1)[1].strip()
    return document.strip()


def case_label(case_id: str) -> str:
    if "id=" in case_id:
        return f"CAAD case {case_id.split('id=')[-1]}"
    return case_id

st.set_page_config(page_title="CAAD Legal RAG", page_icon=":scales:")

st.title("CAAD Legal RAG")

question = st.text_input("Ask a question about the cases")

top_k = st.slider("Number of cases to retrieve", min_value=1, max_value=5, value=3)

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving and answering..."):
        answer, case_ids, docs = answer_question(question, top_k=top_k)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved case IDs")
    if case_ids:
        for index, case_id in enumerate(case_ids):
            summary_text = extract_summary(docs[index]) if index < len(docs) else ""
            st.markdown(f"- [{case_label(case_id)}]({build_caad_url(case_id)})")
            if summary_text:
                st.caption(summary_text)
    else:
        st.write("None")
