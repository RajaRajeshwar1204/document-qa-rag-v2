import streamlit as st

from src.graph import ask_question


st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="centered"
)


st.title("📄 Document Q&A")
st.caption(
    "Ask questions about the documents in the knowledge base."
)


question = st.text_input(
    "Ask a question",
    placeholder="e.g. When was Acme Cloud launched?"
)


if st.button("Ask Question", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):

            answer, verification, sources, attempts = ask_question(
                question
            )

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Verification")

        if verification == "SUPPORTED":
            st.success("SUPPORTED")
        elif verification == "NOT_APPLICABLE":
            st.info("NOT APPLICABLE")
        else:
            st.warning(verification)

        if sources:
            st.markdown("### Sources")

            unique_sources = list(dict.fromkeys(sources))

            for source in unique_sources:
                st.write(f"📄 `{source}`")

        if attempts > 1:
            st.caption(
                f"Answer verified after {attempts} attempts."
            )


st.divider()

st.caption(
    "Built with Python • LangGraph • ChromaDB • "
    "Sentence Transformers • Ollama"
)