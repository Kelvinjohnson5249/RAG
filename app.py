import streamlit as st
from rag_system import rag_advanced, rag_retreiver, llm

# Page configuration
st.set_page_config(
    page_title="AI RAG Assistant",
    layout="wide"
)

st.title("📚 AI RAG Assistant")
st.caption("Retrieval-Augmented Generation using ChromaDB + SentenceTransformers + Groq")

# Sidebar settings
st.sidebar.header("⚙ Retrieval Settings")
top_k = st.sidebar.slider("Top K Results", 1, 10, 3)
min_score = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.1)

# User input
query = st.text_input("Ask a question about your documents")

# Button trigger
if st.button("Generate Answer"):

    if not query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Retrieving documents and generating answer..."):

            result = rag_advanced(
                query=query,
                retriever=rag_retreiver,
                llm=llm,
                top_k=top_k,
                min_score=min_score,
                return_context=True
            )

        # ------------------ Answer ------------------
        st.subheader("Answer")
        st.write(result["answer"])

        # ------------------ Confidence ------------------
        st.subheader("Confidence Score")
        st.progress(float(result["confidence"]))

        # ------------------ Sources ------------------
        st.subheader("Sources")

        if result["sources"]:
            for i, source in enumerate(result["sources"], 1):
                with st.expander(f"Source {i}"):
                    st.write(f"File: {source['source']}")
                    st.write(f"Page: {source['page']}")
                    st.write(f"Similarity Score: {round(source['score'], 3)}")
                    st.write(source["preview"])
        else:
            st.write("No sources found.")

        # ------------------ Retrieved Context ------------------
        with st.expander("View Retrieved Context"):
            st.write(result.get("context", ""))
