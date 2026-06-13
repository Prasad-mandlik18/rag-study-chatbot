import streamlit as st

from src.loader import save_uploaded_file, extract_text_from_pdf
from src.chunker import chunk_text
from src.embeddings import generate_embeddings, generate_query_embedding
from src.vector_store import VectorStore
from src.retriever import retrieve_chunks
from src.llm import generate_rag_answer
# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="RAG Study Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Study Chatbot 🚀")
st.write("Upload your study materials and ask questions about them.")

# ---------------------------
# File Upload
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------------------
# Process Uploaded Files
# ---------------------------
if uploaded_files:

    all_chunks = []

    for uploaded_file in uploaded_files:
        st.subheader(f"📄 {uploaded_file.name}")

        # Extract text
        # Save PDF to disk
        saved_path = save_uploaded_file(uploaded_file)

        # Read text from saved PDF
        extracted_text = extract_text_from_pdf(saved_path)

        # Create chunks
        chunks = chunk_text(extracted_text)

        # Collect chunks from all uploaded PDFs
        all_chunks.extend(chunks)

        st.success(f"Created {len(chunks)} chunks from {uploaded_file.name}")

        with st.expander(f"View first 3 chunks of {uploaded_file.name}"):
            for i, chunk in enumerate(chunks[:3]):
                st.markdown(f"### Chunk {i+1}")
                st.write(chunk)

    # ---------------------------
    # Generate Embeddings
    # ---------------------------
    embeddings = generate_embeddings(all_chunks)

    st.success("✅ Embeddings generated successfully!")

    st.write(f"Total Chunks: {len(all_chunks)}")
    st.write(f"Total Embeddings: {len(embeddings)}")
    st.write(f"Embedding Dimension: {len(embeddings[0])}")

    # ---------------------------
    # Create FAISS Vector Store
    # ---------------------------
    vector_store = VectorStore()

    vector_store.add_embeddings(
        embeddings=embeddings,
        chunks=all_chunks
    )

    vector_store.save()

    st.success("✅ FAISS database saved successfully!")

    st.success("✅ Embeddings stored in FAISS!")

    # ---------------------------
    # Ask Questions
    # ---------------------------
    st.divider()
    st.subheader("💬 Ask Questions")

question = st.text_input("Ask a question about your uploaded PDFs")

if question:
    # Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks(
        vector_store=vector_store,
        question=question,
        top_k=3
    )

    # Generate answer with Gemini
    answer = generate_rag_answer(
        question=question,
        context_chunks=retrieved_chunks
    )

    st.subheader("🤖 Answer")
    st.write(answer)

    with st.expander("📚 Retrieved Context"):
        for i, chunk in enumerate(retrieved_chunks, start=1):
            st.markdown(f"### Chunk {i}")
            st.write(chunk)