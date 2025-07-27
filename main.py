import streamlit as st
import tempfile
import os
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import requests
from huggingface_hub import InferenceClient
from streamlit_pdf_viewer import pdf_viewer

hf_token = <Your Token> # Refer: https://huggingface.co/settings/tokens
# HuggingFace Inference API call for LLM
def hf_inference_api_call(messages, model="meta-llama/Llama-3.1-8B-Instruct", hf_token=hf_token):
    client = InferenceClient(
        provider="auto",
        api_key=hf_token,
    )
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return completion.choices[0].message.content

# Embedding API call
API_URL_EMBED = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
headers_embed = {"Authorization": f"Bearer {hf_token}"}

def get_embedding(text):
    response = requests.post(API_URL_EMBED, headers=headers_embed, json={"inputs": text})
    response.raise_for_status()
    return response.json()  # This will be a list of floats (the embedding)

st.set_page_config(layout="centered")
st.title("PDF Retrieval-Augmented Generation (RAG) Q&A")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.subheader("Your document")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    pdf_preview_placeholder = st.empty()

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # PDF Preview or download in the sidebar
    with st.sidebar:
        try:
            st.markdown("**PDF Preview:**")
            pdf_bytes = open(tmp_path, "rb").read()
            pdf_viewer(input=pdf_bytes, width=350)
        except Exception:
            st.info("PDF preview not supported. You can download and view the file:")
            st.download_button("Download PDF", data=open(tmp_path, "rb").read(), file_name=uploaded_file.name)

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    all_text = "\n".join([doc.page_content for doc in documents])

    # Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(all_text)
    st.info(f"Document loaded and split into {len(chunks)} chunks.")

    # Embed all chunks using the API
    with st.spinner("Embedding chunks via HuggingFace API..."):
        chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

    # Robust check for empty chunks or embeddings
    if len(chunks) == 0 or len(chunk_embeddings) == 0:
        st.error("No text chunks or embeddings were generated from the document. Please check your PDF content.")
    else:
        chunk_embeddings = np.array(chunk_embeddings).astype('float32')
        index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
        index.add(chunk_embeddings)

        # Display previous chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Assistant:** {msg['content']}")

        # Question input and answer area in the main area
        question = st.text_input("Ask a question about the document:")
        if question:
            # Embed the question using the API
            with st.spinner("Embedding question via HuggingFace API..."):
                question_embedding = np.array(get_embedding(question)).astype('float32').reshape(1, -1)
            # Search FAISS for top 5 relevant chunks
            D, I = index.search(question_embedding, 5)
            relevant_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
            context = "\n".join(relevant_chunks)
            st.write(context)
            # Build messages for LLM: previous chat + new context/question
            messages = st.session_state.chat_history.copy()
            # Add context as system message
            messages.append({"role": "system", "content": f"Context: {context}"})
            # Add the new user question
            messages.append({"role": "user", "content": question})
            with st.spinner("Querying LLM via HuggingFace API..."):
                try:
                    answer = hf_inference_api_call(messages, model="meta-llama/Llama-3.1-8B-Instruct", hf_token=hf_token)
                    st.success("Answer:")
                    st.write(answer)
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error from HuggingFace API: {e}")

    # Clean up temp file
    os.remove(tmp_path) 