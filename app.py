import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# Estilo personalizado
st.markdown("""
    <style>
        .stApp {
            background-color: #f9f4ef;
        }
        h1, h2, h3, h4, h5, h6, p, label {
            color: #222 !important;
        }
        .stButton>button {
            background-color: #2d2d2d;
            color: white;
            border-radius: 10px;
            padding: 0.5em 1em;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #000000;
        }
        .stTextInput, .stTextArea, .stFileUploader {
            background-color: #fff;
            border: 1px solid #ccc;
            border-radius: 8px;
        }
        .stTextInput>div>input, .stTextArea textarea {
            color: #222 !important;
        }
        .stSidebar, .stSidebar .sidebar-content {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Chatea con tu PDF")
st.markdown("Interactúa con el contenido de un documento PDF cargado por ti. El modelo responderá tus preguntas con base en el texto del archivo.")
st.write("Versión de Python:", platform.python_version())

# Imagen
try:
    image = Image.open("robot.jpg")
    st.image(image, width=450)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Clave de API
ke = st.text_input("Clave de OpenAI", type="password")
if ke:
    os.environ["OPENAI_API_KEY"] = ke
else:
    st.warning("Ingresa tu clave de API para continuar")

# Carga de PDF
pdf = st.file_uploader("Carga un archivo PDF", type="pdf")

if pdf is not None and ke:
    try:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.info(f"Texto extraído: {len(text)} caracteres")

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.subheader("Consulta el contenido del documento")
        user_question = st.text_area("Escribe tu pregunta...")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            st.markdown("#### Respuesta:")
            st.markdown(response)

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API para continuar")
else:
    st.info("Carga un PDF para empezar")
