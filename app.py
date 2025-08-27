import os
from string import Template
from urllib.parse import quote_plus
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import numpy as np
import ast

# LangChain imports
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

try:
    from langchain.sql_database import SQLDatabase
    from langchain.agents.agent_toolkits import SQLDatabaseToolkit
except ImportError:
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain.agents import create_sql_agent, initialize_agent, AgentType
from langchain_groq import ChatGroq
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# Tools
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools import DuckDuckGoSearchRun

# For uploads
import tempfile
from PyPDF2 import PdfReader
import requests

# -------------------
# Load environment
# -------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")

encoded_password = quote_plus(SUPABASE_PASSWORD) if SUPABASE_PASSWORD else ""
supa_uri = f"postgresql+psycopg2://postgres:{encoded_password}@db.gsztvwtfgplgyrgasiyi.supabase.co:5432/postgres"

# -------------------
# LLM Initialization
# -------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192",
    streaming=True
)

# -------------------
# Tools Initialization
# -------------------
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
duck_tool = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())
general_tools = [wiki_tool, arxiv_tool, duck_tool]

# -------------------
# Supabase DB Init
# -------------------
@st.cache_resource(ttl=7200, show_spinner=False)
def get_supabase_engine(uri: str):
    resolved_uri = Template(uri).safe_substitute(os.environ)
    return create_engine(resolved_uri, connect_args={"sslmode": "require"}, pool_pre_ping=True)

try:
    engine = get_supabase_engine(supa_uri)
except SQLAlchemyError as e:
    st.error(f"❌ Database connection failed: {e}")
    st.stop()

# -------------------
# Pinecone Init
# -------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "langchain-demo"

# Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
dummy_vector = embeddings.embed_query("dimension check")
embedding_dim = len(dummy_vector)

# Ensure index exists
existing_indexes = {i["name"]: i for i in pc.list_indexes()}
if index_name in existing_indexes:
    info = pc.describe_index(index_name)
    if info.dimension != embedding_dim:
        pc.delete_index(index_name)
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
else:
    pc.create_index(
        name=index_name,
        dimension=embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# -------------------
# Upload Functions
# -------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df.infer_objects()

def upload_to_datalake(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel.")
            return

        df = clean_dataframe(df)
        table_name = uploaded_file.name.split(".")[0]

        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
        st.success(f"✅ Uploaded {uploaded_file.name} as table `{table_name}` into Supabase Datalake.")
    except Exception as e:
        st.error(f"Upload failed: {e}")

def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def upload_to_esgdb(file=None, url=None, raw_text=None):
    try:
        if file:
            if file.name.endswith(".pdf"):
                text = extract_text_from_pdf(file)
            elif file.name.endswith(".txt"):
                text = file.read().decode("utf-8")
            else:
                st.error("Unsupported file format. Please upload PDF or TXT.")
                return
        elif url:
            resp = requests.get(url)
            text = resp.text
        elif raw_text:
            text = raw_text
        else:
            st.error("No valid input provided.")
            return

        if not text.strip():
            st.warning("No text extracted from the input.")
            return

        # Embed & upload
        doc = Document(page_content=text, metadata={"source": file.name if file else url or "manual_input"})
        vector_store.add_documents([doc])
        st.success("✅ ESG DB updated successfully.")
    except Exception as e:
        st.error(f"ESG Upload failed: {e}")

# -------------------
# Auto Agent Function
# -------------------
def auto_agent(query: str) -> str:
    results = {}

    # General
    try:
        agent = initialize_agent(
            tools=general_tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        results["General"] = agent.run(query)
    except Exception as e:
        results["General"] = f"Error: {e}"

    # Supabase
    try:
        db = SQLDatabase(engine)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        sql_agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )
        results["Datalake"] = sql_agent.run(query)
    except Exception as e:
        results["Datalake"] = f"Error: {e}"

    # ESG DB
    try:
        docs = vector_store.similarity_search(query, k=2)
        results["ESG DB"] = [res.page_content for res in docs]
    except Exception as e:
        results["ESG DB"] = f"Error: {e}"

    # Filter empty
    filtered = {
        k: v for k, v in results.items()
        if v and str(v).strip() not in ["I don't know", "[]", "Error: {}"]
    }
    if not filtered:
        return "⚠️ No useful result found."

    # Rank
    query_vec = embeddings.embed_query(query)
    best_src, best_ans, best_score = None, None, -1

    for src, ans in filtered.items():
        ans_text = ans if isinstance(ans, str) else " ".join(ans)
        ans_vec = embeddings.embed_query(ans_text)
        score = np.dot(query_vec, ans_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(ans_vec))
        if score > best_score:
            best_src, best_ans, best_score = src, ans_text, score

    return f"✅ Best Answer (from {best_src}):\n\n{best_ans}"

# -------------------
# Streamlit App
# -------------------
st.title("🔍 Sustainable AI Assistant")

search_mode = st.radio(
    "Select Mode:",
    ["General Search", "Datalake", "ESG DB", "Auto"],
    horizontal=True
)

# -------------------
# Upload Section
# -------------------
st.sidebar.header("📤 Upload Data")

upload_target = st.sidebar.radio(
    "Upload Target:",
    ["None", "Datalake", "ESG DB"],
    key="upload_target_radio"
)

if upload_target == "Datalake":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel for Datalake",
        type=["csv", "xls", "xlsx"],
        key="datalake_uploader"
    )
    if uploaded_file and st.sidebar.button("Upload to Datalake", key="upload_datalake_btn"):
        upload_to_datalake(uploaded_file)

elif upload_target == "ESG DB":
    file = st.sidebar.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        key="esgdb_file_uploader"
    )
    url = st.sidebar.text_input("Or enter a URL", key="esgdb_url")
    raw_text = st.sidebar.text_area("Or paste text directly", key="esgdb_textarea")
    if (file or url or raw_text) and st.sidebar.button("Upload to ESG DB", key="upload_esgdb_btn"):
        upload_to_esgdb(file=file, url=url, raw_text=raw_text)

# -------------------
# Query input with session-state prefill
# -------------------
if "query" not in st.session_state:
    st.session_state.query = ""

query = st.text_input("Enter your query:", value=st.session_state.query)

# -------------------
# Initialize chat history
# -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------
# Run search button
# -------------------
if st.button("Run Search") and query:
    st.session_state.query = query  # Save current query
    st.write(f"### Mode: {search_mode}")

    if search_mode == "General Search":
        agent = initialize_agent(
            tools=general_tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        with st.spinner("Searching Wikipedia, Arxiv, and DuckDuckGo..."):
            response = agent.run(query)
        st.success("✅ Result:")
        st.write(response)

    elif search_mode == "Datalake":
        try:
            db = SQLDatabase(engine)
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            sql_agent = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=lambda e: "🤖 I couldn't parse the model's response. Try rephrasing your question.",
            )
            with st.spinner("Querying Supabase Datalake..."):
                response = sql_agent.run(query)
            st.success("✅ Result from Supabase:")
            st.write(response)
        except Exception as e:
            st.error(f"SQL Agent failed: {e}")

    elif search_mode == "ESG DB":
        with st.spinner("Querying ESG VectorDB..."):
            results = vector_store.similarity_search(query, k=3)
        st.success("✅ Top Matches from ESG DB:")
        for res in results:
            st.markdown(f"- **{res.page_content}**  \n  _{res.metadata}_")

    elif search_mode == "Auto":
        with st.spinner("Running Auto Agent..."):
            best_result = auto_agent(query)

        # Save chat history
        st.session_state.chat_history.append({
            "question": query,
            "answer": best_result
        })

        st.success(best_result)

        # -------------------
        # Next Question Suggestions (Clickable)
        # -------------------
        try:
            recommendation_prompt = f"""
            Based on the previous question: "{query}" 
            and its answer: "{best_result}", suggest 3 relevant follow-up questions that a user might ask next.
            Return them as a Python list of strings.
            """
            recommended = llm(recommendation_prompt)

            try:
                recommendations = ast.literal_eval(recommended)
            except:
                recommendations = [q.strip("- ").strip() for q in recommended.split("\n") if q.strip()]

            if recommendations:
                st.markdown("### 💡 Suggested Next Questions:")
                for idx, q in enumerate(recommendations[:3], 1):
                    if st.button(f"{idx}. {q}"):
                        st.session_state.query = q
                        st.experimental_rerun()
        except Exception as e:
            st.error(f"Next question suggestions failed: {e}")

# -------------------
# Sidebar: Chat History
# -------------------
with st.sidebar.expander("💬 Chat History", expanded=True):
    if st.session_state.chat_history:
        for entry in reversed(st.session_state.chat_history[-10:]):  # show last 10
            st.markdown(f"**Q:** {entry['question']}")
            st.markdown(f"**A:** {entry['answer']}")
            st.markdown("---")
    else:
        st.info("No chat history yet.")
