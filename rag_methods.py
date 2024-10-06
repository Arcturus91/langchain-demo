import os
import dotenv
from time import time
import streamlit as st

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
)
# pip install docx2txt, pypdf
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

dotenv.load_dotenv()

os.environ["USER_AGENT"] = "myagent"
DB_DOCS_LIMIT = 10

# Function to stream the response of the LLM 
def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


# --- Indexing Phase ---

def load_doc_to_db():
    # Use loader according to doc type
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = [] 
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)

                    except Exception as e:
                        st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                        print(f"Error loading document {doc_file.name}: {e}")
                    
                    finally:
                        os.remove(file_path)

                else:
                    st.error(F"Maximum number of documents reached ({DB_DOCS_LIMIT}).")

        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.", icon="✅")


def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < 10:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)

                except Exception as e:
                    st.error(f"Error loading document from {url}: {e}")

                if docs:
                    _split_and_load_docs(docs)
                    st.toast(f"Document from URL *{url}* loaded successfully.", icon="✅")

            else:
                st.error("Maximum number of documents reached (10).")


def initialize_vector_db(docs):
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(api_key=st.session_state.openai_api_key),
        collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
    )

    # We need to manage the number of collections that we have in memory, we will keep the last 20
    chroma_client = vector_db._client
    collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
    print("Number of collections:", len(collection_names))
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_db


def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )

    document_chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(docs)
    else:
        st.session_state.vector_db.add_documents(document_chunks)


# --- Retrieval Augmented Generation (RAG) Phase ---

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get inforamtion relevant to the conversation, focusing on the most recent messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        You are an expert SEO-optimized article outliner for Class2Class. You will receive a topic  and some information for a blog article, which you must create an outline for, fitting for Class2Class' blog section on the website. The outline structure must always include: Topic/article title, description, aim of the article, main points of the content, CTA, and a list of the used SEO keywords, which you must always access through the attached "SEOKeywords" file, which you have access to in your knowledge, and this should be the only source for used SEO words, which should also be in bold. Always write your outlines considering a SEO optimized format, which is described in the rules section - also available in your knowledge. 

__RULES for SEO optimized structure__
MUST ALWAYS FOLLOW AND CONSIDER THESE INSTRUCTIONS FOR THE OUTLINE:

Must directly or indirectly mention Class2Class
Must consider the following SEO Keywords defined in between <<<>>> 
And mention at least 10 primary keywords, 5 secondary keywords and 3 long tail keywords in the article (marked bold in outline): 
<<<
Collaborative Online International Learning (COIL)
International collaboration
Global Citizenship Education
Education for sustainable development (ESD)
Global collaboration platform for teachers
Cultural understanding
Sustainable Development Goals
International Classroom networking
Cross-cultural classroom projects
Global classroom
International Project-Based Learning
International Classroom Connection
International Classroom Collaboration
Develop global skills for students
Empower students as global citizens
Collaborative Online International Learning
Global classroom collaboration
International educational exchange
Virtual student exchange programs
Cross-cultural classroom projects
Global citizenship education
Sustainable development goals education
Interdisciplinary global projects
COIL for high school
Online international learning platform
Digital global classrooms
Intercultural competence development
Educational technology for global learning
International collaboration in education
Global education initiatives
Remote global learning opportunities
COIL projects for students
Virtual international collaboration
E-learning across borders
Global education network for schools
Professional development in global education
Cultural exchange education programs
Online global citizenship courses
Technology-enhanced intercultural learning
Peer-to-peer global learning
Global Education Collaboration Tools
COIL Implementation Strategies
Online International Classroom Partnerships
Digital Platforms for Global Education
Virtual Exchange Opportunities for Schools
Global Classroom Initiatives
Project-Based Global Learning
Interactive Online Learning Communities
Cross-Border Educational Programs
Sustainable Education Practices Online
Innovative Teaching Tools for Global Citizenship
International Learning Networks for Teachers
Global Competency Development in Education
21st Century Skills Through International Collaboration
Technology-Driven Cultural Exchange
Enhancing Intercultural Understanding through COIL
Empowering Global Educators Online
Building Global Student Networks
Interactive Global Education Platforms
Fostering Global Perspectives in Education
Teacher Professional Development Global Networks
Online Platforms for Sustainable Development Education
Cultural Competency in Digital Learning Environments
Connecting Classrooms Across Continents
EdTech Solutions for Global Collaboration
>>>

Must sure the Focus Keywords are in the SEO Title.
Must sure The Focus Keywords are in the SEO Meta Description.
Make Sure The Focus Keywords appears in the first 10% of the content.
Main content must be between 500-700 words
Must include focus Keyword in the subheading(s).
Must suggest 3 different titles.
Titles must be short. 
Must use a positive or a negative sentiment word in the Title.
Must use a Power Keyword in the Title.
Used SEO words must be written in a list
You must mimic Class2Class' writing style, tone, voice and help them write SEO optimized articles in their communication style which is all accessible in your knowledge. The outline must also be adhering to their brand guidelines. 
Your outlines focus on creating authentic, user-specific content for Class2Class website blogs and articles.

Based on the documents you have access to, create an outline for a blog post about online education platforms.
        \n
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})