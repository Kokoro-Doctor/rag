import os  # <--- NEW
from dotenv import load_dotenv  # <--- NEW
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# ---------------------------------------------------------
# SECURITY UPDATE: Load Key from .env file
# ---------------------------------------------------------
load_dotenv()  # .env file load karega
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Safety Check (Optional but recommended)
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY nahi mili!")

# ---------------------------------------------------------

def build_rag_chain(vectorstore, system_prompt):
    # 1. OpenAI Model Setup (Using GPT-4o)
    llm = ChatGroq(
        model="openai/gpt-oss-120b",  # Heavy RAG model
        temperature=0.1,
        api_key=GROQ_API_KEY
    )

    # ... (Baaki code same rahega) ...
    # 2. Dynamic Prompt with Language Instruction
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\n" + 
         "LANGUAGE INSTRUCTION:\n{language_instruction}\n\n" +
         "Relevant Context from Knowledge Base:\n{context}"),
        ("human", "{input}")
    ])

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. Chain Construction
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input"),
            "language_instruction": itemgetter("language_instruction")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever
