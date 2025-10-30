from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def build_rag_chain(vectorstore):
    llm = OllamaLLM(model="llama3", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are MedBot, a reliable AI specializing in medical topics.
         Use the provided context to answer clearly and accurately.
         If the answer isn't in the context, say so honestly. Do not hallucinate.

         Context: {context}"""),
        ("human", "{input}")
    ])

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever
