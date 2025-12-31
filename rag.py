from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def build_rag_chain(vectorstore, system_prompt):
    # Using low temperature for factual accuracy
    llm = OllamaLLM(model="llama3", temperature=0.1)

    # Dynamic Prompt Injection
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nRelevant Context from Knowledge Base:\n{context}"),
        ("human", "{input}")
    ])

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever