import os
import openai
import sys
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
import config

doclist = ["docs/blueplanetsix-lastresort.pdf"]
llm_name = config.llm_name
persist_directory = config.persist_directory

def load_db(doclist):
    loaders=[]
    for doc in doclist:
        loaders.append(PyPDFLoader(doc))

    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs

def load_vectordb(docs=doclist, persist_directory=persist_directory):
    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150
    )

    splits = text_splitter.split_documents(docs)

    embedding = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )

    print(vectordb._collection.count())

"""
Context is a list of data from the document. We want the "content" field, but need to flatten it
for our template below.

Also, we need to make sure that it doesn't have too many elements or get too big.

"""
def clean_context(context):
    # TODO -- once it hits MAXSIZE we need to send the text off to the LLM to
    # summarize it.
    for i in range(len(context)):
        context[i] = context[i]['content']
    return " ".join(context)


# write a function that uses and maintains the user's context and can answer questions based on the vectordb chroma that we set up
# this function should take in a question and return an answer based on the context
def ask_document_with_state(session_id, question):
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    """
    format is:
    Document(page_content='Blue Planet Six and Owl...', metadata={'page': 1, 'source': 'docs/blueplanetsix-lastresort.pdf'})
    for doc in docs:
        print(doc['page_content'])
        print(doc['score'])
        print("")
    """
    llm = ChatOpenAI(model_name=llm_name, temperature=0)

    # Build prompt
    doc_question_template = """Use the following pieces of context to answer the question at the end.\
          If you don't know the answer, just say that you don't know, don't try to make up \
          an answer. Use three sentences maximum. Keep the answer as concise as possible. \
    Question: {question}
    Helpful Answer:""".format(question=question)

    retriever=vectordb.as_retriever(search_type="mmr")

    # 'stuff', 'map_reduce', "refine", "map_rerank"
    chain_type = "refine"
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type=chain_type, retriever=retriever, return_source_documents=True)

    result = qa({"query": doc_question_template})
    return result['result']

load_db=False
if __name__ == '__main__':
    if load_db:
        docs=load_db(doclist)
        load_vectordb(docs, persist_directory)
    
    question_list=[
        "Who are the main characters of the story?",
        "Who is the main character of the story and why do you say that?",
        "Where does the story take place?",
        "Who owns the resort?",
        "Does Miskers owns that location?",
        "What are the names of the raccoons in the story?",
        "What type of creature is Pudding?"
    ]

    count=1
    for question in question_list:
        result=ask_document_with_state(1, question)
        print(count, question, result)
        count +=1