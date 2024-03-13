# import libraries
import os
import openai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Load PDF 
def pdf_loader(pdf_path):
    # pdf path
    loader = PyMuPDFLoader(pdf_path)
    # load the pdf
    doc = loader.load()
    return doc

#transforming data
def text_splitter(documents):
    # text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50,
    )
    #split text
    documents= text_splitter.split_documents(documents)
    return documents

# load  into FAISS 
def load_to_index(documents):
    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small"
    )
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()
    return retriever

# query FAISS 
def query_index(retriever, query):
    retrieved_document = retriever.invoke(query)
    return retrieved_document 


# answer prompt
def create_answer_prompt():
    template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':
    Context: 
    {context}
    
    Question: 
    {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def generate_answer(retriever, answer_prompt, query):
    primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

    retrieval_augmented_qa_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context = itemgetter("context"))
        | {"response": answer_prompt | primary_qa_llm, "context": itemgetter("context")}
    )
    result = retrieval_augmented_qa_chain.invoke({"question": query})
    return result


def index_initialization():
    # load the pdf
    cwd = os.path.abspath(os.getcwd())
    data_dir = "data"
    pdf_file = "nvidia10k.pdf"
    pdf_path = os.path.join(cwd, data_dir, pdf_file)
    doc = pdf_loader(pdf_path)
    doc_splits = text_splitter(doc)
    retriever = load_to_index(doc_splits)
    return retriever 


def main():
    retriever = index_initialization()
    # query = "Who is the E-VP, Operations"
    query = "what is the reason for the lawsuit"
    retrieved_docs = query_index(retriever, query)
    print("retrieved_docs: \n", len(retrieved_docs))
    answer_prompt = create_answer_prompt()
    print("answer_prompt: \n", answer_prompt)
    result = generate_answer(retriever, answer_prompt, query)
    print("result: \n", result["response"].content)

if __name__ == "__main__":
    main()