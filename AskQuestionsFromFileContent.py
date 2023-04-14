# todo  - write other pip packages here
# pip install chromadb
# pip install unstructured
# pip install python_magic_bin-0.4.14-py2.py3-none-win32.whl
# pip install chromadb
# pip install tiktoken

# pre-req
# - get open ai key and set as env variable OPENAI_API_KEY

from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
# hugging face (instead of OpenAI)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
# general packages
import nltk
import os
import ssl


# download starts - install below, needed for 1st time. un/comment till `download ends` during subsequent runs
#try:
#    _create_unverified_https_context = ssl._create_unverified_context
#except AttributeError:
#    pass
#else:
#    ssl._create_default_https_context = _create_unverified_https_context

#nltk.download()
#nltk.download('punkt')
## download ends

loader = DirectoryLoader("data/", glob="**/*.txt" )

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

#embdeddings is changing from string and converting to vector space
#embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY']) procing error
embeddings = HuggingFaceEmbeddings()
doc_search = Chroma.from_documents(texts, embeddings) # create a vector store for these 'texts' using the 'embeddings'

llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature": 0, "max_length":64})  # can use gpt2 also as repo_id
qna = RetrievalQA.from_chain_type(llm= llm, chain_type="stuff", retriever=doc_search.as_retriever())

query = "which all holidays we have, give me list"
print(qna.run(query))

query = "what is the condition for availing gratuity"
print(qna.run(query))


