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

# document loaders - https://python.langchain.com/en/latest/modules/indexes/document_loaders.html
loader = DirectoryLoader("data/", glob="**/*.txt" )

documents = loader.load()
# Text splitter - https://python.langchain.com/en/latest/modules/indexes/text_splitters.html
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# embdeddings is changing from string and converting to vector space
# vectorstore - https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
embeddings = HuggingFaceEmbeddings()
doc_search = Chroma.from_documents(texts, embeddings) # create a vector store for these 'texts' using the 'embeddings'

llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature": 0, "max_length":64})  # can use gpt2 also as repo_id
# repo_id = Model name to use. List is defined at https://huggingface.co/models . 
#   Details of mentioned model - https://huggingface.co/google/flan-t5-xl
# model_kwargs - Key word arguments to pass to the model
#   temperature - i/p to model. controls the amount of randomness in the language model’s response. A value of 0 makes the
#                  response deterministic, meaning every time we’ll get the same response. A higher value will make the output more random.
#   max_length = response length 


# RetrievalQA - Chain for question-answering against an index.
#   index - Indexes refer to ways to structure documents so that LLMs can best interact with them
#           The primary index and retrieval types supported by LangChain are currently centered around vector databases
#           https://python.langchain.com/en/latest/modules/indexes.html
qna = RetrievalQA.from_chain_type(llm= llm, chain_type="stuff", retriever=doc_search.as_retriever())

query = "which all holidays we have, give me list"
print(qna.run(query))

query = "what is the condition for availing gratuity"
print(qna.run(query))

query = "what is the cab time from brookefields?"
print(qna.run(query))

query = "what is ravi's phone number"
print(qna.run(query))

query = "what is date for hiliday in May"
print(qna.run(query))
