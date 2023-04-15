from langchain.llms import OpenAI
from langchain import HuggingFaceHub
from langchain import PromptTemplate # for multi sentence question or forcing llm to think step by step
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader

from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA


# open AI - might get rate limiting error
#llm = OpenAI()
#text = "When is India's independence day"
#print(llm(text))

# Hugging face  - works
llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature": 0, "max_length":64})  # can use gpt2 also as repo_id
#print(llm("translate to German: How are you"))

# for prompt template - works
#template =  """ Question: {question}
#Let's think step by step.
#
#Answer: """

#prompt = PromptTemplate(template=template, input_variables=["question"])
#question = "Who started ecomomic liberliation in India and what was the imnpact of it on India economically?"
#llm_chain=LLMChain(prompt=prompt,llm=llm)

#print(llm_chain.run(question))

# document loaders - https://python.langchain.com/en/latest/modules/indexes/document_loaders.html
print("from text document")
loader = DirectoryLoader("data/", glob="**/*.txt" )
documents = loader.load()

# https://python.langchain.com/en/latest/modules/indexes.html 

print("doing text splitting")
# Text splitter - https://python.langchain.com/en/latest/modules/indexes/text_splitters.html
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents=documents)

print("embeddings")
# embeddings- https://python.langchain.com/en/latest/reference/modules/embeddings.html
embeddings = HuggingFaceEmbeddings()

print("vector store")
# vectorstore - https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
doc_search = Chroma.from_documents(docs, embeddings) # create a vector store for these 'texts' using the 'embeddings'

# Questin And Answer
qna = RetrievalQA.from_chain_type(llm= llm, chain_type="stuff", retriever=doc_search.as_retriever())


query = "sene me full list of conacts from contact list"
print(qna.run(query))





