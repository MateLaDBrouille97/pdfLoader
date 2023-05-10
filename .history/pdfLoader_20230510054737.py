import os 
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from  langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import DuckDuckGoSearchRun




with st.sidebar:
  st.header("PDF")
  pdf = st.file_uploader("Upload your PDF", type="pdf")
  st.header("API KEY")
  openaikey = st.text_input("Your openai key", value="", type="password",placeholder="openai key")


st.markdown(f"""
    ## \U0001F60A! Question Answering with your PDF file
    1) Upload a PDF. 
    2) Enter OpenAI API key. This costs $. Set up billing at [OpenAI](https://platform.openai.com/account).
    3) Type a question and Press 'Enter'.🎈 """)
# st.set_page_config(page_title="Ask your PDF")
st.header("Ask your PDF 💬")

# col1, col2 = st.columns(2)

    # upload file
# with col1:
#      pdf = st.file_uploader("Upload your PDF", type="pdf")
# with col2:
#      openaikey = st.text_input("Your openai key", value="", type="password",placeholder="openai key")


user_question = st.text_input("Ask a question about your PDF:")


###### Wikipedia
#Prompt
script_template =PromptTemplate(
     input_variables=['wikipedia_research'],
     template='Do some research using {wikipedia_research} and summarize the information in 400 words with an introduction, a body and a resume of the information  '
)

referencies_template =PromptTemplate(
     input_variables=['wikipedia_research'],
     template='Do some research using {wikipedia_research} and give me 10 clickable links to leverage the information'
)

#Memory
script_memory = ConversationBufferMemory(input_key='wikipedia_research',memory_key='chat_history')





def qa(pdf, query ):
    # load document
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()

    # split the documents into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    chunks = text_splitter.split_text(text)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    # expose this index in a retriever interface
    docs = knowledge_base.similarity_search(query)
        
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=query)
          print(cb)
    print(response)
    return response
# result = qa("example.pdf", "what is the total number of AI publications?")


def qa_result():
    if openaikey is not None:
     os.environ["OPENAI_API_KEY"] = openaikey
    
    # save pdf file to a temp file 
    if pdf is not None:
        prompt_text = user_question
        if prompt_text:
            result = qa(pdf=pdf, query=prompt_text)
            # st.write(result)     
    return result


#########   Research

duckduck=DuckDuckGoSearchRun()

def research1 (result_final):
   os.environ["OPENAI_API_KEY"] = openaikey
   llm= OpenAI(temperature=0.9,openai_api_key=openaikey)
   script_chain= LLMChain(llm=llm,prompt=script_template,verbose=True,output_key='script',memory=script_memory)
   if result_final is not None:
       result=script_chain.run(result_final)
   return result

def research2 (result_final):
   os.environ["OPENAI_API_KEY"] = openaikey
   llm= OpenAI(temperature=0.9,openai_api_key=openaikey)
   referecies_chain= LLMChain(llm=llm,prompt=referencies_template,verbose=True,output_key='referencies',memory=script_memory)
   if result_final is not None:
       referencies=referecies_chain.run(result_final)
   return referencies


def main():
      
      if pdf is not None:
          pdf_reader = PdfReader(pdf)
          text = ""
          for page in pdf_reader.pages:
            text += page.extract_text()
         
          with st.expander(' PDF '):
            st.write(text)
      if user_question:
        result_final=qa_result()
        wikipedia_result=research1(result_final)
        duckDuck_result=duckduck.run(result_final)
        referencies_result=research2 (result_final)
        st.text_area("Result",value=result_final,height=350)
       
        with st.expander('Search History'):
           st.text_area("Search_Result",wikipedia_result,height=300)
        
        with st.expander('DuckDuckGo History'):
           st.text_area("DuckDuckGo_Result",duckDuck_result,height=300)
           
        with st.expander('Memory History'):
           st.info(script_memory.buffer)

        with st.expander('Referencies History'):
           st.text_area("Referencies_Result",referencies_result,height=300)

        
           
        
if __name__ == '__main__':
    main()