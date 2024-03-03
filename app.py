from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import time

st.set_page_config(page_title="AttroneyGPT")
col1, col2, col3 = st.columns([1,8,1])
with col2:
    st.image("logo.png")

st.markdown(
    """
    <style>
    div[data-baseweb="input"] input {
            border-color: #000000;
        }
    margin-top: 0 !important;
div.stButton > button:first-child {
    background-color: #808080;
    color:white;
}
div.stButton > button:active {
    background-color: #808080;
    color : white;
}

   div[data-testid="stStatusWidget"] div button {
        display: none;
        }
    
    .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    button[title="View fullscreen"]{
    visibility: hidden;}
        </style>
""",
    unsafe_allow_html=True,
)

def reset_conversation():
  st.session_state.messages = []
  st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history",return_messages=True) 

embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",model_kwargs={"trust_remote_code":True,"revision":"289f532e14dbbbd5a04753fa58739e9ba766f3c7"})
db = FAISS.load_local("ipc_vector_db", embeddings)
db_retriever = db.as_retriever(search_type="similarity",search_kwargs={"k": 4})

prompt_template = """<s>[INST]This is a chat template and As a legal chat bot specializing in Indian Penal Code queries, your primary objective is to provide accurate and concise information based on the human's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the human's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question', 'chat_history'])

# You can also use other LLMs options from https://python.langchain.com/docs/integrations/llms. Here I have used TogetherAI API
TOGETHER_AI_API= os.environ['TOGETHER_AI']="369ad480a9e695f40803b5d9554e288e752a9228d62909a45074943c0509ff84"
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=f"{TOGETHER_AI_API}"
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

for message in st.session_state.messages:
    role = message.get("role")
    content = message.get("content")
    
    with st.chat_message(role, avatar="user.svg" if role == "human" else "attorney.svg"):
        st.write(content)

input_prompt = st.chat_input("message LAWGpt.....")

if input_prompt:
    with st.chat_message("human",avatar="user.svg"):
        st.write(input_prompt)

    st.session_state.messages.append({"role":"human","content":input_prompt})
    full_response = " "
    with st.chat_message("bot",avatar="attorney.svg"):
        with st.spinner("Thinking..."):
            result = qa.invoke(input=input_prompt)

            message_placeholder = st.empty()

            full_response = "⚠️ **_Note: This offers basic legal advice and is not a complete substitute for consulting a human attorney_** \n\n\n"
        for chunk in result["answer"]:
            full_response+=chunk
            time.sleep(0.02)
            
            message_placeholder.markdown(full_response+" ▌")
        st.button('Reset All Chat 🗑️', on_click=reset_conversation)

    st.session_state.messages.append({"role": "ai", "content": result["answer"], "avatar": "attorney.svg"})

