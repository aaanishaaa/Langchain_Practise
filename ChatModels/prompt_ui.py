from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from dotenv import load_dotenv

import streamlit as st;

load_dotenv();


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # chat-tuned model
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
st.header('Research Tool')
input=st.text_input(
    'Enter your prompt'
);

if st.button('Summarize'):
    result = model.invoke(input)
    st.write(result.content)
