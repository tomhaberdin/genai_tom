
import streamlit as st
from langchain.llms import openai
from langchain.llms import OpenAI

st.title('my first gen AI app_tom')

st.markdown("""
this is a sample markdown
try it on your 'own'
""")

openai_api_key = st.sidebar.text_input("open AI key")
name = st.text_input("enter some text", "enter here")
option = st.radio("choose one option:", options = ["Option1", "Option2"], index=0)

value = st.slider("Enter a value: ", 0, 100, 20)
print(value)
print(option)

def gen_response(txt):
    llm = OpenAI(temperature = 0.7, openai_api_key=openai_api_key)
    st.info(llm(txt))

with st.form("sample app"):
    txt = st.text_area("enter text:", "what GPT stands for")
    subm = st.form_submit_button("submit")
    if subm:
        gen_response(txt)
        