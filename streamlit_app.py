
import streamlit as st
from langchain.llms import openai
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os 

st.title('My First Gen AI')

st.markdown("""
Welcome to the Pirate App!
""")

openai_api_key = st.sidebar.text_input("open AI key")

# possible elements to set in the app
#name = st.text_input("enter some text", "enter here")
#option = st.radio("choose one option:", options = ["Option1", "Option2"], index=0)


# setting tempature
temperature_value = st.slider("Enter a value: ", 0.0, 2.0, 0.3)

print(temperature_value)
print(openai_api_key)

os.environ["OPENAI_API_KEY"] = openai_api_key


def gen_response(txt):
    #llm = OpenAI(temperature_value = 0.7, openai_api_key=openai_api_key)
    # starting llm model
    llm = ChatOpenAI(model="gpt-4o-mini")
    # setting system and human question
    messages = [
                ("system","you are a very friendly pirate and always ansewr in a cool pirat song, and the reply must be in hebrew"), # machine part role
                ("human",txt), # the human asking the question
                ]
    ans = llm.invoke(messages,config={"temperature":temperature_value})
    st.info(ans.content)

with st.form("sample app"):
    txt = st.text_area("enter text:", "what GPT stands for")
    subm = st.form_submit_button("submit")
    if subm:
        gen_response(txt)
        


