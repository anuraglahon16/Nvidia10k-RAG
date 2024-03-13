# OpenAI Chat completion
import os
from openai import AsyncOpenAI  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools
from dotenv import load_dotenv
from src.rag import index_initialization, pdf_loader, text_splitter, load_to_index, query_index, create_answer_prompt, generate_answer

load_dotenv()

retriever = index_initialization()

@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    cl.user_session.set("settings", settings)


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")

    client = AsyncOpenAI()

   

    query = message.content
    retrieved_docs = query_index(retriever, query)
    answer_prompt = create_answer_prompt()
    result = generate_answer(retriever, answer_prompt, query)

    msg = cl.Message(content="")

    msg.content = result["response"].content

   
    await msg.send()