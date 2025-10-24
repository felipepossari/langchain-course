from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

system = ("system", "you are an assistant that answers questions in a {style} style")
user = ("user", "{question}")

chat_prompty = ChatPromptTemplate([system, user])
messages = chat_prompty.format_messages(style="funny", question="Who is Alan Turing?")


for msg in messages:
    print(f"{msg.type}: {msg.content}")

model = ChatOpenAI(model_name="gpt-5-nano", temperature=0.5)
result = model.invoke(messages)
print(result.content)