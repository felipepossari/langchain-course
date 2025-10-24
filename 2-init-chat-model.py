from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

gemini = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
message = gemini.invoke("Hello world!")
print(message.content)