from langchain_core.prompts import PromptTemplate
template = PromptTemplate(
    input_variables=["name"],
    template="Hello {name}!"
)

text = template.format(name="LangChain")
print(text)