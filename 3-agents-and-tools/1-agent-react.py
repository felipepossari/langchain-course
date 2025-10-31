from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


@tool("calculator", return_direct=True)
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression and returns the result."""
    try:
        result = eval(expression)  # be careful with this because it's a security risk
    except Exception as e:
        return f"Error: {e}"
    return str(result)


@tool("web_search_mock")
def web_search_mock(expression: str) -> str:
    """Mocked web search tool. Returns a hardcoded result."""

    data = {"Argentina": "Buenos Aires", "Bolivia": "Sucre", "Brazil": "Brasília", "Chile": "Santiago",
            "Colombia": "Bogotá", "Ecuador": "Quito", "Guyana": "Georgetown", "Paraguay": "Asunción", "Peru": "Lima",
            "Suriname": "Paramaribo", "Uruguay": "Montevideo", "Venezuela": "Caracas"}

    for country, capital in data.items():
        if country.lower() in expression.lower():
            return f"The capital of {country} is {capital}."

    return f"I don't know the capital of {expression}."


llm = ChatOpenAI(model="gpt-5-mini", disable_streaming=True)
tools = [calculator, web_search_mock]
#tools_names = [calculator, web_search_mock]

prompt = PromptTemplate.from_template(
    """
    Answer the following questions as best you can. You have access to the following tools.
    You are prohibited from search for answers from internet case you don't get the answer from the tools provided
    {tools}
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Question: {input}
    Thought: {agent_scratchpad}
    """
)

agent_chain = create_react_agent(llm=llm, prompt=prompt, tools=tools, stop_sequence=False)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent_chain,
                                                    tools=tools,
                                                    verbose=True,
                                                    handle_parsin_errors=True)

#print(agent_executor.invoke({"input": "How much is 10 + 10?"}))
#print(agent_executor.invoke({"input": "What is the capital of Brasil?"}))
print(agent_executor.invoke({"input": "What is the capital of Japan?"}))
