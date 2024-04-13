# https://python.langchain.com/docs/modules/agents/how_to/custom_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper 
from langchain_community.retrievers import WikipediaRetriever
from langchain.agents import tool

system_prompt = """You are very powerful assistant."""

"""
create a simple tool that returns the length of a word
"""
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


"""
ddg backends: suggestion, translate, maps, news, videos, images, answers ????
"""
@tool
def ddg_specialized_search(query, backend="news") -> str:
    """Searches for news on a given query."""
    #search = DuckDuckGoSearchRun()

    wrapper = DuckDuckGoSearchAPIWrapper(region="us-en", time="d", max_results=2)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper, backend=backend, safesearch="moderate")
    return search.run(query)

@tool
def ddg_general_search(query: str) -> str:
    """Searches for general web information on a given query."""
    #search = DuckDuckGoSearchRun()

    wrapper = DuckDuckGoSearchAPIWrapper(region="us-en", time="d", max_results=10)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper, safesearch="moderate")
    return search.run(query)


@tool
def query_wikipedia(query: str):
    """Queries Wikipedia for relevant documents with in-depth information on a topic."""
    retriever = WikipediaRetriever()
    docs = retriever.get_relevant_documents(query=query, n_docs=3)
    return docs


# https://python.langchain.com/docs/integrations/tools/reddit_search/


def get_info(memory_key, input) -> str:
    model="gpt-4"
    llm = ChatOpenAI(model=model, temperature=0)
    tools = [ddg_general_search, query_wikipedia, ddg_specialized_search]

    # now create a prompt with memory key

    MEMORY_KEY = "chat_history"
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # set up a list to track the chat history
    chat_history = []

    # bind tools to the llm - expects a list of functions that are tools
    llm_with_tools = llm.bind_tools(tools)

    # now we create the agent

    """ agent CHAIN
    a component for formatting intermediate steps (agent action, tool output pairs) to input 
    messages that can be sent to the model, and a component for converting the output message into an agent action/agent finish.

    includes the memory/chat_history
    """
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # now we run it

    chat_history=[]

    """
    chat_history.extend(
        [
            HumanMessage(content=input),
            AIMessage(content=result["output"]),
        ]
    )
    """
    result = agent_executor.invoke({"input": input, "chat_history": chat_history})

    return result


if __name__ == "__main__":
    question="What are the top news stories of the past week about the Seattle area?" # that are NOT sports related?"
    result=get_info("news", question)
    print(result['output'])

