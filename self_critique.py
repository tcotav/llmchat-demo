#!/usr/bin/env python
# coding: utf-8

from langchain.prompts import PromptTemplate
from langchain_experimental.smart_llm import SmartLLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config

# # How to use a SmartLLMChain
# 
# A SmartLLMChain is a form of self-critique chain that can help you if have particularly complex questions to answer. Instead of doing a single LLM pass, it instead performs these 3 steps:
# 1. Ideation: Pass the user prompt n times through the LLM to get n output proposals (called "ideas"), where n is a parameter you can set 
# 2. Critique: The LLM critiques all ideas to find possible flaws and picks the best one 
# 3. Resolve: The LLM tries to improve upon the best idea (as chosen in the critique step) and outputs it. This is then the final output.
# 
# SmartLLMChains are based on the SmartGPT workflow proposed in https://youtu.be/wVzuvf9D9BU.
# 
# Note that SmartLLMChains
# - use more LLM passes (ie n+2 instead of just 1)
# - only work then the underlying LLM has the capability for reflection, which smaller models often don't
# - only work with underlying models that return exactly 1 output, not multiple
# 
# This notebook demonstrates how to use a SmartLLMChain.

MODEL_NAME=config.llm_name



def query_llm(system_prompt, user_input, temp):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user","{input}")
    ])
    llm = ChatOpenAI(temperature=temp, model_name=MODEL_NAME)
    chain = prompt | llm
    output = chain.invoke({"input": user_input})
    return output.content

"""
https://github.com/langchain-ai/langchain/blob/master/cookbook/plan_and_execute_agent.ipynb

Plan-and-execute agents accomplish an objective by first planning what to do, then executing the sub tasks. This idea is largely inspired by BabyAGI and then the "Plan-and-Solve" paper.

The planning is almost always done by an LLM.

The execution is usually done by a separate agent (equipped with tools).
"""
from langchain.chains import LLMMathChain
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.tools import Tool
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain_openai import ChatOpenAI, OpenAI

from langchain.agents import tool
@tool
def query_wikipedia(query):
    """Query Wikipedia for relevant documents."""
    retriever = WikipediaRetriever()
    docs = retriever.get_relevant_documents(query=query, n_docs=3)
    return docs

def plan_and_execute(system_prompt, input_question):
    search = DuckDuckGoSearchAPIWrapper()
    llm = OpenAI(temperature=0)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events",
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
        ),
    ]
    model = ChatOpenAI(temperature=0)
    planner = load_chat_planner(llm=model, system_prompt=system_prompt)
    executor = load_agent_executor(llm=model, tools=tools, include_task_in_prompt=True, verbose=True)
    agent = PlanAndExecute(planner=planner, executor=executor)
    return agent.invoke(input_question)


def summarize_output(llm_output):
    # This function will summarize the output from the LLM
    # It will return the resolution of the output
    summary_prompt="""Summarize the key points and crucial details in a maximally compact form, 
        retaining nuanced interpretations as much as possible. The summary should convey the 
        essence of the input accurately but concisely, with minimal token count."""
    return query_llm(summary_prompt, llm_output, 0)

"""
Based on SmartLLM workflow

A SmartLLMChain is a form of self-critique chain that can help you if have particularly complex 
questions to answer. Instead of doing a single LLM pass, it instead performs these 3 steps:

- Ideation: Pass the user prompt n times through the LLM to get n output proposals (called "ideas"), 
    where n is a parameter you can set
- Critique: The LLM critiques all ideas to find possible flaws and picks the best one
- Resolve: The LLM tries to improve upon the best idea (as chosen in the critique step) and outputs it. 
    This is then the final output.
"""
def self_critique_ideation(llm_output, system_prompt, input_question, show_work=False):
    # So, we first create an LLM and prompt template. To experiment, we use three different temperatures to throw wider the creativity.

    prompt = PromptTemplate.from_template("{} {}".format(system_prompt, input_question))
    llm_ideation = ChatOpenAI(temperature=0.8, model_name=MODEL_NAME)
    llm_critique = ChatOpenAI(temperature=0.3, model_name=MODEL_NAME)
    llm_resolver = ChatOpenAI(temperature=0, model_name=MODEL_NAME)
    verbose=False # change this to True to get the llm to show its work in between steps -- spews to console so don't
    return_intermediate_steps=show_work # return intermediate steps in the output in addition to the resolution

    # Now we can create a SmartLLMChain
    chain = SmartLLMChain(ideation_llm=llm_ideation,
                        critique_llm=llm_critique,
                        resolver_llm=llm_resolver,
                        prompt=prompt, n_ideas=3, 
                        return_intermediate_steps=return_intermediate_steps,
                        verbose=verbose)


    # Now we can use the SmartLLM as a drop-in replacement for our LLM. E.g.:
    llm_output = chain.invoke({})

    return llm_output

if __name__ == "__main__":
    input_question = """We want to create four different summer class-like activities for a group of 2-5 teens. An example 
is work through the process of creating your own youtube video."""
    # should put this in a loop and react as we accumulate output
    llm_output = ""
    #llm_output = self_critique_ideation(llm_output, system_prompt, input_question)
    #print(llm_output)
    #print(llm_output["resolution"])

    plan_question="""What are some events going on in Seattle, WA next week for a visiting high school class of twice-exceptional 
    teenagers? Please include the date and times for each event."""

    print(plan_and_execute(system_prompt, plan_question))



