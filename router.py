"""
The idea here is that we run an initial query to decide if we should use:
- the default LLM search
- use news search for current and live events
- interrogate one of the local documents
"""

from basic_llm_request import basic_query
from bpschat import ask_document_with_state
from self_critique import self_critique_ideation
from agent import get_info
from prompts import router_prompt, two_e_teacher_system_prompt, general_system_prompt


accepted_router_responses=["default_llm_search", "news_search", "complex_question", "document_search"]

def determine_request_route(input, model="gpt-3.5-turbo", temperature=0.0):
    # TODO take this out maybe -- this seems like it's gonna take WAAAAAAY too long
    retry_count=3
    while retry_count>0:
        val =basic_query(model, router_prompt, input, temperature)
        if val not in accepted_router_responses:
            retry_count-=1
            print("Invalid response from router. Retrying...")
        else:
            return val


def follow_route_for_query(route_type, input, model="gpt-3.5-turbo", temperature=0.0):
    """
    Function that takes the route type as determined by function `router_query` and then routes the query to the appropriate function.
    """
    if route_type=="default_llm_search":
        return basic_query(model, general_system_prompt, input, temperature)
    elif route_type=="news_search":
        return get_info("1", input)
    elif route_type=="complex_question":
        val= self_critique_ideation(input, two_e_teacher_system_prompt, input, show_work=False)
        return val['resolution']
    elif route_type=="document_search":
        return ask_document_with_state(1, input)
    else:
        raise ValueError("Invalid route type: {}".format(route_type))
    

def test_and_timings_route():
    input_list=[
        "What is the capital of France?",
        "What is the biggest news in Europe right now?",
        "How would I go about creating a website?",
        "In the local documents, who are the villians of the story?",
        r"In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?"
    ]

    import time
    llm_model=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    for model in llm_model:
        print(model)
        start_time = time.time()
        for input_prompt in input_list:
            # for each model, let's also calculate and print how long it takes to get the response
            print(" - ", basic_query(model, router_prompt, input_prompt, 0.0))
        print("Time taken: ", time.time()-start_time)

if __name__ == "__main__":

    input_list=[
        "What is the capital of France?",
        "What is the biggest news in Europe right now?",
        "How would I go about creating a website?",
        "In the local documents, who are the villians of the story?",
        r"In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?"
    ]

    import time
    for input in input_list:
        start_time = time.time()
        return_type = determine_request_route(input)
        print("Input", input)
        print("Route type: ", return_type)
        print("Response: ", follow_route_for_query(return_type, input))
        print("#"*20)
        print("Time taken: ", time.time()-start_time)
