from dotenv import load_dotenv
load_dotenv()
import regex
import inspect

import ollama
from langsmith import traceable

MAX_ITERATIONS = 100
MODEL = "gpt-oss:latest"

@traceable(run_type="tool")
def get_product_price(product_name: str) -> float:
    """Get the price of a product from the catalog"""
    print(f"Getting price for {product_name}")
    prices = { "laptop": 1000.99, "mouse": 10.90, "keyboard": 20.877 }
    return prices.get(product_name, 0.0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price based on the discount tier. Available discount tiers are: bronze, silver, gold"""
    print(f"Applying discount {discount_tier} to {price}")
    discounts = { "bronze": 0.1, "silver": 0.2, "gold": 0.3 }
    return round(price * (1 - discounts.get(discount_tier, 0.0)), 2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

def get_tools_description(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tools_description = get_tools_description(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES:
1. You MUST use the tools provided to you to answer the question.
2. Never guess the price of a product. Use the tools to get the price.
3. Only apply the discount after getting the price of the product from the tool.
4. If the user does not specify the discount tier, ask them to specify the discount tier.

Answer the following questions as best as you can. You have the following tools available:

{tools_description}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, one of [{tool_names}]
Action Input: the input to the action as comma separated values
Observation: the result of the action
... this thought/action/observation can repeat as many times as needed
Thought: I now know the final answer
final answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""



@traceable(name="Ollama chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)

@traceable(name="Langchain Tool Calling Agent Loop")
def run_agent(question: str):

    
    print(f"Question: {question}")
    prompt = react_prompt.format(question=question)
    scratchpad = ""


    for i in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {i}")
        full_prompt = prompt + scratchpad

        response = ollama_chat_traced(model=MODEL, messages=[{"role": "user", "content": full_prompt}],options={"stop":["\nObservation"],"temperature":0.0})
        
        ai_message = response.message

        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        tool_call=tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"Tool Call: {tool_name} with args: {tool_args}")

        tool_to_use=tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")
        
        observation = tool_to_use(**tool_args)

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    print("ERROR: Max iterations reached without finding a solution")
    return None
        



if __name__ == "__main__":
    print("Hello LangChain Agent (.bindTools)!")
    print()
    result = run_agent("What is the price of the 2 laptops, 3 keyboard and 2 mice? Laptop is in gold tier, keyboard is in bronze tier and mice is in silver tier.")