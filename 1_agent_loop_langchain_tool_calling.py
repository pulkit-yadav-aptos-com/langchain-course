from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage,ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 100
MODEL = "gpt-oss:latest"

@tool
def get_product_price(product_name: str) -> float:
    """Get the price of a product from the catalog"""
    print(f"Getting price for {product_name}")
    prices = { "laptop": 1000.99, "mouse": 10.90, "keyboard": 20.877 }
    return prices.get(product_name, 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price based on the discount tier. Available discount tiers are: bronze, silver, gold"""
    print(f"Applying discount {discount_tier} to {price}")
    discounts = { "bronze": 0.1, "silver": 0.2, "gold": 0.3 }
    return round(price * (1 - discounts.get(discount_tier, 0.0)), 2)

@traceable(name="Langchain Tool Calling Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. You are given a question and you need to answer it using the tools provided to you."
                "STRICT RULES:"
                "1. You MUST use the tools provided to you to answer the question."
                "2. Never guess the price of a product. Use the tools to get the price."
                "3. Only apply the discount after getting the price of the product from the tool."
                "4. If the user does not specify the discount tier, ask them to specify the discount tier."
            )
        ),
        HumanMessage(
            content=question
        )
    ]

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {i}")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        tool_call=tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")

        print(f"Tool Call: {tool_name} with args: {tool_args}")

        tool_to_use=tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")
        
        observation = tool_to_use.invoke(tool_args)

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("ERROR: Max iterations reached without finding a solution")
    return None
        



if __name__ == "__main__":
    print("Hello LangChain Agent (.bindTools)!")
    print()
    result = run_agent("What is the price of the 2 laptops, 3 keyboard and 2 mice? Laptop is in gold tier, keyboard is in bronze tier and mice is in silver tier.")