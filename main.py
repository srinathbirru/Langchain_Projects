from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import os

load_dotenv()

MAX_ITERATIONS = 10
MODEL = os.environ["MODEL_NAME"]

#Tool declarations
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"***Executing the get_product_price({product})***")
    prices = {"laptop": 1299.00, "headphones": 149.90, "keyboard": 128.00}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount tier to a price and return the final price.
      Available tiers: bronze, silver, glod.
    """
    print(f"Executing apply_discount(price={price}, discount_tier={discount_tier})")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier,0)
    return round(price * (1 - discount / 100), 2)

#Agent Loop
def run_agent(question: str):
    tools = [get_product_price,apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"groq:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question:{question}")
    print("*" * 30)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant."
                "You have access to product catalog tool"
                "and a discount tool. \n\n"
                "STRICT RULES - You must folloe these exactly\n"
                "1. NEVER guess or assume any product price."
                "You must call get_product_price to get the real price.\n"
                "2. Only call apply_discount AFTER you have received."
                "a price from get_product_price. Pass the exact price"
                "returned by the get_product_price - do not pass a made-up number. \n"
                "3. NEVER calculate discounts youself using math."
                "Always use the apply_discounts tool \n"
                "4. if the user does not specify a discount tier,"
                "ask them which tire to use, don't assume one."
            )
        ),
        HumanMessage(
            content=question
        )
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"---Iteration: {iteration}---")
        
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        print("tool_calls:", tool_calls)

        #if no tools, this is the final answer
        if not tool_calls:
            print("Final answer:", ai_message.content)
            return ai_message.content
        #Process only the First tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"tool selected: {tool_name} with args: {tool_args}")

        tools_to_use = tools_dict.get(tool_name)
        if tools_to_use is None:
            raise ValueError(f"Tool {tool_name} is not found")
        observation = tools_to_use.invoke(tool_args)
        print(f"Tool result: {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )



def main():
    print("Hello from langchain-projects!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount")
    print("result:", result)


if __name__ == "__main__":
    main()
