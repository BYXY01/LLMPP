# Example: reuse your existing OpenAI setup
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

function_descriptions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "description": "The temperature unit to use. Infer this from the users location.",
                    "enum": ["celsius", "fahrenheit"]
                },
            },
            "required": ["location", "unit"],
        },
    }
]

completion = client.chat.completions.create(
    model="meta-llama-3.1-8b-instruct",
    messages=[
        # {"role": "system", "content": "总是押韵回答，已注册{get_current_weather}函数"},
        {"role": "user", "content": "调用get_current_weather函数，查查波士顿天气怎么样？"}
    ],
    # temperature=0.7,
    tools=[function_descriptions],
    # function_call="auto",
)

print(completion.choices[0].message.tool_calls)