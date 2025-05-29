# Example: reuse your existing OpenAI setup
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

completion = client.chat.completions.create(
    model="deepseek-r1-distill-qwen-14b",
    messages=[
        {"role": "system", "content": "总是押韵回答"},
        {"role": "user", "content": "介绍介绍你自己"}
    ],
    temperature=0.7,
)

print(completion.choices[0].message)