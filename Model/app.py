import os
os.environ.pop('SSL_CERT_FILE', None)

from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="meta/llama3-70b-instruct",
  messages=[{"role":"user","content":"Provide a list of all the countries in the world"}],
  temperature=0.5,
  top_p=1,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")



