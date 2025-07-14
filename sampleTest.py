import ollama

# Simple text generation
response = ollama.chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': 'Write a REST API endpoint in Python using Flask',
    },
])
print(response['message']['content'])

# # Stream responses
# stream = ollama.chat(
#     model='llama3.2',
#     messages=[{'role': 'user', 'content': 'Tell me a story'}],
#     stream=True,
# )
#
# for chunk in stream:
#     print(chunk['message']['content'], end='', flush=True)