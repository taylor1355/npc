from openai import OpenAI

openai_client = None

GPT_35 = 'gpt-3.5-turbo-1106'
GPT_4 = 'gpt-4-1106-preview'

def authenticate_openai(api_key):
    global openai_client
    openai_client = OpenAI(api_key=api_key)


def get_llm_completion(prompt, system_prompt, model=GPT_35):
    completion = openai_client.chat.completions.create(model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ])
    return completion.choices[0].message.content