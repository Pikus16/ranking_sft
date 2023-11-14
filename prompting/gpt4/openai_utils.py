import openai
import os

OPENAI_KEY = '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'

def set_openai_key():
    os.environ['OPENAI_API_KEY'] = OPENAI_KEY
    openai.api_key = OPENAI_KEY

def get_client(api_key = OPENAI_KEY):
    return openai.OpenAI(api_key=OPENAI_KEY)

def get_openai_response(prompt, client=None, model_name = 'gpt-4'):
    if openai.__version__ == '0.28.0':
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        return completion.choices[0].message
    else:
        if client is None:
            client = get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content
    