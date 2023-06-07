import os
import openai
import logging
import requests


log = logging.getLogger(__name__)


def get_gpt_responses(systems, prompts, model="gpt-4", temperature=0.4):
    conversation_history = [
        {"role": "system", "content": systems},
        {"role": "user", "content": prompts}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=conversation_history,
        n=1,  # Number of responses you want to generate
        temperature=temperature,  # Controls the creativity of the generated response
    )
    assistant_message = response['choices'][0]["message"]["content"]
    usage = response['usage']
    log.info(assistant_message)
    return assistant_message, usage


def get_gpt_responses_test(systems, prompts, model="gpt-4", temperature=0.4, base_url=None, key=None):
    end_point = r"/v1/chat/completions"
    if base_url is None:
        base_url =  r"https://api.openai.com" + end_point
    if key is None:
        key = os.getenv("OPENAI_API_KEY")

    url = base_url + end_point

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {key}'  # <-- 把 fkxxxxx 替换成你自己的 Forward Key，注意前面的 Bearer 要保留，并且和 Key 中间有一个空格。
    }

    message = [{"role": "system", "content": systems},
                {"role": "user", "content": prompts}]
    data = {
        "model": model,
        "message": message,
        "temperature": temperature
    }
    response = requests.post(url, headers=headers, json=data)
    response = response.json()
    return response['choices'][0]["message"]["content"]


if __name__ == "__main__":
    pass
