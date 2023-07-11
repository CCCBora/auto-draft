import os
import openai
import ast
from tools import functions, TOOLS

MAX_ITER = 99

openai.api_key = os.getenv("OPENAI_API_KEY")
default_model = os.getenv("DEFAULT_MODEL")
if default_model is None:
    default_model = "gpt-3.5-turbo-16k"

import chainlit as cl

async def process_new_delta(new_delta, openai_message, content_ui_message, function_ui_message):
    if "role" in new_delta:
        openai_message["role"] = new_delta["role"]
    if "content" in new_delta:
        new_content = new_delta.get("content") or ""
        openai_message["content"] += new_content
        await content_ui_message.stream_token(new_content)
    if "function_call" in new_delta:
        if "name" in new_delta["function_call"]:
            openai_message["function_call"] = {
                "name": new_delta["function_call"]["name"]}
            await content_ui_message.send()
            function_ui_message = cl.Message(
                author=new_delta["function_call"]["name"],
                content="", indent=1, language="json")
            await function_ui_message.stream_token(new_delta["function_call"]["name"])

        if "arguments" in new_delta["function_call"]:
            if "arguments" not in openai_message["function_call"]:
                openai_message["function_call"]["arguments"] = ""
            openai_message["function_call"]["arguments"] += new_delta["function_call"]["arguments"]
            await function_ui_message.stream_token(new_delta["function_call"]["arguments"])
    return openai_message, content_ui_message, function_ui_message


system_message = "You are a mighty cyber professor. Follow the following instructions: " \
                "1. You always response in the same language as your student." \
                "2. Ask your student for further information if necessary to provide more assistance. " \
                "3. If your student asks you to do something out of your responsibility, please say no. "

@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": system_message}],
    )


@cl.on_message
async def run_conversation(user_message: str):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": user_message})

    cur_iter = 0

    while cur_iter < MAX_ITER:
        # OpenAI call
        openai_message = {"role": "", "content": ""}
        function_ui_message = None
        content_ui_message = cl.Message(content="")
        async for stream_resp in await openai.ChatCompletion.acreate(
            model=default_model,
            messages=message_history,
            stream=True,
            function_call="auto",
            functions=functions,
            temperature=0.9
        ):

            new_delta = stream_resp.choices[0]["delta"]
            openai_message, content_ui_message, function_ui_message = await process_new_delta(
                new_delta, openai_message, content_ui_message, function_ui_message)

        message_history.append(openai_message)
        if function_ui_message is not None:
            await function_ui_message.send()

        if stream_resp.choices[0]["finish_reason"] == "stop":
            break

        elif stream_resp.choices[0]["finish_reason"] != "function_call":
            raise ValueError(stream_resp.choices[0]["finish_reason"])

        # if code arrives here, it means there is a function call
        function_name = openai_message.get("function_call").get("name")
        arguments = ast.literal_eval(
            openai_message.get("function_call").get("arguments"))

        if function_name == "find_research_directions":
            function_response = TOOLS[function_name](
                research_field=arguments.get("research_description"),
            )
        else:
            function_response = TOOLS[function_name](
                title=arguments.get("title"),
                contributions=arguments.get("contributions"),
            )
        message_history.append(
            {
                "role": "function",
                "name": function_name,
                "content": f"{function_response}",
            }
        )

        await cl.Message(
            author=function_name,
            content=str(function_response),
            language='json',
            indent=1,
        ).send()
        cur_iter += 1