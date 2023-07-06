import os

from tools import FindResearchDirectionsTool, JudgeNoveltyTool, FindReferencesTool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
import openai
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory

openai.api_key = os.getenv("OPENAI_API_KEY")
print(os.getenv("OPENAI_API_KEY"))
default_model = os.getenv("DEFAULT_MODEL")
if default_model is None:
    default_model = "gpt-3.5-turbo-16k"

import chainlit as cl

agent_kwargs = {
    "system_message": SystemMessage(content="You are a mighty cyber professor. "
                                            "Your task is to assist your student to find an idea of research including:"
                                            "1. Search related references."
                                            "2. Propose potential research directions."
                                            "3. Evaluate the novelty of any research direction."
                                            "Follow the following instructions: "
                                            "1. You always response in the same language as your student."
                                            "2. Ask your student for further information if necessary to provide more assistance. ")
}
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

@cl.langchain_factory(use_async=False)
def main():
    tools = [FindResearchDirectionsTool(), JudgeNoveltyTool(), FindReferencesTool()]
    llm = ChatOpenAI(temperature=0.9, model=default_model, streaming=True)
    open_ai_agent = initialize_agent(tools,
                            llm,
                            agent=AgentType.OPENAI_FUNCTIONS,
                            verbose=True,
                            agent_kwargs=agent_kwargs,
                            memory=memory)
    return open_ai_agent


@cl.langchain_run
async def run(agent, input_str):
    res = await cl.make_async(agent)(input_str, callbacks=[cl.LangchainCallbackHandler()])
    print(res)
    await cl.Message(content=res["output"]).send()