# GPT赛博导师 (Cyber-Supervisor) 🚀🤖

让ChatGPT为你的研究助力！提供研究课题，提供参考文献，帮助分析论文创新点. 

## 使用OpenAI API来运行这个项目
1. 在环境变量中添加`OPENAI_API_KEY`.
2. 默认模型使用`gpt-3.5-turbo-16k`. 可以通过修改环境变量中的`DEFAULT_MODEL`来进行修改. 
3. 在命令行中运行`chainlit run cyber-supervisor-openai.py`.

## 基本原理
目前提供了三个函数
1. `find_research_directions`: 为你的研究课题寻找研究方向
2. `find_references`: 为你的论文提供参考文献
3. `judge_novelty`: 让赛博导师帮助分析你提出的想法的创新性 
基于OpenAI API的Function Call功能, ChatGPT会自主选择调用哪一个工具.