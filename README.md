# Auto-Draft: 论文结构辅助工具

这个项目旨在为学术论文写作创建一个全自动辅助工具：搜索相关文献，拟定论文结构，生成论文模板。希望这个工具在未来可以让人们摆脱繁琐的论文写作，将工作的重心放在纯粹的科研上。

***Features***： 
* 提供论文标题，AI完成所有。
* 真实的引用：所有引用的文献可以查找到真实的原文。
* LaTeX模板直接编译。

# Requirements

需要能使用GPT-4的API Key. 生成一篇论文需要15000 Tokens(大约0.5到0.8美元). 总共过程需要大约十分钟. 

# 使用方法 
1. 克隆此仓库：
```angular2html
git clone https://github.com/CCCBora/auto-draft
```
2. 安装依赖：
```angular2html
pip install -r requirements.txt
```
3. 在`auto-draft`文件夹中，创建`api_key.txt`并在其中输入OpenAI API密钥。
4. 编辑`auto_draft.py`以自定义论文标题和描述，然后运行
```angular2html
python auto_draft.py
```

# 示例
直接运行 `auto_draft.py`的原始输出，直接编译得到 (全部输出存放在`outputs/outputs_20230420_114226`中):

Page 1            |  Page 2
:-------------------------:|:-------------------------:
![](assets/page1.png "Page-1") |  ![](assets/page2.png "Page-2") 



