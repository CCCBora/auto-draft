# Auto-Draft: Literatures Summarization Assistant

This project aims to automatically summarize a field of research. 

***Features***: 
* Automatically search related papers. 
* Every claim has an appropriate citation.
* Output LaTeX template.   

This script requires GPT-4 access. One round of generation will take around 10 minutes and 15,000 tokens (0.5 to 0.8 US dollars).  

# Demo

The following link provides a free demo of basic functions. 
If you need more customized features, please refer to *Usage* for local deployment and modification. 

https://huggingface.co/spaces/auto-academic/auto-draft

# Usage
1. Clone this repo：
```angular2html
git clone https://github.com/CCCBora/auto-draft
```
2. Install dependencies:
```angular2html
pip install -r requirements.txt
```
3. Set the `OPENAI_API_KEY` in the environment variables.
4. Edit `auto_backgrounds.py` to customize the topic you want to explore, and then run
```angular2html
python auto_backgrounds.py
```

# Example Outputs
The `outputs` folder contains some original outputs for given inputs. 
They can be directly compiled using Overleaf. 

Page 1            |  Page 2
:-------------------------:|:-------------------------:
![](assets/page1.png "Page-1") |  ![](assets/page2.png "Page-2") 



