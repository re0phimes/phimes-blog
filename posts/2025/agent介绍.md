---
title: agent概述 2025-01-10
tags:
  - LLM
  - Agent
categories:
  - agents
date: 2025-1-10
---

# 概述

最早学习大模型的时候，被大模型五花八门的概念弄得云里雾里。国内在翻译的时候也存在理解偏差。比如agents被翻译为代理。这篇主要为了简要的区分models、agents、workflows，并初步介绍agents在当前（2025-01-10），具体是什么，有哪些功能，有哪些工具或者框架。

# Agent介绍

Agent本质是一个自主决策的智能体，它通过持续观察环境、制定计划、执行动作来实现特定目标。与简单的问答系统不同，Agent具备任务规划、工具使用、环境交互和记忆管理等核心能力。其关键特征包括：目标导向的自主决策能力，即能够根据用户意图自主规划任务步骤；环境感知与交互能力，可以观察和理解当前状态并作出相应反应；工具使用能力，能够选择和调用适当的工具完成任务；持续学习与适应能力，可以从交互过程中积累经验并优化决策。Agent的运行通常遵循"感知-规划-执行"的循环模式，通过不断迭代来逐步实现复杂目标。

> 从最基本的形式来看，智能体可以被定义为一个通过观察世界并使用其可用工具采取行动来实现目标的应用程序。智能体是自主的，可以独立于人类干预而行动，特别是当被赋予适当的目标或要实现的目标时。智能体在实现目标的方法上也可以是主动的。即使在没有来自人类的明确指令集的情况下，代理也可以推理下一步应该做什么来实现其最终目标。为了理解代理的内部运作，智能体的行为、行动和决策的基础组件。这些组件的组合可以被描述为认知架构，通过这些组件的混合和匹配可以实现许多这样的架构。聚焦于核心功能，代理的认知架构中有三个基本组件[^1]

## Agent、model以及workflow
![google《agents》 models vs agents](https://image.phimes.top/img/202501100954832.png)

google已经清晰阐述了models和agents的对比：
- 知识方面，agents可以基于工具对外部知识进行扩展
- 交互方面的，models本社你不具备逻辑，主要是基于用户的输入进行预测，而agents通过维护一个session histroy（chat history）来获得多次的交互。
- agents支持工具调用。
- models需要通过prompts来激活CoT、ReAct等，而agents则不需要，agents已经集成了相关的逻辑思考模式。

### workflow [[workflow概述]]

> workflow通过定义好的流程协调多个推理/预测。管理预测/行动发生的顺序和条件。内置流程控制逻辑，包括条件判断、循环、错误处理和并行执行能力。提供工具集成和编排的框架。可以协调不同组件间的工具使用。

workflow其实并不能和models或者agents并列去比较。workflow可以和models或agents结合，更专注于流程的编排、执行和知识的协调。依然很多人把agents和workflow互相混淆，其原因是确实有不少人把agents定义成了一个可以独立自主运行的自动化系统。anthropic的[Building effective agents \ Anthropic](https://www.anthropic.com/research/building-effective-agents)中提到：

>"Agent" can be defined in several ways. Some customers define agents as fully autonomous systems that operate independently over extended periods, using various tools to accomplish complex tasks. Others use the term to describe more prescriptive implementations that follow predefined workflows. At Anthropic, we categorize all these variations as **agentic systems**, but draw an important architectural distinction between **workflows** and **agents**:
>- **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
>- **Agents**, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

所以workflows具有如下特点：
- 明确定义的输入输出
- 支持工具、models、agents通过指定的程序逻辑进行进行执行和调用
- 具有一定的条件分支、决策控制、并行处理或者循环迭代的能力

![一种workflow的例子](https://image.phimes.top/img/202501101128193.png)

# agent的组成

```ad-question[注意]
为什么agents具备自主决策、主动推理下一步的功能？
```

根据google的《agents》，google把agents分为三个部分：`model`、 `tools` 、`Orchestration`。三者共同组成了一个agent。
**可以发现，除了model，它还包含了orchesration和tools两个部分**
- orchestration给予了大模型记忆、编排、根据输入进行特定方式思考的能力
- tools则给予了大模型访问外界知识和工具的能力

![image.png](https://image.phimes.top/img/202501101129372.png)
## model
model是模型的基座，离线的如Qwen2.5系列、Llama系列等。这里就不再赘述
## 工具

### 概述tools calling（function calling）
工具调用赋予了大模型访问外界知识的能力。通常来说大模型不具备回答已学习到的知识以外的能力。但是通过在训练数据中包含了工具调用的示例，并在推理时提供模板生成的工具调用指令来执行外部工具。

### 工具调用的实现逻辑

1. 预训练阶段： 
	- 模型在大规模数据上预训练时学习了基本的语言理解能力 
	- 也学习了程序设计、API调用等相关知识 - 建立了自然语言和程序调用之间的关联 
1. 特定训练/微调： 
	- Qwen文档提到:"Qwen2预先训练了多种支持函数调用的模板" 
	- 这意味着模型经过专门训练，认识特定格式的system prompt 
	- 学会了如何响应这类包含函数描述的提示 
1. 模板机制： 看看文档中提到的几种模板： 
	- ReAct Prompting格式 
	- Hermes风格的工具调用 
	- Qwen自己的函数调用模板

```ad-question[注意]
如何在大模型中使用tools？
```

### 大模型中使用工具的两种模式

这里不深入讨论哪种方法更好，现有情况下我们可以根据框架，选择自己喜欢的。

#### 第一种：prompt函数调用模板+解析执行器
这种方法，我们通常通过在prompt中按照一套模板格式定义tools，这个tools是一个list，而list中的每一个object对象都通过了`name`、`parameters`等关键参数把执行逻辑定义清晰。然后通过通用执行器把每个param实现可执行方法。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 定义实际的函数
def get_weather(location: str) -> str:
    """实际执行天气查询的函数"""
    return f"天气晴朗，温度25度"

# 2. 定义提示模板
PROMPT_TEMPLATE = """你是一个天气助手。当用户询问天气时，你需要：
1. 思考：分析用户的问题是否需要查询天气
2. 行动：如果需要查询天气，使用格式 <call>get_weather|城市名</call> 来调用天气查询
3. 回答：基于查询结果给出友好的回复

例如：
用户：北京天气怎么样？
思考：用户想知道北京的天气，我需要查询
行动：<call>get_weather|北京</call>
回答：根据查询结果，北京现在天气晴朗，温度25度，是个不错的天气呢！

记住：只在需要查询天气时使用<call>标签。现在请回答用户的问题。

用户：{query}
思考："""

# 3. 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-2.5", device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-2.5", trust_remote_code=True)

# 4. 处理对话
def chat_about_weather(query: str):
    # 构建初始提示
    prompt = PROMPT_TEMPLATE.format(query=query)
    
    # 获取模型回复
    response, history = model.chat(tokenizer, [{"role": "user", "content": prompt}])
    
    # 处理可能的函数调用
    if "<call>get_weather|" in response:
        # 解析函数调用
        start = response.find("<call>get_weather|") + len("<call>get_weather|")
        end = response.find("</call>")
        location = response[start:end]
        
        # 执行函数调用
        weather_info = get_weather(location)
        
        # 将结果返回给模型继续对话
        full_context = prompt + response + f"\n获取到的天气信息：{weather_info}\n请基于这个信息给出最终回答："
        final_response, _ = model.chat(tokenizer, [{"role": "user", "content": full_context}])
        return final_response
    
    return response

# 5. 测试
test_queries = [
    "北京天气怎么样？",
    "上海今天天气如何？",
    "你觉得明天适合出门吗？"
]

for query in test_queries:
    print(f"\n用户: {query}")
    response = chat_about_weather(query)
    print(f"助手: {response}")
```

#### 第二种：注册函数
```python
# 1. 定义函数
def get_weather(city: str):
    return f"{city}的天气..."

# 2. 定义工具
tools = [{
		  "name": "get_weather",
		  "description": "Get weather information for a location",
		  "parameters": { 
		  "type": "object", 
		  "properties": {
			  "location": {
			  "type": "string", 
			  "description": "The city name" }
		  },
		  "required": ["location"]
		}
	}
]


# 3. 在模型初始化时注册工具
from qwen_agent import QwenAgent
agent = QwenAgent( model_path="Qwen/Qwen-7B-Chat", tools=tools )
```


### 现在的函数调用模板

#### ReAct Prompting

>例如，可以使用ReAct Prompting实现带有额外规划元素的函数调用：
>- **Thought**：显而易见的推理路径，分析函数和用户查询，并大声“说”出来
>- **Action**：要使用的函数以及调用该函数时应使用的参数
>- **Observation**：函数的结果



```bash
Answer the following questions as best you can. You have access to the following tools:

{function_name}: Call this tool to interact with the {function_name_human_readable} API. What is the {function_name_human_readable} API useful for? {function_desciption} Parameters: {function_parameter_descriptions} {argument_formatting_instructions}

{function_name}: Call this tool to interact with the {function_name_human_readable} API. What is the {function_name_human_readable} API useful for? {function_desciption} Parameters: {function_parameter_descriptions} {argument_formatting_instructions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{function_name},{function_name}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}
Thought: {some_text}
Action: {function_name}
Action Input: {function_arguments}
Observation: {function_results}
Final Answer: {response}
```

#### Qwen2.5的函数调用模板

```bash
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

Current Date: 2024-09-30

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \"City, State, Country\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \"celsius\"."}}, "required": ["location"]}}}
{"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \"City, State, Country\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \"Year-Month-Day\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \"celsius\"."}}, "required": ["location", "date"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
What's the temperature in San Francisco now? How about tomorrow?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "San Francisco, CA, USA"}}
</tool_call>
<tool_call>
{"name": "get_temperature_date", "arguments": {"location": "San Francisco, CA, USA", "date": "2024-10-01"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}
</tool_response>
<tool_response>
{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}
</tool_response><|im_end|>
<|im_start|>assistant
The current temperature in San Francisco is approximately 26.1°C. Tomorrow, on October 1, 2024, the temperature is expected to be around 25.9°C.<|im_end|>
```

## 编排层

不同的框架在编排层塞入的模块和逻辑也不大相同。编排层是赋予模型的自主决策能力、存储上下文（长短期记忆能力）、反馈机制、知识库或者其他功能的能力。即使一些简化的框架，也会至少提供：

- Memory模块：用于提供长期、短期记忆，通常配合一些向量库
- 思维模式：比如CoT，ToT，ReAct，这可以从prompt激活。
- 任务规划：通过一些逻辑控制代码赋予一定的管理执行流程、中间状态处理以及停止条件等。


# 参考

[^1]:Schluntz, E., & Zhang, B. (2024, December 20). Building effective agents. Anthropic. https://www.anthropic.com/research/building-effective-agents

[^2]: Jarvis, C., & Palermo, J. (2023). How to call functions with chat models. OpenAI Cookbook. Retrieved from https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models

[^3]:[函数调用 - Qwen](https://qwen.readthedocs.io/zh-cn/latest/framework/function_call.html#)
