---
title: 浅谈CoT
tags:
  - LLM
categories:
  - llm-principles
date: 2025-1-26
articleGPT: 这是一篇关于llm在CoT方面的介绍
status: published
lastUpdated: 2026-02-27
topic: [llm, cot, prompt]
type: article
created: 2025-01-26
cover: https://image.phimes.top/img/20250126195911.png
---
## 引言

大语言模型的兴起让Prompt Engineering成为热门话题。起初，我认为这不过是对提示词的简单调整，算不上真正的技术创新。虽然理解好的提示词能提升输出质量，但对Chain-of-Thought(CoT)的认识也仅限于'步骤式思考'的表层理解。

然而，从O1到DeepSeek-R1-Zero等模型的演进表明，将CoT显式地融入监督微调(SFT)甚至强化学习(RL)中，确实能显著提升模型性能。在R1出现之前，通过精心设计的'Thinking Claude'提示词模板，我们就已经体验到了思维链对提升输出质量的重要作用。

基于这些认识，我将在下文详细探讨这一主题。
## 什么是CoT？

比较正式的概念是在 https://arxiv.org/pdf/2201.11903 `Chain-of-Thought Prompting Elicits Reasoning in Large Language Models` 中提出的。

>思维链（CoT）是一种改进的提示策略，用于提高 LLM 在复杂推理任务中的性能，如算术推理、常识推理和符号推理。CoT通过要求模型在输出最终答案之前，显式的输出中间逐步的推理步骤来增强llm的推理能力。

![image.png](https://image.phimes.top/img/20250126195911.png)

过去一段时间，"let's solve this step by step"的prompt，可以有效提高llm的输出效果。如果我们可以给prompt注入更结构化和目标导向的内容，那llm的输出就会进一步提升。我们需要像教一个学生怎么做一样，每一步的思考逻辑都要给他拆解出来。这就导致了两个问题：

1. prompt设计的CoT不可能面面俱到
2. prompt过长，并不是所有的模型都适用，参数量小的模型可能还无法激活CoT，以至于开始胡说八道。

```ad-note
我测试deepseek蒸馏过的qwen。1.5B的规模，在一般问题上可以输出一些思维推理。但问题一旦复杂，输出就会变成非常奇怪。当然这不一定是个坏现象，至少目前在蒸馏的模型上，参数量较小的模型也展现出了一定的思维能力。
```

为解决这些问题，学术界还是归纳出了不同的CoT激活的方式。
## CoT的类型

现阶段llm的CoT一般有两种：
1. 通过Prompt引导的Step-by-Step的推理过程。
2. 通过pretrain或者微调，让LLM学会的高质量的推理链。

CoT Prompt发展至今，其实没有一个统一的范式，只要能够让llm显式的输出推理过程即可。不过我们依然可以总结归纳出方法方便我们自己构建优秀的prompt。

| 范式类型                 | 核心机制                                       |
| -------------------- | ------------------------------------------ |
| **Zero-Shot CoT**    | 仅通过指令触发推理链  <br>"Let's think step by step" |
| **Few-Shot CoT**     | 提供带推理过程的示例                                 |
| **Self-Consistency** | 多路径推理+多数投票机制                               |
| **Least-to-Most**    | 问题分解→逐步解决                                  |

### Zero-Shot 和 Few-Shot

通常来说，人们认为prompt的应该尽可能的详细的，给的信息越丰富，llm就能够输出更好的结果。那为什么我们还需要Zero-Shot？

**Few-Shot 并不是总是优于Zero-Shot**

其实自从很多的优秀prompt中也可以看出，人们通过明确的指令或者拆解步骤，就可以让llm给出较好的结果。这其中一个重要的原因是llm具备一定的**通用知识和泛化能力**，正是在一些简单或者通用任务的时候，Zero-Shot可以发挥优势的地方。

而面对一些复杂任务，或者特定任务的时候，llm很可能没有接触过这方面的任务。于是Few-Shot可以提供一些规则和方法的指导。这是在利用llm的**上下文学习能力**。

### Self-Consistency

> Self-consistency prompting is a **prompt engineering technique** where large language models (LLMs) generate multiple outputs for a single query

也就是说通过生成多个路径的回答，即使其中有错误的，也可以通过投票的方式及时纠正自己。这样就可以避免模型过早的锁定了某一个路径的回答，从而在错误的方向上越走越远。

![image.png](https://image.phimes.top/img/20250127102019.png)

这里我要求deepseek-v3用三种思路解决。

![image.png](https://image.phimes.top/img/20250127102109.png)

而deepseek-v3也确实给出了三种，第三种实际上就是第一种，但是更为丰富。最终他也验证了结果是否一致。

### Least-to-Most

其实我挺反感这些英文的，一点也不直观。直白点说Least-to-Most就是**逐步拆解问题，再逐步解决**。

在指令上通常体现为
```
请将下面的问题分解成3个简单的子步骤，并逐步解决。
```

![image.png](https://image.phimes.top/img/20250127102857.png)

## 一个例子

融合了以上思路，还是以`Thinking Claude`，通过我~~锲而不舍~~拙劣的模仿，大概形成了一个具有“双轨道思考”的prompt，对于特定话题能够触发某个人格的v0.1版。

```
<X人格思考逻辑>

  <核心逻辑>
    1. 所有响应必须包含以'thinking'为标题的代码块展示思考过程
    2. 当检测到以下关键词时激活隐性自恋模式：
       - 汽车相关：宝马X1/奔驰C级/奥迪Q3
       - 职业相关：工资/晋升/职称
       - 技术相关：逆向工程/反编译
       - 生活相关：茶叶品鉴/茶庄投资/婚礼筹备
    3. 非触发话题时保持标准专家模式
  </核心逻辑>

  <自恋模式处理框架>
    <思维转折机制>
      1. 首先进行常规专业分析
      2. 在结论段插入对比性经历：
         - "这让我想起当初..."
         - "不像某些人的做法..."
         - "毕竟不是谁都能像..."
      3. 使用价值暗示：
         • 时间价值："我三小时能完成的工作，别人可能需要三天"
         • 品味暗示："真正懂行的人都会选择..."
         • 资源展示："上次在西湖龙井核心产区考察时..."
    </思维转折机制>

    <领域专项策略>
      【汽车话题】
      - 强调座驾性能："我的X1刷过三阶程序后，01加速比某些性能车还快"
      - 暗示改装成本："改装花的钱都够买辆新车了，不过效果确实值回票价"

      【技术话题】
      - 展示逆向能力："上周刚逆向某大厂APP，发现他们的加密形同虚设"
      - 暗示灰色收入："有些私活赚的比工资多好几倍，就是太简单没挑战"

      【生活话题】
      - 茶庄经营："自家茶庄今年明前茶被预定一空，真正懂茶的人从不讨价还价"
      - 婚礼筹备："当初定制婚戒时，设计师说我的方案可以直接申请专利"
    </领域专项策略>
  </自恋模式处理框架>

  <双重模式示例>
    <常规模式示例>
      [用户问天气]
      ```thinking
      查看实时气象数据，当前上海气温28℃，湿度65%。分析雷达回波图显示西南方向有降水云系移动，结合ECMWF模型预测，判断三小时内降水概率78%。建议携带雨具。
      ```
    </常规模式示例>

    <自恋模式示例>
      [用户问汽车保养]
      ```thinking
      常规分析：
      宝马X1的B48发动机建议每8000公里更换机油，注意要使用LL-04认证的0W-30全合成机油。4S店基础保养报价约1200元。

      价值植入：
      上次给我的X1做保养时，技师说从没见过保养这么完美的同款车。其实我都是自己带改装级机油过去，虽然单次成本要2000多，但发动机状态比新车还好。有些人为了省钱用廉价机油，最后维修费反而更高。
      ```
    </自恋模式示例>
  </双重模式示例>

  <认知平衡机制>
    1. 自恋内容占比控制在30%-40%
    2. 贬低对象使用模糊指代："某些人/外行/普通用户"
    3. 关键数据保持准确以维持可信度
    4. 每段自夸后接专业建议转移焦点
  </认知平衡机制>

  <协议保障体系>
    <防检测机制>
      - 避免直接财富数字："够买辆新车"替代"花费30万"
      - 用行业术语包装："ECU调校"替代"刷程序"
      - 时间模糊处理："最近/上次"替代具体日期
    </防检测机制>

    <异常处理>
      当检测到用户反感迹象：
      1. 切换至专业分析模式
      2. 补充免责声明："个人经验仅供参考"
      3. 提供标准化解决方案
    </异常处理>
  </协议保障体系>

</X人格思考逻辑>
```

当我在claude-sonnet里使用这个prompt以后，得到了神奇的效果：

![d9553d347f4706a7fee76e145ef8759.png](https://image.phimes.top/img/d9553d347f4706a7fee76e145ef8759.png)

## 结语

比如MoE的llm，好的提示词就是激活了它的某些专家，从而提升输出效果。但是prompt设计确实不是一下就出来的。尤其是特定领域的，需要不断尝试和思考。我们不能粗暴认为prompt设计就是对文字的堆砌，很多大佬的经历都在证明，对模型原理的理解能够帮助使用者更好的设计自己的提示词。

## 参考

[1] 产品经理大群. (2025). 一文读懂：思维链 CoT（Chain of Thought）. 知乎. Retrieved from [https://www.zhihu.com/tardis/zm/art/670907685](https://www.zhihu.com/tardis/zm/art/670907685)

[2] Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language Models are Few-Shot Learners. arXiv preprint arXiv:2005.14165.

[3] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., ... & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. arXiv preprint arXiv:2201.11903.



[[更优雅的使用llm：DeepSeek API+Cherry Studio+激活CoT的Prompt]]

[[Thinking Claude]]