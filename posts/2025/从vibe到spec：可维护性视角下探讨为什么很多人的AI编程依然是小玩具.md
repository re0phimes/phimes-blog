---
name: 从vibe到spec：可维护性视角下探讨为什么很多人的AI编程依然是小玩具
tag: 公众号 经验分享 技术杂谈
---

## 1 前言

本来没打算写这种经验性或者思考的内容。不过最近~~团队出故障比较多~~问的人比较多，想想以后还是开上一个分系列，说说一些经验和探索。

## 2 Vibe 到 Spec

### 2.1 vibe coding

Vibe Coding比较火的时候，可以在社交媒体上看到那些：一个人对着屏幕说了句“给我做一个类似 Uber 的界面”，然后 AI 刷刷刷生成了一套漂亮的 UI。

![vibe coding时代](https://image.phimes.top/img/202512261503955.png)


并且现在，“一句话生成XX”依然是模型基座能力评估一种方法。这种“一句话生成 ”通常只能产出一个看起来像模像样的独立模块。一旦规模变大，与其他模块交互变多，业务内容复杂起来。或者试图修改其中的一个核心逻辑，事情就麻烦起来了。

### 2.2 spec coding

于是spec coding出现了。

腾讯的codebuddy文档里给的定义是：

>规约编程顾名思义以规范文档驱动的编程（Specification Driven Development, 简称 SDD ），一种以规范为核心的编程方法，旨在通过明确的需求和规则定义，提升软件开发的效率、质量和协作性。在__传统开发中，Spec 是"指导性文件"**——写完就束之高阁，真正的工作还是靠人工编码，人工评审进行驱动优化。但在 AI 编程时代，Spec 不仅仅是指导，而是变成了“可执行的源代码”**，即直接让 AI 根据 Spec 生成完整的代码实现。

其发展大概是从`.cursorrules`/`agents.md`/`claude.md`里写约束规范开始，越来越多的plan mode到kiro里出现spec mode，然后一些影响力较大的社区和作者开发了[speckit](https://github.com/github/spec-kit)和[openspec](https://github.com/Fission-AI/OpenSpec)等等。

spec本质逻辑依然可以解释为：先规划，后执行。不过这个规划被明确的流程约束了，并且spec允许规划到不同的粒度。对于一些从0-1的项目，通常包括：

- 项目基本原则
- 项目需求
- 细节澄清
- 技术方案选型、架构设计、数据模型
- 任务清单
- 项目一致性审核
- 执行

别看这么多任务，框架可能会几个步骤一起，只要你的输入足够丰富，像openspec，其实只需要2-3个指令就可以生成足够文件，让模型根据这些文件来执行。

另一方面，为了能够持续的执行任务。spec通常会用文档或者csv等方法来实现一种状态机，以记录task的完成状态（尚未开始，进行中，已完成等等）。现在大部分还是用markdown去修改，不过我看到一些csv实现的效果也很不错。

这看起来只是**新瓶装旧酒**，但胜在**确实好用**

遵循spec coding，确实让我们的代码一把出的概率大大上升了。得益于部分模型基座强大的coding能力，配合正确的规范的确实提效。但是其依然没有解决我认为核心的问题 **”可维护性“**。

## 3 可维护性的缺失

![可维护性的缺失导致时间成本转移](https://image.phimes.top/img/202512261502487.png)


可维护性很少作为宣传的一个标准存在。这主要归结于大家的前置知识和编程水平不同，导致难以量化。所以市面上大部分能看到的标准通常是以下几点：
- 实际效果：也就是这次AI编程是不是跑的通
- 持续时长：一次AI编程了多久，按照spec后的tasks持续执行了
- 自我debug：遇到一些冲突的问题，是否自己debug验证解决了。
- token的消耗量：有人喜欢消耗的多，作为噱头宣传，当然同样的准确度下，我觉得消耗的越少越好。

但是对于个人而言，如果你不了解自己的能力边界，看得懂多少代码，那就代表着 **“不可维护”**。

或许小打小闹无所谓，但如果你在做一个长期项目，不知道修改哪里，或者你的独立站出现了bug，那成本就是非常高的了。

这里有两个典型的反面教材：

**第一种是“全自动敷衍”。** 

这是我真见到的一个场景，休息的时候听到，隔壁组在吵架，原因是其中一个人提交了一份文档，导致领导破口大骂。AI 生成的一坨文档和实际项目里有太多冲突的东西，已经烂到需要大改了。但是这种东西提交上去，总的有人得去大刀阔斧的改。于是说出 **“下次再交这种你自己都没看过的东西就给我滚蛋吧”**。

**第二种是标准的生产事故**  

这是我认为最常见也最严重的一种。AI coding（甚至不是spec）之后，大部分人的时间成本转移到了code review上，尽管很多代码一把出后看着跑通了。但是一些边界用例，过度的偶尔或者过度抽象的代码都有可能。但是很多人review都不review，直接就往生产环境里丢。

一种迂回的方式是**左脚踩右脚**，也就是

> 自己提需求->AI写代码->AI review->AI 部署->出了bug->AI 改bug

这种**后置一个模型进行检查**的方式能够减少问题出现，但是也**仅仅是减少**，对可维护来说并没有本质上的帮助。

尽管现在大家也做了这样的探索也就是多个AI的结合。比如[cccc](https://github.com/ChesterRa/cccc)。一种多个AI工具并行结合的方案。我自己也试过类似codex mcp来做协同编码的工作。这些工作**很优秀**，但是如果编程上面那种自动化review的流程的话，我认为 **”还不够“**。

![cccc仓库的多AI工具协调例子](https://image.phimes.top/img/202512261253025.png)

其缺陷十分明显，我相信这是所有在一线写代码的人都有过的经验，出了BUG，AI给了解决方案，但是几个小时过去了，怎么改都不对，最后搜索到一个帖子说了同样的情况，一试还真好了。

百试百灵的AI失效的时候，左脚踩右脚就行不通了。

## 4 如何提升可维护性

从技术底层逻辑看，提升可维护性的手段与 MCP、Skills 并无二致，其本质依然是**高密度的提示词注入**。但从工程角度看，这种注入是**对 AI 和自身的认知对齐**。

### 4.1 提升可维护性的前提

说了那么多，似乎一直在提倡程序员的专业能力。实际上不是的，首先我们必须承认的是，每个人的专业技能是有差距的。所以可维护性的前提应该是，在AI编程之前，充分理解自己个人的能力边界。

所以也正如之前所说的可维护性很难作为一个模型能力的基准去体现，人和人的能力不同，作为一个无法量化的标准，我们需要的事定制化的修改。

也就是说，有的人可能代码功底深一点，可以多用一些高级语法糖，或者自己的经验。那么约束就是按照部分来。

但是有些人可能接触的比较少，或者接触一门新的语言，那么就需要保证使用简单方式去实现，而不是看着代码优美，实则什么都不是。

**代码风格的约束，本质上是人的认知负载约束。**

### 4.2 错误的例子

AI 很喜欢做一件事：过度抽象。如果代码有一定的重复，那么他大概率要抽象成更高级的方法。通常情况下没有毛病，但是需求是会变的，所以两个分离模块的相似代码，不一定需要抽象。或者有些时候，发展成为了抽象而抽象，导致看不懂了。

举个例子：一个嵌套json的排序。

```python
users = [
    {"id": 1, "name": "Alice", "stats": {"score": 85, "level": 3}},
    {"id": 2, "name": "Bob",   "stats": {"score": 92, "level": 4}},
    {"id": 3, "name": "Charlie", "stats": {"score": 78, "level": 2}}
]
```

**简单粗暴版本**

如果是我，我可能就这么写了。一行解决。

```python
sorted_users = sorted(users, key=lambda x: x["stats"]["score"], reverse=True)
```

**特定规范版**

比如有些团队约定了，不要用lambda表达式，或者程序员本身就不爱用lambda表达式。

```python
def get_user_score(user):
    """
    提取用户的分数用于排序
    """
    return user["stats"]["score"]

# 使用
sorted_users2 = sorted(users, key=get_user_score, reverse=True)
```

**AI过度优化版**

AI有一段自己独有的思考，似乎还有模有样。“用户其他地方也可能用到这个排序，所以我要给他抽象成工具级别.....”，然后给出了一段读起来费劲的代码。

```python
from typing import List, Dict, Any, TypeVar, Callable, Optional
from abc import ABC, abstractmethod
from functools import reduce

T = TypeVar('T')

class SortStrategy(ABC):
    """定义排序策略的抽象基类"""
    @abstractmethod
    def sort(self, data: List[T]) -> List[T]:
        pass

class DeepKeyExtractor:
    """
    负责处理嵌套键的提取逻辑
    例如: 'stats.score' -> item['stats']['score']
    """
    def __init__(self, key_path: str, separator: str = "."):
        self.key_path = key_path
        self.separator = separator

    def __call__(self, item: Dict[str, Any]) -> Any:
        try:
            # 这里用 reduce 处理嵌套层级，虽然很炫技，但增加了理解成本
            return reduce(lambda d, k: d.get(k, {}), self.key_path.split(self.separator), item)
        except (AttributeError, TypeError):
            # 过度防御：吞掉错误有时候比抛出错误更糟糕
            return float('-inf')

class GenericNestedSorter(SortStrategy):
    """
    具体的排序实现类，支持动态路径解析
    """
    def __init__(self, key_path: str, descending: bool = True):
        self.extractor = DeepKeyExtractor(key_path)
        self.descending = descending

    def sort(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 这里的 key 使用了封装好的 extractor 实例
        return sorted(data, key=self.extractor, reverse=self.descending)

# --- 使用代码 ---

# 你需要先实例化一个 Sorter
sorter = GenericNestedSorter(key_path="stats.score", descending=True)
# 然后再调用
sorted_users = sorter.sort(users)
```

### 4.3 可维护性的本质

其实聊到这里，可维护性的本质也就比较清晰了。就是 **“能看懂”**。

![能看懂](https://image.phimes.top/img/202512261500568.png)

我们制定代码规范，以前是为了让团队协作顺畅，现在是为了**保护人类的能力边界**。

如果 AI 为了展示它的“聪明”，写了一堆复杂的元编程（Metaprogramming）、多层装饰器或者晦涩的函数式链式调用，而你的水平只能轻松驾驭常规的面向对象编程，那么这段代码就是**越界**的。

> [!question] 
> 为什么 AI 喜欢一些过度设计或者大量的防御逻辑？

它看过太多 GitHub 上的“最佳实践”，它试图把所有可能性都考虑到。但在实际工程中，你并不需要那么多最佳实现。你需要的是你能介入，如果AI做的事情让你介入变得无比复杂，那你实际要耗费更长的时间在这**最后一公里的优化上**。

![认知减负](https://image.phimes.top/img/202512261504430.png)

## 5 抛砖引玉：spec之外的辅助约束

其实正如之前所言，因为每个人的编码能力不同，阅读代码能力不同，认知不同。所以我没法给出焚诀，然后告诉你”拿去用吧“。

我依然推荐在Agents.md或者Claude.md中加入自己认为合适的约束。

基于“人必须能读懂”的原则，我的推荐是下面三种原则进行扩充和修改：

```markdown
1.  **可读性 > 抽象**：代码的可读性是绝对的最高优先级。除非能显著减少代码重复或降低复杂度，否则不要创建抽象。坚决避免“过度设计”。

2.  **务实使用高级特性**：仅当有明确的性能优势或架构必要性时，**才可以使用**高级模式（如单例模式、装饰器、工厂模式）。

3.  **显式优于隐式**：避免使用隐藏过多逻辑的“魔法”代码。显式定义返回类型和属性。
```

这段规范的目的不是限制 AI 的能力，而是强制它 **“讲人话”**。

>[!question]
> 不对啊，我看spec-kit已经有对代码风格的约束了。你这是没遵循spec规范？

spec-kit确实有这个部分，但正如之前所说，spec-kit适合0-1项目。但是更多的时候，我们是已经拆分好了功能，然后对项目进行持续的更新，我更多用的openspec，或者自己编写的一些方案。通过agents.md或者claude.md里加入约束，我可以全局生效，是我目前比较喜欢的偏好，我也集成到了codex中的命令里，有我常用的两天不同于openspec的设计约束。

![codex中的自定义prompt](https://image.phimes.top/img/202512261441427.png)

## 6 冲突与权衡：规范不是死教条

最后，聊聊一些传统规范的取舍。我们需要在“读得懂”和“开发效率”之间找平衡。举两个例子。
### 6.1 类型注解的执念

早期我在用 AI 编程时，往往要求极其严格的类型定义，比如要求所有必须用 `pydantic` 或 `dataclasses`，禁止使用 `Dict` 或 `Any`。

但反思一下，这真的对吗？

对于很多中间方法的临时数据结构，或者是一次性的数据清洗脚本，强行定义完整的 `dataclass` 会导致代码量激增，反而增加了阅读负担。在注释清晰的情况下，其实你已经读得懂了，没有必要增添负担。

### 6.2 Docstring 与重构时机

生成代码的时候，有一种观点是：先让 AI 一把梭，生成代码，然后再让它重构、加注释。

但在实际操作中，特别是使用 Opus 这种级别的模型时，如果你的 Spec 规范写得好，**单模块开发往往可以“一把出”**。

如果任务分解（Task）做得足够细，指令跟随做得好，生成的代码本身就已经逻辑自洽了。这时候如果可以保证风格和你自己的一致性，你能看得懂，模型也按你说的做。这时候强行要求“先写草稿再重构”，反而是在浪费 Token 和时间。
## 7 写在最后

这次算是一个抛砖引玉，希望大家都能基于自身能力认知来匹配团队规范，个人规范，以保证项目的可持续性。而不是继续堆积无法维护的屎山代码。（惨痛教训了）

## 8 参考

[1] Fission-AI. OpenSpec: Spec-driven development (SDD) for AI coding assistants [EB/OL]. (2025-12-23) [2025-12-26]. [https://github.com/Fission-AI/OpenSpec](https://github.com/Fission-AI/OpenSpec).

[2] GitHub. spec-kit: Toolkit to help you get started with Spec-Driven Development [EB/OL]. (2025-12-04) [2025-12-26]. [https://github.com/github/spec-kit](https://github.com/github/spec-kit).

[3] ChesterRa. cccc: Two always-on AI peers co-drive your repository as equals [EB/OL]. (2025-12-13) [2025-12-26]. [https://github.com/ChesterRa/cccc](https://github.com/ChesterRa/cccc).

[4] linuxdo.【教程】如何让codex任劳任怨跑几个小时(https://linux.do/t/topic/1353223) [EB/OL]. [2025-12-26]. [https://linux.do/t/topic/1353223](https://linux.do/t/topic/1353223).

[5] 腾讯云. 腾讯云代码助手 CodeBuddy IDE - AI 时代的智能编程伙伴 [EB/OL]. [2025-12-26]. [https://copilot.tencent.com/blog/Spec-Kit](https://copilot.tencent.com/blog/Spec-Kit).