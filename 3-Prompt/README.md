
## 什么是Prompt

Prompt是一种用于指导以大语言模型为代表的**生成式人工智能**生成内容(文本、图像、视频等)的输入方式。它通常是一个简短的文本或问题，用于描述任务和要求。

Prompt可以包含一些特定的关键词或短语，用于引导模型生成符合特定主题或风格的内容。例如，如果我们要生成一篇关于“人工智能”的文章，我们可以使用“人工智能”作为Prompt，让模型生成一篇关于人工智能的介绍、应用、发展等方面的文章。

Prompt还可以包含一些特定的指令或要求，用于控制生成文本的语气、风格、长度等方面。例如，我们可以使用“请用幽默的语气描述人工智能的发展历程”作为Prompt，让模型生成一篇幽默风趣的文章。

总之，Prompt是一种灵活、多样化的输入方式，可以用于指导大语言模型生成各种类型的内容。

![](https://img-blog.csdnimg.cn/img_convert/05b84c9109a4ad9c285e4c9986c5aac9.png)
## 什么是提示工程

提示工程是一种通过设计和调整输入(Prompts)来改善模型性能或控制其输出结果的技术。

在模型回复的过程中，首先获取用户输入的文本，然后处理文本特征并根据输入文本特征预测之后的文本，原理为**next token prediction**。

提示工程是模型性能优化的基石，有以下六大基本原则：

- 指令要清晰
- 提供参考内容
- 复杂的任务拆分成子任务
- 给 LLM“思考”时间(给出过程)
- 使用外部工具
- 系统性测试变化

## 提示设计框架

- CRISPE，参考：[https://github.com/mattnigh/ChatGPT3-Free-Prompt-List](https://github.com/mattnigh/ChatGPT3-Free-Prompt-List)

  - **C**apacity and **R**ole (能力与角色)：希望 ChatGPT 扮演怎样的角色。​
  - **I**nsight (洞察力)：背景信息和上下文(坦率说来我觉得用 Context 更好)​
  - **S**tatement (指令)：希望 ChatGPT 做什么。​
  - **P**ersonality (个性)：希望 ChatGPT 以什么风格或方式回答你。​
  - **E**xperiment (尝试)：要求 ChatGPT 提供多个答案。

  写出的提示如下：

  ```
  Act as an expert on software development on the topic of machine learning frameworks, and an expert blog writer. The audience for this blog is technical professionals who are interested in learning about the latest advancements in machine learning. Provide a comprehensive overview of the most popular machine learning frameworks, including their strengths and weaknesses. Include real-life examples and case studies to illustrate how these frameworks have been successfully used in various industries. When responding, use a mix of the writing styles of Andrej Karpathy, Francois Chollet, Jeremy Howard, and Yann LeCun.
  ```

- CO-STAR，参考：[https://aiadvisoryboards.wordpress.com/2024/01/30/co-star-framework/](https://aiadvisoryboards.wordpress.com/2024/01/30/co-star-framework/)

  ![](https://img-blog.csdnimg.cn/img_convert/fe7e6c67bbd9ed8d0d7e817cf0bbbf42.png)

    - **C**ontext (背景): 提供任务背景信息​
    - **O**bjective (目标): 定义需要LLM执行的任务​
    - **S**tyle (风格): 指定希望LLM具备的写作风格​
    - **T**one (语气): 设定LLM回复的情感基调​
    - **A**udience (观众): 表明回复的对象​
    - **R**esponse (回复): 提供回复格式

    完成的提示如下：

    ```
  # CONTEXT # 
  I am a personal productivity developer. In the realm of personal development and productivity, there is a growing demand for systems that not only help individuals set goals but also convert those goals into actionable steps. Many struggle with the transition from aspirations to concrete actions, highlighting the need for an effective goal-to-system conversion process.
  
  #########
  
  # OBJECTIVE #
  Your task is to guide me in creating a comprehensive system converter. This involves breaking down the process into distinct steps, including identifying the goal, employing the 5 Whys technique, learning core actions, setting intentions, and conducting periodic reviews. The aim is to provide a step-by-step guide for seamlessly transforming goals into actionable plans.
  
  #########
  
  # STYLE #
  Write in an informative and instructional style, resembling a guide on personal development. Ensure clarity and coherence in the presentation of each step, catering to an audience keen on enhancing their productivity and goal attainment skills.
  
  #########
  
  # Tone #
   Maintain a positive and motivational tone throughout, fostering a sense of empowerment and encouragement. It should feel like a friendly guide offering valuable insights.
  
  # AUDIENCE #
  The target audience is individuals interested in personal development and productivity enhancement. Assume a readership that seeks practical advice and actionable steps to turn their goals into tangible outcomes.
  
  #########
  
  # RESPONSE FORMAT #
  Provide a structured list of steps for the goal-to-system conversion process. Each step should be clearly defined, and the overall format should be easy to follow for quick implementation. 
  
  #############
  
  # START ANALYSIS #
  If you understand, ask me for my goals.
    ```


## 优质文档推荐
非常好的提示词工程教程：[提示工程指南](https://www.promptingguide.ai/zh)