# ai-infra-notes📖
## 已更

- AI Infra cpp基础
- 量化基础

## 待更新

- Docker基础
- Cuda基础

## 学习路线

大模型推理优化技术概述图

![image-20250707161136069](https://img.zimei.fun/image-20250707161136069.png)

> *[[2404.14294\] A Survey on Efficient Inference for Large Language Models](https://arxiv.org/abs/2404.14294)*
>
> *Tsinghua Universit*



为了掌握图中提到的大型语言模型（LLM）高效推理技术，笔者认为应该了解如下技术：

###  1.**计算机体系结构基础**

- CSAPP《深入理解计算机系统》推荐章节
  - 第一章：导读
  - 第二章：信息表示处理
  - 第三章：程序的机器级表示
  - 第六章：存储器层次结构

> 涉及的知识点（部分）：
>
> 计算
>
> - 流水线、向量化操作、进制转换、浮点数
>
> 访存
>
> - 数据预取、存储器层次结构

### 2. **前置数学与编程知识**

- **线性代数**：矩阵运算、特征分解等基础知识，对于优化模型计算至关重要
- **概率与统计**：理解 **概率论** 和 **统计学**，尤其是在量化和知识蒸馏中的应用
- **编程能力**：熟悉 **Python**、**C++** 等编程语言，并掌握主流深度学习框架（如 **PyTorch**、**TensorFlow**），熟悉**Linux**开发环境

### 3. **机器学习与深度学习基础**

- **神经网络基础**：了解基本的神经网络架构，随后深入 **Transformer** 模型的原理
- **优化算法**：熟悉常见的优化算法，以及如何在大规模模型中应用它们
- **损失函数与评估指标**：了解各种损失函数（如交叉熵损失、MSE）及其在不同任务中的应用，掌握 **F1 分数**、**准确率**、**困惑度** 等常见评估指标

### 4. 模型训练技术（了解）

- **数据处理**：清洗文本，去除噪声，分词和归一化等
- **优化与正则化**：学习率，过拟合
- **预训练**：在大规模文本上训练基础模型
- **微调**：在特定任务上对预训练模型进行调整
- **分布式训练**：数据并行等技术
- **超参数调整**：调参

### 5. **高效推理技术**

> 前面是公共的前置知识，到这里就可以选择一个具体方向深入研究

- **高效结构设计（Efficient Structure Design, Sec. 5.1）**：
  - **高效前馈网络设计（Efficient FFN Design）**：通过优化前馈神经网络结构，减少计算量和模型参数
  - **高效注意力机制设计（Efficient Attention Design）**：优化注意力机制，减少计算复杂度，如 **低复杂度注意力** 和 **多查询注意力**
  - **Transformer替代（Transformer Alternate）**：探索Transformer架构的替代品，以提升效率。
- **模型压缩（Model Compression, Sec. 5.2）**：
  - **量化（Quantization）**：将模型权重从浮动精度转为低精度（如8位），减少内存占用并加速推理
  - **稀疏化（Sparsification）**：通过剪枝等方式去除不重要的参数，减小模型规模
  - **知识蒸馏（Knowledge Distillation）**：将大模型的知识迁移到小模型，提升小模型的性能并减少计算量

- **图算优化（Graph and Operator Optimization）**：优化计算图、算子，提高计算效率
- **内存管理（Memory Management）**：优化内存的使用，减少内存占用并加速计算过程
- **调度（Scheduling）**：优化任务调度，确保计算资源的合理分配
- **分布式系统（Distributed Systems）**：利用多台机器进行分布式推理，扩展计算能力

## 相关技术博客

| 📖 类别 | 📖 标题                                                       | 📖 作者                                                       |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 量化   | [目前针对大模型进行量化的方法有哪些？ - 知乎](https://www.zhihu.com/question/627484732/answer/3261671478) | @吃果冻不吐果冻皮                                            |
| 量化   | [【模型量化-上】一个Excel说清楚核心公式_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1CVLAzNEzh/?spm_id_from=333.1387.favlist.content.click&vd_source=ac53754f6533097757863a1d248f5406) | @[费曼学徒冬瓜](https://space.bilibili.com/367678065/?spm_id_from=333.788.upinfo.detail.click) |
| 量化   | [[LLMs inference\] quantization 量化整体介绍（bitsandbytes、GPTQ、GGUF、AWQ）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1FH4y1c73W/?spm_id_from=333.337.search-card.all.click&vd_source=ac53754f6533097757863a1d248f5406) | @[五道口纳什](https://space.bilibili.com/59807853/?spm_id_from=333.788.upinfo.detail.click) |
| 量化   | [目前针对大模型进行量化的方法有哪些？ - 知乎](https://www.zhihu.com/question/627484732/answer/13816157360) | @不归牛顿管的熊猫                                            |
|        | 待更新 ...                                                   |                                                              |

## Github 仓库

| 📦 标题                                                       | 📖 作者          | 📖 备注                                            |
| ------------------------------------------------------------ | --------------- | ------------------------------------------------- |
| [LeetCUDA: Modern CUDA Learn Notes with PyTorch for Beginners](https://github.com/xlite-dev/LeetCUDA?tab=readme-ov-file) | @xlite-dev      | **CUDA**编程实战与理论                            |
| [llm-action: 本项目旨在分享大模型相关技术原理以及实战经验（大模型工程化、大模型应用落地）](https://github.com/liguodongiot/llm-action) | @liguodongiot   | 微调，分布式，**模型压缩**                        |
| [AISystem: AISystem 主要是指AI系统，包括AI芯片、AI编译器、AI推理和训练框架等AI全栈底层技术](https://github.com/Infrasys-AI/AISystem) | @Infrasys-AI    | **AI编译器**与体系理论，**B站ZOMI酱**             |
| [leedl-tutorial: 《李宏毅深度学习教程》（李宏毅老师推荐👍，苹果书🍎）](https://github.com/datawhalechina/leedl-tutorial) | @datawhalechina | **DL**理论                                        |
| [A curated list of Awesome LLM/VLM Inference Papers with Codes](https://github.com/xlite-dev/Awesome-LLM-Inference) | @xlite-dev      | AI Infra领域**论文**集，且开源代码                |
| [KuiperInfer: 校招、秋招、春招、实习好项目！带你从零实现一个高性能的深度学习推理库](https://github.com/zjhellofss/KuiperInfer) | @zjhellofss     | CPP**手搓推理库**实战，有软广                     |
| [llm_note: LLM notes, including model inference, transformer model structure, and llm framework code analysis notes.](https://github.com/harleyszhang/llm_note?tab=readme-ov-file) | @harleyszhang   | Triton**手搓推理框架**实战，含HPC博客推荐，有软广 |

