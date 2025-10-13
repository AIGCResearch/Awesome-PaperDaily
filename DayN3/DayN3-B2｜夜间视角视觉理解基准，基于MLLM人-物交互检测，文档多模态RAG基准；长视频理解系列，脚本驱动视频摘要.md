# Topic: Multi-modal 
## EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark 
2025-10-07｜Sofia U, Sofia University, ECNU, HKUST(GZ), NKU, FDU 

<u>http://arxiv.org/abs/2510.06218v1</u>  
### 概述 
  
EgoNight是首个专注于夜间视角视觉理解的综合性基准数据集，核心任务为视觉问答（VQA）。现有的多数第一视角视觉研究集中于白天或良好光照环境，忽视了夜间低光照条件下的实际应用需求。为填补这一空白，EgoNight整合了合成环境与真实世界的室内外视频，特别设计了日夜对齐的视频对，利用白天数据辅助夜间标注，揭示光照变化带来的性能差异。数据集包含三部分：模拟合成视频（EgoNight-Synthetic）、保加利亚索菲亚实录视频（EgoNight-Sofia）及牛津现有数据（EgoNight-Oxford），覆盖多样场景与光照条件。基于此，构建了多样化的VQA任务体系，涵盖物体识别、空间推理、动作识别、计数、文本识别及新颖的照明动态和非常识推理等，全面考察模型在低光环境下的理解与推理能力。此外，EgoNight引入了夜间的深度估计和日夜对应检索任务，拓展了评测维度，推动夜间第一视角视觉理解的研究。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/82.jpg)  
EgoNight的数据采集与标注方法主要包括以下三个方面：  

1. **视频采集与日夜对齐设计**：采用三种视频来源保证数据多样性与对齐精度。合成数据通过Blender精确控制场景、摄像机轨迹和光照，生成像素级对齐的日夜视频对。索菲亚实录采用视频引导录制策略，白天录制视频作为夜间录制的参考，确保动作和视角的空间时间一致性，辅以后期剪辑优化对齐。牛津数据则作为补充，虽无严格对齐但提供更多夜间场景。  
2. **多样化VQA任务设计**：定义12类问答任务，分为可日夜对比的“配对”任务（如物体识别、空间推理）和仅夜间有效的“非配对”任务（如照明识别、动态检测），涵盖静态和动态视频分析，提升任务的全面性和挑战性。  
3. **三阶段自动标注与人工校验流程**：首先利用多模态大语言模型（MLLM）生成夜间视频的详细描述，针对不同任务类型定制化提示确保关键内容覆盖；其次基于描述生成多样化问题；最后结合对应白天视频生成伪答案以提升准确性，未对齐视频则直接基于夜间内容。所有自动生成的问答对经过人工筛查、修改和补充，确保高质量标注，累计投入300余小时人力。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/83.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/84.jpg)  
在EgoNight的三个子数据集上，评测了包括闭源GPT-4.1、Gemini2.5Pro及多款开源模型（Qwen2.5-VL系列、InternVL3等）和专门的第一视角模型EgoGPT。结果显示所有模型在夜间条件下表现显著下降，最高准确率仅约31%，且夜间性能普遍比白天低25%至33%。闭源模型整体优于开源模型，但模型大小并非性能的唯一决定因素。任务层面，感知型任务（如物体和文本识别）夜间降幅更大，而推理型任务（如导航、场景序列）虽整体较难但受光照影响相对较小。新增的照明识别、动态检测和非常识别任务表现尤为低迷，暴露当前模型在低光复杂推理上的不足。此外，日夜对应检索任务中，GPT-4.1在空间检索表现优异，但时间定位能力较弱，表明时间推理仍是难点。夜间深度估计任务显示所有模型表现不佳，鱼眼镜头专用方法略有优势，凸显夜间和第一视角深度估计的挑战。

### 通俗易懂  
这项研究的核心是让计算机理解我们戴着摄像头在夜晚看到的东西。为了做到这一点，研究者们收集了三种视频：电脑生成的完全一样场景的白天和夜晚视频，现实中人们在白天和夜晚走同样路线录制的视频，以及一些现成的夜间视频。这样做的好处是可以直接对比白天和夜晚的差别，帮助模型学习在低光照下也能看清楚。标注数据时，研究者先用智能语言模型描述视频内容，再让它根据描述提出问题，最后结合白天更清晰的视频答案生成夜间视频的答案。整个过程还经过人工仔细检查，确保问答准确。通过这些方法，他们建立了一个既真实又丰富的夜间视觉理解测试平台。实验结果显示，现有最先进的视觉语言模型在夜晚的表现远不如白天，尤其是识别物体和文字时困难很大。这说明要让智能设备在夜晚像白天一样聪明，还需要更多针对低光环境的技术突破。 

# Topic: Multi-modal｜Image 
## HOI-R1: Exploring the Potential of Multimodal Large Language Models for Human-Object Interaction Detection 
2025-10-07｜UEC 

<u>http://arxiv.org/abs/2510.05609v1</u>  
<u>https://github.com/cjw2021/HOI-R1</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/85.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/86.jpg)  
本文提出了HOI-R1，一种基于多模态大语言模型（MLLM）的全新人体-物体交互检测（HOID）框架。传统HOID方法依赖复杂的目标检测器和视觉语言模型（VLM）来提取交互特征，导致架构复杂且难以扩展。HOI-R1创新性地摒弃了目标检测器，直接利用MLLM通过自然语言推理完成交互检测，极大简化了流程。该方法设计了专门的提示模板和推理过程，结合监督微调（SFT）和强化学习（RL）两阶段训练策略，有效提升模型对交互关系的理解和泛化能力。实验在HICO-DET数据集上验证，HOI-R1在准确率上实现了基线模型两倍的提升，且训练效率显著优于传统方法，展示了MLLM在结构化视觉任务中的巨大潜力。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/87.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/88.jpg)  
HOI-R1方法核心包括以下几个部分：  

1. **语言驱动的HOID预测**：通过设计包含任务指令、推理引导和格式示例的多段式提示模板，引导MLLM输出符合HOID标准格式的自然语言结果，实现对人体和物体边界框及交互动作的联合预测。  
2. **思维蒸馏的监督微调（SFT）**：利用GPT4o-mini作为教师模型，生成包含详细推理步骤的示范答案，学生模型通过模仿这些推理链条和最终HOI标签，内化复杂的交互逻辑，强化对任务的理解。  
3. **强化学习（RL）对齐**：采用基于Group Relative Policy Optimization（GRPO）的强化学习框架，设计了多维度奖励函数，包括格式正确性、对象与动作标签准确率以及基于IoU的空间匹配奖励，进一步提升模型预测的结构完整性和空间精度。该训练策略有效避免了奖励欺骗现象，确保模型生成合理且准确的HOI输出。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/89.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/90.jpg)  
实验在HICO-DET数据集上进行，使用mAP作为主要评估指标。首先，基础的Qwen2.5-VL-3B模型通过提示模板即获得优于传统HOID方法的性能，展现了MLLM的强大先验知识。经过SFT训练后，模型性能显著提升，训练仅1个epoch即达到16.77 mAP，约为基线的两倍。进一步结合GRPO强化学习，模型在Rare和Non-Rare类别上分别实现18.33和19.02 mAP，超越了更大规模的MLLM和传统Transformer基线。消融实验表明，推理链条提示和任务指令对性能提升至关重要，奖励函数中标签准确率和IoU奖励均不可或缺，缺一均导致明显性能下降。定性分析显示，强化学习阶段显著提升了模型对复杂交互的定位和识别能力，验证了方法设计的有效性和实用性。

### 通俗易懂  
HOI-R1的核心思想是让一个聪明的语言模型直接“看懂”图片里人和物体之间的互动，而不需要传统的先找出人和物体框框再分析它们的动作。具体做法是先让一个强大的“老师”模型用文字一步步解释图中发生了什么，学生模型通过模仿老师的解释来学习识别这些交互。接着，学生模型再通过一个类似游戏的训练方式，得到奖励来鼓励它输出格式正确、动作和物体标签准确、并且框选位置合理的答案。这个过程就像教一个学生先理解题目，再通过不断练习和评分提升答题能力。最终，这种训练让模型能快速准确地告诉我们图片中“谁在做什么”，而且比传统方法更简单高效，能更好地应用到实际场景中。 
## UNIDOC-BENCH: A Unified Benchmark for Document-Centric Multimodal RAG 
2025-10-04｜Salesforce 

<u>http://arxiv.org/abs/2510.03663v2</u>  
<u>https://github.com/SalesforceAIResearch/UniDOC-Bench</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/91.jpg)  
本文提出了UniDoc-Bench，这是首个面向文档中心多模态检索增强生成（MM-RAG）的统一大规模基准，涵盖70,000页真实PDF文档，跨8个领域。该基准通过提取并关联文本、表格和图像中的证据，生成了1,600个多模态问答对，涵盖事实检索、比较、摘要和逻辑推理四种问题类型。UniDoc-Bench支持文本、图像、文本-图像融合和多模态联合检索四种检索范式的公平比较，采用统一的候选池、提示和评估指标。实验表明，文本-图像融合的RAG系统在答案完整性和准确性上优于单模态和联合多模态嵌入检索，揭示了当前多模态嵌入的不足。该基准不仅推动了多模态文档智能系统的评测标准化，也揭示了视觉信息如何补充文本证据及系统的系统性失败模式，为未来MM-RAG系统设计提供了实用指导。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/92.jpg)  
本研究的方法主要包括三个核心步骤：  

1. **文档收集与过滤**：利用PDFA数据源，设计字段标签系统对文档进行领域、子领域、语言、模态等多维度筛选，最终构建涵盖金融、法律、医疗等8个领域、共70,000页的高质量多模态文档库。  
2. **问答对合成**：通过PDF解析提取文本块、表格和图像，构建跨模态知识图谱连接相关证据。设计4类问题模板（事实检索、比较、摘要、逻辑推理），结合GPT-4和Gemini-Pro多轮生成与验证，确保问答对的多样性、事实性和跨模态依赖。问答对经过重写以保证自包含性和符合人类查询意图，20%样本由多名注释员和专家复核。  
3. **统一评测框架**：设计统一的候选检索池和评价指标，支持文本检索、图像检索、文本-图像融合检索和多模态联合检索的公平比较。通过映射检索结果回原文档页码，实现跨模态和多页证据的准确匹配，评估检索精度和召回率，以及最终生成答案的完整性和可信度。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/93.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/94.jpg)  
实验部分系统评估了四种检索模型（文本、图像、联合多模态嵌入、文本-图像融合）和六种RAG系统的性能。结果显示，图像检索召回率较高但精度较低，文本检索精度较好但召回略低，融合检索综合表现最佳。多模态联合嵌入模型虽然召回率接近融合模型，但精度显著较低，说明当前多模态嵌入技术尚不成熟。端到端评测中，文本-图像融合RAG系统在答案完整性上领先（68.4%），优于文本单模态（65.3%）、图像单模态（54.5%）和多模态联合检索（64.1%）。图像依赖型问题对所有系统仍具挑战性，表明未来改进应聚焦于视觉信息的深度理解。实验还揭示，不同问题类型对系统表现影响有限，而不同领域中图像内容的丰富程度显著影响难度。成本和延迟分析表明，多模态融合系统在保持较低计算代价的同时提升性能。

### 通俗易懂  
简单来说，本文搭建了一个大规模的“考试卷”，用来测试人工智能系统在处理包含文字、图片和表格的复杂文档时的表现。首先，研究团队从大量真实的PDF文件中挑选出涵盖多个领域的文档，并把里面的文字、图片和表格都拆分出来，建立起它们之间的联系。接着，他们设计了多种类型的问题，比如找事实、比较信息、总结内容和推理判断，然后用先进的语言模型自动生成这些问题和答案，并让人工审核确保质量。最后，研究人员设计了一个公平的“评分标准”，让不同的AI系统都能在同样的条件下比拼，看谁能更准确地从文字和图片中找到答案。结果发现，同时利用文字和图片信息的系统表现最好，单独用文字或图片都不够完美。这个工作帮助大家更好地理解如何让AI更聪明地阅读和理解复杂文档，也为未来改进提供了方向。 
# Topic: Multi-modal｜Video 
## VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization 
2025-10-07｜BUPT, MUC｜ICCV 2025 

<u>http://arxiv.org/abs/2510.06040v1</u>  
<u>https://github.com/caoxinye/VideoMiner</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/100.jpg)  
本文针对时长达小时级别的长视频理解，提出了一种新颖的层级树结构模型VideoMiner，旨在有效定位关键帧并保持时间连贯性。长视频中大量冗余信息会干扰多模态大语言模型（MM-LLMs）对关键内容的识别，传统均匀采样方法难以应对信息过载问题。VideoMiner通过迭代分割、生成事件描述并基于文本聚类构建分层树状结构，从粗到细逐步细化视频内容，兼顾空间和时间信息。为实现关键帧的精准定位，本文设计了树结构专用的强化学习策略优化算法T-GRPO，该方法结合事件级空间-时间信息和用户查询引导树的动态扩展，显著提升了长视频理解的准确性和效率。实验结果表明，VideoMiner在多个长视频理解基准上均优于十余种先进方法，且T-GRPO激发了模型自主生成推理链的能力，展现了方法的创新性和实用价值。  

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/101.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/102.jpg)  
VideoMiner的核心流程包括三个阶段：1）视频场景分割与事件描述生成；2）基于事件描述的聚类形成树状层级结构；3）利用T-GRPO强化学习策略在树结构中迭代探索关键帧。具体方法如下：  
- 场景分割：对长视频进行均匀采样，计算相邻帧的相似度，自动确定分割点，将视频划分为多个事件，保持时间信息连续性。  
- 事件描述与聚类：利用视觉语言模型（VLM）结合用户提问，为每个事件生成文本描述，再通过密度聚类算法将语义相近事件合并，形成树状节点，构建多层次视频结构。  
- T-GRPO策略优化：设计针对树结构的组相对策略优化算法，输入节点描述、用户问题及节点深度，输出三类动作（接受、继续扩展、删除），并通过节点级和树级奖励函数引导策略动态调整探索深度，兼顾准确性与效率。该算法采用强化学习训练，激励策略生成连贯推理链，提升关键帧定位的精度。  

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/103.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/104.jpg)  
在四个主流视频理解数据集及九个子任务上，VideoMiner均展现出领先性能，特别是在长视频任务中，其优势随视频时长增长而显著扩大。相比端到端方法，VideoMiner通过事件级聚类更好地保留了时间信息，避免了冗余帧干扰。与其他层级结构方法相比，T-GRPO策略有效控制树的扩展，提升了关键帧选择的准确率和推理能力。消融实验显示，事件聚类优于帧聚类和无聚类方法，运行效率更高且准确率更好；采用T-GRPO相比传统强化学习算法，显著提升了模型推理效果。案例分析进一步证明，T-GRPO激发了模型生成推理链，增强了对复杂问题的理解能力。实验还探讨了补充长度和树增长率对性能的影响，揭示了平衡效率与准确性的关键参数设置。  

### 通俗易懂  
VideoMiner的核心想法是把一个很长的视频拆成很多小段，每段用一句话描述它的内容，然后把这些描述相似的小段聚在一起，形成一个“树”，树的每个节点代表视频的一个重要部分。接着，系统会像玩“找关键帧”的游戏一样，沿着树的分支一步步探索，决定哪些节点是关键，哪些可以忽略。这个决策过程不是随便做的，而是用一种叫做强化学习的智能方法训练出来的，就像给系统一个奖励机制，鼓励它做出更好的选择。比如，如果系统找到的关键帧能帮它更好地回答问题，它就会得到奖励。这样，系统学会了怎样在海量视频信息中快速找到最重要的部分，避免被无关内容干扰，最终给出更准确的视频理解和回答。 
## Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow 
2025-10-07｜PKU, Peng Cheng Lab 

<u>http://arxiv.org/abs/2510.05836v1</u>  
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/105.jpg)  
长视频理解因时间和空间内容冗余极大，一直是多模态大语言模型（MLLMs）面临的挑战。现有方法多依赖语义先验，如CLIP模型，提取关键帧或语义信息，但这依赖用户指令的详尽程度且易受先验模型误差影响。本文提出Flow4Agent，首次引入光流（optical flow）作为运动先验，辅助MLLM进行长视频理解。该框架通过两个核心模块——时间粒度优化（TGO）和运动令牌剪枝（MTP），分别在帧间和帧内层面减少冗余。TGO利用粗光流分割视频事件并结合语义先验筛除无关场景，MTP则利用细粒度光流剔除帧内重复视觉信息。大量实验验证Flow4Agent在多个长视频基准上的领先表现，特别是在视频时长超过30分钟的任务中表现突出，显著提升了长视频的理解和推理能力。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/106.jpg)  
Flow4Agent方法设计围绕两个核心模块：  

1. **时间粒度优化（TGO）**  
   - 利用HSV色彩空间转换和粗光流估计，计算相邻帧的运动变化，粗略划分动态事件边界。  
   - 通过光流幅度阈值进一步细化事件分割，确保事件划分精准且不丢失重要信息。  
   - 结合语义先验（如CLIP或SigLIP模型）对事件进行筛选，保留与用户查询相关或具有代表性的关键事件，过滤低相关性场景。  
   - 采用事件中心帧作为关键帧，避免逐帧处理的冗余，提升效率。  
2. **运动令牌剪枝（MTP）**  
   - 针对同一事件内帧，利用细粒度光流计算像素级运动，识别动态显著区域。  
   - 通过相机运动补偿和显著性检测，剔除背景和静态区域的视觉令牌，只保留动态变化显著的视觉信息。  
   - 保持关键帧全部令牌，邻近帧则应用剪枝，平衡信息完整性与冗余压缩。  
该方法结合运动与语义双重先验，既保证了关键信息的覆盖，又大幅降低了冗余，适应MLLM有限的上下文窗口。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/107.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/108.jpg)  
本文在六大视频理解基准上进行了全面评测，包括VideoMME、LongVideoBench、MLVU、EgoSchema、PerceptionTest和NextQA，涵盖从短视频到超长视频（最长达2小时）。结果显示，Flow4Agent在所有测试集上均优于现有最先进模型，尤其在长视频任务中优势明显，VideoMME长视频上提升3.6%，超过同类方法。多种基础模型和不同上下文窗口长度的适配实验进一步验证了Flow4Agent的通用性和稳定性。消融实验表明，TGO模块中的动态事件划分和语义筛选协同作用显著提升性能，MTP模块则进一步优化帧内冗余，三者结合效果最佳。视觉化案例展示了Flow4Agent如何有效区分场景事件并去除静态冗余背景，突出动态重要区域。实验还分析了不同光流模型和迭代次数对性能的影响，确认了Sea-RAFT光流模型的优越性及合理迭代配置。

### 通俗易懂  
Flow4Agent的核心想法是：长视频里很多画面其实是重复和没变化的，直接给计算机看所有画面既浪费时间又容易让它“糊涂”。所以，我们先用光流技术来“感知”视频中哪些地方动得多、变化大，这样可以把视频分成几个重要的“事件段”，每个事件挑选最能代表它的关键画面。接着，在每个画面里，我们再用光流找到真正动得显著的部分，像人脸表情、动作这些重要信息，去掉那些没什么变化的背景或静止区域。这样一来，计算机只看最关键的内容，既节省了处理资源，也能更准确理解视频讲了什么。简单来说，就是用“动起来的地方”帮我们挑重点，让机器看长视频时不迷路，理解得更透彻。 
## SD-MVSum: Script-Driven Multimodal Video Summarization Method and Datasets 
2025-10-07｜CERTH-ITI 

<u>http://arxiv.org/abs/2510.05652v1</u>  
<u>https://github.com/IDT-ITI/SD-MVSum</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/109.jpg)  
本文提出了SD-MVSum，一种基于用户提供的长篇脚本，融合视频视觉内容与视频语音内容的多模态视频摘要方法。该方法通过加权跨模态注意力机制，动态建模脚本与视频视觉及转录文本之间的语义相关性，从而精准提取与用户需求最相关的视频片段，生成个性化且多样化的视频摘要。为支持该任务，作者扩展了两个大规模视频摘要数据集（S-VideoXum和MrHiSum），新增了文本描述与音频转录数据，使其适配脚本驱动的多模态视频摘要研究。实验结果显示，SD-MVSum在两个数据集上均优于现有最先进的脚本驱动及通用视频摘要方法，验证了引入语音信息和加权跨模态注意力机制的有效性。该工作不仅提升了视频摘要的定制化水平，也为后续多模态视频理解研究提供了丰富数据资源。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/110.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/111.jpg)  
SD-MVSum的核心在于结合视频视觉帧、用户脚本与音频转录文本三模态信息，通过两个关键技术实现高效融合和筛选：  

1. **多模态编码器**：对视频帧、用户脚本句子及音频转录文本分别编码，生成对应的特征嵌入；  
2. **加权跨模态注意力机制**：引入动态加权的语义相似度矩阵，对视频与脚本、脚本与转录文本的依赖关系进行建模，强化高度相关的内容连接，抑制无关片段，提升注意力权重的语义精准度；  
3. **特征融合与评分**：将融合后的跨模态特征通过线性变换和位置编码后输入Transformer评分器，输出每帧的重要性得分；  
4. **摘要生成**：基于得分和预设的时间预算，采用分段选择策略，拼接高相关度视频片段形成最终摘要。  
此外，针对语音转录数量与视频帧数不匹配问题，设计了转录嵌入的时间扩展策略，确保跨模态注意力计算的对齐。该方法通过动态调整注意力权重，显著提高了视频与脚本之间的语义匹配度，进而提升摘要的内容相关性和多样性。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/112.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/113.jpg)  
实验在扩展后的S-VideoXum和S-MrHiSum数据集上进行，采用F-Score及排名相关系数（Kendall’s τ和Spearman’s ρ）评估摘要质量。与现有脚本驱动方法SD-VSum及CLIP-It相比，SD-MVSum在两数据集均取得更优表现，F-Score提升约1%。与通用视频摘要方法（如A2Summ、CSTA）对比，SD-MVSum在满足用户脚本需求的定制化摘要方面表现更佳。消融实验验证了音频转录信息和加权跨模态注意力机制的贡献，去除转录或动态加权均导致性能下降，说明语音内容和动态权重调整对提升摘要效果至关重要。定性分析展示了SD-MVSum更准确地捕捉用户关注的关键视频片段，生成内容更符合用户脚本描述的摘要。整体实验结果充分证明了多模态融合与动态注意力机制在脚本驱动视频摘要任务中的优势。

### 通俗易懂  
简单来说，SD-MVSum就像一个聪明的视频编辑助手。它不仅看视频画面，还“听”视频里说的话，同时还会仔细阅读用户写的详细脚本。它会先把视频的每一帧画面、用户脚本的每句话以及视频里说的内容都转换成数字“特征”，然后用一种特别的“注意力机制”来判断这些内容之间的关联程度。这个机制会根据内容的相似度给不同部分“打分”，越相关的部分得分越高。接下来，这些得分会被送入一个Transformer模型，让它决定哪些视频片段最重要。最后，系统会挑选出这些最相关的片段，按照用户希望的长度拼接成一个简短的视频摘要。这个过程让视频摘要不仅内容丰富，还特别符合用户的具体需求，而不是简单地拼接视频中最热门的部分。 
