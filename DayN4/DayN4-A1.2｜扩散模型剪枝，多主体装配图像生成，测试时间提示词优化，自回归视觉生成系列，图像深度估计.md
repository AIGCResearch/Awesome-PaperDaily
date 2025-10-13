# Topic: Image Generation

## OBS-Diff: Accurate Pruning For Diffusion Models in One-Shot 
2025-10-08｜Westlake, ZJU 

<u>http://arxiv.org/abs/2510.06751v1</u>  
<u>https://github.com/Alrightlone/OBS-Diff</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/119.jpg)  
本文针对大规模文本到图像的扩散模型提出了OBS-Diff，一种创新的单次剪枝框架，旨在解决扩散模型因迭代去噪特性而难以直接应用传统剪枝方法的问题。扩散模型参数庞大，计算资源消耗巨大，限制了其广泛应用。现有剪枝技术多依赖训练或微调，且多针对特定架构，难以适用于结构复杂且多样的现代扩散模型。OBS-Diff通过复兴经典的最优脑外科手术（Optimal Brain Surgeon，OBS）方法，结合扩散模型的时序动态特点，实现了无需训练、一次性完成的高精度压缩。该方法支持多种剪枝粒度，包括非结构化、半结构化（N:M模式）及结构化剪枝（如多头注意力机制中的头和前馈网络中的神经元），有效提升推理效率且保持生成图像的高质量。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/120.jpg)  
OBS-Diff方法包含三大核心创新：  

1. **时序感知Hessian构建**：考虑扩散模型迭代生成过程中的误差累积，设计了一个时间步权重衰减机制，赋予早期去噪步骤更高权重，从而在Hessian矩阵构建中突出关键阶段参数的重要性，减少早期误差放大对最终结果的影响。  
2. **模块包分组顺序剪枝**：为缓解全模型逐层校准的高昂计算代价，将模型层划分为若干“模块包”，在每个包内并行采集激活数据并同时剪枝，包与包之间顺序更新网络状态。此策略在保证剪枝效果的同时，大幅降低了校准时间和内存开销。  
3. **多粒度剪枝扩展**：在非结构化剪枝基础上，OBS-Diff自然扩展至半结构化剪枝（如2:4稀疏模式）和结构化剪枝。结构化剪枝通过聚合神经元或注意力头的权重重要性评分，结合多模态输入的特殊注意力结构，采用排名融合策略统一头的重要性排序，实现高效且稳定的结构化剪枝。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/121.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/122.jpg)  
在多款主流文本到图像扩散模型（包括Stable Diffusion系列及Flux.1-dev）上，OBS-Diff展现出卓越的剪枝性能。相比传统基于权重大小的剪枝和最新的LLM剪枝方法，OBS-Diff在CLIP分数和ImageReward等语义和视觉一致性指标上均取得显著优势，尤其在高稀疏率下表现稳定，避免了基线方法生成图像质量严重下降的问题。半结构化剪枝实验中，OBS-Diff在保持硬件友好模式的同时，显著提升语义一致性。结构化剪枝中，OBS-Diff远超基于L1范数的剪枝方法，能在较高剪枝率下维持近似原模型的性能。效率方面，OBS-Diff单次剪枝过程耗时短，推理速度提升明显，且消耗资源可通过调整模块包大小灵活平衡。消融实验进一步验证了时序加权和模块包策略对性能和效率的关键作用。

### 通俗易懂  
OBS-Diff的核心思想是“聪明地剪掉不重要的部分”，但它不仅仅看哪个参数小，而是结合了扩散模型生成图像的特殊过程。扩散模型生成图像是一步步“去噪”，越早的步骤错误对最终结果影响越大。因此，OBS-Diff会给早期步骤的参数更高的“重要性分”，确保这些关键步骤的参数被尽可能保留。为了避免剪枝时需要反复运行整个模型消耗大量时间，OBS-Diff把模型分成几个大块，一次性收集这些块的运行数据，然后再一起剪枝，这样既节省时间又节约内存。它还能根据需求，灵活地剪掉单个参数、参数组，甚至整个神经元或注意力头，保证模型在变小的同时还能保持生成图片的质量。简单来说，OBS-Diff就像是一个既懂得关键部位重要性，又懂得如何高效操作的大师，帮扩散模型“瘦身”而不伤元气。 
## SIGMA-GEN: Structure and Identity Guided Multi-subject Assembly for Image Generation 
2025-10-07｜MAU, Adobe 

<u>http://arxiv.org/abs/2510.06469v1</u>  
<u>https://oindrilasaha.github.io/SIGMA-Gen/</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/142.jpg)  
本文提出了SIGMA-GEN，一种统一的多主体身份保持图像生成框架，首次实现了在单次推理中同时基于结构和空间约束生成多个身份明确的主体图像。现有文本到图像生成模型缺乏对主体身份和场景布局的精细控制，限制了其实用性。SIGMA-GEN通过引入单视角RGB图像作为身份描述，结合3D空间布局和深度信息，实现了从粗糙的二维/三维边界框到像素级分割和深度的多层次用户控制。为支持该方法，作者构建了SIGMA-SET27K，一个包含27k张图像、超过10万唯一身份的合成数据集，提供了身份、结构和空间信息的丰富监督信号。通过广泛的评估，SIGMA-GEN在身份保持、图像质量和生成速度上均显著优于现有方法，尤其在多主体场景中表现突出。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/143.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/144.jpg)  
SIGMA-GEN的核心方法包括以下几个关键部分：  

1. **数据集构建（SIGMA-SET27K）**：利用大语言模型生成多主体描述，结合现有文本到图像模型生成目标图像，随后通过分割工具获得主体掩码，采用深度估计模型预测深度图，并利用姿态调整技术生成多样化身份图像，最终配合二维和三维边界框完成数据集构建。  
2. **统一多模态条件编码**：输入包括文本提示、多主体身份图像和空间控制图像，均通过预训练的变分自编码器编码，利用统一注意力机制实现多模态信息的交互融合。  
3. **多主体空间控制表示**：将空间控制分为路由控制（确定主体在图像中的位置）和结构控制（描述深度等整体场景结构），通过构建多层次的空间控制图像实现不同粒度的控制，采用双向合成策略解决重叠区域的可见性问题。  
4. **多层次结构控制支持**：框架支持从像素级掩码与深度图到粗糙的二维/三维边界框的多种结构输入，保证灵活适应不同用户需求和场景复杂度。  

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/145.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/146.jpg)  
实验设计涵盖单主体和多主体个性化生成，采用DINO和SigLIP指标评估身份保持，SigLIP文本图像相似度评估文本一致性，深度均方误差衡量结构控制精度，CLIP-IQA和MUSIQ评价图像质量。结果显示，SIGMA-GEN在身份一致性和图像质量上均优于包括OminiControl、UniCombine、Flux Kontext及Insert Anything\*等主流方法，尤其在多主体场景中优势明显。此外，SIGMA-GEN支持单次生成多主体，避免了迭代插入带来的效率低下和质量下降问题。消融实验验证了深度信息和完整文本提示对性能提升的重要性。此外，作者展示了该方法在主体插入和姿态调整等应用中的灵活性和扩展性，进一步体现其实用价值。

### 通俗易懂  
SIGMA-GEN就像是一款非常聪明的画家助手，能根据你的描述和具体要求帮你画出一幅画，画中有多个你指定的人物，每个人的样子都能被准确还原。首先，系统会先准备一个大“素材库”，里面有很多人物的照片、他们在不同场景中的位置和姿势信息。然后，当你告诉它想画什么内容和人物时，它会把这些信息编码成一种“语言”，让计算机理解你要的画面结构和每个人的样子。接着，它会通过一个统一的模型，一次性把所有人物放到正确的位置，保持他们的身份特征，还能根据你给的空间信息调整他们的大小、朝向和遮挡关系。这样，你不需要一遍遍地调整或插入每个人，既节省时间，也保证了画面质量和人物的真实性。无论你是只给出大致的框框，还是精确的轮廓和深度信息，SIGMA-GEN都能灵活应对，帮你生成满意的多主体图像。 
## GenPilot: A Multi-Agent System for Test-Time Prompt Optimization in Image Generation 
2025-10-08｜CAS-IA, UCAS, Baichuan, FMRC, Objecteye, BUPT｜EMNLP 2025 

<u>http://arxiv.org/abs/2510.07217v1</u>  
<u>https://github.com/27yw/GenPilot</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/151.jpg)  
本文提出了GenPilot，一种针对文本生成图像（T2I）任务的多智能体系统，专注于测试时的提示词优化。当前文本到图像生成模型在处理复杂且冗长的提示时常出现语义不一致和细节缺失，且现有的微调方法依赖模型训练，自动提示优化缺乏系统化的错误分析，测试时扩展方法多局限于固定提示和噪声采样，缺乏灵活性。GenPilot通过将提示优化视为搜索问题，直接在输入文本空间进行动态迭代优化，集成错误分析、基于聚类的自适应探索、细粒度验证及记忆模块，提升了生成图像的语义一致性和结构连贯性。该方法无须模型微调，适用于多种T2I模型，实验结果显示在多个基准数据集上，GenPilot显著优于传统提示工程和测试时扩展方法，尤其在复杂长提示下表现卓越，展示了良好的通用性和鲁棒性。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/152.jpg)  
GenPilot的核心设计分为两个阶段：错误分析与测试时提示优化。  

1. **错误分析阶段**：  
   - **提示分解**：将输入提示拆解为包含对象、关系和背景信息的若干子句，覆盖更全面的语义层次。  
   - **多模态问答（VQA）与图像描述比对**：通过生成针对图像的细粒度问答和自动生成的图像描述，与原始提示进行对比，发现语义不一致。  
   - **错误整合与定位**：将VQA与描述比对所得错误整合，利用智能体映射错误到具体提示片段，支持后续针对性修正。  
2. **测试时提示优化阶段**：  
   - **候选提示生成**：基于错误分析结果，提示优化智能体生成多个多样化的候选提示。  
   - **评分与聚类**：利用多模态大语言模型（MLLM）对生成的图像与提示进行评分，并通过K-Means聚类筛选出表现最优的提示组。  
   - **记忆模块**：保存历史优化结果和错误反馈，辅助下一轮优化迭代，形成闭环提升。  
该方法将提示优化转化为在文本输入空间的迭代搜索问题，结合多模态理解和动态反馈机制，实现高效且可解释的提示优化。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/153.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/154.jpg)  
实验在多个公开基准数据集（DPG-bench和GenEval）上进行，涵盖多种主流T2I模型（如DALL-E3、Stable Diffusion系列、FLUX.1schnell及Sana-1.0）。结果显示，GenPilot在整体性能指标上均优于传统提示工程（PE）、测试时扩展（TTS）及其他自动提示优化方法，平均提升幅度达5%至17%。在复杂提示的细节绑定、空间关系、颜色属性等子任务中表现尤为突出，显著减少了生成图像中的语义错误和不必要元素。定性分析进一步证明，GenPilot能有效排除提示中排斥的对象，准确表达复杂语义。消融实验表明，错误整合、聚类筛选及记忆模块均为性能提升的关键组成部分。不同多模态大语言模型和图像描述模块的替换实验也体现了系统的模块化设计和适应性。尽管引入了额外的推理时间，GenPilot通过并行和早停策略在实际应用中保持合理效率。

### 通俗易懂  
GenPilot就像一个聪明的团队，帮助计算机更好地理解和执行复杂的“画画指令”。首先，它会把这条复杂的指令拆开，像拆拼图一样，把每个部分都仔细检查。然后，它会“问问题”来确认画面中每个细节是不是都符合指令，比如“这里有几个红色的球？”“这只猫是在桌子上吗？”同时，它还会把生成的图片和原始指令做对比，找出哪里不匹配。接下来，这个团队会根据发现的问题，想出好多不同的改进版本的指令，然后用一个聪明的评分系统来挑选表现最好的版本。整个过程会反复进行，每次都记住之前的经验，逐步让指令变得更准确，画出来的图像也更符合期待。这样，GenPilot就能帮助各种不同的画图模型，在不需要重新训练的情况下，快速提升画图的准确度和细节表现。 
# Topic: Image Generation｜Diffusion/Autoregress/VQ 
## IAR2: Improving Autoregressive Visual Generation with Semantic-Detail Associated Token Prediction 
2025-10-08｜SJTU 

<u>http://arxiv.org/abs/2510.06928v1</u>  
<u>https://github.com/sjtuplayer/IAR2</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/155.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/156.jpg)  
本文提出了IAR2，一种改进的自回归视觉生成框架，针对视觉数据的内在结构特性进行优化。传统自回归模型使用单一预训练码本，难以兼顾重构精度和生成质量，且硬聚类导致语义分组不准确。IAR2创新性地引入了语义-细节关联双码本，将图像表示拆解为捕获全局语义的语义码本和编码细粒度纹理的细节码本，显著提升了表示能力，从线性扩展到多项式规模。基于此，设计了层次化的自回归预测策略，先预测语义token，再条件化预测细节token，结合局部上下文增强模块，强化空间一致性。此外，针对条件生成，提出了渐进式注意力引导的自适应CFG机制，动态调整不同token的引导强度，提升条件对齐度和生成真实感。大量实验表明，IAR2在ImageNet 256×256上实现FID 1.50，超越现有方法且计算效率更优，验证了结构化粗细粒度生成策略的有效性。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/157.jpg)  
IAR2的核心方法包括以下三个模块：  

1. **语义-细节关联双码本量化**：将图像patch的编码分为两步，先用较小的语义码本捕获全局语义特征，再对残差进行细节码本编码，形成联合表示。该设计提升了码本的表达能力，避免了单一大码本带来的预测难度。  
2. **层次化语义-细节自回归预测**：在自回归模型中，针对每个patch同时预测语义和细节token，先预测语义token，再将其作为条件输入预测细节token，利用token融合机制保持序列长度不变。引入局部上下文增强自回归头，通过压缩邻近token的隐藏状态并融合全局上下文，强化局部空间依赖，提升细节预测的准确性和生成图像的空间连贯性。  
3. **渐进式注意力引导的自适应CFG**：传统CFG对所有token采用统一引导尺度，容易在背景产生伪影。IAR2根据token的空间相关性（通过注意力权重衡量）和生成进度动态调整引导强度，重点强化语义相关区域的条件信号，同时逐步增强引导力度，确保生成过程从粗到细逐步精准对齐条件。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/158.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/159.jpg)  
实验部分系统验证了IAR2的设计优势。首先，通过在ImageNet 256×256数据集上对比不同码本大小，发现单一大码本虽提升重构精度，但生成质量下降，验证了单码本的权衡难题。IAR2采用双码本结构，扩展表达能力同时降低预测复杂度，显著提升生成性能。其次，层次化预测策略和局部上下文增强模块有效提升细节表现和空间一致性，消除了传统模型中细节模糊和局部不协调问题。最后，渐进式注意力引导CFG机制在条件生成任务中表现出更优的条件对齐和图像质量，减少背景伪影。整体来看，IAR2在FID指标上大幅优于当前最先进的自回归模型，且训练和推理效率更高，展示了其在视觉生成领域的强大潜力。

### 通俗易懂  
IAR2的核心创新是把图像的表达拆成“整体大意”和“细节补充”两部分。就像写故事时，先写故事大纲（语义码本），再写具体细节（细节码本）。这样做有两个好处：一是让模型先确定大体内容，避免细节预测时迷失方向；二是细节部分只需在大纲基础上补充，降低预测难度。模型在生成图像时，先预测每个小块的“主题”，再根据主题补充细节，这样生成的图像更连贯、细节更丰富。为了让生成的每个小块更协调，模型还会参考附近小块的信息，保证纹理和边缘自然过渡。最后，针对有条件生成（比如给定类别），IAR2会根据图像不同区域的重要性和生成进度，动态调整条件引导力度，重点强化主体部分，避免背景出现不自然的痕迹。整体来说，IAR2像是在画画时先画草图，再慢慢加细节，且根据画面重点灵活调整画笔力度，既保证了整体结构，也丰富了细节，生成效果更好。 
## Heptapod: Language Modeling on Visual Signals 
2025-10-08｜ByteDance 

<u>http://arxiv.org/abs/2510.06673v1</u>  
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/160.jpg)  
本文提出了Heptapod，一种基于语言建模原则的视觉信号自回归模型，旨在解决视觉生成领域中“下一令牌”定义模糊及对外部语义依赖的问题。传统视觉生成模型往往依赖外部语义信息（如分类器无指导CFG或预训练自监督模型SSL）来弥补视觉数据的复杂性，导致生成质量受限且多样性降低。Heptapod创新性地将下一令牌预测从一维序列扩展到二维空间，采用因果Transformer结合重构导向的视觉分词器，训练模型并行预测图像二维格点上所有位置的令牌分布。这种学习目标融合了自回归序列建模和掩码自编码的自监督思想，使模型能隐式捕获图像的全局语义。实验表明，Heptapod在ImageNet生成任务中无需依赖CFG即可显著超越现有因果自回归模型，展示了视觉语言建模的全新范式。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/161.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/162.jpg)  
Heptapod的核心方法包括以下几个关键点：  

1. **重构导向的视觉分词器**：采用VQ-VAE或VAE对图像进行压缩，专注于忠实重建而非预先注入语义，确保令牌表达的完整性和细节保留。  
2. **下一二维分布预测目标**：区别于传统模型仅预测单一“下一令牌”，Heptapod在每个时间步并行预测二维空间内所有未观察位置的令牌分布，消除视觉数据中“下一令牌”顺序的模糊性，迫使模型学习全局空间结构和语义依赖。  
3. **因果Transformer架构**：保持标准1D因果注意力机制，输入为带有空间位置信息的视觉令牌序列，输出通过专门设计的二维预测头映射到整个二维格点的概率分布。  
4. **二维预测头设计**：包括全局预测头（双向Transformer，捕获全图长程依赖）和局部预测头（跨注意力结合局部双向注意力，提升计算效率），两者均实现单步触发对全二维空间的预测。  
5. **统一的概率建模**：针对离散令牌采用交叉熵损失，连续令牌采用扩散模型的均方误差损失，统一视为条件概率分布建模，兼顾训练效率和生成质量。  
该方法有效解决了视觉数据的局部相关性过强导致模型陷入低级纹理插值的难题，促进了视觉语义的内生式学习。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/163.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/164.jpg)  
实验在ImageNet-1K 256×256图像生成任务上进行，全面评估Heptapod的生成质量和训练效率。  
- **训练设置**：禁用CFG，确保评价模型内在生成能力。采用VQGAN和VAE两种视觉分词器，分别对应离散和连续令牌。训练800轮，使用大规模批次和AdamW优化。  
- **训练动态**：VQ分词器模型初期收敛快、训练稳定，但最终生成质量略逊于VAE分词器，后者在训练中后期表现优异，反映分词器的重构质量对生成性能有决定性影响。  
- **二维预测头设计对比**：全局预测头（覆盖全图）显著优于局部预测头（局部区域），表明扩大预测窗口对捕获全局语义至关重要。增加监督密度（每步预测更多位置）在全局窗口下提升训练效果，局部窗口中效果有限。  
- **性能对比**：Heptapod-H模型以2.70的FID和229.8的IS指标，超越了多种先前因果自回归视觉生成模型，且无须依赖外部语义注入或CFG，验证了方法的有效性与实用性。  
综上，实验结果支持Heptapod通过下一二维分布预测实现视觉语言模型的突破，兼顾生成质量和模型架构的纯粹性。

### 通俗易懂  
传统的视觉生成模型在“预测下一步内容”时，遇到了一个难题：文本是按顺序排列的词语，下一词很明确；但图像是二维的，没有固定的“下一块像素”，所以模型很难学会理解图像的整体内容，只能靠猜邻近地方的样子。为了解决这个问题，Heptapod让模型在每一步不仅预测一个位置，而是预测图像中所有未完成位置的可能样子，相当于让模型同时猜测整个剩余图像的内容。这样，模型必须学会理解图像的整体结构和语义，不能只靠简单的局部复制。模型先用一个“压缩器”把图片变成一堆小块（令牌），然后用一种叫因果Transformer的网络，结合一个特别设计的预测模块，来预测这些小块的分布。这样做既保留了语言模型的核心架构，也让模型能更好地理解和生成复杂的视觉内容。实验表明，这种方法让模型在没有外部帮助的情况下，就能生成高质量的图像，证明了这种“二维同时预测”的新思路非常有效。 
# Topic: Image
## Pixel-Perfect Depth with Semantics-Prompted Diffusion Transformers 
2025-10-08｜HUST, Xiaomi, ZJU｜NeurIPS 2025 

<u>http://arxiv.org/abs/2510.07316v1</u>  
<u>https://pixel-perfect-depth.github.io/</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/147.jpg)  
本文提出了Pixel-Perfect Depth，一种基于像素空间扩散变换器的单目深度估计模型，旨在生成高质量且无飞像素的点云。当前生成式深度估计模型通常基于Stable Diffusion的潜变量空间，通过VAE压缩深度图，导致边缘细节模糊和飞像素问题。为解决这一缺陷，Pixel-Perfect Depth直接在像素空间进行扩散生成，避免了VAE引入的失真。该方法通过引入语义提示扩散变换器（SP-DiT）和级联变换器设计（Cascade DiT）两大创新，有效提升了高分辨率像素空间生成的效率和精度。SP-DiT利用视觉基础模型提取的高层语义信息，保持全局语义一致性并增强细节表现；级联设计则采用由粗到细的Token处理策略，兼顾全局结构和局部细节。实验结果显示，该模型在五个公开基准上均取得最佳性能，尤其在边缘感知点云评估中显著优于现有方法，展示了其在实际三维重建和机器人操作等应用中的潜力。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/148.jpg)  
Pixel-Perfect Depth的核心是直接在像素空间执行扩散生成，避免了传统VAE潜变量压缩导致的飞像素问题。具体方法包括：  

1. **生成式框架**：采用Flow Matching方法构建连续的高斯噪声到深度样本的变换，通过一阶常微分方程学习噪声场的去噪速度，实现高质量深度图生成。  
2. **语义提示扩散变换器（SP-DiT）**：基于纯Transformer架构，将输入图像与噪声拼接后分块编码为Token。引入预训练视觉基础模型（如DINOv2、DepthAnything v2）提取的高层语义特征，并通过归一化和多层感知机融合进Transformer Token中，提升模型对全局语义结构的感知和细节恢复能力。  
3. **级联变换器设计（Cascade DiT）**：将Transformer块分为两阶段，前半部分使用较大Patch尺寸减少Token数量，专注于捕捉全局低频结构；后半部分采用较小Patch尺寸增加Token数量，强化对高频细节的建模。该设计有效降低计算成本同时提升精度。  
4. **训练细节**：模型在多个合成及真实数据集上训练，深度值经过对数变换和归一化处理，优化目标为预测噪声速度场的均方误差。该方法实现了从噪声到深度图的高效稳定转换。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/149.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/150.jpg)  
实验部分涵盖多方面评估，验证了Pixel-Perfect Depth的有效性与优越性。  
首先，在五个主流零样本相对深度估计基准（NYUv2、KITTI、ETH3D、ScanNet、DIODE）上，模型均以较低的绝对相对误差（AbsRel）和较高的准确率（δ）领先所有已发布的生成式深度估计方法。  
其次，通过引入边缘感知点云评估指标，专门衡量物体边缘处的飞像素情况，模型在Hypersim高质量测试集上表现出显著优势，生成的点云边缘更为清晰，飞像素数量大幅减少。  
此外，消融实验表明，SP-DiT模块极大提升了模型对全局语义和细节的把握，带来高达78%的性能提升；级联设计则显著降低推理时间约30%，同时进一步提升精度。不同视觉基础模型的语义提示均有效提升性能，显示该设计的通用性和鲁棒性。  
最后，定性比较中，Pixel-Perfect Depth生成的深度图细节丰富且边缘锐利，明显优于基于VAE潜变量的Marigold和DepthAnything v2等模型，验证了其在复杂场景下的泛化能力和鲁棒性。

### 通俗易懂  
这项研究的核心是让计算机从单张图片中准确判断每个像素的深度，换句话说，就是告诉它每个点离摄像头有多远。传统方法通常先把深度图压缩成一种“简化版”的表示（潜变量），再进行生成，这样做虽然节省计算，但会模糊边缘，导致生成的三维点云出现“飞像素”，看起来不自然。  
本研究没有压缩深度图，而是直接在原始像素级别上进行生成，这样可以更精准地捕捉物体边界和细节。为了解决直接生成时计算量大、训练难的问题，研究人员设计了两大关键技术：  

1. **语义提示扩散变换器（SP-DiT）**：它先用一个强大的预训练模型理解图片的整体语义信息，比如识别出“这是个人”，“这是个桌子”，然后把这些信息作为提示，帮助生成模型更好地理解图片的全局结构和细节。  
2. **级联变换器设计**：这个设计先用粗略的“视角”快速抓住图片的大致结构，再逐步细化到更小的区域，专注于细节的生成。这样既保证了全局一致性，也让细节更丰富，同时节省了计算资源。  
总结来说，这种方法让计算机能更聪明地“看懂”图片，生成的深度图更准确，点云更干净，没有乱飞的像素，适合用在机器人导航、3D重建等实际应用中。 