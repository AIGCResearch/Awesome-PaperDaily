# Topic: Image Generation

## DreamOmni2: Multimodal Instruction-based Editing and Generation 
2025-10-08｜CUHK, HKUST, HKU, ByteDance｜⭐️⭐️ 

<u>http://arxiv.org/abs/2510.06679v1</u>  
<u>https://github.com/dvlab-research/DreamOmni2</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/123.jpg)  
本文提出了DreamOmni2，一个面向多模态指令驱动的图像编辑与生成的新框架，突破了传统仅依赖文本指令或单一具体对象输入的限制。该工作聚焦于两个新任务：多模态指令基础的编辑和生成，支持同时利用文本和多张参考图像，且涵盖具体对象与抽象属性（如纹理、姿态、风格等）的处理。DreamOmni2解决了训练数据匮乏和模型架构无法处理多图像输入的难题，通过创新的数据合成流程和多图像编码机制，实现了更智能、灵活且符合实际需求的图像创作工具。该系统不仅提升了用户交互体验，也推动了统一生成与编辑模型的研究进展，具备广泛应用潜力。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/124.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/125.jpg)  
DreamOmni2的核心方法包括三个关键部分：  

1. **三阶段数据合成流程**：  
   - 阶段一采用特征混合机制，在模型的注意力层交换不同批次的特征，生成包含相同抽象属性或具体对象的高质量图像对，提升数据质量和分辨率，避免内容混叠。  
   - 阶段二利用阶段一训练的提取模型，从目标图像提取指定对象或属性，生成参考图像，并结合指令编辑模型修改目标图像，形成包含源图、目标图、参考图及指令的多模态编辑训练样本。  
   - 阶段三基于阶段二的源图提取关键词，生成多参考图，构建多模态生成训练数据。  
2. **多图像输入处理框架**：  
   - 引入索引编码与位置编码偏移策略，确保模型准确区分多张输入图像，避免像素混淆和复制粘贴效应。  
3. **联合训练策略**：  
   - 结合视觉语言模型（VLM）与生成/编辑模型，通过VLM预训练增强对复杂、非结构化指令的理解能力，提升模型在真实场景中的表现。  
该方法体系兼顾数据质量和模型结构创新，显著提升多模态编辑与生成的准确性和多样性。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/126.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/127.jpg)  
在多模态指令基础的图像编辑与生成任务中，DreamOmni2通过与多款开源及闭源模型（如GPT-4o、NanoBanana、DreamO等）对比，展现了显著的性能优势。定量评测指标包括Google的Gemini和字节跳动的Doubao，以及专业工程师的人工评估，结果显示DreamOmni2在具体对象和抽象属性的编辑与生成准确率均领先其他模型。视觉对比实验也表明其编辑结果更精准、一致性更好，生成图像更符合复杂指令要求。联合训练策略和多图像编码方案的引入，进一步提升了模型对复杂多图输入的处理能力和指令理解深度。所构建的DreamOmni2基准测试集涵盖多样真实场景，验证了模型的泛化能力和实用价值，推动了多模态图像创作技术的发展。

### 通俗易懂  
DreamOmni2的核心是让电脑能够更聪明地理解我们对图片的“多种要求”，不仅听懂我们说的话（文字指令），还能看懂我们给的参考图片。它通过三个步骤训练：第一步，电脑学会从两张图片中找到相同的东西，比如颜色或风格；第二步，电脑学会根据这些提示，修改图片，让它变得符合我们的要求；第三步，电脑再用这些学到的知识，创造出全新的图片。为了让电脑能同时处理多张参考图，研究者设计了一种“标签+位置偏移”的方法，帮助电脑区分不同图片，避免弄混。更厉害的是，DreamOmni2还让一个专门理解语言和图片的大脑（视觉语言模型）和生成图片的模型一起学习，这样它能更好地理解我们复杂的指令。简单来说，DreamOmni2就像一个既能听懂话又能看懂图，还能帮你画图和改图的智能助手，帮你轻松实现各种创意。 

## StyleKeeper: Prevent Content Leakage using Negative Visual Query Guidance 
2025-10-08｜Yonsei, NAVER｜ICCV 2025｜⭐️ 

<u>http://arxiv.org/abs/2510.06827v1</u>  
<u>https://github.com/naver-ai/StyleKeeper</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/114.jpg)  
本文针对文本到图像生成中的视觉风格提示问题，提出了一种名为StyleKeeper的方法。传统方法在利用参考图像作为风格提示时，常出现内容泄露现象，即参考图像中不希望转移的具体内容（如姿态、布局等）被错误地带入生成图像，影响文本内容的准确表达和图像多样性。StyleKeeper通过扩展无分类器引导（CFG）并引入负视觉查询引导（NVQG），有效分离风格与内容，实现了更精准的风格迁移和内容保持。此外，本文还提出了针对真实图像作为风格提示时的随机编码和色彩校准策略，进一步提升了风格反映的准确性和图像质量。大量实验验证了该方法在多样风格和复杂文本提示下的优越性能，显著优于现有的训练免费风格迁移技术。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/115.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/116.jpg)  
StyleKeeper方法包含四个核心组成部分：  

1. **交换自注意力机制（Swapping Self-Attention）**：在文本到图像的扩散模型中，将参考图像的关键（Key）和值（Value）特征替换到生成过程的自注意力层中，借此引入风格信息，同时保留查询（Query）特征以保持内容。该机制只在扩散模型的后半部分（上采样块）应用，避免内容泄露与结构混乱。  
2. **无分类器引导（CFG）结合交换自注意力**：将CFG扩展到交换自注意力操作，使生成过程既能保持文本内容，又能准确反映视觉风格。  
3. **负视觉查询引导（NVQG）**：通过对查询特征施加负向引导，抑制参考图像中不希望转移的内容信息，进一步减少内容泄露。NVQG通过模拟查询交换的反向过程实现，对内容和风格的分离更为清晰。  
4. **随机编码与色彩校准**：针对真实图像风格提示，采用随机噪声编码替代传统DDIM反演，减少累积误差和伪影；通过调整生成图像与参考图像的颜色统计，确保色彩风格的精确匹配。  

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/117.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/118.jpg)  
实验部分系统评估了StyleKeeper在文本对齐、风格相似度、多样性及内容泄露控制上的表现。首先，通过消融研究验证了交换自注意力、NVQG、随机编码和色彩校准各自的贡献，均显著提升了风格反映和文本一致性。其次，层级分析确定了仅在上采样块应用交换自注意力的最佳策略，避免了早期层的内容泄露和图像混乱。与多种先进方法（如StyleAligned、IP-Adapter、Dreambooth-LoRA等）定性和定量对比，StyleKeeper在保持文本内容准确的同时，实现了更自然且多样的风格迁移。最后，方法在图像到图像风格迁移任务中也表现出色，尤其在真实图像作为风格参考时，通过随机编码和色彩校准显著提升了生成质量和风格一致性。

### 通俗易懂  
StyleKeeper的核心思想是让生成的图片既能准确表现文本描述的内容，又能很好地借鉴参考图片的风格，但不会“偷”参考图里的具体内容（比如姿势或布局）。它通过“交换自注意力”来实现：想象生成过程中的大脑有三个部分——查询（想表达什么）、关键（风格信息）和值（风格细节）。我们用参考图片的关键和值替换生成图的大脑对应部分，但保留查询部分，这样生成图能借鉴风格却不会复制内容。为了防止参考图的内容偷偷跑进来，StyleKeeper还加入了“负向引导”，就像给生成过程一个“不许偷内容”的提醒。最后，为了让真实照片作为风格参考时效果更好，方法加入了随机编码和色彩调整，让生成图的颜色和风格更贴近参考图。整体上，这套方法让AI画图既能听懂文字，又能学会“画风”，而不会把参考图的具体内容照搬过来。 
## Lumina-DiMOO: An Omni Diffusion Large Language Model for Multi-Modal Generation and Understanding 
2025-10-07｜Shanghai AI Lab, Shanghai Inno, SJTU, USYD, NJU, CUHK, THU｜⭐️⭐️ 

<u>http://arxiv.org/abs/2510.06308v1</u>  
<u>https://github.com/Alpha-VLLM/Lumina-DiMOO</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/136.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/137.jpg)  
Lumina-DiMOO是一款开源的基础多模态大模型，创新性地采用纯离散扩散建模统一处理文本与图像等多种模态的输入输出。相比传统自回归（AR）或混合AR-扩散模型，Lumina-DiMOO在采样效率上实现了显著提升，支持文本生成、文本到图像、高分辨率图像生成、图像编辑、风格迁移、主题驱动生成、多视图生成及图像理解等多样任务。该模型基于预训练的离散扩散语言模型，整合了视觉和语言的统一词汇表，并引入了新颖的特殊标记以区分图像边界和辅助信息。Lumina-DiMOO在多个公开多模态基准测试中表现优异，领先于现有开源统一多模态模型，且首次在UniGenBench排行榜中位列开源模型第一，展现了强大的多模态生成与理解能力。其设计不仅提升了速度和质量，还支持零样本图像修补和交互式局部图像润色，极大拓展了应用灵活性。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/138.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/139.jpg)  
Lumina-DiMOO的核心方法基于统一的离散扩散框架，关键设计包括：  

1. **统一离散扩散建模**：将文本和图像统一编码为离散token序列，训练时随机遮蔽部分token，模型并行预测被遮蔽的原始token，优化遮蔽交叉熵损失，实现多模态统一生成与理解。  
2. **多模态词汇扩展**：基于预训练的LLaDA文本模型词汇，整合了8192个视觉token及特殊标记（如\<IMAGE\>、\</IMAGE\>、\<end-of-line\>等），支持图像边界及结构信息的显式表达，保障图像任意分辨率的处理能力。  
3. **高效采样策略**：采用基于MaskGIT的分阶段并行采样，图像生成时对被遮蔽token并行预测和采样，动态调整遮蔽比例，结合分类器无条件引导提升生成质量；图像理解采用块状半自回归策略，结合早停机制避免冗余计算。  
4. **Max Logit缓存加速**：利用token预测概率的最大logit值稳定性，缓存高置信度token的表示，跳过重复计算，实现训练无额外代价的推理速度翻倍提升。  
5. **训练流程设计**：分四阶段进行，从多模态预训练、丰富任务中训练、监督微调到基于轨迹一致性的自我强化学习（Self-GRPO），通过联合优化文本到图像生成和多模态理解，提升模型的综合能力和指令对齐度。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/140.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/141.jpg)  
Lumina-DiMOO在多项公开多模态生成与理解基准上进行了广泛评测。文本到图像生成方面，模型在GenEval、DPG、UniGenBench、OneIG-EN和TIIF五大数据集上表现优异，全面超越主流开源统一模型，尤其在复杂场景理解和细节捕捉上展现出强大能力。其速度较代表性自回归模型Lumina-mGPT 2.0提升约32倍，结合Max Logit缓存加速后进一步加快采样效率。图像编辑、风格迁移、主题驱动生成等图像到图像任务也获得了显著提升，支持高分辨率和多样化的图像操作。多模态理解任务中，模型能够准确回答涉及图像内容的复杂问题，表现出色。自我强化学习阶段（Self-GRPO）通过联合优化生成与理解，进一步提升了模型的语义一致性和用户指令响应能力。整体实验结果验证了Lumina-DiMOO在多模态领域的领先地位和广泛适用性。

### 通俗易懂  
Lumina-DiMOO可以理解为一个“多才多艺”的人工智能，它不仅能看懂文字，还能看懂和生成图片。它的秘诀在于用一种叫“离散扩散”的方法，把文字和图片都变成一串串“代码”，然后随机把其中一些“代码”遮住，让模型猜出来，反复训练让它越来越聪明。这样，模型就能同时处理文字和图片，生成高质量的图像，也能回答关于图片的问题。它还设计了特殊的“标签”来告诉模型哪里是图片的边界，甚至能处理各种大小和形状的图片。为了让生成速度更快，模型会记住哪些部分已经猜得很准确了，下次就不用重新算，节省时间。训练时，模型先学会文字和图片的关联，然后学习各种图像编辑和生成任务，最后通过“自我强化学习”不断改进自己，让生成的图片和理解的内容更加贴合用户需求。简单来说，Lumina-DiMOO就像一个既会看图又会说话的全能助手，既快又准，能帮你完成各种复杂的图文任务。 
## Ming-UniVision: Joint Image Understanding and Generation with a Unified Continuous Tokenizer 
2025-10-08｜InclusionAI, Ant Group 

<u>http://arxiv.org/abs/2510.06590v1</u>  
<u>https://github.com/inclusionAI/Ming-UniVision</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/128.jpg)  
Ming-UniVision旨在统一视觉理解与生成任务，突破传统视觉tokenization中离散潜变量带来的量化误差限制。该方法提出了MingTok，一种基于连续潜在空间的视觉tokenizer，兼顾生成任务对紧凑低维编码的需求与理解任务对高维语义特征的需求。通过三阶段架构（低级编码、语义扩展、像素重建），实现视觉内容的高效压缩与丰富表达。Ming-UniVision基于MingTok构建统一的自回归多模态模型，将视觉理解和生成统一为共享连续空间中的下一个token预测，支持多轮上下文交互，如迭代理解、生成与编辑。该框架显著简化架构复杂度，提升训练收敛速度和推理效率，达成视觉语言任务的先进性能，推动了连续域视觉tokenization的统一研究。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/129.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/130.jpg)  
MingTok视觉tokenizer采用三阶段顺序架构：  

1. **低级编码器**：将输入图像压缩为紧凑的连续潜变量，减少token数量，优化自回归生成效率，采用全注意力机制捕获空间依赖。  
2. **语义解码器**：自回归地将低维潜变量扩展为高维语义特征，支持视觉语言推理，采用因果注意力确保生成顺序。  
3. **像素解码器**：从语义特征重建原始图像，利用像素重排提升细节还原能力，采用全注意力捕获长距离依赖。  
训练采用多任务掩码图像建模，分别对潜变量和语义特征执行掩码预测，利用预训练视觉模型监督，保证潜空间结构紧凑且语义丰富；同时像素解码器在掩码与非掩码条件下训练，增强重建质量和鲁棒性。Ming-UniVision基于MingTok，将视觉语义特征作为统一输入，结合大规模语言模型，通过统一的自回归next-token预测机制，实现视觉与语言的无缝融合，支持理解、生成及多轮编辑任务。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/131.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/132.jpg)  
在多模态理解任务中，Ming-UniVision在多项公开基准上表现出与专用视觉语言模型相当的性能，尤其在语义推理和幻觉检测任务中表现优异，但在细粒度字符识别任务上略显不足，反映潜空间压缩和因果结构的限制。在视觉生成任务中，模型在GenEval和DPG-Bench中取得领先，尤其在属性控制和空间推理子任务表现突出，显示统一语义空间对图像合成的指导作用。图像重建实验表明，MingTok在高压缩率下实现了良好的结构和像素一致性，联合训练后性能进一步提升。图像编辑评测显示模型支持多轮上下文编辑，尽管整体分数稍逊于部分专用编辑模型，表明未来需加强多模态大规模序列预训练和细节保留能力。消融研究证实统一连续视觉表示显著提升理解与生成性能，且训练收敛更快，推理更高效。

### 通俗易懂  
MingTok就像一个聪明的“图像压缩大师”，它把一张图片先压缩成少量的“连续小块”，这些小块既包含细节信息，也带有丰富的语义含义。然后，这些小块被逐步“放大”成对图片内容的深刻理解，最后再被用来“还原”成高质量的图片。整个过程像是先用简洁的语言描述图片，再用详细的故事丰富这个描述，最后根据故事重新绘制图片。Ming-UniVision利用这个过程，把图像理解和生成统一在一个框架里，让电脑能像人一样边看边画，边画边改。这样，无论是识别图片里的内容，还是根据文字生成新的图片，甚至是多次修改图片，都能在同一个“语言”里轻松完成，既省时间又更准确。这种连续的表示方式避免了传统方法中“硬切割”带来的信息损失，让模型更聪明、更灵活。 
## VUGEN: Visual Understanding priors for GENeration 
2025-10-08｜Meta 

<u>http://arxiv.org/abs/2510.06529v1</u>  
### 概述 
  
本文提出了VUGEN，一种创新的视觉语言模型（VLM）图像生成框架，旨在充分利用VLM预训练的视觉理解先验，实现高效且高质量的图像生成。现有方法多依赖重构型自编码器或复杂的桥接机制，导致理解与生成表征不匹配或架构复杂。VUGEN通过将VLM视觉编码器的高维潜在空间降维至一个更低维且易处理的分布，最大限度地保留视觉信息，使模型能够在此简化空间中采样生成，确保生成与理解能力的紧密对齐。随后，专门的像素解码器将生成的潜变量映射回图像空间。实验表明，VUGEN在COCO数据集上显著提升图像质量和语义对齐指标，同时保持VLM的理解能力，突破了以往生成与理解任务之间的鸿沟，展示了统一视觉理解与生成的强大潜力。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/133.jpg)  
VUGEN的核心方法包括三个关键步骤：  

1. **理解嵌入空间降维**：VLM的视觉编码器输出高维理解嵌入，信息丰富但难以直接建模。VUGEN设计了一个可训练的降维模块，将高维嵌入映射至低维潜在空间，既保留了生成所需的视觉信息，又简化了生成模型的学习难度。该降维模块与像素解码器联合训练，确保降维后的潜空间最适合图像重建。  
2. **生成模型训练**：在降维潜空间上，采用基于流匹配的扩散模型训练生成器，使模型能够从文本条件下采样潜变量。通过冻结理解编码器和降维器，仅训练生成塔，保证生成过程与预训练视觉理解保持一致。  
3. **像素解码器设计**：将生成的低维潜变量映射为图像。VUGEN探索了两种解码器：一种是基于预训练文本到图像的潜空间扩散模型的微调，另一种是轻量级的无VAE的像素空间扩散解码器，后者简化架构且性能不输于复杂模型。整体设计摆脱了传统依赖VAE潜空间的限制，实现理解与生成的无缝融合。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/134.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/135.jpg)  
VUGEN在两个不同规模的数据集（StockMix和ImageNet）上进行了广泛评估。与基线方法相比，VUGEN在FID、CLIP Score、DPG-Bench和GenEval等多项指标上均表现出显著优势，提升了图像质量和语义对齐能力。定量结果显示，VUGEN在COCO数据集的FID值从11.86降至9.06，DPG分数提升至74.32，且在训练速度上也更快，表现出更高的训练效率。定性分析展示了VUGEN在细节表现和语义准确性上的显著提升，生成的图像更加丰富多样且符合文本提示。消融实验验证了联合训练降维模块优于PCA降维，且像素空间扩散解码器在保持性能的同时极大降低了模型复杂度和计算成本。与当前最先进的统一视觉语言模型相比，VUGEN在图像生成质量方面处于领先地位，证明了其方法的有效性和前瞻性。

### 通俗易懂  
VUGEN的核心想法是让模型用它自己“理解”图像的方式来“画”图，而不是用传统的复杂工具。首先，模型的视觉编码器会把一张图片转换成一个很大很复杂的数字“地图”，这个地图包含了图片的各种信息，但太大太复杂了，不好直接用来生成图片。于是，VUGEN设计了一个“压缩器”，把这个大地图缩小成一个更简单的版本，同时保留重要信息。接下来，模型学会在这个简化的地图上“画画”，也就是生成新的数字地图。最后，另一个专门的“画家”模块把这些数字地图变回真实的图片。这样做的好处是，生成的图片和模型理解图片的方式完全一致，避免了之前方法中理解和生成用不同语言沟通的问题，同时让整个过程更简单、更高效。简单来说，VUGEN让模型用自己理解世界的“眼睛”直接创作图像，效果更好也更自然。 
