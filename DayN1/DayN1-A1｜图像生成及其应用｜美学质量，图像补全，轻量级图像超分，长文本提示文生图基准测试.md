# Topic: Image Generation
## PEO: Training-Free Aesthetic Quality Enhancement in Pre-Trained Text-to-Image Diffusion Models with Prompt Embedding Optimization 
2025-10-02｜KU Leuven 

<u>http://arxiv.org/abs/2510.02599v1</u>  
<u>https://github.com/marghovo/PEO</u> 
### 概述 
![](https://img.blenet.top/file/paper-daily/aigc_research/2dbf99f35f1c4a1cb4bcd94276d2d1ed.png) 
![](https://img.blenet.top/file/paper-daily/aigc_research/cb9c4051b03e42c1b356007b3316f93a.png)  
本文提出了一种名为Prompt Embedding Optimization（PEO）的新颖方法，旨在提升预训练文本到图像扩散模型（Text-to-Image Diffusion Models）在输入简单提示词时生成图像的美学质量。PEO通过优化给定简单提示词的文本嵌入，增强生成图像的视觉表现力，同时保持与优化后文本嵌入的高度一致性，并限制与原始提示词的偏差。该方法无需额外训练，且对所用的扩散模型骨干网络无依赖性，具有广泛适用性。通过定量指标和用户研究，PEO在提升图像美学质量的同时，保证了文本与图像的一致性，表现优于或匹配当前最先进的文本到图像生成及提示词适配技术，展示了其在艺术创作、设计及内容生成等领域的应用潜力。

### 方法 
![](https://img.blenet.top/file/paper-daily/aigc_research/f3af575299bb4e1889f8bb21f6368891.png)  
PEO的核心是一个三部分组成的目标函数，分别针对图像美学质量、文本与图像特征的对应性以及文本嵌入的保持性进行优化：  

1. **美学提升项**：利用LAION Aesthetic Predictor V2对生成图像进行评分，提升图像的视觉吸引力。  
2. **文本-图像一致性项**：通过CLIP模型计算生成图像特征与优化后文本嵌入之间的余弦相似度，确保图像内容与优化文本保持一致。  
3. **提示词保持项（Prompt Preservation Term, PPT）**：限制优化后的文本嵌入与初始文本嵌入的偏差，防止生成内容偏离原始提示词意图。  
优化过程中，PEO固定无条件文本嵌入，仅更新条件文本嵌入，结合Adam优化器在文本嵌入空间迭代更新，逐步提升生成图像的美学质量。该方法不依赖于模型的再训练，且适用于不同的扩散模型骨干，如SD-v1-5和SDXLTurbo。

### 实验 
![](https://img.blenet.top/file/paper-daily/aigc_research/e440876c81ba42a2ac6fce374db2a7d2.png) 
![](https://img.blenet.top/file/paper-daily/aigc_research/6403f1e3fc1d4e1ea7dc814797f09759.png)  
实验选用DiffusionDB、COCO及自制PEO数据集，涵盖多种简单提示词，采用SD-v1-5和SDXLTurbo两种扩散模型作为骨干。通过LAION-AesPredv2、HPSv2和CLIPScore三项指标评估图像美学质量及文本-图像一致性。结果显示，PEO在提升美学评分的同时，保持甚至略微提升了文本与图像的对应关系，优于基线方法Promptist及原始扩散模型。用户调研中，PEO生成图像在整体美学和文本匹配度上显著优于对比方法，获得超过11%的偏好率提升。消融实验验证了三项目标函数成分的协同作用。尽管存在优化发散和高分图像难以进一步提升的失败案例，PEO依然表现出稳定的性能和广泛的适用性。

### 通俗易懂  
PEO方法的核心思想是：“让计算机更聪明地理解你的简单描述，画出更美的图像。”它通过调整输入的文字描述在计算机内部的“数字表达”，让生成的图片不仅更漂亮，还更符合文字的意思。具体来说，PEO会反复尝试微调这个数字表达，目标是三个方面同时做到最好：一是让图片看起来更有美感；二是确保图片内容和文字描述相符；三是避免偏离你原本想表达的内容。这个过程就像在画家脑海中不断修改草图，直到画作既美观又精准。不同于需要重新训练模型或设计复杂提示词，PEO只需简单操作，且适用于多种已有的图像生成模型，极大地简化了提升图像质量的流程。用户实验也证明，这种方法生成的图像更受人喜欢，说明它真的能让AI画出更好看的作品。 
# Topic: Image Generation
## GeoComplete: Geometry-Aware Diffusion for Reference-Driven Image Completion 
2025-10-03｜NUS｜NeurIPS 2025 

<u>http://arxiv.org/abs/2510.03110v1</u>  
<u>https://bb12346.github.io/GeoComplete/</u> 
### 概述 
![](https://img.blenet.top/file/paper-daily/aigc_research/a752a1f82a1244519df5d070096306c2.png)  
本文提出了GeoComplete，一种融合显式三维几何信息的参考驱动图像修复框架，专门针对目标视角与参考视角差异较大的场景。传统基于几何的修复方法依赖多阶段估计（如相机位姿、深度重建等），易受误差累积影响而失败；而现有生成式方法虽能合成缺失区域，但缺乏几何约束，常导致内容错位或不合理。GeoComplete通过引入两大核心创新：一是将投影点云作为几何条件输入扩散模型，显式注入三维结构信息；二是采用目标感知遮罩策略，聚焦参考图像中对目标视角有补充价值的区域，从而提高修复的准确性和一致性。该框架采用双分支扩散架构，一支处理目标图像的缺失区域，另一支处理几何信息，两者通过联合自注意力机制实现信息融合。实验结果显示，GeoComplete在多个基准数据集上较现有方法提升了17.1%的PSNR，显著增强了几何准确性和视觉质量。

### 方法 
![](https://img.blenet.top/file/paper-daily/aigc_research/0944e4a35a524584aea5f8ddb6d3ac3f.png) 
![](https://img.blenet.top/file/paper-daily/aigc_research/c485b9bfd94848eaa0c07b49de273ab4.png)  
GeoComplete方法包含三大关键模块：  

1. **点云生成**：利用Visual Geometry Grounded Transformer（VGGT）联合估计相机参数和深度图，结合Language Segment Anything（LangSAM）通过文本提示分割并滤除动态物体，确保点云构建基于静态场景，提高几何估计的鲁棒性。点云随后被投影到目标和参考视角，作为几何引导。  
2. **目标感知遮罩**：通过将目标视角投影至参考视角，识别参考图像中目标视角缺失的“信息丰富区域”，在训练时对这些区域进行选择性遮罩。该策略避免模型过度依赖冗余信息，促使其学习利用对目标视角补充性强的内容。  
3. **双分支扩散模型**：包含目标分支和点云分支，分别编码目标图像的缺失区域和投影点云的几何信息。两分支的特征在联合自注意力机制中融合，允许掩码区域的视觉特征与对应的几何特征直接交互，有效提升修复的结构一致性。训练过程中通过条件扩散损失优化，推理时以目标图像和点云为条件生成缺失内容。

### 实验 
![](https://img.blenet.top/file/paper-daily/aigc_research/4b67ec8aacba44b88ba751564656b372.png) 
![](https://img.blenet.top/file/paper-daily/aigc_research/1df1fe7e573f476abeeaf8f12fff3686.png)  
GeoComplete在RealBench和QualBench两个具有挑战性的参考驱动图像修复数据集上进行了全面评测。与文本提示驱动的生成方法（如SDInpaint、GenerativeFill）及现有基于参考图像的先进方法（Paint-by-Example、TransFill、RealFill）相比，GeoComplete在PSNR、SSIM、LPIPS等低层次指标和DINO、CLIP等高层语义一致性指标上均取得显著提升，PSNR提升超过5dB。定性结果显示其在大视角差异、动态场景等复杂条件下依然能保持几何一致性，避免错位和结构幻觉。此外，消融实验验证了双分支扩散、联合自注意力和目标感知遮罩对性能的贡献，且通过条件点云遮罩和自注意力机制增强了模型对上游几何估计误差的鲁棒性。用户研究进一步确认了GeoComplete生成结果的自然性和一致性优势。

### 通俗易懂  
GeoComplete的核心思想是让图像修复不仅“看图”，还“懂空间结构”。它先用一种智能工具（VGGT）来估计每张参考图和目标图的拍摄角度和景深，同时用另一种工具（LangSAM）把动态物体“擦掉”，只留下稳定的背景。这样就能把这些图像信息变成一个三维点云，像搭积木一样重建场景的空间结构。然后，GeoComplete用一个双分支的“画家”来修复缺失部分：一个分支专注于目标图像的内容，另一个分支专注于三维点云提供的空间信息。它们通过一种特殊的“交流”机制相互合作，确保修复的内容不仅看起来合理，还和场景的真实结构吻合。为了不让模型被无关的信息干扰，GeoComplete还会智能地遮住参考图中对目标图没用的部分，专注于那些能补充目标视角缺失信息的区域。这样一来，修复出来的图像既清晰又符合真实世界的空间关系。 
## PocketSR: The Super-Resolution Expert in Your Pocket Mobiles 
2025-10-03｜THU, JoyFuture, HKUST(GZ) 

<u>http://arxiv.org/abs/2510.03012v1</u>  
### 概述 
![](https://img.blenet.top/file/paper-daily/aigc_research/dd2363e83555412c83e894232c7b5c1d.png)  
本文针对现实世界中手机拍摄等场景下的图像超分辨率（RealSR）问题，提出了PocketSR，一种极致轻量且高效的单步扩散模型。传统基于大规模生成模型的方法虽然效果显著，但因计算资源消耗大、推理速度慢，难以在边缘设备上应用。PocketSR通过设计轻量化的编码器-解码器架构（LiteED）及创新的在线退火剪枝策略，大幅压缩模型参数（仅146M，约为现有主流模型的十分之一），同时保持高质量的超分辨率效果。该方法不仅实现了4K图像0.8秒内的快速处理，还在多个真实世界数据集上达到了与多步扩散模型相媲美的性能，显著推动了扩散模型在移动端和边缘设备的实用化。

### 方法 
![](https://img.blenet.top/file/paper-daily/aigc_research/76a966e725214e9caab17cb3502498fd.png) 
![](https://img.blenet.top/file/paper-daily/aigc_research/273ee2593a674fe8b1811644915f6c6a.png)  
PocketSR的核心方法包括三大创新点：  

1. **LiteEncoder&Decoder（LiteED）**：替代原始Stable Diffusion（SD）中计算密集的变分自编码器（VAE），采用极简卷积层设计编码器，配合轻量解码器，参数减少97.5%。为缓解信息压缩带来的性能下降，引入自适应跳跃连接和双路径特征注入机制，增强特征表达能力。  
2. **在线退火剪枝（Online Annealing Pruning）**：针对扩散模型中U-Net结构的多个模块（残差块、自注意力、前馈网络等），设计渐进式剪枝策略。通过并行替换轻量模块并逐步降低原模块权重，实现平滑知识迁移，避免性能骤降。  
3. **多层特征蒸馏**：在剪枝过程中，利用多层、多尺度特征蒸馏稳定训练，强化轻量模型对原模型生成先验的保留，确保剪枝后模型依然拥有丰富细节恢复能力。训练分两阶段，先训练LiteED，后冻结其参数并对U-Net进行剪枝与蒸馏。  

### 实验 
![](https://img.blenet.top/file/paper-daily/aigc_research/9750c9863b204a0fab222893f9585785.png) 
![](https://img.blenet.top/file/paper-daily/aigc_research/b1f5bf9d5c2c42f1bef3b2fbb04ae9eb.png)  
实验在多个真实世界超分辨率数据集（RealSR、DRealSR）上展开，全面比较了PocketSR与多步及单步主流方法。结果显示，PocketSR以146M参数和225G MACs，推理速度远超其他方法（512×512图像单步推理仅0.016秒，4K图像0.8秒），实现了超过6倍的计算效率提升。性能指标方面，PocketSR在LPIPS、DISTS及NIQE等感知质量指标上均优于现有顶尖方法，且在PSNR和SSIM等保真度指标上保持竞争力。消融实验进一步验证了LiteED设计、在线退火剪枝及多层蒸馏的有效性，展示了各组件对模型性能和效率的贡献。定性结果表明，PocketSR在细节恢复和纹理还原上表现出色，适合实时边缘设备应用。

### 通俗易懂  
PocketSR的设计核心是让超分辨率模型既小又快，同时保证图片变得更清晰。它首先用一个“轻量级编码器-解码器”替代传统复杂的图像理解模块，这个轻量级模块用更少的计算就能提取和还原图像细节。接着，针对模型中负责生成高清图像的“U-Net”部分，采用了一种“在线退火剪枝”策略。简单来说，就是在训练时慢慢把复杂的部分换成更简单的版本，一步步教新模块学会旧模块的知识，避免突然换掉导致画质变差。最后，通过“多层特征蒸馏”，模型在多个层次上学习原始大模型的表现，确保新模型能保留丰富的细节信息。这样一来，PocketSR既能快速处理高分辨率图片，又能生成细腻自然的高清图像，非常适合手机和其他算力有限的设备使用。 
# Topic: Image｜Multi-modal
## TIT-Score: Evaluating Long-Prompt Based Text-to-Image Alignment via Text-to-Image-to-Text Consistency 
2025-10-03｜SJTU 

<u>http://arxiv.org/abs/2510.02987v1</u>  
### 概述 
![](https://img.blenet.top/file/paper-daily/aigc_research/201677e9db4b48d18ec09e2589b473c2.png)  
随着大型多模态模型（LMMs）的迅速发展，当前文本生成图像（T2I）模型在短文本提示下表现优异，但在处理长且复杂的提示时仍存在理解不足、生成不一致的问题。为此，本文提出了LPG-Bench，这是首个专门针对长文本提示的综合评测基准。LPG-Bench包含200条平均超过250字的详尽提示，涵盖多样主题，接近多款商业模型的输入上限。基于这些提示，作者使用13个先进的T2I模型生成了2600张图像，并进行了大规模的人类排序注释。研究发现，现有主流的T2I对齐评测指标与人类偏好在长提示场景下相关较弱。为弥补这一不足，本文提出了一种新颖的零样本评测框架——TIT（Text-to-Image-to-Text consistency），通过比较原始提示与由视觉语言模型（VLM）生成的图像描述之间的一致性来衡量文本与图像的对齐程度。该框架包含高效的基于向量相似度的TIT-Score和基于大语言模型（LLM）的TIT-Score-LLM两种实现。实验表明，TIT方法在长提示图像生成评测中显著优于CLIP-score等传统指标，TIT-Score-LLM在配对准确率上领先最强基线7.31%。LPG-Bench与TIT框架为长文本理解与生成提供了坚实的评测基础，推动了T2I模型的深入发展。

### 方法 
![](https://img.blenet.top/file/paper-daily/aigc_research/ec947d279a82428c81fff532616d68e8.png)  
本文提出的TIT评测框架核心在于将复杂的文本-图像对齐评估任务拆解为两个阶段：  

1. **视觉感知与描述生成**：利用强大的视觉语言模型（VLM）对生成的图像进行“描述”，将图像内容转化为丰富的文本描述。此步骤避免了传统直接评分中主观性与不稳定性，确保描述客观且细致。  
2. **文本语义对齐计算**：将原始长文本提示与VLM生成的文本描述输入到文本相似度计算模块。该模块有两种实现方式：  
   - **TIT-Score**：使用先进的文本嵌入模型编码两段文本，计算余弦相似度，提供高效且稳健的对齐评分。  
   - **TIT-Score-LLM**：调用顶尖大语言模型（如Gemini 2.5 Pro）直接对两段文本进行相似度打分，借助LLM强大的语义理解能力提升评测准确性。  
该设计避免了端到端大模型评分的稳定性问题，充分利用视觉模型和语言模型的专长，实现了长文本提示下的高精度对齐评估。此框架无需额外训练，适用性广泛且易于部署。

### 实验 
![](https://img.blenet.top/file/paper-daily/aigc_research/b85743f525a34592a11e9f4a5fb88e73.png)  
实验在LPG-Bench上进行，涵盖2600张由13个主流T2I模型生成的图像及12832对非平局的人类偏好对比。评测指标包括配对准确率、Spearman和Kendall相关系数以及归一化折损累计增益（nDCG）。结果显示，TIT-Score和TIT-Score-LLM均显著优于传统指标（如CLIP-Score、BLIP-Score、VQA-Score等），TIT-Score-LLM配对准确率达到66.51%，较最强基线LMM4LMM提升7.31%。此外，TIT-Score作为轻量级版本也取得了66.38%的高准确率，兼顾效率与性能。消融研究表明，端到端大模型直接评分准确率低，验证了分步设计的必要性；使用更强大的LLM作为评分器也能进一步提升表现。模型排名与人类偏好高度一致（SRCC达0.929），证明方法的可靠性。定性分析进一步展示了TIT评分对细节差异的敏感性及其与人类判断的高度契合。

### 通俗易懂  
这个方法的核心思路是先让一个“看图能力强”的模型把生成的图片用文字描述出来，然后再用一个“读懂文字能力强”的模型来比对这段描述和原始的长文本提示，看它们有多相似。这样做的好处是，把复杂的“看图和理解长文本是否匹配”的任务分成了两个简单的步骤：先把图片变成文字，再用文字比文字。这样避免了直接让一个模型给出分数时可能出现的混乱和不稳定。具体来说，第一步用视觉语言模型（比如能看懂图片的AI）来写“这张图里有什么”；第二步用文本嵌入模型或者大语言模型（比如聊天机器人）来判断这段描述和原始提示的相似度。相似度越高，说明图片和提示越匹配。这个方法不需要额外训练，准确率比传统方法高很多，尤其适合处理那些特别长、特别复杂的文字提示，让机器更好地理解并生成符合要求的图片。