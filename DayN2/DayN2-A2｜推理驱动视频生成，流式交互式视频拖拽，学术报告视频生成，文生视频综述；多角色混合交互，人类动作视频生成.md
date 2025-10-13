
# Topic: Video Generation
## VChain: Chain-of-Visual-Thought for Reasoning in Video Generation 
2025-10-06｜NTU, Eyeline Labs 

<u>http://arxiv.org/abs/2510.05094v1</u>  
<u>https://eyeline-labs.github.io/VChain</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/56.jpg)  
本文提出了VChain，一种创新的推理驱动视频生成框架，旨在解决现有视频生成模型难以捕捉复杂动态和因果连贯性的问题。传统视频生成多聚焦于视觉平滑和画质，但往往忽略物理规律、状态转变及因果关系，导致生成内容缺乏合理性。VChain借助大型多模态模型（如GPT-4o）的强大视觉推理能力，自动推断并生成一系列关键的视觉思维帧（Visual Thoughts），这些帧作为场景中关键状态的稀疏快照，描绘事件的因果链条。通过在推理时刻对预训练视频生成器进行轻量级调优，VChain引导模型生成更具逻辑连贯性和物理合理性的动态视频，无需额外数据或重训练，显著提升了视频的因果一致性和视觉质量。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/57.jpg)  
VChain框架包含三大核心步骤：  

1. **视觉思维推理（Visual Thought Reasoning）**：利用GPT-4o对用户文本提示进行多步推理，预测事件的因果后果，生成描述关键状态的文本和对应的图像关键帧，形成“视觉思维链”。这一步通过迭代生成图像和文本，捕捉场景中重要的视觉变化节点。  
2. **稀疏推理时调优（Sparse Inference-Time Tuning）**：将上述视觉思维链中的关键帧作为稀疏监督信号，采用低秩适配（LoRA）技术对预训练视频生成模型进行轻量调优。该调优仅针对关键帧，确保模型聚焦于因果关键状态的准确生成，提升视频连贯性且计算成本低。  
3. **视频采样（Video Sampling）**：将所有生成的文本思维拼接成完整提示，驱动经调优的视频生成器合成最终视频。此流程确保视频既符合推理出的因果链，也保持视觉上的自然流畅。  

该方法避免了大规模再训练和密集标注，利用多模态模型的推理能力实现高效且逻辑合理的视频生成。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/58.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/59.jpg)  
实验基于Wan预训练视频生成模型，采用包含20个复杂多步骤场景的测试集，结合VBench评测框架，全面评估视频的画面质量、时间一致性、文本对齐度及因果推理能力。结果显示，VChain在物理规律遵循、常识推理和因果一致性等关键维度上明显优于基线模型和消融版本，且不牺牲视频的视觉质量。人类评审也证实VChain生成的视频更符合因果逻辑，动态更自然，物体交互更真实。消融实验进一步证明视觉思维帧和稀疏调优两者缺一不可，缺少视觉思维帧会导致空间理解不足，缺少稀疏调优则产生帧间错位和视觉伪影。尽管存在图像生成轻微过度平滑和API调用成本等限制，VChain仍展现出推理能力与视频生成结合的巨大潜力。

### 通俗易懂  
简单来说，VChain就像给视频生成模型装上了“思考大脑”。当你告诉它“冰块放在纸上晒太阳”，它不会直接开始画视频，而是先用强大的多模态AI帮你想一想：冰块会慢慢融化，水会渗透纸张，纸张可能变形。这个过程被拆成几个关键画面，比如“冰块完整”、“冰块部分融化”、“形成水渍”等。接着，VChain用这些关键画面告诉视频生成器：“你看，这些是故事发展的重要节点，生成视频时一定要保证这些场景真实合理。”视频生成器只针对这些关键画面做小调整，快速学会如何表现冰块融化的过程。最后，视频生成器根据这个思考链，生成连贯又符合物理常识的视频。这样，生成的视频不仅好看，还能讲得通，像人类思考后拍出来的电影一样。 
## Streaming Drag-Oriented Interactive Video Manipulation: Drag Anything, Anytime! 
2025-10-03｜NTU, HFUT 

<u>http://arxiv.org/abs/2510.03550v1</u>  
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/60.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/61.jpg)  
本论文提出了一个全新的任务——流式拖拽交互式视频操作（REVEL），旨在实现用户在视频生成过程中对任意内容进行细粒度、实时的拖拽式编辑和动画制作。相比以往仅支持静态编辑或轨迹引导动画的工作，REVEL统一了拖拽操作下的视频编辑与动画两大范畴，支持用户指定的平移、形变及二维/三维旋转效果，极大丰富了交互的多样性和灵活性。然而，该任务面临两大核心挑战：一是拖拽操作引起的潜变量空间分布漂移，导致操作中断；二是上下文帧干扰，造成生成内容不自然。针对这些问题，本文提出了训练免费且可无缝集成于现有自回归视频扩散模型的解决方案——DragStream，并通过大量实验验证了其在保证高质量视频操控的同时，避免了昂贵训练成本的优势。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/62.jpg)  
DragStream方法设计围绕两个关键技术策略展开：  

1. **自适应分布自我校正（ADSR）**：针对潜变量空间的分布漂移问题，ADSR利用邻近帧潜变量的统计特征（均值与方差）动态调整当前帧潜变量的分布，抑制累积扰动，确保拖拽过程的连续性和稳定性。  
2. **时空频率选择性优化（SFSO）**：结合空间和频率域的信息选择机制，SFSO在迭代优化潜变量时，利用频率过滤（如巴特沃斯滤波器）随机切换传递的频段，防止高频噪声主导生成过程，同时采用基于高斯权重的空间梯度选择，聚焦拖拽手柄区域，减少背景过度优化，缓解上下文干扰。  
整体流程中，用户拖拽指令被映射为特定区域的平移或旋转变换，通过迭代潜变量优化实现视频帧的编辑或动画生成。ADSR和SFSO策略在该优化过程中相辅相成，确保生成内容既符合用户期望，又保持视觉自然和连贯。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/63.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/64.jpg)  
实验部分针对REVEL任务进行了全面评估。首先，视觉结果展示了DragStream在实现平移、变形及旋转等多样拖拽操作时，生成的视频帧在结构和外观上均优于现有训练免费方法SG-I2V和DragVideo，表现出更少的失真、伪影和操作失败。其次，定量指标（如FID、FVD、ObjMC和DAI）进一步验证了DragStream在视频质量和拖拽精度上的显著优势。消融实验明确表明，ADSR和SFSO两大核心组件对整体性能贡献巨大，缺一不可；其中，SFSO的频率切换策略优于固定频率过滤，能够更好地平衡上下文信息利用与干扰抑制。综合来看，DragStream不仅提升了拖拽式视频操作的质量，也实现了训练成本的极大节约，具备良好的实用价值和推广潜力。

### 通俗易懂  
简单来说，DragStream就像给视频生成加了双“护盾”和“聚光灯”。“护盾”就是ADSR，它能感知周围视频帧的正常状态，防止你拖动视频里物体时，画面突然变形或出现奇怪的颜色变化，就像帮你保持画面稳定不跑偏。而“聚光灯”则是SFSO，它聪明地选择只关注那些对画面细节有用的信息，避免被杂乱的背景干扰，确保你拖动的物体动作自然流畅。整个过程不需要重新训练模型，只要在生成视频的过程中，动态调整和优化视频的“内部代码”，就能让你随时拖动任何物体，实时看到它们被平移、拉伸或旋转的效果。这种方法既省时又省力，让视频编辑变得像玩积木一样简单有趣。 
## Paper2Video: Automatic Video Generation from Scientific Papers 
2025-10-06｜NUS 

<u>http://arxiv.org/abs/2510.05096v2</u>  
<u>https://showlab.github.io/Paper2Video/</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/128.jpg)  
本文针对学术报告视频的自动生成问题，提出了Paper2Video基准和PaperTalker生成框架。学术演示视频在科研传播中作用重要，但传统制作耗时费力，且涉及长文本、多模态信息（文字、图表）及多通道内容协调（幻灯片、字幕、语音、主持人形象），现有自然视频生成技术难以直接应用。为此，作者收集了101篇论文及其作者录制的演示视频、幻灯片和演讲者元数据，构建了首个涵盖多模态、多任务的学术演示视频生成基准。基准中设计了四种评价指标，包括内容与人类制作的相似度、观众偏好比较、知识传递效果测试及作者影响力记忆度，全面评价生成视频的质量与学术传播效果。PaperTalker框架则通过多代理协作，集成幻灯片生成、字幕与光标定位、个性化语音合成及真人头像视频渲染，实现高效且信息丰富的学术演示视频自动制作。该工作为自动化学术视频制作提供了实用路径，推动了AI辅助科研传播的发展。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/129.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/130.jpg)  
PaperTalker框架包含四个核心模块协同工作：  

1. **幻灯片生成**：输入论文全文，利用大型语言模型自动生成LaTeX Beamer代码，采用迭代编译反馈机制修正语法错误，确保代码正确无误。针对幻灯片布局问题，引入“树搜索视觉选择”策略，通过规则调整字体大小、图形缩放等参数，生成多种版本幻灯片图像，再由视觉语言模型评分选出最佳布局，提升幻灯片的视觉效果和内容表达。  
2. **字幕生成与光标定位**：将生成的幻灯片转为图片输入视觉语言模型，自动生成与幻灯片内容对应的句级字幕及视觉焦点提示。随后，结合计算机界面模拟模型和WhisperX语音时间戳技术，实现字幕句子与光标位置的空间时间精准对齐，方便观众关注关键内容。  
3. **语音合成与真人头像渲染**：基于作者提供的声音样本，使用个性化文本转语音模型合成演讲音频；结合作者肖像，采用长时音频驱动的人脸动画模型生成同步口型和表情的真人头像视频。  
4. **并行化生成**：考虑到幻灯片间内容独立，采用幻灯片级别的并行生成策略，大幅提升视频生成效率，缩短制作时间。  
该框架通过多模态、多任务模块的有机结合，实现了学术演示视频从论文到成品的自动化生成。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/131.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/132.jpg)  
在Paper2Video基准上，作者对比了三类方法：端到端自然视频生成模型、多代理组合模型及PaperTalker及其变体。实验采用多维度指标评估生成视频的内容相似度、观众偏好、知识覆盖率及作者影响力记忆度。结果显示，PaperTalker在语音和幻灯片内容相似度上均优于其他方法，体现了个性化语音合成和精细幻灯片布局优化的优势。视频语言模型评测的观众偏好中，PaperTalker胜率最高，表明其视频整体质量和观感优越。知识问答测试中，PaperTalker准确率领先，说明其生成视频更有效地传递论文核心信息。加入光标提示和真人主持人视频显著提升了记忆效果，增强了学术影响力。此外，通过人类主观评价，PaperTalker生成的视频获得了接近人工制作的好评。消融实验验证了树搜索视觉选择和光标定位对提升幻灯片质量和视频信息传递的重要性。整体实验表明，PaperTalker不仅生成效果优异且效率提升显著，具备实际应用潜力。

### 通俗易懂  
PaperTalker就像一个智能“学术视频制作助手”，它能自动把一篇复杂的科研论文变成一段清晰、生动的演示视频。首先，它会把论文内容转化成一套精美的幻灯片，这一步就像让电脑帮你写幻灯片代码，然后通过“树搜索视觉选择”方法，自动调整字体大小和图片大小，确保幻灯片看起来既美观又整洁。接着，它会根据幻灯片内容生成对应的字幕，并且用光标指示重点内容，帮助观众更好地跟上讲解。然后，系统会用作者的声音样本合成演讲语音，并结合作者照片制作一个会说话的虚拟主持人视频，仿佛作者本人在屏幕前讲解。最后，整个视频制作过程被拆分成每张幻灯片单独处理，多个步骤同时进行，大大加快了制作速度。这样，研究人员就不用花费大量时间去录制和剪辑视频，PaperTalker帮他们高效地生成专业、个性化的学术演示视频，让科研成果更容易被别人理解和传播。  
## Bridging Text and Video Generation: A Survey 
2025-10-06｜SRMIST 

<u>http://arxiv.org/abs/2510.04999v1</u>  
### 概述 
  
本文系统综述了文本生成视频（Text-to-Video, T2V）领域的发展历程、主流模型架构、数据集、训练配置及评估指标。T2V技术通过将自然语言描述转换为连贯的视频序列，极大地推动了教育、营销、娱乐及辅助技术等多领域的应用。相比文本生成图像（T2I），T2V面临更复杂的时序一致性、语义对齐及计算资源挑战。文章回顾了从早期的生成对抗网络（GANs）、变分自编码器（VAEs）到最新的扩散模型（Diffusion Models）和Transformer架构的演进，解析了各类模型如何解决前辈存在的质量、连贯性和控制问题。文中还详述了主流训练数据集的规模与多样性，训练硬件和超参配置，及当前评估指标的优缺点，强调了结合人类评价的综合评测趋势。最后，论文总结了领域面临的核心难题及未来研究方向，为后续研究提供了全面的理论基础和实践指导。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/138.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/139.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/140.jpg)  
本文将T2V生成方法归纳为三大类：  

1. **生成对抗网络（GANs）**：通过对抗训练提升视频的真实感，代表模型如MoCoGAN分离内容与运动潜变量，利用双判别器确保单帧质量和时序一致性，NUWA则采用3D Transformer编码器-解码器，利用局部时空注意力提升长序列生成效率。  
2. **变分自编码器（VAEs）**：利用概率建模学习视频的潜在分布，VideoGPT结合VQ-VAE与自回归Transformer实现高效视频生成，GODIVA则引入3D稀疏注意力处理文本条件下的帧级编码，CogVideo设计双通道Transformer分别捕获空间与时间信息，实现多阶段关键帧生成与插帧。  
3. **扩散模型（Diffusion Models）**：通过逐步去噪生成高质量视频，Make-A-Video基于预训练T2I模型扩展时空卷积与注意力，VideoFusion采用分解噪声生成基础帧与残差帧平衡空间一致性和动态变化，Latent-Shift引入无参数时序偏移模块实现潜空间的时序连贯性，Free-Bloom结合大型语言模型实现零样本视频生成等。各方法均针对时序一致性、语义对齐及计算效率提出创新架构和训练策略。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/141.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/142.jpg)  
本文综述了多种T2V模型在主流数据集上的训练配置和评估表现。训练方面，模型普遍依赖大规模多模态数据集，使用高性能GPU集群，批量大小、学习率及优化器等超参数设计因模型架构和任务而异。评估指标涵盖传统的FID、IS等图像质量度量，以及专门针对视频的时序一致性和文本对齐的定量指标。文中指出，单一指标难以全面反映视频的真实感和语义准确性，故结合人类主观评价成为趋势。实验分析表明，扩散模型在生成质量和时序连贯性方面表现优异，而GANs虽生成速度较快但稳定性较差，VAEs则在潜空间控制方面有优势。最新模型如LaVie和DreamVideo通过多阶段训练和模块化设计，显著提升了视频分辨率和生成时长，展现了良好的实用潜力。

### 通俗易懂  
文本生成视频技术就像让计算机“看懂”一句话，然后“表演”出一段视频。为了实现这一点，科学家们开发了三种主要方法。第一种叫GANs，它们像一场游戏，有两个角色：一个负责“画画”，另一个负责“挑错”，两者互相竞争，最终画出更逼真的视频。第二种是VAEs，它们先把视频压缩成“秘密代码”，再根据这些代码重建视频，这样可以更好地控制视频内容。第三种是扩散模型，它们从一张“杂乱的噪点图”开始，一点一点地“擦掉噪点”，最终变成清晰的视频。扩散模型虽然计算复杂，但能生成高质量且连贯的视频。为了让视频和文字描述匹配，模型还会“学习”文字和视频之间的关系，确保视频内容和描述一致。整个过程需要大量数据和强大的计算资源，但随着技术进步，未来我们能用更简单的方法生成更精彩的视频。 
# Topic: Video Generation｜Human
## Character Mixing for Video Generation 
2025-10-06｜MBZUAI 

<u>http://arxiv.org/abs/2510.05093v1</u>  
<u>https://tingtingliao.github.io/mimix</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/65.jpg)  
本研究聚焦于文本驱动的视频生成中多角色混合交互的挑战，尤其是如何在不同风格和世界观中实现角色的自然互动。传统方法多依赖单一角色的视觉参考图像，难以捕捉角色独特的行为特征和动作逻辑，且跨角色、跨风格的混合生成存在“风格错觉”问题，即现实角色被卡通化或卡通角色被现实化，导致视觉不协调。为此，本文提出了一套创新框架，旨在保持角色身份、行为特性和原始风格的同时，实现角色间的合理互动。研究团队构建了包含卡通与真人剧集共计81小时、5.2万剪辑的多角色视频数据集，并设计了细粒度的角色与风格标注体系，奠定了多角色视频生成的首个基准。实验结果表明，该方法在身份保持、动作一致性、互动自然度及风格稳定性方面均显著优于现有技术，推动了跨风格、多角色生成的故事叙述新可能。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/66.jpg)  
本文方法核心包括两个创新模块：跨角色嵌入（CCE）和跨角色增强（CCA）。  

1. **跨角色嵌入（CCE）**：通过设计角色-动作的文本标注格式，明确区分不同角色的身份与行为，避免多角色视频中身份混淆。利用GPT-4自动生成包含角色名和动作的详细字幕，结合视频、音频和剧本多模态信息，训练模型学习角色的视觉与行为特征。模型采用LoRA微调策略，针对每个角色单独微调，确保角色行为和身份的独立表达。  
2. **跨角色增强（CCA）**：针对卡通与真人剧集风格差异带来的风格错觉问题，采用基于SAM2的角色分割技术，将不同风格角色合成至异风格背景中，生成合成训练数据。通过这种风格混合增强，模型学会在保持角色原始风格的同时，实现跨风格的自然互动。合成视频配以明确的风格标签，辅助模型风格控制。实验证明，少量合成数据即可显著提升模型对风格的识别与保持能力。  
该方法通过细致的文本标注与风格增强，有效解决了跨角色身份混淆和风格错觉两大难题，实现多角色跨风格的视频生成。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/67.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/68.jpg)  
实验部分设计了全面的评价体系，涵盖视频质量、动作连贯性、角色身份保持和多角色互动自然度等多个维度。采用VBench指标评估视频质量与时间一致性，使用视觉语言模型（Gemini-1.5-Flash）评测角色身份（Identity-P）、动作表现（Motion-P）、风格一致性（Style-P）及交互质量（Interaction-P）。基于四个不同来源（两部卡通，两部真人剧集）的10个核心角色，生成单角色及多角色视频进行对比。结果显示，本文方法在所有指标上均优于VideoBooth、DreamVideo、Wan2.1-I2V及SkyReels-A2等先进基线，尤其在多角色跨风格交互场景中表现出更好的身份识别和风格保持能力。消融实验进一步验证了结构化文本标注和合成增强数据在提升模型性能中的关键作用。通过合理的合成数据比例，模型在风格保持与交互自然度之间取得最佳平衡，避免了过度合成带来的质量下降。整体实验充分展示了方法在多角色视频生成领域的领先优势及应用潜力。

### 通俗易懂  
这项研究的目标是让不同动画和真人剧集里的角色能够“走到一起”，在同一个视频里自然地互动，就像让《憨豆先生》和《猫和老鼠》里的角色一起表演一样。为了实现这一点，研究者做了两件重要的事情：  
第一，他们给每个角色和动作都贴上了“标签”，就像给每个人写了详细的身份卡和行为说明，这样模型就能清楚地知道谁是谁，做了什么动作，避免把不同角色搞混。第二，为了让各种风格的角色（卡通和真人）能和谐地出现在同一场景里，他们把角色从原视频中“剪下来”，放到另一个风格的背景里，制作了很多“合成训练视频”，让模型学会在保持角色本来样貌的同时，也能适应不同的场景风格。这样，模型不仅能识别角色，还能让它们表现出自己的特色动作，还能自然地和其它角色互动。通过这些方法，模型生成的视频看起来更真实、角色更鲜活，甚至能让以前从未见过面的人物一起“演戏”！ 
## Generating Human Motion Videos using a Cascaded Text-to-Video Framework 
2025-10-04｜UMich, ETH Zurich, Yonsei 

<u>http://arxiv.org/abs/2510.03909v1</u>  
<u>https://hyelinnam.github.io/Cameo/</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/133.jpg)  
本文提出了CAMEO，一种级联式文本驱动人类动作视频生成框架，旨在克服现有视频扩散模型（VDM）在人类视频生成领域的局限。传统方法多依赖图像到视频的转换，或局限于特定领域如舞蹈视频，且往往缺乏对动作信号来源的系统整合。CAMEO创新性地将文本到动作（T2M）模型与条件视频扩散模型紧密结合，通过设计专门的文本提示和视觉条件策略，确保动作描述、条件信号与生成视频之间的鲁棒对齐。此外，引入了摄像机感知的条件模块，自动选择与文本输入一致的视角，提升视频连贯性和减少人工干预。该框架在MovieGen和新建的HuMoBench基准测试中表现优异，展示了其在多样化应用场景中的适用性和稳定性，显著提升了复杂动作的表现力和视频质量。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/134.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/135.jpg)  
CAMEO方法由两个核心部分组成：训练策略和推理流程。  

1. **训练策略**：  
   - 对文本提示进行细致拆分，区分动作相关和语义上下文信息，避免混淆，提升训练效率和生成准确性。  
   - 利用视觉运动条件信号，采用SMPL三维人体模型渲染动作骨骼，结合光照和身体部位颜色编码，提供丰富的空间结构信息。  
   - 训练时采用ControlNet风格的条件机制，调整扩散模型在不同时间步的条件采样分布，强化大尺度动作控制与细节刻画。  
2. **推理流程**：  
   - 第一阶段由T2M模型生成动作序列，输出基于SMPL的三维顶点数据，再投影为二维条件信号。  
   - 第二阶段设计摄像机视角选择模块，利用视频扩散模型生成的初步视频，自动推断最佳摄像机参数，实现动作与视角的自然匹配。  
   - 最终，视频扩散模型结合语义提示和视觉动作条件生成高质量、连贯的人类动作视频，实现端到端文本到视频的生成。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/136.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/137.jpg)  
实验部分采用MovieGen和新建的HuMoBench两大基准测试，全面评估CAMEO在动作一致性、图像质量和文本对齐度等多个维度的表现。结果显示，CAMEO在动作连贯性和文本语义匹配上显著优于基线模型和现有方法（如HMTV、CamAnimate），尤其在复杂动作场景中表现稳定，避免了动作扭曲和视觉伪影。消融实验验证了文本细化策略和摄像机视角选择模块对性能提升的关键作用。定性分析表明，CAMEO能生成多样化且符合语义的摄像机视角，增强视频表现力。此外，扩展实验展示了该框架在动作编辑和摄像机视角调整上的灵活性，进一步证明了其广泛应用潜力。整体实验验证了CAMEO在多样场景下的鲁棒性和生成质量。

### 通俗易懂  
CAMEO的核心思想是把“说动作”变成“看动作视频”的过程分成两步，先让电脑根据文字描述想象出一个动作的骨架，再根据这个骨架画出真实的人物动作视频。第一步用的是一个专门把文字变成动作序列的模型，它会生成一个三维人体骨骼的运动轨迹。第二步是把这个三维骨骼“拍照”成二维图片，作为提示给视频生成模型。为了让视频看起来更自然，CAMEO还自动帮动作视频选合适的摄像机角度，不用人手动调整。这样，整个流程就像是先画了动作草图，再给草图上色和拍摄，确保动作连贯且视频画面真实。通过这种分工合作，电脑能更准确地理解文字里的动作细节，生成的视频也更稳定、更符合预期。
