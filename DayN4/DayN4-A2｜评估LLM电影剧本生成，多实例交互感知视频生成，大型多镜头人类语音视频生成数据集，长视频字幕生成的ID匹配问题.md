# Special Topic: Movie and Film
## CML-Bench: A Framework for Evaluating and Enhancing LLM-Powered Movie Scripts Generation 
2025-10-01｜HKUST, Lehigh, FDU, Emory, UCF｜⭐️ 

<u>http://arxiv.org/abs/2510.06231v1</u>  
<u>https://github.com/DuNGEOnmassster/CML-Bench</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/165.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/166.jpg)  
本研究聚焦大语言模型（LLMs）在电影剧本生成中的表现，指出尽管LLMs在结构化文本生成上表现优异，但在电影剧本所需的细腻叙事和情感深度方面存在明显不足。为此，作者构建了CML-Dataset，收录100部经典高质量电影剧本的多场景片段及其摘要，作为研究和评估的基准。通过对这些真实剧本的深入分析，确定了评估剧本质量的三大核心维度：对话连贯性（Dialogue Coherence）、角色一致性（Character Consistency）和剧情合理性（Plot Reasonableness）。基于这些维度，提出了CML-Bench评测框架，能量化评估剧本质量并揭示LLM生成剧本的不足。为提升LLM生成质量，设计了CML-Instruction提示策略，引导模型生成结构更合理、叙事更富感染力的剧本。整体工作为电影剧本的自动生成提供了系统的评估与优化手段，推动LLM在创意写作领域的应用进步。

### 方法 
  
本研究方法包括三部分核心内容：  

1. **数据集构建**：筛选IMDb评分高的经典电影剧本，统一格式，并利用LLM自动抽取15-20场景的连贯叙事片段，配以精炼摘要，形成100个高质量（内容，摘要）对，构建CML-Dataset。  
2. **质量维度分析**：针对剧本的核心叙事特征，定义三大评估维度：  
   - 对话连贯性（DC）：分析相邻对话主题的语义连续性、话题集中度及语言创造力。  
   - 角色一致性（CC）：评估角色情感变化的稳定性、语言风格的一致性及其意图与行为的匹配度。  
   - 剧情合理性（PR）：衡量事件序列的语义连贯性、因果关系密度及叙事创新性。  
3. **CML-Bench设计**：基于以上维度，开发八个可解释的量化指标，结合结构化解析、LLM特征抽取和嵌入相似度计算，确保评估的细粒度和客观性。  
此外，提出CML-Instruction，作为详细的结构化提示，指导LLM在生成剧本时注重场景组织、角色对话和事件逻辑，显著提升生成剧本的质量。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/167.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/168.jpg)  
实验部分通过对11款主流LLM在基础提示和CML-Instruction提示下的剧本生成表现进行评测，验证了CML-Bench的有效性和CML-Instruction的改进效果。结果显示，基线模型在对话连贯性、角色一致性和剧情合理性三大维度均显著低于人类剧本，尤其在角色原创性和语言创造力方面表现弱。引入CML-Instruction后，所有模型在各指标上均有显著提升，部分指标甚至接近人类水平。案例研究覆盖多种电影类型，进一步证明CML-Bench的广泛适用性和稳定性。人类评审也表明，CML-Bench评分与专家主观评价高度相关（Spearman相关系数0.80），验证了自动评测的可靠性。整体实验表明，该框架不仅能准确评估LLM生成剧本质量，还能有效指导模型产出更具电影感和叙事深度的作品。

### 通俗易懂  
这项研究的核心是帮助电脑写出更像电影剧本的故事。首先，研究人员收集了100部经典电影的剧本片段和简短总结，作为“好剧本”的标准。然后，他们发现好剧本主要有三个关键点：对话要连贯，角色要行为和说话都一致，剧情要合理且有逻辑。基于这些发现，他们设计了一套评分系统，能自动检查电脑写的剧本有没有这三个特点。接着，他们还给电脑写作时提供了详细的“写作指南”，告诉电脑怎么安排场景、让角色说话更自然、故事发展更合乎逻辑。通过这些方法，电脑写出的剧本质量明显提升，更加像真正的电影剧本。简单来说，这项工作就像给电脑装上了“电影剧本写作的眼睛和大脑”，既能帮它看清好剧本的样子，也能教它怎么写出更好的故事。 
## Stacked Regression using Off-the-shelf, Stimulus-tuned and Fine-tuned Neural Networks for Predicting fMRI Brain Responses to Movies (Algonauts 2025 Report) 
2025-10-02｜Université Paris Cité, ​​Ind. Res, Oxford, Turin, UL, MPSC 

<u>http://arxiv.org/abs/2510.06235v1</u>  
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/169.jpg)  
本报告介绍了团队Seinfeld参加Algonauts 2025挑战赛的方案，旨在预测人类大脑对电影刺激的fMRI反应。该挑战提供了丰富的多模态数据，包括视频、音频及文本转录，强调模型的泛化能力。团队方法融合了多种预训练神经网络的内部表征，涵盖语言模型（如Llama、Qwen）、视觉模型（如slow r50、InternVL）、音频模型（Whisper）及视频语言模型。通过增强文本输入、刺激特定调优和微调策略，提升了模型对脑活动的预测准确性。最终，利用堆叠回归融合多模型预测，取得了第10名的成绩。该工作不仅展示了多模态深度学习模型在脑编码领域的潜力，也公开了代码与资源，助力未来研究。  

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/170.jpg)  
本研究方法主要包括以下几个方面：  

1. **预训练模型表征提取**：分别从语言、视觉、音频和多模态模型中提取内部特征，采用PCA降维后用线性回归拟合fMRI信号。  
2. **文本增强与多模态融合**：通过添加场景摘要、说话人信息等丰富文本内容，提升语言模型的上下文理解；利用InternVL融合视觉与语言信息，探索不同层次的表征对脑预测的贡献。  
3. **刺激特定调优与微调**：对Llama模型进行基于电影对话结构的持续预训练（刺激调优），以及对语言和视觉模型进行LoRA微调，直接优化对fMRI的预测能力。  
4. **堆叠回归融合**：将多模型预测结果通过线性组合学习最优权重，实现信息互补和性能提升。  
此外，还尝试了基于隐藏马尔可夫模型的fMRI时序建模和对比学习的视频-脑编码器，但未纳入最终提交。  

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/171.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/172.jpg)  
实验基于Algonauts 2025公开数据，涵盖4名受试者长时间电影观看的fMRI数据。团队通过系统调试模型参数、选择最佳层次和时间窗口，评估各模型在“Friends”系列及其他电影上的泛化表现。文本增强提升了InternVL模型的预测性能，刺激调优略微改善了Llama模型在训练分布内的表现，但对外部分布效果有限。视觉模型slow r50经微调后显著优于未微调版本。堆叠回归融合了Whisper、Llama、InternVL和微调slow r50的预测，最终提升整体预测相关性。对比学习的视频编码器和隐藏马尔可夫模型虽有潜力，但因时间限制未被采纳。最终方案在挑战赛中获得第10名，验证了多模态融合和微调策略的有效性。  

### 通俗易懂  
简单来说，这项研究就是用多种“聪明的电脑模型”来预测人脑看电影时的大脑活动。首先，他们用已经训练好的模型分别理解电影里的文字、画面和声音，提取这些信息的“内部想法”。然后，他们让模型看到更详细的电影文字说明，比如谁在说话、场景描述，帮助模型更好理解内容。接着，他们针对电影的对话风格让语言模型稍作调整，让它更适合电影语言。还有，他们对视觉模型做了专门训练，让它更精准地预测大脑反应。最后，他们把这些模型的预测结果“混合”起来，找出最合适的组合方式，这样预测就更准确了。通过这样的步骤，他们成功用机器预测了人脑看电影时的反应，效果在比赛中表现不错。 
# Topic: Video Generation 
## MATRIX: Mask Track Alignment for Interaction-aware Video Generation 
2025-10-08｜KAIST 

<u>http://arxiv.org/abs/2510.07310v1</u>  
<u>https://cvlab-kaist.github.io/MATRIX/</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/173.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/174.jpg)  
本文针对视频扩散变换器（Video Diffusion Transformers, DiTs）在多实例及主体-客体交互建模上的不足，提出了一种交互感知的视频生成方法。作者首先构建了MATRIX-11K数据集，包含11K带有交互描述的多实例视频及其实例掩码轨迹，为分析视频DiTs内部如何表征交互提供了基础。通过系统性分析，发现视频DiTs在少数“交互主导层”中实现了语义绑定（语义落地）和语义传播（跨帧一致性），即名词与动词对应的视觉区域及其关系在特定层表现突出。基于此观察，提出MATRIX框架，通过对这些关键层的注意力进行掩码轨迹对齐，显著提升了交互的准确性和时序一致性。文章还设计了交互感知评测协议InterGenEval，综合衡量生成视频的交互语义对齐度和稳定性，验证了方法在交互保真度和语义一致性上的优越性。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/175.jpg)  
MATRIX方法的核心在于对视频DiT的注意力机制进行正则化，具体包括：  

1. **数据准备**：构建MATRIX-11K数据集，利用大语言模型（LLM）从交互描述中提取主语、谓语、宾语三元组及实例ID，结合视觉语言模型（VLM）和SAM2分割模型生成并验证多实例掩码轨迹，确保语义与视觉的精确对应。  
2. **交互主导层识别**：通过Attention Alignment Score（AAS）评估视频到文本（video-to-text）和视频到视频（video-to-video）注意力在各层的语义落地和传播效果，筛选出语义绑定和传播最显著的少数层作为交互主导层。  
3. **正则化设计**：引入两种损失函数——语义落地对齐损失（SGA）和语义传播对齐损失（SPA），分别约束交互主导层中视频到文本和视频到视频的注意力分布，使其与实例掩码轨迹空间和时间上对齐。  
4. **轻量级解码器**：设计一个轻量因果解码器将注意力映射到像素级掩码空间，确保监督信号与掩码轨迹在时空分辨率上的匹配。  
5. **训练策略**：基于CogVideoX-5B-I2V模型，采用LoRA微调技术，仅更新输入层、交互主导层的注意力权重及解码器参数，避免破坏模型原有的生成能力。  

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/176.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/177.jpg)  
实验部分在合成和真实场景数据集上评估了MATRIX方法，主要通过InterGenEval协议测量交互语义对齐度（KISA）、语义落地完整性（SGI）和综合交互保真度（IF）。结果显示，MATRIX在这三项指标上均明显优于现有先进模型，同时在人类解剖学合理性（HA）、视频运动流畅度（MS）和图像质量（IQ）上也保持或提升了性能。消融实验进一步验证了交互主导层选择的重要性及SGA与SPA两种损失的互补性：单独使用SGA或SPA均提升部分指标，而两者结合则实现综合性能最优。定性分析表明，MATRIX能有效减少语义错误、身份漂移和实例重复，生成更符合文本描述的多实例交互视频。整体实验证明，针对交互语义的注意力对齐正则化是提升视频DiT交互感知能力的有效途径。

### 通俗易懂  
简单来说，MATRIX就是教会视频生成模型“看懂”视频里“谁在做什么”和“和谁互动”。首先，研究人员准备了一个大数据集，里面的视频都有清楚标注“谁是谁”和“他们在做什么”的信息。然后，他们发现模型内部有几层特别重要，这几层的注意力机制就像模型的“眼睛”，专门用来关注这些交互。接着，他们设计了一个方法，强制让模型这几层的“注意力”对齐真实的“人或物的轮廓和动作轨迹”，就像让模型学会把句子里的词和视频中的对应人物动作准确对应起来。为了做到这一点，他们还做了一个小工具，把模型的注意力映射回视频画面上的具体位置，方便对比和调整。最后，他们只微调模型中这几层的参数，避免影响整体视频质量。这样一来，模型在生成视频时，就能更准确地表现多个人或物体之间的互动，生成的画面更符合描述，动作也更连贯自然。 
# Topic: Video & Audio Generation｜Human 
## TalkCuts: A Large-Scale Dataset for Multi-Shot Human Speech Video Generation 
2025-10-08｜UMass Amherst, Tencent, FDU, Sony, UC San Diego｜⭐️⭐️  

<u>http://arxiv.org/abs/2510.07249v1</u>  
<u>https://talkcuts.github.io/</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/178.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/179.jpg)  
本文提出了TalkCuts，一个大规模、多镜头的人类语音视频生成数据集，专门针对长格式、多镜头的人类演讲视频合成任务。与现有主要聚焦单镜头、固定视角的数据集不同，TalkCuts包含164,000个片段、超过500小时的高清视频，涵盖多种镜头类型（特写、半身、全身等）及丰富的说话者身份（超过10,000人），并配备2D关键点、3D SMPL-X动作注释和详细文本描述，支持多模态学习。基于此数据集，作者设计了Orator，一个由大语言模型（LLM）引导的多模态生成框架，能够根据文本脚本和参考图像自动规划镜头切换、说话者动作和语音调控，实现连贯的长格式多镜头视频合成。实验表明，利用TalkCuts训练的模型在镜头连贯性、动作自然度和身份保持方面显著优于现有方法，展示了数据集和框架在多镜头人类视频生成领域的潜力和应用价值。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/180.jpg)  
本研究方法核心包括TalkCuts数据集构建和Orator生成系统设计。  

1. **TalkCuts数据集构建**：  
   - 采集多样化场景（TED演讲、脱口秀等）中的高清视频，经过人工筛选保证质量。  
   - 利用PySceneDetect分割视频，RTMDet检测人类主体，DWPose提取133个2D全身关键点，结合SMPLer-X及HaMeR、EMOCA对3D姿态和表情进行高质量估计。  
   - 定义六种镜头类型（特写、半特写、中景等），通过关键点检测判定镜头类别，并用Qwen2.5-VL生成包含动作、表情、镜头运动及环境描述的文本注释。  
2. **Orator系统设计**：  
   - 由Director LLM担任“导演”，输入文本脚本，输出分镜头规划、动作指导和语音情感调控指令。  
   - 采用检索增强生成（RAG）策略，结合相似语料的镜头切换示例，提升镜头规划的准确性与连贯性。  
   - 语音生成模块（SpeechGen）基于CosyVoice文本转语音模型，结合导演指令实现情感丰富的语音合成。  
   - 视频生成模块（VideoGen）基于CogVideoX和Hallo3架构，融合参考图像、动作指令和语音，利用扩散模型生成高质量多镜头视频，保证身份一致性和动作自然流畅。  

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/181.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/182.jpg)  
实验分为三个部分验证TalkCuts数据集和Orator系统的有效性。  

1. **镜头切换规划**：通过IoU、准确率和镜头匹配准确度（SMA）指标评估多种语言模型（GPT-4o、LLaMA、Qwen等）在镜头切换任务上的表现。结果显示，基于RAG的GPT-4o模型在镜头规划的准确性和连贯性方面优于其他方法，证明了检索增强策略的有效性。  
2. **音频驱动视频生成**：在50个测试视频上，使用FID、FVD、PSNR、SSIM、LPIPS等多维度指标评估生成视频质量及口型同步度。与SadTalker、EchoMimicV2、Hallo3等现有方法相比，Orator在视频质量、身份保持、动作细节及同步性方面均表现最佳，尤其在手部细节和动态表现上有明显优势。  
3. **姿态驱动视频生成**：通过对MusePose、AnimateAnyone、ControlNeXt等模型在TalkCuts上的微调，验证数据集对姿态引导视频生成的提升作用。微调后模型在图像质量、视频连贯性和身份保持方面均有显著提升，体现了TalkCuts在多镜头、多视角动作合成训练中的重要价值。  

### 通俗易懂  
这个工作主要做了两件事：先是收集了一个超大规模的演讲视频数据集，里面不仅有各种不同的镜头，比如特写、半身和全身镜头，还有每个视频的详细动作和表情标注。这样，机器就能学会在不同镜头间自然切换，模仿真实视频的感觉。然后，他们设计了一个叫Orator的“导演”系统，这个系统就像电影导演一样，读懂演讲稿后，告诉机器什么时候切换镜头，什么时候做手势，声音要怎么变换。它还会生成和动作同步的语音，最后把这些信息融合，用先进的视频生成模型输出一段连贯、真实感强的多镜头演讲视频。简单来说，就是让电脑学会拍电影：根据内容安排镜头，配合动作和声音，生成像真人演讲一样的视频。实验表明，这套方法生成的视频比以前的更自然、连贯，动作和声音也更匹配，效果很棒。 
# Topic: Video｜Multi-modal 
## Addressing the ID-Matching Challenge in Long Video Captioning 
2025-10-08｜SJTU, Alibaba 

<u>http://arxiv.org/abs/2510.06973v1</u>  
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/188.jpg)  
本文聚焦于长视频自动生成字幕中的关键难题——ID匹配问题，即如何准确识别视频中多帧出现的同一人物。该问题对视频理解和文本生成至关重要，但传统方法如人脸识别和行人重识别存在泛化能力差、依赖特定视角和点对点匹配等缺陷，难以处理复杂多变的长视频场景。本文创新性地利用大型视觉语言模型（LVLM），尤其是GPT-4o，挖掘其内在的ID匹配潜力，提出了RICE方法（Recognizing Identities for Captioning Effectively），旨在提升长视频字幕中人物身份的连续追踪能力。为此，作者构建了RICE-benchmark，该基准包含专门标注多人物及其出现帧的长视频数据集、自动提取预测ID序列的方法以及针对ID匹配的评估指标，为该领域提供了系统的研究工具和评测标准。实验表明，RICE在ID匹配精度和召回率上均显著优于现有方法，能够有效解决长视频字幕中人物身份混淆的问题。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/189.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/190.jpg)  
本文方法体系围绕提升LVLM在长视频字幕中的ID匹配性能，分为以下几个核心部分：  

1. **三种视频字幕生成策略分析**：包括单轮对话（ST，所有帧同时输入）、同一上下文多轮对话（MTSC，逐帧对话且可访问历史图像信息）和跨上下文多轮对话（MTDC，仅文本信息传递）。ST因能直接对多帧图像进行联合注意，视觉信息利用最充分，ID匹配效果最佳。  
2. **增强视觉信息利用**：通过多帧窗口输入实现ST模式，强化跨帧视觉特征的关联，提升ID识别的准确性。  
3. **丰富人物描述信息量**：构建包含36种常用人物特征的描述体系，通过“场景格式”设计，灵活控制单帧描述的详细程度。实验发现，增加描述特征数量显著提升文本中ID匹配的连贯性和准确率，尤其在MTDC模式下效果尤为突出。  
4. **RICE方法设计**：结合多帧窗口输入（MF）和强化文本特征描述（ETF），通过仅基于人物特征而非动作或环境信息的摘要，避免因相似动作或场景导致的误识别。此设计有效融合视觉和文本信息，提升长视频中人物身份的持续追踪能力。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/191.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/192.jpg)  
实验部分采用构建的RICE-benchmark和ActivityNet数据集，重点评估RICE方法在字幕质量和ID匹配性能上的表现。通过用户偏好调查、GPT-4评分、文本覆盖率以及专门设计的ID匹配精度和召回率指标，全面验证方法效果。结果显示，RICE在用户偏好和自动评分中均超越现有的ShareGPT4Video基线，在ID匹配的精确率和召回率上分别从50%提升至90%和从15%提升至80%。消融实验进一步表明，多帧窗口（MF）和增强文本特征（ETF）两大组件均对性能提升贡献显著。RICE方法不仅在GPT-4o上表现优异，迁移到其他LVLM如Qwen-VL2.5和Deepseek v3时，也展现出一致的提升，验证了方法的通用性和鲁棒性。整体实验充分证明了通过视觉信息强化和丰富人物描述，能有效解决长视频字幕中的ID匹配难题。

### 通俗易懂  
长视频里经常有很多人反复出现，给视频自动生成字幕时，怎么准确说清楚“是谁做了什么”特别难。传统方法像人脸识别只能对特定角度的人有效，且容易把不同的人搞混。本文的方法用了一种聪明的AI模型——它能同时看多张图片，把这些图片里的同一个人连接起来。然后，我们让AI给每个人写更详细的描述，比如穿什么衣服、有什么特别的发型，而不是只看他们在做什么或在哪个地方。这样，AI就能更准确地识别出视频中反复出现的是同一个人，不会搞错。我们还做了一个专门的测试集，专门用来考察AI在这方面的表现。实验结果很棒，AI能把识别的准确率提高很多倍，字幕也更清楚更连贯。简单来说，我们让AI“多看几张照片”，再“多记一些细节”，就能帮它更聪明地认人，写出更好的视频字幕。 
