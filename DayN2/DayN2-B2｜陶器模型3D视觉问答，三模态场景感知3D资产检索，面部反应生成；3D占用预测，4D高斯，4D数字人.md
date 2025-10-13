# Topic: 3D/4D Generation 

## VaseVQA-3D: Benchmarking 3D VLMs on Ancient Greek Pottery 
2025-10-06｜PKU, BJTU, LTU 

<u>http://arxiv.org/abs/2510.04479v1</u>  
<u>https://github.com/AIGeeksGroup/VaseVQA-3D</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/76.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/77.jpg)  
本文针对古希腊陶器这一文化遗产领域，提出了首个专门的3D视觉问答（VQA）数据集VaseVQA-3D，收录664个高质量3D陶器模型及其对应的问答对，填补了现有视觉语言模型（VLMs）在专业文化遗产数据上的空白。鉴于传统VLM在处理此类长尾、专业化数据时面临数据稀缺和领域知识不足的双重挑战，作者设计了完整的数据构建流程，从2D图像筛选、3D重建到问答对生成，确保数据的考古准确性和视觉质量。同时，提出了专门针对陶器分析的VaseVLM模型，通过域适应训练显著提升了模型在该领域的表现。实验结果显示，该模型在R@1准确率和词汇相似度上分别较前沿方法提升12.8%和6.6%，为数字文化遗产的智能分析与保护提供了新的技术路径。  

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/78.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/79.jpg)  
本文方法包括以下关键步骤：  

1. **数据收集与筛选**：基于已有的VaseVQA二维陶器图像数据集，应用ResNet-50分类器进行质量筛选，剔除模糊和残缺图像；利用CLIP模型进行陶器碎片检测和多视角图像最佳选择，确保最终数据的完整性和代表性。  
2. **3D模型生成**：采用TripoSG技术将筛选后的二维图像转化为高保真3D模型，经过VaseEval验证集的严格评估，确保几何和纹理的准确还原。  
3. **问答对与描述生成**：结合考古元数据和GPT-4增强的文本生成，构建包含六大核心属性（织物、工艺、形状、年代、装饰、归属）的结构化问答对和丰富描述。  
4. **模型训练**：基于Qwen2.5-VL基础模型，分两阶段进行训练——先通过LoRA技术进行监督微调，利用360度旋转视频和考古描述建立基础能力；后采用GRPO强化学习框架，设计多维度可验证奖励机制（RLVR），细致评估模型回答的准确性和文化适切性，进一步提升模型性能。  

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/80.jpg)  
实验部分系统验证了数据筛选、3D生成和模型训练的有效性。数据质量过滤从3万张原始图像中筛选出3880张高质量图像，最终生成664个3D模型，数据保留率为2.2%。在3D生成方法比较中，TripoSG在几何重建和语义一致性上优于Hunyuan3D，被选为主力生成技术。多模型性能评估显示，专门微调的VaseVLM在FID、CLIP分数及检索准确率上均优于多种3D专用及通用VLM，尤其是7B参数版本的强化学习模型表现最佳。专家人类评价进一步证实，VaseVLM在考古描述准确性和文化适宜性上领先，强化学习训练带来约0.4分的显著提升。实验结果体现了专门设计的数据集和训练策略对提升文化遗产领域视觉语言理解的关键作用。  

### 通俗易懂  
简单来说，研究团队想让电脑更好地“看懂”古希腊陶器的3D模型，并能回答关于陶器的专业问题。首先，他们从3万张陶器照片中挑出清晰、完整的图片，然后用一种叫TripoSG的技术把这些2D照片变成真实感很强的3D模型。接着，他们用人工智能帮忙写出关于这些陶器的详细介绍和问答内容。最后，他们让一个基础的视觉语言模型先学习这些3D陶器的旋转视频和介绍，再用一种叫强化学习的方法，教它如何更准确地回答陶器相关的问题。这个强化学习方法特别聪明，会从陶器的材质、工艺、形状等六个方面来评估模型的回答，确保答案既准确又符合考古知识。结果显示，这种专门训练的模型比普通模型聪明很多，能更好地理解和描述古希腊陶器，有助于数字化保护和研究文化遗产。 
## MetaFind: Scene-Aware 3D Asset Retrieval for Coherent Metaverse Scene Generation 
2025-10-05｜NWU(USA), NYU｜NeurIPS 2025 

<u>http://arxiv.org/abs/2510.04057v1</u>  
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/146.jpg)  
MetaFind是一种面向元宇宙场景生成的三模态场景感知3D资产检索框架，旨在从大规模3D资产库中高效检索与当前场景空间、语义和风格一致的3D对象。它解决了传统检索方法忽视空间布局和风格一致性的问题，同时填补了专门针对3D资产检索缺乏标准范式的空白。MetaFind创新地支持任意组合的文本、图像和3D点云多模态查询，结合对象级特征与场景级布局结构，实现空间推理和风格一致性的联合建模。其核心贡献包括引入具备旋转平移等变性的布局编码器ESSGNN，能够捕捉空间关系和对象属性，确保检索结果在不同坐标系变换下保持上下文和风格的一致。该框架支持迭代式场景构建，动态适应场景更新，实现更自然的场景生成，显著提升了检索的空间与风格一致性。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/147.jpg)  
MetaFind采用双塔架构，分别为查询编码器和库中资产编码器，均基于ULIP-2多模态嵌入骨干。查询编码器灵活处理文本、图像、点云及场景布局任意组合的输入，通过模态感知融合策略整合多模态特征，支持部分模态缺失。场景布局被建模为结构化图，节点代表带有空间坐标和语义特征的对象，边表示空间和语义关系。ESSGNN布局编码器基于Equivariant Graph Neural Network扩展，具备SE(3)等变性，能稳定捕获空间和语义边特征，避免因坐标变换导致的嵌入不稳定。训练分两阶段：第一阶段在无布局的对象级数据上进行跨模态对齐预训练，第二阶段在带布局的房间级数据上微调查询编码器和布局编码器，引入场景上下文。检索过程采用迭代式场景合成策略：每次检索并放置一个对象，更新场景图并重新编码布局，保证后续检索考虑最新空间上下文，提升整体场景连贯性和真实感。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/148.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/149.jpg)  
实验涵盖对象级和场景级两大任务，分别使用Objaverse-LVIS和ProcTHOR数据集。对象级检索测试多模态组合查询下的准确率，MetaFind在无布局编码器时即优于多种基线，显示出强大的多模态融合能力。加入ESSGNN后，虽然单纯检索准确率略有下降，但场景级评估显示布局感知显著提升了场景的审美、风格一致性、空间连贯性和物理合理性。场景级评价结合GPT-4o自动评分与人工打分，MetaFind均取得最高分。消融实验验证了ESSGNN、融合策略、模态掩码率等设计对性能的贡献。迭代检索策略相比一次性检索显著提升了场景布局的合理性和视觉和谐度。整体结果表明，MetaFind不仅具备灵活多模态检索能力，还能通过布局感知实现高质量、语义丰富的3D场景生成。

### 通俗易懂  
MetaFind就像一个聪明的3D场景设计助手，能帮你从海量3D模型库里挑选最合适的物品放进你的虚拟空间。它的特别之处在于，不仅根据你的文字描述、图片或3D模型来找东西，还会考虑这些物品在场景里的位置关系和风格搭配。比如，如果你在设计一个客厅，它会理解沙发和茶几应该怎么摆放，颜色和风格是否协调。它通过一个叫ESSGNN的“场景感知大脑”来理解物体之间的空间和语义关系，这个大脑能识别物体的位置和它们之间的联系，即使场景旋转或移动也不会迷糊。检索时，MetaFind不是一次性找全所有物品，而是一步步找，放一个更新场景信息，再找下一个，这样每次都能基于最新的布局做出更合理的选择。简单来说，MetaFind帮助你用各种信息组合，智能地挑选和摆放3D物品，让虚拟空间看起来更自然、更协调。 
## ReactDiff: Fundamental Multiple Appropriate Facial Reaction Diffusion Model 
2025-10-06｜Monash, UoE 

<u>http://arxiv.org/abs/2510.04712v1</u>  
<u>https://github.com/lingjivoo/ReactDiff</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/97.jpg)  
本文针对人机交互中生成多样且自然的面部反应问题，提出了ReactDiff，一种基于扩散模型的时序生成框架。传统方法难以捕捉人类面部反应的随机性和动态连续性，且往往忽视面部肌肉动作间的空间依赖和时间行为规律，导致生成的表情出现抖动、不连贯或不自然。ReactDiff通过引入两类关键先验知识——时序面部行为运动学和面部动作单元（AU）之间的依赖关系，有效约束生成过程，保证反应的平滑性和解剖学合理性。该方法不仅提升了反应的多样性和适切性，还显著改善了生成质量。大量实验基于REACT2024数据集验证了ReactDiff在多样性、真实性及反应匹配度等方面的先进性能，展示了其在实时对话环境中生成符合上下文的多样面部反应的能力。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/98.jpg)  
ReactDiff设计了一个融合时空约束的扩散生成网络，核心包括以下三部分：  

1. **时序面部行为运动学约束**：通过计算相邻时间窗口内面部表情变化的速度，确保生成的面部反应在时间上平滑连贯，避免突兀的快速变化。该约束以速度匹配损失形式引导模型学习真实人类面部运动节奏。  
2. **面部动作单元空间依赖约束**：基于面部肌肉动作单元之间的对称性、共现性和互斥性关系，设计空间约束损失，防止生成不符合人体面部解剖结构的表情组合，提升表情的自然度和真实感。  
3. **在线扩散生成策略**：区别于传统一次生成完整序列的离线方法，ReactDiff采用滑动时间窗口，结合前一时间段生成的反应段，实时生成当前段落，保证跨窗口的连续性和上下文一致性。训练中通过逐步注入高斯噪声并学习去噪过程，结合分类器自由引导技术，增强生成多样性和灵活性。整体模型采用U-Net架构，通过联合优化数据拟合损失、时序运动学损失和空间动作单元损失，实现高质量、多样化的面部反应生成。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/99.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/100.jpg)  
实验基于REACT2024数据集，涵盖1594个训练视频及806个测试视频，采用多样性（FRDvs、FRVar、FRDiv）、真实性（FVD）、适切性（FRCorr）和同步性（FRSyn）四类指标评估。对比多种主流方法，包括基于LSTM、VAE及传统扩散模型的生成器，ReactDiff在多样性指标上显著领先，表明其能生成更丰富的反应样本；在真实性和适切性指标上也取得最优，反映生成面部表情更贴近真实人类反应且符合对话上下文。定性分析展示ReactDiff生成的表情变化自然，避免了传统方法常见的表情失真和抖动现象。消融实验验证了时序索引和空间动作单元约束对模型性能的贡献，强化了模型对面部运动规律和解剖结构的学习能力。整体结果证明ReactDiff在实时多样面部反应生成任务中的有效性和先进性。

### 通俗易懂  
ReactDiff的核心思想是让计算机“学会”生成像人一样自然且多样的面部表情反应。它通过两条“规则”来指导生成过程：首先，面部表情的变化要像人一样平滑，不会突然跳动或变形，这就像画动画时要保证动作连贯；其次，面部的不同肌肉动作之间有一定的关系，比如两边的嘴角通常会同时上扬，某些表情不会同时出现，这些解剖学规律帮助模型避免生成奇怪或不自然的表情。ReactDiff不是一次性生成完整的表情序列，而是分段生成，每次都参考之前生成的内容，确保表情连续且符合对话情境。训练时，模型先学习如何从带噪声的表情数据中还原出真实的表情，通过反复练习，逐渐掌握生成自然表情的技巧。这样，最终生成的面部反应不仅多样，还能根据对话内容做出恰当且真实的表现，提升了人机交互的自然感和亲切感。
# Topic: 3D/4D Reconstruction 
## Progressive Gaussian Transformer with Anisotropy-aware Sampling for Open Vocabulary Occupancy Prediction 
2025-10-06｜HKUST, ZEEKR Automobile R&D Co., Ltd 

<u>http://arxiv.org/abs/2510.04759v2</u>  
<u>https://yanchi-3dv.github.io/PG-Occ</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/150.jpg)  
本文针对自动驾驶中基于视觉的3D占据预测任务，提出了一种创新的Progressive Gaussian Transformer（PG-Occ）框架，实现了开放词汇的3D占据预测。传统方法受限于固定语义类别，难以识别新颖物体；而现有文本对齐特征预测方法在稀疏高斯表示和密集表示之间存在效率与细节捕获的权衡。PG-Occ通过渐进式的在线加密策略，逐步细化3D高斯表示，有效捕获场景细节，实现更精准的场景理解。同时引入了各向异性感知采样策略，结合时空特征融合，针对不同尺度和阶段的高斯分布自适应调整感受野，增强特征聚合能力。大量实验表明，PG-Occ在Occ3D-nuScenes基准上实现了14.3%的mIoU相对提升，显著优于现有最优方法，展现了其在开放词汇3D占据预测中的领先性能和效率优势。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/151.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/152.jpg)  
PG-Occ框架核心是将场景表示为一组带文本对齐特征的3D高斯体积，并通过渐进式的高斯Transformer逐层细化表示。具体方法包括：  

1. **3D特征高斯体积建模**：利用带有位置、尺度、旋转、透明度及512维文本对齐特征的高斯体积，替代传统稠密体素，实现稀疏而高效的语义表达。  
2. **渐进式在线加密（POD）**：从粗糙的基础高斯集合开始，基于深度图和当前预测误差，动态识别欠采样区域，通过远点采样增添新的高斯体积，实现逐步细化，避免了昂贵的梯度优化。  
3. **非对称自注意力（ASA）**：新加入的高斯体积只能从已优化的高斯中学习，防止未优化体积影响已训练表示，保证训练稳定性。  
4. **各向异性感知采样（AFS）**：根据高斯体积的尺度和旋转，生成多个采样点投影到2D特征图，通过多视角时空特征插值与聚合，提升特征提取的空间准确性和丰富性。  
5. **训练与推理**：利用2D图像的伪深度和文本对齐特征进行监督，采用深度和特征渲染损失训练模型。推理时，将最终高斯特征与任意文本提示匹配，转换为稠密3D占据网格，实现开放词汇语义占据预测。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/153.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/154.jpg)  
在Occ3D-nuScenes数据集上，PG-Occ在开放词汇3D占据预测任务中取得了15.15的mIoU，相较于最近的GaussTR方法提升14.3%，且不依赖LiDAR数据，优于多模态融合方法VEON。细粒度中等尺寸物体检测表现尤为突出。nuScenes检索任务中，PG-Occ实现了21.2的可见点平均精度（mAP(v)），优于LangOcc的18.2，验证了其在语言驱动三维检索的有效性。深度估计方面，PG-Occ在多视角深度一致性约束下，误差指标显著优于伪深度标签，提升了18.2%。消融实验显示，渐进式在线加密（POD）对性能提升关键，去除后mIoU下降明显；各向异性采样（AFS）和非对称自注意力（ASA）模块同样对提升语义精度和稳定性贡献显著。效率方面，PG-Occ在训练时间和推理速度上均优于同类方法，兼顾准确性和实时性。

### 通俗易懂  
PG-Occ的方法可以简单理解为“逐步画细节的3D拼图”。首先，它用一堆带有语义信息的“模糊球体”（高斯体积）来粗略拼出整个场景的形状和位置。接着，它会检查哪些地方细节不够清楚，就自动添加更多小球体，逐步把这些区域的细节画得更清楚。为了让这些球体之间的信息传递更合理，它设计了一种“单向交流”的机制：新加的小球可以向老球学习，但不会反过来干扰，保证已经学好的部分稳定。更特别的是，每个球体不只是一个点，而是有形状和方向的椭圆体，这样它们在采集图像特征时会根据自己的形状调整采样范围，能更准确地获取多视角信息。训练时，模型只用2D图片上的深度和语义信息来学习，不需要复杂的3D标签。这样，PG-Occ既能快速地搭建场景，也能灵活地根据用户输入的文字描述识别各种物体，实现了既高效又精准的开放词汇3D占据预测。 
## Optimized Minimal 4D Gaussian Splatting 
2025-10-04｜Yonsei, SNU, POSTECH, SKKU 

<u>http://arxiv.org/abs/2510.03857v1</u>  
<u>https://minshirley.github.io/OMG4/</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/155.jpg)  
本文提出了OMG4（Optimized Minimal 4D Gaussian Splatting），旨在解决4D高斯斑点动态场景表示中存储开销巨大的难题。4D高斯斑点技术通过在空间和时间维度联合建模，实现了复杂动态场景的实时渲染，但通常需要数百万个高斯基元，导致模型体积庞大，难以应用于实际设备。现有压缩方法虽有所改进，但在压缩率或视觉质量上仍存不足。OMG4设计了一个多阶段的高斯基元压缩流程，逐步筛选、剪枝并合并冗余基元，同时引入隐式外观编码和4D子向量量化技术，有效压缩模型尺寸。实验结果表明，OMG4在保证重建质量的前提下，可将模型大小缩减超过60%，在严格的3MB内存预算下，达到甚至超越当前最先进方法的表现，极大推动了4D动态场景紧凑表示的发展。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/156.jpg)  
OMG4的核心是一个多阶段的高斯基元优化流程，具体包括以下步骤：  

1. **高斯采样（Gaussian Sampling）**：利用静态-动态评分（SD-Score）评估每个高斯基元对渲染质量的贡献，结合空间梯度和时间梯度，筛选出对静态和动态区域均重要的基元，初步减少基元数量。  
2. **高斯剪枝（Gaussian Pruning）**：对采样结果进一步剔除静态和动态评分均低的冗余基元，通过双阈值筛选确保保留关键细节，同时在两个阶段间进行优化迭代，提升压缩稳定性。  
3. **高斯合并（Gaussian Merging）**：基于空间-时间网格，对相似度高的基元进行聚类，利用学习权重融合聚类内基元，减少重复表达，进一步压缩基元数量。  
4. **属性压缩（Attribute Compression）**：采用改进的子向量量化（SVQ）技术，结合隐式外观编码，将高维属性向量分割并量化，分阶段优化静态与动态属性，保证压缩后的视觉质量。该方法结合传统的霍夫曼编码和LZMA压缩，极大降低存储需求。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/157.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/158.jpg)  
OMG4在N3DV和MPEG数据集上进行了全面评测。结果显示，OMG4将原始Real-Time4DGS模型从约2GB压缩至3MB左右，压缩比高达99%，且视觉质量几乎无损失。与近期先进方法GIFStream相比，OMG4在存储上节省65%，同时保持甚至略微提升PSNR指标。应用于更复杂的MPEG动态场景，同样表现出优异的压缩与重建能力。此外，OMG4还成功扩展至FreeTimeGS框架，实现了90%的存储缩减，展示了高度的通用性和适应性。消融实验验证了多阶段流程的必要性，单独应用采样或剪枝效果有限，分阶段优化显著提升压缩效率与视觉表现。整体而言，OMG4实现了高效的动态场景紧凑表达，兼顾速度与质量，具备广泛应用潜力。

### 通俗易懂  
OMG4的核心思想是“聪明地挑选和合并”成千上万的高斯点，以减少模型的体积，同时保证画面效果不变。首先，系统会给每个高斯点打分，看看它在静止画面和动态变化中的重要程度，只保留那些对画面影响大的点。接着，在这批重要的点中，去掉那些既不重要又重复的点。最后，把长得很像的点合成一个“超级点”，这就像把一堆相似的拼图块拼成一个更大的块，减少了拼图的数量。除了减少点的数量，OMG4还用了一种聪明的编码方法，把每个点的颜色、透明度等信息压缩起来，像是把复杂的颜色数据拆分成小块，分别压缩，效果更好又节省空间。通过这三步：筛选重要点、去掉冗余点、合并相似点，再加上高效的数据压缩，OMG4能把原本庞大的动态场景模型缩小几百倍，既节省存储空间，又能快速渲染出高质量的动态画面。 
# Topic: 3D/4D Reconstruction｜Human 
## DHQA-4D: Perceptual Quality Assessment of Dynamic 4D Digital Human 
2025-10-04｜SJTU 

<u>http://arxiv.org/abs/2510.03874v1</u>  
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/159.jpg)  
随着3D扫描和重建技术的快速发展，动态4D数字人（即动态4D网格数字人）因其在游戏、动画制作、虚拟现实及远程沉浸式通信等领域的广泛应用，逐渐成为研究热点。然而，在采集、压缩和传输过程中，动态4D数字人网格容易受到多种噪声和失真影响，严重降低用户的观看体验。针对这一问题，本文提出了首个大规模动态4D数字人质量评估数据集DHQA-4D，包含32个高质量真实扫描的4D人类网格序列及1920个带纹理失真的网格样本和832个无纹理失真样本，涵盖11种不同类型的失真。基于此数据集，系统地分析了各种失真对纹理和几何形状感知的影响，并提出了DynaMesh-Rater，一种基于大规模多模态模型（LMM）的无参考动态4D网格质量评估方法，综合视觉、运动和几何特征，显著优于现有评估方法。该工作为动态4D数字人质量优化和相关应用提供了重要基础和工具。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/160.jpg)  
本文提出的DynaMesh-Rater框架主要包括以下几个核心模块：  

1. **视觉特征编码**：从动态4D数字人渲染生成的稀疏二维视频帧中提取视觉特征，采用预训练的视觉Transformer（InternViT），并通过多层感知机(MLP)映射至大语言模型（LLM）输入空间。  
2. **运动特征编码**：将视频均匀分割为多个片段，利用SlowFast模型提取运动信息，再通过MLP映射至LLM空间，捕捉动态变化带来的质量差异。  
3. **几何特征编码**：基于4D网格的邻接面夹角（对二面角）计算几何特征，统计其分布的均值、方差及参数化分布（广义高斯和伽马分布），反映网格表面平滑度和细节变化，通过MLP转换为LLM可接受的特征。  
4. **质量回归**：将视觉、运动和几何三类特征融合后输入大语言模型，利用低秩适配（LoRA）技术进行指令调优，实现对动态4D数字人质量的连续分数预测，替代传统离散等级评分，提升评估精度和泛化能力。  

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/161.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-10/162.jpg)  
在DHQA-4D数据集上，DynaMesh-Rater与多种传统和先进质量评估方法进行了对比。数据集被划分为带纹理和无纹理两个子集，涵盖多种几何和纹理失真。实验采用SRCC、PLCC和KRCC指标衡量相关性和预测准确度。结果显示，DynaMesh-Rater在两个子集上均取得最高性能，显著优于全参考指标（如PSNR、SSIM）和无参考图像/视频质量评估模型，尤其在捕捉动态特征和复杂失真方面表现突出。通过k折交叉验证保证了评估的鲁棒性。消融实验进一步验证了LoRA调优对视觉编码器和语言模型的贡献。失真类型分析表明，当前方法对某些纹理压缩和时间连续性失真识别能力有限，提示未来研究方向。整体实验充分证明了LMM结合多模态特征在动态4D数字人质量评估中的有效性和先进性。

### 通俗易懂  
DynaMesh-Rater的核心思想是“多角度看问题”。想象你在看一个动态的3D人物模型，不仅要看它的外观（视觉特征），还要关注它的动作是否流畅（运动特征），以及它的形状有没有变形（几何特征）。首先，我们把这个3D模型转换成一段视频，从中抽取关键帧来观察人物的细节和颜色，这就是视觉部分。接着，我们把视频分成几个小段，分析人物动作的变化，捕捉动作带来的质量变化，这就是运动部分。再者，我们测量模型表面相邻部分的夹角，了解模型表面的平滑度和细节，这就是几何部分。最后，把这三部分的信息合起来，送进一个强大的语言模型，让它根据这些信息预测这个动态模型的整体质量分数。通过这种方法，系统不仅能更准确地判断模型的好坏，还能适应各种不同类型的失真，让我们在看动态3D人物时获得更好的视觉体验。 
