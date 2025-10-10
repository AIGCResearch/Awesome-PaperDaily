# Topic: 3D/4D Generation
## Memory Forcing: Spatio-Temporal Memory for Consistent Scene Generation on Minecraft 
2025-10-03｜CUHK(SZ), Shenzhen Loop Area Institute, HKU, Voyager Research, Microsoft Research 

<u>http://arxiv.org/abs/2510.03198v1</u>  
<u>https://junchao-cs.github.io/MemoryForcing-demo/</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/32.jpg)  
本文针对基于自回归视频扩散模型的交互式场景生成，尤其是在Minecraft游戏环境中的应用，提出了Memory Forcing框架。现有模型在有限计算资源下面临时空记忆权衡：仅依赖时间记忆虽能探索新场景但空间一致性差，加入空间记忆虽提升回访一致性却可能抑制新场景生成质量。Memory Forcing通过结合几何索引的空间记忆和创新训练策略，实现了在固定上下文窗口内灵活切换时间记忆与空间记忆的依赖，兼顾新场景探索与回访一致性。该方法通过混合训练和链式前向训练引导模型根据场景状态动态调整记忆使用，同时利用点到帧检索和增量3D重建实现高效、鲁棒的空间记忆访问。大量实验表明，Memory Forcing在长时序空间一致性、生成质量和计算效率方面均优于现有方法，显著提升了Minecraft等复杂环境下的交互式视频生成能力。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/33.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/34.jpg)  
Memory Forcing方法包含三大核心组成：  

1. **混合训练（Hybrid Training）**：设计两类训练数据分布，分别模拟新场景探索和回访场景，促使模型在探索时侧重时间记忆，在回访时融合空间记忆。具体为固定上下文窗口一半采用时间上下文，另一半根据场景动态分配给空间记忆或扩展时间上下文。  
2. **链式前向训练（Chained Forward Training, CFT）**：在训练中逐步用模型预测替代真实帧，形成连续的预测窗口链，放大视角漂移，逼迫模型依赖空间记忆以保持长时序一致性，同时减少推理时误差累积。  
3. **几何索引空间记忆（Geometry-indexed Spatial Memory）**：通过增量3D重建维护全局点云，每个点关联其来源帧，实现点到帧的高效检索。仅选取关键帧更新空间记忆，避免冗余，确保检索复杂度与空间覆盖度相关而非时间长度相关。空间记忆通过跨窗口尺度对齐和3D几何信息增强的交叉注意力模块融入视频扩散模型中，提升回访场景的空间一致性。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/35.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/36.jpg)  
在Minecraft基准数据集上，Memory Forcing通过长时序记忆、泛化能力和新环境生成性能三方面进行了全面评估。与Oasis、NFD及WorldMem等基线模型比较，Memory Forcing在FVD、PSNR、SSIM和LPIPS等指标上均取得显著优势，展现出更优的视觉质量和空间一致性。定性结果显示其在回访场景中表现稳定，避免了视角漂移和几何错误。泛化测试中，模型能适应多种未见地形，生成连贯自然的场景。生成性能测试进一步验证了其在新环境下的响应性和细节表现。空间记忆检索效率实验显示，该方法相比WorldMem减少了98.2%的存储需求，检索速度提升7.3倍，且随着序列长度增长优势更明显。消融研究验证了混合训练与链式前向训练的协同作用，以及几何索引空间记忆相较传统姿态检索的性能提升，充分体现了设计的有效性和实用性。

### 通俗易懂  
Memory Forcing的核心思想是让模型学会在不同情况下“聪明地”使用两种记忆：一种是“时间记忆”，记住最近发生的事情，适合探索新地方；另一种是“空间记忆”，记住以前见过的地方的详细信息，适合回头看已经去过的地方。为了实现这个目标，研究者设计了两种训练方法。第一种叫混合训练，模型在训练时会接触两种不同类型的数据，有的数据更多靠时间记忆，有的更多靠空间记忆，让模型学会区分什么时候用哪种记忆。第二种叫链式前向训练，就是让模型自己预测未来的画面，然后用这些预测结果继续预测下去，模拟真实使用时的情况，帮助模型学会纠正自己预测中的错误，更好地依赖空间记忆保持场景稳定。空间记忆通过构建三维地图来实现，模型可以快速找到当前视角对应的历史画面，避免重复存储和计算。这种方法让模型既能自由探索新场景，又能保证回访时画面连贯自然，效果比以前的模型更好，运行也更快。 
## Towards Scalable and Consistent 3D Editing 
2025-10-03｜ECUST, SMU 

<u>http://arxiv.org/abs/2510.02994v1</u>  
<u>https://www.lv-lab.org/3DEditFormer/</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/43.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/44.jpg)  
本研究聚焦于3D编辑技术，旨在实现对3D资产几何形状或外观的局部、精准修改，满足沉浸式内容创作、数字娱乐及AR/VR等应用需求。传统2D编辑技术难以直接迁移至3D领域，因其需保证多视角一致性、结构完整性及细粒度控制，现有方法多存在速度慢、易产生几何畸变或依赖人工3D掩码等缺陷，限制了实际应用。为此，本文提出两大突破：一是构建了3DEditVerse，这是迄今最大规模、质量高、具备多视角一致性和语义对齐的配对3D编辑数据集，涵盖11.6万训练对和1500测试对，编辑涵盖姿态驱动的几何变化和基于基础模型的外观修改；二是设计了3DEditFormer，一种结构保留的条件Transformer模型，通过双重引导注意力机制和时间自适应门控，有效分离可编辑区域与保持结构，支持无需手工掩码的高效、精确、跨视角一致的3D编辑。实验结果显示，该框架在多项3D和2D指标上均显著优于现有先进方法，推动了3D编辑技术向实用化和规模化迈进。  

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/45.jpg)  
本文方法由两部分组成：  

1. **3DEditVerse数据集构建**：采用两条互补流水线生成配对数据。  
  - 姿态驱动几何编辑：利用公开3D角色及动画序列，选取多样化姿态，通过嵌入去重确保多样性，形成5.4万对结构变化数据。  
  - 文本引导外观编辑：基于4,585词汇表，利用多个基础模型（DeepSeek-R1生成多样文本提示，Flux合成源图像，Qwen-VL自动生成编辑指令及区域，Trellis进行3D重建），结合多视角掩码投影和重绘策略，实现64,123对外观编辑数据生成。  
2. **3DEditFormer模型设计**：基于Trellis图像到3D生成框架，提出三大创新：  
  - 双重引导注意力模块，设有两个跨注意力分支，分别关注晚期扩散步骤的细粒度结构特征和早期步骤的语义过渡特征；  
  - 多阶段特征提取，分别从不同扩散时间截取源3D资产的结构和语义信息；  
  - 时间自适应门控机制，根据扩散时间动态调整两类特征权重，早期强调语义转变，后期强化结构保留。  
该设计实现了编辑与未编辑区域的有效区分，保证了编辑的局部性和结构一致性，无需人工3D掩码辅助。  

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/46.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/47.jpg)  
实验部分在3DEditVerse测试集上，采用多维3D几何指标（Chamfer距离、法线一致性、细节保留F1分数）及2D渲染指标（PSNR、SSIM、LPIPS、DINO-I语义一致性）进行评估。定性对比显示，3DEditFormer在保持结构完整性和纹理细节方面明显优于EditP23、Instant3dit和依赖掩码的VoxHammer，后者对掩码精度高度敏感，易导致编辑失真。量化结果表明，3DEditFormer在无需掩码的条件下，3D指标整体提升13%以上，且在多数2D指标中表现最佳。消融实验验证了双重引导注意力和时间自适应门控的有效性，逐步提升了模型的结构保留和编辑准确性。整体来看，3DEditFormer结合3DEditVerse数据集，展现了高效、精准且实用的3D编辑能力，显著优于现有技术。  

### 通俗易懂  
这项研究开发了一种让3D模型“动起来”更简单的方法。想象你有一个3D玩偶，想给它戴帽子或者换颜色，传统方法要么很慢，要么改了帽子却把玩偶身体也变形了，或者需要你手动标记要修改的部分，非常麻烦。研究人员做了两件事：第一，收集了大量“前后对比”的3D模型数据，告诉电脑“原来是这样，改完是这样”，这些数据既包括姿势变化，也包括颜色和细节的变化。第二，设计了一个聪明的模型，叫3DEditFormer，它能同时看懂“原始模型”和“想要的样子”，并且知道哪些地方需要改，哪些地方要保持不变。它通过两种不同的“注意力”机制，一边关注细节结构，一边理解整体语义，再用一个智能开关根据不同阶段调整这两种信息的重要性。这样，模型就能精准地在3D空间里局部修改，保持其他部分完好，而且不需要你手动帮它标记修改区域。简单来说，这就像给3D模型做局部整形手术，既快又精准，还省力。 
# Topic: 3D/4D Reconstruction
## ROGR: Relightable 3D Objects using Generative Relighting 
2025-10-03｜Google, Google DeepMind, TUM｜NeurIPS 2025 

<u>http://arxiv.org/abs/2510.03163v1</u>  
<u>https://tangjiapeng.github.io/ROGR</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/37.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/38.jpg)  
本文提出了ROGR，一种创新的3D物体重建与可重光照渲染方法。该方法基于生成式重光照模型，能够从多视角拍摄的、未知光照条件下的图像中，重建出可在任意环境光照下实时渲染的神经辐射场（NeRF）。与传统的逆渲染方法不同，ROGR通过生成多视角、多光照条件下的图像数据集，训练一个光照条件化的NeRF，实现对新光照环境的快速响应，无需针对每种光照进行单独优化。该方法在TensoIR和Stanford-ORB等合成及真实数据集上均表现出优越的性能，尤其在捕捉高光和复杂反射方面显著优于现有技术，且具备高效的实时渲染能力，适用于电影和游戏等需要动态光照变化的场景。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/39.jpg)  
ROGR方法包含两个核心步骤：  

1. **多视角重光照扩增数据集构建**：利用多视角扩散模型对原始多视角图像进行重光照生成，输入为原始图像及其相机姿态和目标环境光照，输出视角一致且光照变化丰富的图像集合。该扩散模型基于Latent Diffusion Model，采用跨视角注意力机制保证视角一致性，环境光照通过高动态范围（HDR）和低动态范围（LDR）编码融合。  
2. **光照条件化神经辐射场训练**：将生成的多光照、多视角图像作为监督数据，训练一个基于NeRF-Casting的神经辐射场。该NeRF采用双分支结构，分别编码一般光照信息和镜面反射光照，后者通过多尺度高斯模糊预滤波环境贴图实现高频镜面高光的准确捕捉。训练时，环境光照通过Transformer编码器转换为紧凑的128维向量，作为条件输入，使模型具备对任意新光照环境的泛化能力，实现无需额外优化的即时重光照渲染。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/40.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/41.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/42.jpg)  
实验部分在多个公开数据集上验证了ROGR的性能。首先，在合成的TensoIR数据集上，ROGR在PSNR、SSIM和LPIPS等指标上均超过了包括IllumiNeRF和NeuralGaffer在内的多种先进方法，特别在复杂镜面反射的“hotdog”和“ficus”场景中表现卓越。其次，在真实捕获的Stanford-ORB数据集上，ROGR同样取得最高的PSNR和SSIM分数，特别适合反射性强的物体。实验还展示了ROGR在自然光照下的实物重光照效果，成功还原了金属和木质材料的高光和阴影细节。消融研究表明，多尺度镜面光照编码和环境光照高分辨率输入对模型性能提升关键，多视角联合训练也显著增强了视角一致性和渲染质量。方法实现了约0.38秒每帧的实时渲染速度，兼顾质量与效率。

### 通俗易懂  
ROGR的核心思想是让计算机“学习”如何在不同光照条件下看同一个物体。首先，我们用一种叫扩散模型的AI技术，给物体拍摄的多张照片“换灯光”，生成很多不同光照下的照片，但这些照片的视角和物体形状保持一致。接着，我们用这些丰富的照片训练一个神经网络（NeRF），这个网络不仅学会了物体的三维形状，还学会了光线如何在物体表面反射和折射。训练好的网络就像一个万能的虚拟摄影师，能在任何光照条件下快速渲染出物体的样子，无需重新训练。为了更好地表现镜面反射的高光，我们还特别设计了双重光照编码，一个负责整体光照，一个专门处理镜面高光，保证渲染细节丰富。这样一来，无论光源怎么变，物体看起来都真实自然，还能快速生成新视角的图像，方便用于电影特效或游戏中动态光照的需求。 
# Topic: Motion Generation｜AAA10.6 
## MoGIC: Boosting Motion Generation via Intention Understanding and Visual Context 
2025-10-03｜HKUST(GZ) 

<u>http://arxiv.org/abs/2510.02722v1</u>  
<u>https://github.com/JunyuShi02/MoGIC</u> 
### 概述 
  
本文提出了MoGIC，一种统一的多模态人类动作生成框架，旨在解决现有文本驱动动作生成方法中忽视动作背后因果逻辑和人类意图建模的不足。传统方法多将动作生成视为语言与动作的单向或双向映射，难以捕捉动作执行的内在动机与细节，且缺乏视觉信息的辅助导致生成动作缺乏精确性和个性化。MoGIC通过联合优化动作生成与意图预测，揭示潜在人类目标，融合视觉先验信息，增强生成的多样性和可控性。为支持该框架，作者构建了包含21个高质量数据集、总计440小时动作数据的Mo440H基准。实验结果表明，MoGIC在HumanML3D和Mo440H数据集上分别实现了38.6%和34.6%的FID显著降低，且在动作描述生成任务中优于基于大型语言模型的方法，进一步拓展了视觉条件下动作补全和多模态条件生成的能力。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/48.jpg)  
MoGIC框架主要包括以下核心模块：  

1. **多模态编码器**：分别对动作序列、文本描述和视觉帧进行编码，动作采用时序卷积自编码器降维，文本利用冻结的CLIP编码器，视觉通过低帧率图像序列编码并通过注意力机制聚合成全局视觉表示。  
2. **条件掩码变换器（CMT）**：核心模块，融合多模态条件信息进行动作潜空间的掩码建模。其包含两层调制机制：语义级别的自适应层归一化调制，确保全局多模态上下文一致性；以及混合注意力机制，动态选择最相关的文本和视觉片段，实现局部与全局的精细对齐。  
3. **解耦生成头**：分为意图预测头和动作生成头。意图预测头基于T5解码器，生成描述动作背后目标的文本；动作生成头采用基于扩散模型的连续时间插值器，生成连续的动作轨迹。  
4. **联合训练策略**：通过五大任务（语言到动作、视觉语言到动作、视觉到动作、动作到动作、意图预测）联合优化，结合扩散模型的速度场匹配损失和意图预测的交叉熵损失，实现动作与意图的协同学习。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/49.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/50.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/51.jpg)  
作者构建了大规模的Mo440H数据集，涵盖单人动作、人与人交互及人与物体交互，配备文本描述和视觉帧，确保多模态训练的丰富性。实验在HumanML3D和Mo440H上进行，结果显示MoGIC在文本驱动动作生成任务中显著优于多种先进方法，FID指标降低超过30%，R-Precision和匹配度均有提升。意图预测任务中，MoGIC能够准确推断动作背后的高层目标，且生成的动作与预测意图高度一致。视觉模态的引入极大提升了动作生成的精确性和可控性，特别是在视觉语言联合条件下，模型能生成更符合现实环境约束的动作序列。动作补全任务中，MoGIC同样表现优异，无需专门微调即可实现对缺失动作片段的合理补全。此外，轻量级的意图预测头在动作描述生成上也超过了依赖大型语言模型的基线，展示了高效且有效的多模态学习能力。

### 通俗易懂  
MoGIC就像一个聪明的“动作导演”，不仅根据文字描述指导演员做动作，还能理解演员的“动机”和“目标”，比如知道为什么要做这个动作。它不仅听文字，还能看视频，知道场景里有什么东西，这样动作就更真实、更符合实际。它先把文字、视频和动作信息变成“数字语言”，然后用一个特别的“注意力机制”去找出哪些文字和视频片段对当前动作最重要，确保动作和描述紧密匹配。接着，模型分两步生成结果：一部分负责预测动作背后的意图（告诉你演员想做什么），另一部分负责生成具体动作（告诉你怎么做）。同时，模型在训练时同时学习理解意图和生成动作，让动作更自然、准确。简单来说，MoGIC就像既能听懂导演讲故事，又能看现场环境，还能指导演员做出合适动作的智能系统。 
