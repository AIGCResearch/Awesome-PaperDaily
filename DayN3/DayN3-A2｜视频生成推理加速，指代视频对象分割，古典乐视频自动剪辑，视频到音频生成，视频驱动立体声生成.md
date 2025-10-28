# Topic: Video Generation 
## LightCache: Memory-Efficient, Training-Free Acceleration for Video Generation 
2025-10-06｜Tulsa, Clemson , Arizona, NEU, Microsoft 

<u>http://arxiv.org/abs/2510.05367v1</u>  
<u>https://github.com/NKUShaw/LightCache</u> 
### 概述 
![](https://img.blenet.top/file/paper-daily/aigc_research/0c196a52387649849f6539ff76624e0f.png)  
本文针对基于扩散模型的视频生成中高计算时延和显存消耗的问题，提出了一种训练无关的加速框架LightCache。扩散模型推理过程中，潜变量的冗余为加速提供了天然切入点，但现有基于缓存的加速方法在去噪和解码阶段会引发显存激增，限制了实际应用。作者将推理过程拆分为编码、去噪和解码三个阶段，发现显存峰值主要集中在后两者。为此，提出针对不同阶段的显存优化策略，既保证了加速效果，也有效控制了显存使用，且生成质量在可接受范围内。实验表明，LightCache在多种视频扩散模型上实现了显著的推理速度提升和显存节省，优于现有的DeepCache和FME方法。该方法无需额外训练，兼容多种采样调度器，且代码开源，便于后续研究与应用推广。

### 方法 
![](https://img.blenet.top/file/paper-daily/aigc_research/df2354fcf6fd401a9df6d6a534f4aeb1.png) 
![](https://img.blenet.top/file/paper-daily/aigc_research/9ae429d17c3f498bb433d2da98c85f46.png) 
![](https://img.blenet.top/file/paper-daily/aigc_research/d04553d76a2e460b8d9f5606ba2a2c24.png)  
LightCache基于对扩散模型推理流程的深入分析，设计了三大核心策略：  

1. **异步缓存交换（Asynchronous Cache Swapping）**：借鉴模型权重CPU卸载技术，将缓存的特征图在GPU和CPU间异步交换，避免GPU显存峰值激增，同时通过后台传输减少交换带来的延时。  
2. **特征分块（Feature Chunk）**：针对去噪阶段特征图空间维度（高、宽）进行分块处理，减小单次处理的显存占用，平衡内存节省与生成质量。  
3. **切片解码（Sliced Decoding）**：解码阶段将视频帧视为多张图片，逐帧解码，避免一次性处理所有帧导致显存暴涨，适配多帧同时处理的条件无条件引导模型。  
这三种策略分别针对推理的去噪和解码阶段，显著降低了显存峰值，并且整体时间开销小于加速收益，确保加速效果。框架兼容多种扩散模型架构（如U-Net和DiT），且无需修改模型结构或重新训练。

### 实验 
![](https://img.blenet.top/file/paper-daily/aigc_research/a160bafc015c4929b17c96417d631892.png)  
实验在多款主流视频扩散模型（如AnimateDiff-Light、Stable-Video-Diffusion-Img2vid-XT、EasyAnimate）上进行，使用4块NVIDIA L40 GPU，分辨率512×512及1024×576。结果显示，LightCache在保持生成质量（LPIPS、PSNR、SSIM指标稳定）的同时，实现了1.3到2.9倍的推理速度提升，显存峰值较基线及DeepCache减少8GB以上。相比之下，DeepCache虽加速明显但显存激增，FME虽降低显存但速度反而下降且质量受损。消融实验进一步验证了三大策略各自对显存优化的贡献，且方法对不同采样调度器均表现稳定。视觉对比展示了LightCache生成视频与原模型高度一致，证明其有效性与实用价值。

### 通俗易懂  
LightCache的核心思想是“聪明地管理和利用计算资源”，让视频生成更快、更省显存。它把生成过程分成三个阶段：编码、去噪和解码。  
- 在去噪阶段，模型会生成很多“中间画面”，这些画面有很多相似的内容。LightCache把这些画面分成小块，逐块处理，避免一次处理太大数据占用太多显存。  
- 同时，它会把暂时不用的画面数据“搬”到电脑的内存（CPU），需要时再“搬”回来，这样GPU显存就不会被塞满。这个搬运过程是后台悄悄进行，不会拖慢生成速度。  
- 在解码阶段，视频其实是很多连续的图片，LightCache不一次性解码所有图片，而是一张一张来，避免显存爆满。  
这样一来，视频生成既快又省显存，不用重新训练模型，直接用现有模型就能享受加速效果，非常实用。 
## Deforming Videos to Masks: Flow Matching for Referring Video Segmentation 
2025-10-07｜SGITAI Lab, UCSD, HKUST, Tokyo, Cambridge, ZJUT, Baidu 

<u>http://arxiv.org/abs/2510.06139v1</u>  
<u>https://github.com/xmz111/FlowRVS</u> 
### 概述 
![](https://img.blenet.top/file/paper-daily/aigc_research/0db8ea143eec4549aa637de47963dbb9.png)  
本文针对指代视频对象分割（Referring Video Object Segmentation，RVOS）任务，提出了一种全新的框架FlowRVS。RVOS的核心难点在于如何将抽象的语言描述准确地映射到视频中的像素，并保持时间上的连续一致性。传统方法多采用“先定位再分割”的两阶段流水线，这种设计会导致信息瓶颈，语言语义被简化为粗糙的几何提示，且分割过程与语言定位解耦，难以维持时序连贯。FlowRVS创新地将RVOS问题重新定义为一个条件连续流（conditional continuous flow）问题，通过学习从视频整体表示到目标掩码的语言引导形变，实现端到端的联合建模，避免信息丢失。该方法利用预训练的文本到视频生成模型（Text-to-Video，T2V）的细粒度像素控制、语义对齐和时序推理能力，显著提升了复杂语言和动态视频场景下的分割表现，刷新了多个RVOS基准测试的性能纪录。

### 方法 
![](https://img.blenet.top/file/paper-daily/aigc_research/f7c88fa76d234466a6419a141d4a8d32.png) 
![](https://img.blenet.top/file/paper-daily/aigc_research/47464aa1d7a9465fa9c8f5cdd5779170.png)  
FlowRVS的核心创新是将RVOS视为一个语言条件下的视频到掩码的连续形变过程，借助常微分方程（ODE）学习一个速度场，逐步将视频潜在表示变形为对应的掩码潜在表示。具体方法包括：  

1. **边界偏置采样（Boundary-Biased Sampling，BBS）**：针对流的起点（视频潜在表示）进行过采样，强化模型对初始速度的学习，保证语言条件下的精确定位，防止早期误差导致轨迹失败。  
2. **起点增强（Start-Point Augmentation，SPA）**：通过随机扰动视频潜在表示，扩展起点分布，提升模型对邻近状态的鲁棒性和泛化能力。  
3. **直接视频注入（Direct Video Injection，DVI）**：在流的每一步将原始视频潜在表示与当前状态拼接，确保全程保持丰富的上下文信息，防止轨迹漂移，提升细节分割精度。  
此外，FlowRVS针对预训练VAE解码器进行专门微调以适应掩码重建，且训练中冻结文本编码器和VAE编码器，仅微调Diffusion Transformer模块，实现高效且稳定的训练。该方法避免了传统生成模型从噪声到掩码的宽泛探索，转而聚焦于从高维视频到低维掩码的确定性收敛映射，充分发挥T2V模型的多模态推理优势。

### 实验 
![](https://img.blenet.top/file/paper-daily/aigc_research/c7d849f55f974b9d953ca3033acb8a05.png) 
![](https://img.blenet.top/file/paper-daily/aigc_research/de05d0a43a1b44219bd48a3047f622c5.png)  
FlowRVS在MeViS、Ref-YouTube-VOS和Ref-DAVIS17三个主流RVOS基准上进行了广泛评测。结果显示，FlowRVS在复杂运动场景集中的MeViS上取得J&F指标51.1，较前一最佳方法提升1.6个百分点，显著展现其对动态语言和视频理解的优势。在大规模Ref-YouTube-VOS数据集上也表现优异，且在Ref-DAVIS17数据集实现了73.3的零样本迁移J&F分数，超越多种需微调方法，体现出强大的泛化能力。消融实验验证了边界偏置采样、起点增强和直接视频注入三大策略的关键作用，尤其是边界偏置采样对稳定训练和性能提升贡献最大。对比不同范式，单步掩码预测和噪声到掩码的生成方法均表现较差，证明FlowRVS多步视频到掩码流的设计更适合RVOS任务。此外，预训练T2V模型权重的引入对性能至关重要，去除预训练导致性能崩溃，强调了大规模生成模型的基础价值。

### 通俗易懂  
FlowRVS的核心思想是把视频里的目标分割看成一个“变形过程”，就像把一张复杂的动画图片慢慢变成一个清晰的目标轮廓。传统方法先用语言找到大致位置，再去分割，信息会丢失，导致效果不理想。FlowRVS直接让模型学会根据语言描述，逐步把整个视频画面“变形”为目标物体的掩码。这个变形过程是连续的，每一步都会根据视频和语言的提示，调整形状，直到准确分割出目标。为了让这个过程稳定有效，FlowRVS特别强调从视频原始信息开始的第一步很关键，会多次训练模型在这一步做出准确判断，同时在变形过程中不断把原始视频信息注入进去，防止“走偏”。此外，还会让模型学会对视频信息做一些小扰动，增强对不同情况的适应能力。这样，FlowRVS就像一个聪明的雕塑家，从复杂的动态视频中雕刻出符合语言描述的目标形状，效果比传统方法更准确、更连贯。 

# Topic: Video & Audio Generation 
## When and How to Cut Classical Concerts? A Multimodal Automated Video Editing Approach 
2025-10-07｜UOC, UB 

<u>http://arxiv.org/abs/2510.05661v1</u>  
### 概述 
![](https://img.blenet.top/file/paper-daily/aigc_research/57f59fc24832447ca61973054ac5c603.png)  
本研究聚焦于古典音乐多机位演奏会视频的自动剪辑，解决视频编辑中“何时切换镜头”和“如何切换镜头”两个核心子任务。相比于现有视频生成和场景理解的研究，自动视频编辑尤其是多机位剪辑仍较少被深入探讨。作者提出了一种多模态深度学习架构，结合音频的对数梅尔频谱图、时间标量特征及视觉嵌入，旨在准确定位切换点并选择最佳摄像机视角。为支撑模型训练，团队构建了包含100个古典音乐会视频的伪标签数据集，利用自动聚类和语义验证生成高质量的切换点标注。实验结果显示，所提模型在“何时切换”任务上相较于统计基线提升了10个百分点以上，且在“如何切换”任务中也优于传统的ResNet和Xception视觉特征提取方法，显著推动了多模态自动视频编辑领域的发展。

### 方法 
![](https://img.blenet.top/file/paper-daily/aigc_research/17e731523eb744efaa3d34de41998e5e.png) 
![](https://img.blenet.top/file/paper-daily/aigc_research/2d5d910c94394f378543c2b34a3865b7.png) 
![](https://img.blenet.top/file/paper-daily/aigc_research/5187e37b026f4326a7576edf938472fe.png)  
本研究将自动视频编辑拆解为两个关键任务：  

1. **何时切换（Temporal Segmentation）**：将多机位录制的视频划分为若干时间段，识别出最合适的镜头切换时刻。该任务被视作一个二分类问题，输入包括音频的对数梅尔频谱图和自上次切换以来经过的时间标量，采用轻量级卷积块与Transformer层捕捉时序和频率特征。多模态版本进一步引入视觉信息，通过CLIP模型提取切换前最新帧的视觉嵌入，增强语义理解。  
2. **如何切换（Spatial Selection）**：在检测到切换时刻后，从多个摄像机视角中选出最佳镜头。此任务视为多分类匹配问题，利用CLIP提取每个候选镜头帧的视觉特征，采用注意力机制计算与当前镜头的相关性，选出最合适的下一个镜头。  
此外，数据集构建采用伪标签方法，结合传统像素差异检测、CLIP相似度阈值及大型语言模型（Gemini1.5Flash）验证，确保切换点的语义准确性。该多阶段标签流程保证了训练数据的质量和多样性。

### 实验 
![](https://img.blenet.top/file/paper-daily/aigc_research/50221ad66f484b0abea39d9a541742e3.png)  
实验分为“何时切换”和“如何切换”两个部分。针对“何时切换”，作者设计了基于Poisson和指数分布的统计基线，并与提出的深度学习模型进行了对比。结果显示，深度模型在准确率、召回率和F1分数上均显著优于统计基线，且单模态音频模型在召回率和F1分数表现最佳，多模态模型则在精准率上略胜一筹。ROC曲线进一步验证了模型的稳定性和泛化能力。  
在“如何切换”任务中，采用Recall@1和Recall@3指标评估不同视觉特征提取器的性能。CLIP视觉嵌入明显优于传统ResNet和Xception模型，Recall@1达28.49%，Recall@3达51.97%。定性分析展示了模型在不同场景下的镜头选择能力，体现了其对语义信息的敏感度。尽管模型表现优于基线，但仍存在因视角相似而判定为错误的情况，未来计划引入相似度加权评价以更合理衡量模型性能。

### 通俗易懂  
这项研究解决了在古典音乐会视频中，如何自动决定“什么时候换镜头”和“换成哪个镜头”的问题。首先，系统会听音乐会的声音，把声音转换成一种叫做“梅尔频谱图”的图像，类似声音的指纹。同时它会记录自上次换镜头以来经过了多久。然后，系统用一种叫Transformer的智能算法来分析这些声音信息，判断是不是该换镜头了。为了让系统更聪明，还可以给它看当前镜头的画面，提取画面里的重要特征，这样它可以更好地理解现场情况。  
当确定了需要换镜头的时刻，系统就会从所有摄像机拍摄的画面里挑选出最合适的一个。它会比较每个摄像头画面的特征，选择与当前画面内容最相关且视觉效果最好的镜头。为了训练系统，研究者们收集了很多音乐会视频，利用自动方法标记出镜头切换的位置，并通过智能模型确认这些切换是否合理。这样，系统学会了如何像专业剪辑师一样，合理地切换镜头，使视频更流畅和有表现力。 
## FoleyGRAM: Video-to-Audio Generation with GRAM-Aligned Multimodal Encoders 
2025-10-07｜Sapienza｜IJCNN 2025 

<u>http://arxiv.org/abs/2510.05829v1</u>  
### 概述 
![](https://img.blenet.top/file/paper-daily/aigc_research/ed86c4f3ec584d32b7d7325e032e3f9c.png)  
本文提出了FoleyGRAM，一种创新的视频到音频生成方法，核心在于通过GRAM（Gramian Representation Alignment Measure）实现多模态编码器的对齐，确保视频、文本和音频三种模态的语义嵌入在统一的潜在空间中高度一致。该方法基于扩散模型进行音频合成，结合GRAM对齐的语义嵌入和波形包络信息，实现生成音频在语义和时间上的双重匹配。相比现有方法，FoleyGRAM不仅解决了多模态编码器训练分离导致的潜在空间不对齐问题，还提升了生成音频与视频内容的语义一致性。通过在标准的Greatest Hits数据集上的评测，FoleyGRAM在语义对齐和音频质量上均优于当前最先进的视频到音频生成模型，展示了其在影视音效设计和多媒体体验中的广泛应用潜力。

### 方法 
![](https://img.blenet.top/file/paper-daily/aigc_research/8ab075b2e2944253bb7da3ea80c62dd9.png) 
![](https://img.blenet.top/file/paper-daily/aigc_research/08999c8c1c56475fbd12e0e13c458bbe.png)  
FoleyGRAM方法主要包括以下几个关键组成部分：  

1. **多模态编码器对齐（GRAM）**：采用GRAM度量，通过计算视频、文本和音频三模态嵌入构成的高维平行多面体体积，实现三者潜在空间的联合对齐，避免传统基于成对余弦相似度的局限，确保所有模态嵌入在统一空间中语义一致。编码器分别为EVAClip-ViT-G（视频）、BERT-B（文本）和BEATs（音频），先在大规模多模态数据集上预训练，后在目标数据集上微调。  
2. **扩散式音频生成模型**：基于Stable Audio的潜在扩散模型，利用GRAM对齐的多模态嵌入作为条件输入，通过交叉注意力机制引导生成过程，确保语义控制的精确性。  
3. **时间同步机制**：引入音频包络信息作为辅助条件，通过ControlNet结构处理包络特征，实现生成音频的时间动态与输入视频动作同步，保证事件发生的时序准确。  
4. **联合训练与推理**：模型在训练阶段联合优化编码器和生成器，推理时支持多模态联合或单一模态条件输入，灵活控制生成音频的语义特征，满足多样化应用需求。

### 实验 
![](https://img.blenet.top/file/paper-daily/aigc_research/6b54276ad13f4887a93fc05eeee2deb9.png) 
![](https://img.blenet.top/file/paper-daily/aigc_research/15682af39e7a4988809f3f69b0a4723f.png)  
实验基于Greatest Hits数据集展开，该数据集包含多样的击打和摩擦动作视频，配有高质量的44.1kHz立体声音频及文本描述。训练过程中，FoleyGRAM在单GPU环境下以20,000步训练，采用AdamW优化器，利用预训练编码器进行初始化。评估指标涵盖Frechet Audio Distance（FAD，使用不同音频编码器计算）、CLAP分数（音频语义相似度）及Frechet Audio-Visual Distance（FAVD，衡量音视频对齐度）。对比五个公开基线模型（SpecVQGAN、CondFoleyGen、Diff-Foley、SyncFusion、Video-Foley），FoleyGRAM在所有指标上均表现最佳，特别在语义一致性和时间同步方面显著领先。消融实验显示，联合使用视频、文本和音频三模态条件能最大化生成效果，单一模态条件则效果较差。整体结果验证了GRAM对齐潜在空间和结合时间包络控制对提升生成音频质量和语义准确性的关键作用。

### 通俗易懂  
FoleyGRAM的核心想法是让电脑“听懂”视频里的声音应该是什么样的。它通过一种叫GRAM的技术，把视频、文字描述和声音这三种信息放到一个“共同语言”里，这样电脑就能更好地理解它们之间的联系。然后，电脑用一个叫扩散模型的生成方法，结合这三种信息，慢慢“画”出符合视频内容的声音。为了让声音和视频里的动作时间上同步，FoleyGRAM还会分析视频声音的包络，也就是声音的强弱变化曲线，确保声音不会提前或延后出现。训练时，模型学会如何把视频和文字里的信息转化为声音的特征，生成的声音不仅听起来自然，还和视频内容匹配得很准确。简单来说，FoleyGRAM就是教电脑用视频和文字的线索，精准地“配音”，让生成的声音既有意义又和动作同步，适合电影和游戏等需要逼真音效的场景。 
## StereoSync: Spatially-Aware Stereo Audio Generation from Video 
2025-10-07｜Sapienza, Sony｜IJCNN 2025 

<u>http://arxiv.org/abs/2510.05828v1</u>  
### 概述 
![](https://img.blenet.top/file/paper-daily/aigc_research/c35fc02209fe422894e47b1d93f12282.png)  
本文提出了StereoSync，一种创新的视频驱动立体声生成模型，能够生成与视频内容在时间和空间上高度同步且空间感知的音频。当前视频到音频（V2A）生成领域主要关注语义和时间同步，而忽视了空间维度，导致生成的声音缺乏沉浸感和空间连贯性。StereoSync通过引入深度图和目标边界框作为空间线索，结合多模态特征和扩散模型，实现了音频的空间对齐和时序同步。该方法依托预训练基础模型，极大降低了训练成本，同时保持高质量音频合成。实验在Walking The Maps数据集上验证，显示StereoSync在空间和时间对齐方面显著优于无空间条件版本，提升了生成音频的真实感和沉浸感，推动了视频到音频生成技术的前沿。

### 方法 
![](https://img.blenet.top/file/paper-daily/aigc_research/7a10154ee63949b2b8b2d573df5c0ede.png)  
StereoSync方法主要分为三个核心步骤：  

1. **空间特征提取**：利用RollingDepth模型提取视频的深度图，捕获场景几何信息，保证时序一致性；使用MASA框架结合Segment Anything Model（SAM）进行对象跟踪，获得精确的边界框数据，反映动态对象位置。  
2. **多模态条件编码**：将深度图通过EVAClip-ViT-G编码器生成全局空间嵌入，边界框数据用Stable Audio的NumberConditioner编码，语义信息由CLAP音频编码器提取，三者作为跨注意力机制的条件输入。  
3. **扩散音频生成**：基于Stable Audio的潜在扩散模型，结合ControlNet结构利用音频能量包络实现时间同步。模型冻结预训练权重，仅微调条件投影层和ControlNet，保证轻量高效。通过融合空间、语义和时间条件，生成与视频视觉内容空间一致、语义匹配且时间同步的立体声音频。

### 实验 
![](https://img.blenet.top/file/paper-daily/aigc_research/e951c1ed53c64aeb915243d61a5dc52b.png) 
![](https://img.blenet.top/file/paper-daily/aigc_research/5a17e1e69e9248b98a5a63bb1effce89.png)  
实验采用Walking The Maps数据集，包含多款高质量视频游戏中的步行动作视频，配备清晰的步行动作音频。评估指标涵盖音频质量（FAD）、视频音频语义时间对齐（FAVD、E-L1）及空间同步度（Spatial AV-Align）。结果显示，StereoSync在空间对齐指标上显著优于无空间条件版本（0.78 vs. 0.61），接近真实音频的空间表现（0.89），且保持语义和时间对齐的稳定性。定量指标表明引入空间条件不会损害音质和时间同步。实验还展示了模型利用预训练基础模型实现零样本泛化能力，且训练过程轻量快速。整体验证了StereoSync在生成空间感知立体声音频方面的有效性和实际应用潜力。

### 通俗易懂  
StereoSync的核心思想是让电脑根据视频画面“听”到的声音不仅在时间上和画面同步，还能感觉到声音是从哪里发出来的，就像我们用两只耳朵听声音一样有方向感。它先用两个聪明的工具，一个帮它看出画面里物体的远近（深度图），另一个帮它准确找到画面中移动的物体（边界框）。然后，把这些信息转成电脑能懂的数字信号，再结合视频中声音的语义内容，一起输入到一个叫扩散模型的声音生成器里。这个模型就像一个音乐家，能根据这些提示，创作出既符合画面内容又有空间感的立体声音。这样，我们听到的声音不仅时间对得上，听起来还像真的从画面中物体的位置传出来，让观影体验更加真实和沉浸。这个方法还用了预先训练好的模型，省去了大量训练时间，效率很高。 