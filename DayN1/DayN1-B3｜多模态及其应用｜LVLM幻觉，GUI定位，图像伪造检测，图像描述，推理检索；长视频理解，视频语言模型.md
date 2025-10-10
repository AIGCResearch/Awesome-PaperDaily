# Topic: Multi-modal
## MaskCD: Mitigating LVLM Hallucinations by Image Head Masked Contrastive Decoding 
2025-10-03｜​THU SIGS

<u>http://arxiv.org/abs/2510.02790v1</u>  
<u>https://github.com/Deng-Jingyuan/MaskCD</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/106.jpg)  
本文针对大型视觉语言模型（LVLM）在多模态任务中表现出的幻觉问题，即模型生成与输入图像和文本内容不符的错误信息，提出了一种新颖的无训练成本方法——图像头遮蔽对比解码（MaskCD）。幻觉现象严重影响模型的实用性和可信度，现有解决方案主要分为训练相关和无训练两类，前者资源消耗大，后者稳定性和效果有限。MaskCD通过观察模型内部注意力机制，发现部分注意力头（称为“图像头”）对图像信息关注过度。基于此，MaskCD在解码过程中遮蔽这些图像头，构造高质量的对比样本，进而有效抑制幻觉生成。实验结果表明，MaskCD在多个主流评测基准（CHAIR、POPE、AMBER、MME）上均显著优于现有方法，同时保持了模型的整体能力，展现出较好的稳定性和实用价值。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/107.jpg)  
MaskCD方法核心包括以下几个步骤：  

1. **图像头识别**：通过统计模型在生成每个词时各注意力头对图像token的关注度，设定阈值τ筛选出关注度较高的头，称为“图像头”。这些头被认为包含对图像信息的关键关注。  
2. **构造图像头遮蔽掩码**：在推理时，将这些图像头对应的注意力输出置零，实质上“遮蔽”了这些头对图像信息的访问，但不改变模型参数。  
3. **对比解码策略**：利用原始输入和遮蔽图像头后的“损伤”输入分别进行推理，计算两者输出的logits差异，作为对比信号调整最终生成概率。超参数α控制对比强度。  
4. **平衡效果与稳定性**：通过合理选择阈值τ和对比强度α，MaskCD能有效去除幻觉信息而不损害模型正常的视觉理解能力。  
该方法结合了对注意力机制的深度理解与对比解码的优势，构造了更纯粹的负样本，提升了对比解码的效果和稳定性。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/108.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/109.jpg)  
实验部分选用两种主流LVLM：LLaVA-1.5-7b和Qwen-VL-7b，覆盖多种视觉语言任务。评测基准包括CHAIR（图像描述幻觉评估）、POPE（对象识别问答）、AMBER（多维幻觉评测）和MME（多模态综合评测）。结果显示，MaskCD在减少幻觉比例、提高准确率和F1分数方面均优于传统对比解码（VCD、M3ID）和注意力操控（OPERA）方法。特别是在CHAIR指标中，MaskCD在LLaVA-1.5-7b上减少幻觉比例近20%，Qwen-VL-7b上减少超过50%。消融实验进一步验证了图像头遮蔽的必要性和超参数设置的合理性。随机遮蔽非图像头效果较差，说明图像头确实承载了关键视觉信息。整体来看，MaskCD在保证推理效率的同时，兼顾了性能和稳定性，展现出较强的实用潜力。

### 通俗易懂  
MaskCD的核心思想是利用模型内部“注意力头”来发现模型对图像的关注点。想象模型有许多“眼睛”（注意力头），其中一些眼睛特别关注图片内容。MaskCD就是让这些特别关注图片的眼睛暂时闭上，看看模型会产生什么样的输出。通过比较模型正常状态和这些眼睛被遮蔽状态下的输出，MaskCD能识别出哪些信息是幻觉（错误的），哪些是真实的。然后用对比的方法调整生成的文字概率，减少模型说错话的机会。这样做既不需要重新训练模型，也不改变模型本身的知识，只是在推理时巧妙地控制模型的注意力。通过这种“遮眼睛”的方式，模型更少依赖那些可能导致错误的视觉信息，生成的描述更准确、更靠谱。这种方法轻量且有效，适合在各种视觉语言任务中减少模型产生的幻觉，提高用户对模型的信任。 
# Topic: Multi-modal｜Image
## Improving GUI Grounding with Explicit Position-to-Coordinate Mapping 
2025-10-03｜ServiceNow, Mila-Quebec AI Institute, UdeM, York, PolyMTL, McGill, MLIA CIFAR AI Chair 

<u>http://arxiv.org/abs/2510.03230v1</u>  
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/110.jpg)  
本文聚焦于图形用户界面（GUI）定位任务，即将自然语言指令准确映射到界面像素坐标，解决现有视觉语言模型（VLM）在高分辨率显示器上泛化能力差和定位不准确的问题。传统方法将坐标预测视为文本生成任务，模型需隐式地从视觉特征中推断复杂的位置信息，导致准确率下降且难以适应未见过的分辨率。为此，本文提出两项关键创新：一是引入RULER令牌，作为明确的坐标参考标记，模型通过参考这些标记进行坐标调整，而非完全生成坐标；二是设计了交错多维旋转位置编码（I-MROPE），均衡宽高两个维度的空间编码频率，解决了传统编码中频率分配不均的问题。实验在多个公开UI定位数据集上验证，特别是在高分辨率界面上显著提升了定位准确率，展示了良好的泛化与实用性。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/111.jpg)  
本文方法包括两大核心模块：  

1. **RULER令牌（Rotary position-to-pixel mappER）**：  
   - 设计一组辅助令牌，每个令牌对应图像中一个固定位置的像素坐标，且与该位置的视觉补丁共享位置编码。  
   - 模型通过查找与视觉补丁位置最匹配的RULER令牌，复制其坐标值作为参考，并在此基础上进行小范围的数值调整，避免了传统隐式回归的困难。  
   - 通过定期间隔插入RULER令牌，减少计算负担，同时保证空间坐标的显式指导。  
2. **交错多维旋转位置编码（I-MROPE）**：  
   - 传统多维旋转编码将不同频率段顺序分配给高、宽、时三个维度，导致频率分布不均，某些维度缺乏高频或低频信息。  
   - I-MROPE通过交错分配频率，使每个空间维度均匀获得完整频率范围，提升空间位置的表达能力。  
   - 该编码方案与预训练语言模型兼容，保持了文本和视觉模态的连续性。  
整体框架将显式的坐标参考与均衡的空间编码结合，显著增强模型对像素级精确定位的能力。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/112.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/113.jpg)  
实验分为从零训练和微调两部分，均在UGround数据集上训练，测试集包含ScreenSpot、ScreenSpot-V2及更具挑战性的高分辨率ScreenSpot-Pro。  
- **从零训练**：基于LLaVA-NeXT架构，替换默认位置编码为MRoPE或I-MROPE，并加入RULER令牌。结果显示I-MROPE优于传统MRoPE，加入RULER令牌后性能进一步提升，尤其在高分辨率ScreenSpot-Pro上准确率从31.1%提升至32.1%。  
- **微调**：在Qwen2.5-VL模型上引入RULER令牌，显著提升了所有测试集的定位准确率，ScreenSpot-Pro上从34.6%提升至37.2%，验证了RULER的泛化能力。  
- **分析**：调整RULER令牌间隔对性能影响不大，默认间隔8在性能和效率间达到平衡。令牌数量占比极低（不足1%），保证了计算效率。  
整体实验充分证明了显式坐标映射和均衡位置编码对提升GUI定位精度和分辨率泛化的有效性。

### 通俗易懂  
想象你在地图上找一个地点，传统模型就像只给你一张模糊的地图，要求你凭感觉直接说出坐标，既难又容易出错。本文提出的方法则像在地图上画上了清晰的格子线（RULER令牌），每个格子都有明确的编号，模型只需找到最近的格子编号，再根据具体位置稍作调整，就能准确说出坐标。这样做大大减少了猜测的成分，让定位更稳健。而且，地图上的格子线宽度和高度的间隔设计得更均匀（I-MROPE），保证无论是横向还是纵向，模型都能同样精准地感知位置。通过这种“参考格子+均匀编码”的方法，模型不仅能更准确地找到目标位置，还能适应不同大小和分辨率的屏幕，就像无论地图放大多少倍，你都能轻松找到地点一样。这样一来，自动化软件操作变得更可靠，未来还能扩展到视频和更复杂的界面环境。 
## UniShield: An Adaptive Multi-Agent Framework for Unified Forgery Image Detection and Localization 
2025-10-03｜PKU, SCUT 

<u>http://arxiv.org/abs/2510.03161v1</u>  
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/114.jpg)  
随着图像生成技术的迅猛发展，合成图像的真实感显著提升，带来了信息安全和社会信任的严峻挑战。针对图像伪造检测与定位（FIDL）任务，现有方法多局限于特定领域，缺乏跨域泛化能力和统一适配框架，难以满足实际应用需求。本文提出UniShield，一种基于多智能体的统一框架，能够自适应处理包括图像篡改、文档伪造、DeepFake及AI生成图像等多种伪造类型。UniShield通过感知智能体动态分析图像特征，选择最合适的检测模型；检测智能体则整合多领域专家模型，输出结构化且具解释性的检测报告。该系统在多个公开基准上取得了领先性能，展示了其在实用性、适应性及扩展性方面的显著优势，填补了跨域伪造检测领域的空白。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/115.jpg)  
UniShield架构由两个核心模块组成：感知智能体和检测智能体。  

1. **感知智能体**：包括任务路由器和工具调度器。任务路由器基于图像的语义结构和低层视觉特征，推断伪造类型（如图像篡改、文档篡改、DeepFake、AI生成图像），实现伪造域的自动分类。工具调度器根据路由结果，判断应调用基于大语言模型（LLM）的语义检测工具还是非LLM的低层视觉伪造检测工具。该决策依赖于图像所表现出的语义不一致性或视觉伪影。  
2. **检测智能体**：集成了多种专家模型，覆盖四大伪造领域。根据感知智能体的选择，调用对应检测器完成伪造检测与定位，并生成包含检测结果、篡改区域掩码及判断依据的详尽报告。报告融合低层视觉线索（如边缘模糊、光照不符）与高层语义信息（如语境冲突、重复模式），提升系统的可解释性和用户信任度。  
3. **优化策略**：采用基于相对奖励的强化学习（GRPO）对任务路由器及部分检测模型进行微调，优化模型选择策略和检测性能，确保系统在有限标注数据下的稳定高效训练。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/116.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/117.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/118.jpg)  
在涵盖图像篡改（IMDL）、文档篡改（DMDL）、DeepFake检测（DFD）和AI生成图像检测（AIGCD）四大领域的多个公开数据集上，UniShield均显著优于当前最先进的单领域及跨领域检测方法。具体表现包括：IMDL任务中，在CASIA1+和IMD2020数据集上，UniShield在图像和像素级F1分数均领先；DFD任务中，在DF40数据集上，F1分数超出次优方法FakeShield约0.2；DMDL任务中，在RTM数据集上，像素级F1分数提升明显。消融实验验证了动态工具调度及任务路由机制的重要性，固定使用单一检测策略或简单投票机制均导致性能下降。此外，系统生成的检测报告结构清晰，能够详细说明伪造区域及其检测依据，增强了结果的透明度和用户理解。

### 通俗易懂  
UniShield就像一个聪明的侦探团队，里面有两个主要角色：一个是“侦查员”，负责先观察图片，判断它可能是哪种类型的伪造，比如是被篡改的照片、伪造的文件、换脸视频，还是AI生成的假图；另一个是“专家”，根据侦查员的判断，选择最合适的工具来仔细检查图片。侦查员会看图片里的大致内容和细节，比如颜色、光线是否自然，物体之间的关系是否合理，来决定用哪种检测方法。专家则用专门的技术，找到图片里被修改的部分，并且写一份详细的报告，告诉你哪里有问题，为什么觉得这是假的。整个过程自动完成，不需要人工挑选检测工具。这样，UniShield不仅能准确发现各种伪造，还能解释原因，让用户放心使用，避免被假图片欺骗。 
## One Patch to Caption Them All: A Unified Zero-Shot Captioning Framework 
2025-10-03｜CNR-ISTI, Università di Pisa 

<u>http://arxiv.org/abs/2510.02898v2</u>  
<u>https://paciosoft.com/Patch-ioner/</u> 
### 概述 
  
本文提出了Patch-ioner，一种统一的零样本图像描述框架，突破传统依赖全图全局特征的限制，转向以图像中的局部“patch”（图像小块）为基本单元进行描述。该方法无需区域级标注，实现对任意图像区域（单个patch、非连续区域甚至整图）的灵活描述，极大提升了区域级图像描述的泛化能力和适用范围。Patch-ioner基于视觉-语言共享空间的预训练模型，结合能生成文本的解码器，创新地将局部视觉特征聚合并映射到文本空间，支持多种细粒度的零样本描述任务，包括密集描述、区域集合描述以及新提出的鼠标轨迹描述任务。实验表明，采用具备丰富局部语义信息的视觉骨干（如DINO系列）是实现高质量区域描述的关键，Patch-ioner在多个区域级零样本描述任务中均显著优于现有方法，展示了其统一且高效的多粒度描述能力。  

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/119.jpg)  
Patch-ioner框架核心在于：  

1. **Patch级视觉编码**：利用视觉Transformer（如DINOv2）将图像划分为固定大小的patches，提取每个patch的视觉特征，形成密集且语义丰富的patch嵌入。  
2. **无监督区域特征聚合**：定义任意区域为patch集合，通过简单的均值或加权平均聚合patch特征，灵活支持矩形框、掩码、鼠标轨迹等多样区域形态。  
3. **零样本文本解码器**：训练一个文本解码器仅依赖文本数据，将视觉区域特征映射到文本语义空间进行描述生成。为缓解视觉与文本模态间的分布差异，采用两种调和策略：一是基于记忆库的视觉特征投影，二是训练时引入噪声增强解码器对视觉特征的鲁棒性。  
4. **统一多任务支持**：该框架无需区域级标注，统一处理全图描述、密集区域描述、区域集合描述及自由形状轨迹描述，极大简化了多粒度描述任务的模型设计与训练流程。  

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/120.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/121.jpg)  
实验涵盖四个零样本描述任务：鼠标轨迹描述、密集区域描述、区域集合描述和传统整图描述。数据集包括COCO、Visual Genome、Flickr30K及Localized Narratives。通过对比多种视觉编码器，结果表明基于DINOv2的编码器在捕获局部语义细节方面表现优异，显著提升了区域描述的准确性和语义丰富度。Patch-ioner在各任务中均优于传统基于全局CLIP特征的零样本描述方法，尤其在细粒度的轨迹和密集描述任务中优势明显。与依赖区域级监督的模型相比，Patch-ioner在无监督条件下仍能达到甚至超越部分有监督方法的表现，展现出极强的泛化能力和实用价值。此外，实验还验证了不同模态差异缓解策略和patch特征聚合方式对性能的影响，进一步巩固了框架设计的有效性。  

### 通俗易懂  
Patch-ioner的核心思想是把一张图片拆成很多小块（patch），把每个小块看成一个独立的“描述单位”，然后把这些小块的特征合起来，生成对图片任意部分的文字描述。想象你在拼积木，每个积木块都有自己的颜色和形状，拼成不同的组合就能描述不同的场景。传统方法只能描述整张图片，或需要大量标注告诉模型每个区域是什么，而Patch-ioner不需要这些标注，靠预训练模型自动理解每个小块的内容。它还训练了一个只看文字的“翻译器”，能把这些视觉积木块的特征翻译成自然语言。通过这种方式，无论是描述一个小物体、几个分散的区域，还是整张图片，都能灵活生成准确的文字说明。这种方法不仅省去了繁琐的标注过程，还能适应各种复杂的描述需求，非常适合实际应用中的交互式和细粒度图像理解。 
## Retrv-R1: A Reasoning-Driven MLLM Framework for Universal and Efficient Multimodal Retrieval 
2025-10-03｜CUHK, Tencent , ZJU｜NeurIPS 2025 

<u>http://arxiv.org/abs/2510.02745v1</u>  
<u>https://lanyunzhu.site/RetrvR1/</u> 
### 概述 
  
本文提出了Retrv-R1，一种首创的基于推理驱动的多模态大语言模型（MLLM）检索框架，旨在实现通用且高效的多模态信息检索。该框架借鉴了DeepSeek-R1中强化学习（RL）提升推理能力的成功经验，采用逐步推理链（Chain-of-Thought, CoT）策略来产生更精准的检索结果。针对直接将DeepSeek-R1方法应用于检索任务时遇到的高计算成本和训练不稳定性问题，Retrv-R1引入了信息压缩模块和细节检验机制，有效降低了输入token数量，同时保留了对难判候选项的关键信息。此外，设计了包括检索定制的合成CoT数据激活阶段和基于课程学习的奖励机制的训练新范式，显著提升了模型的性能、效率及泛化能力。通过在多个公开基准和任务上的广泛实验，Retrv-R1展示了其在多模态通用检索领域的领先表现和强大适应性。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/122.jpg)  
Retrv-R1的核心方法包括以下几个关键部分：  

1. **两阶段检索流程**：第一阶段利用MLLM生成查询与候选项的嵌入向量，筛选出Top-K候选；第二阶段采用另一个MLLM对这部分候选进行细粒度CoT推理，最终确定最佳匹配。  
2. **信息压缩模块（ICM）**：通过两层注意力机制将每个候选的token序列压缩为两个关键token，分别承载内容和查询关系信息，显著减少token数，降低计算资源消耗。  
3. **细节检验机制**：模型在推理过程中自动识别难判候选，触发检验操作，附加完整token序列辅助判断，缓解压缩带来的信息丢失问题。  
4. **训练范式**：先用合成的CoT检索数据进行监督微调（SFT）激活模型推理能力，再通过强化学习（RL）微调，采用基于课程学习的奖励函数平衡推理性能和效率，逐步强化模型的推理能力和计算资源利用率。  
5. **自对齐预训练**：在训练初期冻结MLLM，仅训练ICM以确保压缩token能有效保留检索相关信息，为后续联合训练打下基础。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/123.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/124.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/125.jpg)  
Retrv-R1在多个多模态检索基准（如M-BEIR）及跨领域任务上进行了全面评估。结果显示，Retrv-R1无论是3B还是7B参数规模版本，都在多种查询与候选组合形式下实现了领先的召回率，显著优于包括CLIP、BLIP、Vision-R1、LamRA等现有先进方法。效率方面，得益于ICM和奖励机制的设计，Retrv-R1在保持高性能的同时，大幅减少了推理时间和显存占用，支持更大候选集的处理。泛化能力测试表明，Retrv-R1在未见过的数据集和任务上依然表现稳健，且在多模态推荐任务中展现出强适应性和竞争力。消融实验进一步验证了信息压缩模块、自对齐预训练、细节检验机制、SFT激活及RL微调对整体性能和效率的关键作用，课程学习策略的引入也被证明有效提升了训练稳定性和模型表现。

### 通俗易懂  
Retrv-R1的核心思想是让模型像人一样“想一想再做决定”，通过逐步推理来挑选最合适的答案。为了避免处理太多信息导致速度慢，Retrv-R1先用一个“压缩器”把每个候选答案的内容和它与问题的关系浓缩成两个小“摘要”，这样模型处理起来更快更省资源。但有些候选答案比较难判断，模型会自动“举手示意”，让系统把这个答案的全部详细信息拿出来，仔细检查。训练时，模型先通过模拟的推理过程学习如何一步步思考，再用强化学习不断调整，既保证推理准确，也控制计算成本。这样设计让Retrv-R1既聪明又高效，能在各种复杂检索任务中快速找到最合适的答案。 
# Topic: Multi-modal｜Video
## AdaRD-key: Adaptive Relevance-Diversity Keyframe Sampling for Long-form Video understanding 
2025-10-03｜UWA, DLUT, Khalifa, Zhejiang Lab 

<u>http://arxiv.org/abs/2510.02778v1</u>  
<u>https://github.com/Xian867/AdaRD-Key</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/126.jpg)  
长视频理解因其时间跨度长、信息密度大而成为多模态大语言模型（MLLMs）的一大挑战。现有方法多采用均匀采样，导致遗漏关键时刻，影响回答准确性。部分关键帧选择方法采用固定时间窗口抑制邻近帧，虽减少冗余，但易错过事件附近的细节信息；另一些方法强调视觉多样性，却忽视了与查询的相关性。本文提出AdaRD-Key，一种无需训练的自适应关键帧采样模块，结合查询相关性与视觉多样性，通过最大化一个统一的“相关性-多样性最大体积”目标函数，选取信息丰富且非冗余的关键帧。针对弱相关查询，设计轻量级相关感知门控，自动切换到仅多样性模式以扩展覆盖范围。该模块兼容现有视觉语言模型，实时运行，显著提升长视频理解性能，尤其在长视频问答和视频摘要任务上表现优异。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/127.jpg)  
AdaRD-Key关键帧选择方法主要包括以下几个核心部分：  

1. **相关性评分计算**：利用BLIP-2模型提取每帧与查询的相似度分数及视觉特征，形成帧级语义表示和相关性评分。  
2. **相关性-多样性最大体积（RD-MV）选择器**：设计一个联合目标函数，平衡帧的查询相关性和视觉多样性。多样性通过所选帧特征的Gram矩阵的对数行列式（log-det）衡量，鼓励选取互补且非冗余的帧。该目标函数是单调次模函数，采用贪心算法和Sherman–Morrison公式高效迭代选帧。  
3. **轻量级相关感知门控**：根据相关性分数的最大值与熵判断查询与视频的对齐程度，若弱相关，则关闭相关性项，仅依赖多样性目标保证信息覆盖。  
4. **变异性-预算自适应调节（VB-Scale）**：动态调整相关性与多样性权重λ，依据相关性分数的变异系数和视频长度（候选帧与选帧比例）自动平衡两者，确保在不同视频和查询条件下均能有效选帧。  
整个流程无需训练，支持单GPU实时运行，直接插拔到任何视觉语言模型中。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/128.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/129.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/130.jpg)  
在LongVideoBench和Video-MME两个长视频理解基准上，AdaRD-Key显著优于均匀采样、AKS和MAXINFO等方法。以Qwen2-VL和LLaVA-Video为基础模型，分别在32帧和64帧预算下进行评测。结果显示，AdaRD-Key在整体准确率上提升3.8个百分点，尤其在中长视频（3-10分钟及15-60分钟）中表现更为突出。此外，在视频字幕缺失条件下，模型依然能保持较高视觉推理能力。视频摘要和问答任务中，AdaRD-Key采样的关键帧更好地捕获了查询相关细节，避免了均匀采样遗漏关键信息和AKS过度关注背景场景的问题。视频字幕生成任务中，采用AdaRD-Key的模型在准确率、覆盖率和一致性指标均有明显提升。消融实验验证了相关性、多样性、门控机制及VB-Scale各模块的独立贡献，整体性能随着模块叠加逐步提升，尤其对长视频效果提升显著。

### 通俗易懂  
AdaRD-Key就是一个聪明的“挑选关键画面”的小帮手，专门帮你从长视频里挑出最有用、最不重复的画面，方便后续模型理解视频内容。它先用一个强大的图像和文本匹配工具（BLIP-2）给每一帧打分，看看这帧和你提出的问题有多相关。然后，它用一个数学“平衡术”来挑选画面：一方面要挑相关性高的画面，另一方面又要保证这些画面之间差异大，避免选出太多相似的重复画面。这个“平衡术”通过计算画面特征之间的“体积”来实现，体积越大，说明画面越丰富多样。还有个小开关，如果问题和视频关系不大，它就自动关闭相关性，只挑多样的画面，保证信息全面。最后，它还能根据视频长短和得分情况自动调节“挑选规则”，让选择更灵活。整个过程不需要额外训练，速度快，直接接入现有视频理解模型，就能让它们更聪明地看懂长视频。 
## Oracle-RLAIF: An Improved Fine-Tuning Framework for Multi-modal Video Models through Reinforcement Learning from Ranking Feedback 
2025-10-02｜LLNL, UCLA, Cornell 

<u>http://arxiv.org/abs/2510.02561v1</u>  
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/131.jpg)  
本文提出了Oracle-RLAIF，一种创新的多模态视频模型微调框架，通过引入基于排序的强化学习方法，提升视频语言模型（VLM）的理解和生成能力。传统微调方法依赖人工反馈构建奖励模型，成本高且受限于数据标注质量。Oracle-RLAIF突破这一瓶颈，利用一个通用的“Oracle”排序器替代复杂的奖励模型，仅需对模型生成的多条候选回答进行排序，而非评分，极大降低了反馈构建难度和成本。为配合这一排序反馈，作者设计了GRPO（Group Relative Policy Optimization）算法的改进版本，直接基于排序信息优化模型策略，避免了传统基于标量奖励的强化学习方法在处理排序反馈时的局限。实验证明，Oracle-RLAIF在多个视频理解基准上均优于现有最先进的微调技术，尤其在时序感知和动作识别等任务中表现突出，展现了其在多模态视频模型高效、灵活微调方面的巨大潜力。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/132.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/133.jpg)  
Oracle-RLAIF框架主要由以下三个核心部分组成：  

1. **Oracle排序器替代奖励模型**：传统RLAIF依赖训练有素的奖励模型对生成回答进行打分，Oracle-RLAIF则引入一个能够对多条候选回答进行质量排序的Oracle模型，简化了反馈机制并提升了适用性。  
2. **基于排序的GRPO优化算法**：针对Oracle提供的排序反馈，作者设计了改进的GRPO算法，将排序差异通过归一化折扣累计增益（nDCG）转化为惩罚信号，构建优势函数，非线性地惩罚排序误差，尤其强调对排名靠前回答的准确排序，保证模型优先学习高质量输出。  
3. **迭代微调流程**：从预训练的监督微调模型开始，生成多条候选回答，由Oracle排序后计算nDCG惩罚，GRPO利用该信号调整模型策略，通过限制策略偏离和引入熵正则化保证训练稳定，反复迭代直至模型输出更符合Oracle排序标准。该方法无需训练价值函数，避免了传统强化学习中对奖励标度和价值估计的依赖。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/134.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-09/135.jpg)  
实验部分系统评估了Oracle-RLAIF在多模态视频理解任务上的表现。首先，在MSVD、MSRVTT和ActivityNet三个视频问答基准上，Oracle-RLAIF相比传统的VLM-RLAIF框架实现了显著的准确率和评分提升，表现出更强的视频内容理解能力。其次，在最新的Video-MME数据集上，Oracle-RLAIF取得了6.2%的整体准确率提升，尤其在时间感知、动作识别和物体推理等关键任务上提升超过10%，说明其在捕捉时序和因果关系方面优势明显。实验还发现，Oracle-RLAIF在空间感知和抽象推理任务上表现稍逊，推测这类任务对模型结构和表征能力要求更高，排序优化效果有限。所有模型均采用相同初始监督微调模型和训练配置，确保性能差异源于微调策略改进。整体结果验证了基于排序反馈的Oracle-RLAIF框架在提升多模态视频模型理解能力和泛化性能方面的有效性和实用性。

### 通俗易懂  
Oracle-RLAIF的核心思想是用“排序”代替“评分”来教会视频理解模型变得更聪明。传统方法需要人类给模型的回答打分，既费时又费钱。Oracle-RLAIF则用一个“裁判”模型，这个裁判不需要给分，只要告诉模型“这几个回答哪个最好、哪个次之”，类似老师让学生排队而不是给每个人打分。然后，作者设计了一个特别的学习方法（GRPO改进版），它能根据这个排序信息来调整模型，让模型学会更喜欢排名靠前的好回答，避免选错答案。这个方法还特别强调把最好的答案排在最前面，确保用户看到的回答质量最高。整个过程像是模型在和裁判玩排序游戏，模型不断根据裁判的排序调整自己，慢慢变得更懂视频内容。这样既节省了大量人工打分成本，也让模型训练更灵活高效，最终让视频理解能力显著提升。 
