# Topic: Image Generation｜Diffusion/Autoregress/VQ｜BBB10.8 
## $\bf{D^3}$QE: Learning Discrete Distribution Discrepancy-aware Quantization Error for Autoregressive-Generated Image Detection 
2025-10-07｜THU｜ICCV 2025 

<u>http://arxiv.org/abs/2510.05891v1</u>  
<u>https://github.com/Zhangyr2022/D3QE</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/41.jpg)  
本文聚焦于视觉自回归（autoregressive, AR）模型生成图像的检测问题。与传统的GAN或扩散模型不同，AR模型通过离散token的预测生成图像，展现出更高的合成质量及其向量量化表示的独特分布特征。针对这一特点，作者提出了离散分布差异感知的量化误差（D3QE）检测框架，利用真实与伪造图像在codebook使用频率上的显著差异。该方法通过引入离散分布差异感知的Transformer，将动态的codebook频率统计信息融入注意力机制，结合语义特征与量化误差潜变量，实现对AR生成图像的高效识别。此外，本文构建了覆盖7个主流视觉AR模型的ARForensics数据集，系统评估了方法的准确性、泛化能力及对现实扰动的鲁棒性，验证了D3QE在多模型、多场景下的优越性能。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/42.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/43.jpg)  
D3QE方法核心基于视觉AR模型的离散编码机制，设计了三大模块：  

1. **量化误差表示模块**：利用预训练的VQVAE编码器将输入图像映射到连续潜空间，再通过codebook进行离散化，计算连续潜向量与其对应离散向量间的量化误差，捕捉真实与生成图像在离散化过程中的差异。  
2. **离散分布差异感知Transformer（D3AT）**：该模块通过动态追踪真实与伪造图像codebook索引的频率统计，计算分布差异向量，并将其嵌入Transformer的自注意力机制中，增强模型对codebook使用频率偏差的敏感度，兼顾局部token分布与全局结构信息。  
3. **语义特征嵌入模块**：采用预训练CLIP模型提取图像的全局语义特征，补充局部离散特征的不足，提升检测的语义理解能力。  
最终，融合离散分布特征与语义特征，通过多层感知机进行二分类判别，实现对AR生成图像的有效检测。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/44.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/45.jpg)  
作者构建了ARForensics数据集，涵盖7种主流视觉AR模型，包含15.2万真实与15.2万生成图像，确保数据多样性与均衡性。实验中，D3QE在AR模型生成图像检测中展现出显著优势，平均准确率达82.11%，超越多种先进基线方法，尤其在对结构差异较大的VAR模型检测中表现突出。跨范式测试中，D3QE同样在GAN和扩散模型生成图像检测中保持强劲性能，显示出良好的泛化能力。针对真实场景中的JPEG压缩和图像裁剪扰动，D3QE表现出较强鲁棒性，准确率和平均精度均优于对比方法。消融研究验证了量化误差、离散分布差异感知机制及语义特征融合对性能提升的贡献。可视化分析进一步揭示了真实与生成图像在codebook激活分布上的显著差异，为方法设计提供了理论依据。

### 通俗易懂  
这项研究的关键是识别由自回归模型生成的假图像。自回归模型生成图像时，会先把真实图像转换成一系列“离散代码”，这些代码就像图像的“拼图块”。不同于连续的像素值，这些拼图块的使用频率在真实图像和假图像之间存在差异。研究者设计了一个特别的“侦探”——一个智能的Transformer网络，它不仅看图像的语义内容，还专门关注这些拼图块的使用频率和它们之间的微小差异。具体来说，这个“侦探”会记录每种拼图块出现的次数，比较真实和假图像中这些次数的差异，并把这些信息融入它的注意力机制中，帮助它更准确地判断图像是真是假。通过结合图像的整体语义和局部拼图块的统计特征，这个方法能更好地发现自回归模型生成图像的独特“指纹”，即使图像经过压缩或裁剪，也能保持较高的检测准确率。简单来说，这个方法就像一个既懂图像内容又能察觉细微统计差异的侦探，帮助我们识别真假图像。 
## Be Tangential to Manifold: Discovering Riemannian Metric for Diffusion Models 
2025-10-07｜Hokkaido 

<u>http://arxiv.org/abs/2510.05509v1</u>  
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/46.jpg)  
扩散模型作为强大的深度生成模型，能够生成高质量且多样化的内容，但其缺乏显式且可解析的低维潜空间来参数化数据流形，限制了对流形结构的感知与操作，如插值和编辑。现有插值方法多沿高密度区域路径进行，但这些路径未必与数据流形对齐，导致视觉上不自然的过渡。本文提出基于扩散模型中的得分函数雅可比矩阵构建的黎曼度量，该度量定义在噪声空间，能够捕获数据流形的切空间信息，鼓励插值路径沿流形切向或平行于流形变化。该方法无需额外训练或架构修改，实验证明其在合成二维数据、图像及视频帧插值任务中，能产生更自然且忠实于数据流形的过渡效果，优于基于密度或传统线性插值方法。

### 方法 
  
本文核心在于构造一种新的黎曼度量，用以度量扩散模型噪声空间中样本间的距离。具体步骤包括：  

1. 利用扩散模型训练得到的得分函数（噪声预测器的梯度近似）计算其雅可比矩阵J；  
2. 定义黎曼度量矩阵G = J^T J，保证其对称且半正定，反映样本在得分函数空间的变化率；  
3. 该度量在噪声空间中刻画了数据流形的局部几何结构，因J在流形切向量方向上较小，在法向量方向较大，从而鼓励插值路径沿流形切向方向变化；  
4. 通过数值优化最小化路径能量函数，求解两点间的测地线，实现流形感知的插值；  
5. 应用DDIM反演将干净样本映射到噪声空间，插值后再映射回数据空间，保证端点一致性。该方法兼容带引导的得分函数，适用于各种扩散模型。  
此度量有效避免了传统线性插值穿越低密度区域和基于密度度量导致的过度平滑问题，提供更语义连贯的样本过渡。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/47.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/48.jpg)  
实验分为三部分：  

1. 合成二维数据验证：在C形分布上训练扩散模型，比较线性插值（LERP）、球面线性插值（SLERP）、基于密度的插值与本文方法。结果显示，本文测地线插值路径更贴合流形，保持端点概率分布，过渡自然平滑。  
2. 图像插值：使用Stable Diffusion v2.1-base在MorphBench动画子集、AnimalFaces-HQ和CelebA-HQ数据集上测试。评价指标包括FID、PPL、PDV和重建误差。本文方法在所有指标上均表现优异，生成的插值图像细节丰富，过渡自然，明显优于传统插值及最新基于密度方法。  
3. 视频帧插值：在DAVIS、Human Pose和RealEstate10K数据集上进行三帧插值，评价MSE和LPIPS。本文方法在保持边缘、物体形状和纹理细节方面优于其他方法，插值结果更接近真实中间帧。  
整体实验验证了本文黎曼度量在捕获扩散模型数据流形结构及提升插值质量方面的有效性。

### 通俗易懂  
这项研究解决了扩散模型在“连接”两张图片时容易出现不自然过渡的问题。通常，扩散模型没有一个简单的低维空间来描述数据的“真实形状”，所以直接在噪声空间做线性插值会穿过“空白”区域，导致图像模糊或细节丢失。研究者发现，通过观察扩散模型中的一个叫“得分函数”的数学工具，可以了解数据真实存在的“曲面”方向。具体来说，他们用得分函数的变化率（雅可比矩阵）来定义一个新的距离度量，这个度量告诉我们如何沿着数据的“自然轨迹”移动。这样，插值路径就不会偏离数据本身的形状，而是沿着它的“表面”滑动。通过数值方法找到这条最短路径（测地线），再将路径上的噪声转换回清晰图像，就能得到更加自然和连贯的过渡效果。简言之，这种方法就像在崎岖山路上找到一条平滑的山脊小道，走这条路比横穿山谷更安全、风景更美。 
## Mitigating Diffusion Model Hallucinations with Dynamic Guidance 
2025-10-06｜SBU, UW-Midison 

<u>http://arxiv.org/abs/2510.05356v1</u>  
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/49.jpg)  
扩散模型因其生成高质量、多样化图像的能力而备受关注，但它们常产生结构不合理的“幻觉”样本，即超出真实数据分布支持范围的图像，如解剖结构错误的人手。幻觉现象主要源于模型学习的得分函数在数据分布不同模态间过于平滑，导致生成过程在低概率区域产生语义不合理的插值。传统引导方法虽能提升样本质量，但通常固定条件，无法动态适应采样轨迹，容易导致误入幻觉区域。本文提出动态引导（Dynamic Guidance）方法，通过在采样过程中动态选择引导目标，针对已知导致幻觉的方向有选择性地锐化得分函数，从而有效抑制幻觉生成，同时保留合理的语义多样性。该方法首次实现了在生成时直接缓解幻觉问题，避免了事后过滤的资源浪费，且在多种受控和自然图像数据集上均显著优于现有基线。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/50.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/51.jpg)  
动态引导的核心在于动态调整采样过程中的引导目标，具体包括以下几个关键步骤：  

1. **幻觉定义与分类标签选择**：针对不同数据集（合成二维高斯混合、几何形状、人体手部图像等）定义幻觉样本的标准，并选取与幻觉对应或相关的类别标签作为引导条件。  
2. **动态类别识别**：在每个去噪步骤，根据当前噪声样本，利用分类器动态确定最可能的类别标签，避免固定条件导致的轨迹偏离。  
3. **选择性得分函数锐化**：通过分类器梯度调整扩散模型的得分函数，仅在幻觉相关的语义方向上增强梯度，减少跨模态的无效插值，同时不干扰其他无关维度，保持语义多样性。  
4. **局部引导时间窗**：动态引导只在采样过程中的特定时间区间内应用，避免全程引导带来的过度偏置。  
该方法在去噪扩散隐式模型（DDIM）和概率模型（DDPM）中均可实现，且通过结合β-VAE学习的潜变量空间，精确定位幻觉相关维度，实现了针对性控制。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/52.jpg)  
本文在多个层面验证了动态引导的有效性：  
- **二维高斯混合玩具数据集**：通过与真实闭式得分函数对比，动态引导显著锐化了模型得分函数，避免了样本陷入低概率幻觉区域。  
- **合成几何形状数据集**：在“单一形状”和“混合形状”设置下，动态引导分别针对幻觉定义的跨类插值和重复形状问题，减少幻觉超过50%，优于基于方差过滤和固定分类器引导的方法。  
- **人体手部数据集**：在仅25步DDIM采样的低步数条件下，动态引导仍能减少约50%的幻觉，显示出强鲁棒性。  
- **ImageNet大规模数据集**：由于幻觉难以精确定义，采用生成样本的精度、召回率和Inception分数等代理指标评估。动态引导在保持召回率的同时，显著提升了精度和Inception分数，表明生成样本更贴近真实数据分布，减少幻觉。  
此外，实验还分析了动态引导相较于固定引导避免了轨迹过冲和欠冲问题，提升了采样效率和样本质量。

### 通俗易懂  
动态引导就像在生成图像的过程中给模型一个“实时导航”，而不是一开始就定死路线。想象你让一个画家画一只手，但他可能会画出五个手指或六个手指（幻觉），这是因为他对手指数量的理解模糊。动态引导的方法会在画家的每一步都检查当前画的手指数，并根据情况给出建议，让他调整画法，避免出现不合理的手指数。具体来说，系统会用一个分类器判断当前生成的图像更像哪个类别（比如“正常手”、“多指手”），然后用这个类别的反馈来调整生成过程中的“方向”，只在那些容易出错的地方加强控制，而不会限制画家自由发挥其他合理的变化。这样既防止了画出奇怪的东西，又保持了作品的多样性和创造力。通过这种动态调整，模型在生成时就能避免产生幻觉，而不是等到画完后再去挑错，节省了时间和计算资源。 
# Topic: Image Generation｜Safety 
## SafeGuider: Robust and Practical Content Safety Control for Text-to-Image Models 
2025-10-05｜USTC, NTU｜CCS 2025 

<u>http://arxiv.org/abs/2510.05173v2</u>  
<u>https://github.com/pgqihere/safeguider</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/53.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/54.jpg)  
文本到图像生成模型（Text-to-Image，简称T2I）如Stable Diffusion展现了从自然语言描述生成高质量图像的强大能力，但其安全性问题日益突出，尤其容易被对抗性提示绕过安全机制，生成有害内容。现有防御方法在抵御多样化攻击时表现有限，同时难以兼顾生成质量和用户体验。本文通过对Stable Diffusion文本编码器的深入实证研究，发现其[EOS]（结束符）标记在语义聚合中扮演关键角色，且在正常与对抗性提示的嵌入空间中呈现显著不同的分布特征。基于此，提出SafeGuider框架，结合嵌入级别的安全识别模型与安全感知特征擦除的束搜索算法，实现对恶意提示的精准识别与安全引导生成，兼顾鲁棒性与实用性。该方法在多种攻击场景下大幅降低攻击成功率，并能生成安全且语义合理的图像，提升了实际部署的可行性和用户体验。  

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/55.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/56.jpg)  
SafeGuider框架设计包括两个核心步骤：  

1. **安全识别**：利用文本编码器提取输入提示的[EOS]标记嵌入，输入至轻量级的三层神经网络分类器，评估提示的安全性。该识别模型基于对抗性与正常提示的嵌入数据训练，能有效区分安全与不安全内容，且具备较强的泛化能力。  
2. **安全引导生成**：对于识别为不安全的提示，SafeGuider启动安全感知特征擦除（SAFE）束搜索算法。该算法通过策略性修改输入提示的嵌入向量，删除或调整潜在有害的特征，同时保持与原始语义的相似性，确保生成的图像既安全又有意义。束搜索在多个候选嵌入中寻找最优解，平衡安全性和语义保真度。  
整体框架不依赖于模型内部参数访问，适用于多种T2I模型架构，实现了对抗攻击的鲁棒防御与高质量生成的有机结合。  

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/57.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/58.jpg)  
本文在多种对抗攻击场景（包括词汇替换和符号注入）下，对SafeGuider进行了广泛评估。结果显示，SafeGuider在攻击成功率方面显著优于现有十种先进防御方法，词汇替换攻击的成功率降至1.34%-5.48%，符号注入攻击更低至0.01%-1.12%，远优于主流商业API。对正常提示，SafeGuider保持了100%的生成成功率和高质量输出，避免了语义漂移和生成拒绝的弊端。此外，SafeGuider在去除色情、暴力等有害内容方面表现出色，移除率超过80%。该框架还验证了跨模型架构的适用性，成功应用于Flux模型，显示出良好的迁移能力和实际部署价值。多维度指标和定性分析均支持SafeGuider在鲁棒性与实用性上的优越表现。  

### 通俗易懂  
SafeGuider的核心思想是先“看清楚”用户输入的描述是否安全，再决定如何生成图片。它通过观察文本编码器中特别的“结束符号”这个点的特征，来判断一句话是不是有潜在危险内容。就像老师听学生讲故事时，能从故事结尾的语气判断故事是否健康。若判断安全，直接生成图片；若不安全，就启动一个“纠错”程序，这个程序会在文字的“隐藏特征”里做适当调整，去掉可能引起不良内容的部分，同时尽量保留故事的原意。这样，生成的图片既安全又符合用户的描述。这个方法不需要改动生成模型本身，只用“看”和“调”这两个简单步骤，就能有效防止坏内容出现，还保证了好内容的质量和用户体验。 
# Topic: Video Generation 
## Drive&Gen: Co-Evaluating End-to-End Driving and Video Generation Models 
2025-10-07｜JHU, Waymo, Google｜IROS 2025 

<u>http://arxiv.org/abs/2510.06209v1</u>  
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/59.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/60.jpg)  
本文提出了Drive&Gen，一个创新的联合评估框架，旨在同时评估端到端（E2E）自动驾驶模型和驾驶视频生成模型。随着视频生成技术和E2E驾驶模型的快速发展，如何验证生成视频的真实性及其对驾驶规划模型的影响成为关键问题。Drive&Gen通过控制相同的交通场景布局和环境条件（如天气、时间），生成与真实视频相匹配的合成视频，并将这些视频输入到E2E规划模型中，比较模型对真实与合成视频的反应差异，从而评估视频生成的真实性及其对规划性能的影响。此外，本文设计了行为置换检验（Behavior Permutation Test, BPT），一种基于规划模型输出轨迹分布的统计检验方法，量化生成视频与真实视频在规划行为上的相似度。最后，研究展示了利用合成视频数据进行模型微调，有效提升了E2E规划模型在分布外场景（如恶劣天气和夜间）下的泛化能力。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/61.jpg)  
Drive&Gen框架包含三大核心模块：  

1. **视频生成模型**：基于扩散模型W.A.L.T扩展，新增多模态条件控制，包括交通参与者的边界框、道路地图、车辆姿态、时间（日照角度编码）和天气等。通过这些条件，模型能生成与真实场景布局一致且可控的多样化驾驶视频，实现对场景光照和环境变化的精细调节。  
2. **端到端驾驶模型**：采用预训练的视觉语言模型PaLI，结合视觉编码与文本输入（包括车辆状态和导航指令），将规划任务转化为视觉问答问题，输出未来轨迹。模型通过对比真实与合成视频输入下的轨迹预测，评估视频生成的真实性及规划性能。  
3. **行为置换检验（BPT）**：设计一种统计测试，通过比较在真实与合成视频输入下规划模型产生的轨迹集合的分布差异，判定两者行为是否来自同一分布。具体使用Chamfer距离作为轨迹集合间的相似度度量，结合置换检验计算p值，显著性水平低于0.05即判定生成视频未能“欺骗”规划模型，体现生成视频与真实视频的差异。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/62.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/63.jpg)  
实验部分首先展示了视频生成模型在控制不同环境条件（如天气、时间）下生成视频的能力，采用Frechet视频距离（FVD）、平均位移误差（ADE）和BPT三种指标综合评估。结果显示，FVD虽然反映视觉质量，但对生成视频的条件一致性和规划行为相似度捕捉不足，BPT和ADE能更准确反映模型的可控性和真实性。通过消融实验验证了采用太阳角度编码时间信息优于传统本地时间编码。随后，利用生成视频构建不同运营设计域（ODD）场景，系统评估E2E规划模型在光照和天气变化下的表现，发现场景布局信息对规划性能影响最大，而天气和时间对性能影响较小但显著。最后，采用合成数据微调E2E规划模型，实验表明结合生成视频与真实数据进行微调，有效提升了规划模型在夜间和雨天等分布外环境下的泛化能力，降低了轨迹预测误差。

### 通俗易懂  
这项研究的核心是让自动驾驶系统“看”由电脑生成的驾驶视频，然后判断这些视频是否足够真实，能否让自动驾驶系统做出和真实视频一样的驾驶决策。首先，研究人员开发了一个能根据道路布局、车辆位置、天气和时间等信息，生成逼真驾驶视频的模型。然后，他们用一个自动驾驶“老司机”模型去看这些视频，并预测未来的行驶路线。接着，他们设计了一种“对比测试”，把自动驾驶模型在真实视频和生成视频上的预测路线做比较，如果两者差不多，说明生成的视频很真实，自动驾驶模型没被“欺骗”。最后，他们用这些生成的视频来训练自动驾驶模型，结果发现这样训练出来的模型在雨天和晚上等复杂环境下表现更好。简单来说，这个方法不仅能检测生成视频的质量，还能利用这些视频帮助自动驾驶系统变得更聪明、更可靠。 
# Topic: 3D/4D Reconstruction｜Human 
## ArchitectHead: Continuous Level of Detail Control for 3D Gaussian Head Avatars 
2025-10-07｜UBC 

<u>http://arxiv.org/abs/2510.05488v1</u>  
<u>https://peizhiyan.github.io/docs/architect/</u> 
### 概述 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/78.jpg)  
本文提出了ArchitectHead，一种创新的3D高斯头部头像创建框架，支持训练后实时连续调节细节层次（LOD）。当前基于3D高斯点的头像方法通常在训练后固定高斯点数量，难以在渲染效率与视觉质量之间灵活平衡。ArchitectHead通过将高斯点参数化到二维UV特征空间，并设计多级可学习的UV特征场，实现了基于UV图分辨率动态重采样高斯点数量的能力。该方法利用轻量神经网络解码器，将UV特征映射转换为3D高斯点属性，支持从最高细节到极简表达的连续LOD调整，显著提升渲染速度同时保持较高视觉质量。实验验证了其在单目视频数据集上的自我与跨身份重现任务中达到甚至超越现有最先进方法的表现，且在最低LOD下仅使用6.2%的高斯点，渲染速度翻倍，质量仅有适度下降，展现出良好的实用性和扩展潜力。

### 方法 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/79.jpg)  
ArchitectHead的核心在于多级UV特征场与连续LOD控制机制：  

1. **UV特征场设计**：构建由不同分辨率的UV特征图组成的金字塔结构，捕捉从高分辨率到低分辨率的空间信息。通过加权重采样策略，根据目标LOD平滑融合不同层级特征，避免简单缩放带来的细节丢失和邻近高斯点属性过于相似的问题。  
2. **参数化高斯点**：利用FLAME头模型生成3D网格，映射到UV位置图以初始化高斯点位置。结合表达、姿态编码和LOD值，经过位置编码后输入轻量MLP解码器，输出高斯点的空间位置偏移、旋转、尺度、不透明度和颜色属性。  
3. **训练策略**：分两阶段训练，第一阶段固定最高LOD训练多级UV特征场和解码器以稳定细节捕获；第二阶段随机采样LOD值训练解码器适应不同细节层次，实现无需重训即可连续控制LOD。  
4. **渲染流程**：解码后的高斯点通过3D高斯点光栅化技术渲染，保证实时性与视觉逼真度的平衡。

### 实验 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/80.jpg) 
![](http://www.huyaoqi.ip-ddns.com:83/C%3A/Users/13756/Documents/GitHub/paper_daily_format/paper_daily_back_flask/result/2025-10-12/81.jpg)  
实验基于PointAvatar和INSTA两个单目视频数据集，涵盖多身份头像重建与驱动任务。与当前主流高斯点头像方法（基于网格绑定和UV映射的FlashAvatar、GaussianAvatars等）进行对比，ArchitectHead在L1误差、PSNR、SSIM和LPIPS感知质量指标上均表现优异，且支持半精度推理加速渲染，保持质量稳定。自我重现实验显示最高LOD下细节丰富，最低LOD下高斯点数量仅为最高LOD的6.2%，渲染速度几乎翻倍，视觉质量仅轻微下降。跨身份驱动实验验证了模型的泛化能力和表达控制的准确性。LOD调节实验展示了连续平滑的细节过渡，无明显视觉跳变。消融研究强调多级UV特征场的重要性，单一分辨率或无特征场均导致性能下降，验证了设计的有效性和必要性。

### 通俗易懂  
ArchitectHead就像是给3D头像装上了一个“变焦镜头”，可以根据需要调整头像的细节多寡。它先用二维的“地图”（UV特征图）来表示头像上的每个小点（高斯点），然后根据你想要的清晰度，动态决定用多少这些小点来展示头像。这个“地图”有多个层次，从粗糙到细致，系统会智能地把不同层次的地图混合起来，保证切换细节时画面不会突然变糙或跳动。这样你可以在需要高质量展示时用更多点，画面更细腻；在需要快速渲染或者设备性能有限时，用更少点，速度更快但仍保持不错的形象。训练时，系统先学会用最高细节表现头像，再学会适应各种不同细节等级，这样用时就能随时切换，不用重新训练。最终，这套方法让3D头像既真实又灵活，适合VR、视频通话等多种应用。 
