# 基于小数基音延迟相关性的自适应多速率语音流隐写分析
[Github Page](https://github.com/junono97/SM-MDNNF),[Paper Link](http://cjc.ict.ac.cn/online/bfpub/th-2022325174908.pdf).

**作者:**
[田晖](https://faculty.hqu.edu.cn/htian/zh_CN/index.htm)(华侨大学), [吴俊彦(华侨大学)](https://orcid.org/0000-0003-2692-5928), 严艳(华侨大学), 王慧东(华侨大学), [全韩彧](https://faculty.hqu.edu.cn/hanyu/zh_CN/index.htm)(华侨大学).

**摘要:** 网络语音流隐写分析是信息隐藏检测领域中的一个研究热点。针对自适应多速率语音流隐写检测问题，本文提出了一种基于小数基音延迟相关性的隐写分析方案。首先通过理论分析和实验对比验证了小数基音延迟相关性作为隐写特征的有效性；其次，摒弃了“手工”寻找特征的传统方式，通过采用深度神经网络获取编码参数的相关性，分别设计了基于局部相关性的检测模型、基于全局相关性的检测模型以及基于特征融合的检测模型；最后，以上述三种模型为基础，结合基于线性回归的多模型融合思想，给出了7种检测模式，即3种单一模型检测模式和4种多模型融合检测模式。通过大量的语音样本，对方案进行了性能评估，并与相关工作进行了实验对比分析。实验结果表明，方案中提出的各种检测模式均是可行和有效的，其中三模型融合检测模式整体性能最优。此外，本文工作填补了基于小数基音延迟隐写检测的空白，且较之已有方案对于各类基音延迟隐写方法在任意的嵌入率和样本长度下均具有更好的检测性能和更低的时间开销，从而实现了更为实时高效的检测。


## 数据集

采用清华大学研究团队提供的[公开语音数据集]((https://github.com/fjxmlzn/RNN-SM))。该数据集中的语音样本均来自互联网，共包含41小时的中文语音和72小时的英文语音：

- 中文语音：

  [Chinese.tar.gz](https://drive.google.com/file/d/1LF2dAXHkd8TmzaDnTg0Zmbs7xVdSovMH/view?usp=sharing): 160 pieces of speech in wav format.

- 英文语音：

  [English.tar.gz](https://drive.google.com/file/d/1Uy7WyEg3y-hvefUczo_6gFyyeeTC6ohg/view?usp=sharing): 160 pieces of speech in wav format.

>该数据集被以下论文广泛采用
>
>- **RSM**:Lin, Z., Huang, Y., & Wang, J. (2018). RNN-SM: Fast steganalysis of VoIP streams using recurrent neural network. IEEE Transactions on Information Forensics and Security, 13(7), 1854-1868. [Github Page](https://github.com/fjxmlzn/RNN-SM).
>
>- **FSM**:Yang, H., Yang, Z., Bao, Y., Liu, S., & Huang, Y. (2019). Fast steganalysis method for voip streams. IEEE Signal Processing Letters, 27, 286-290.[Github Page](https://github.com/YangzlTHU/VoIP-Steganalysis)
>- **SFFN**:Hu, Y., Huang, Y., Yang, Z., & Huang, Y. Detection of heterogeneous parallel steganography for low bit-rate VoIP speech streams. Neurocomputing, 419, 70-79. [Github Page](https://github.com/YangzlTHU/VStego800K/blob/main/Steganalysis).
>- **SCRN**: Gong C, Yi X, Zhao X, et al. Recurrent convolutional neural networks for AMR steganalysis based on pulse position[C]//Proceedings of the ACM Workshop on Information Hiding and Multimedia Security. 2019: 2-13. [Github Page](https://github.com/VOIPsteganalysis/FCBsteganalysis)
>
>etc...

## 预训练模型

训练好的模型下载链接：（1）[GoogleDrive](https://drive.google.com/file/d/1Qzn084beTdEvwaWb1A0PfdwB4WFgcmXG/view?usp=sharing)。

**命名格式**：模型名称\_样本长度\_嵌入率\_隐写名称\_语种.h5。

**模型名称**：

- LCDM：局部相关性检测模型。
- GCDM：全局相关性检测模型。
- FFDM： 特征融合检测模型。
- LR：基于线性回归的多模型融合模型。

**样本长度**：

- 例如1秒样本长度写为"1.0s"。

**嵌入率**：

- 例如100%嵌入率写为"100"。

**隐写名称**：

- S1：Huang Y F, Liu C, Tang S Y, Bai S. Steganography integration into a low-bit rate speech codec. IEEE Transactions on Information Forensics and Security, 2012, 7(6): 1865–1875.
- S2：严书凡,汤光明,孙怡峰.基于基音周期预测的低速率语音隐写.计算机应用研究, 2015, 32(6): 1774-1777.
- S3：Liu X K, Tian H, Huang Y F, Lu J. A novel steganographic method for algebraic-code-excited-linear-prediction speech streams based on fractional pitch delay search. Multimedia Tools and Applications, 2019, 78(7): 8447–8461.

**语种**：

- CN：中文样本集。
- EN： 英文样本集。


# 基于基音延迟的隐写分析方法

- **SM-MDNNF(提出的方法)**: 田晖, 吴俊彦，严艳， 王慧东，全韩彧. 基于小数基音延迟相关性的自适应多速率语音流隐写分析. 计算机学报，2022，45（6）:1308-1325.
- **C-MSDPD**: 	Ren Y Z, Yang J, Wang J, Wang L. AMR steganalysis based on second-order difference of pitch delay. IEEE Transactions on Information Forensics and Security, 2017, 12(6): 1345–1357.
- **PBP**: Liu X K, Tian H, Liu J, Li X, Lu J. Steganalysis of adaptive multiple-rate  speech  using  parity  of  pitch-delay  value//Proceedings  of  the Security  and  Privacy  in  New  Computing  Environments.  Tianjin, China, 2019, 282–297.
- **HYBRID**: Tian H, Huang M, Chang C C, et al. Steganalysis of Adaptive Multi-Rate Speech Using Statistical Characteristics of Pitch Delay[J]. JUCS-Journal of Universal Computer Science, 2019, 25: 1131.
