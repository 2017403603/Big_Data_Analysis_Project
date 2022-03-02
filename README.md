# Big_Data_Analysis_Project

本仓库用于存储研究生期间大数据分析课程的竞赛大作业——图书推荐系统，比赛网址：[https://www.datafountain.cn/competitions/542](https://www.datafountain.cn/competitions/542)，；文件包含比赛所用代码：1. NCF 2. SSR 3. LF

<img src=".\图片资源\图1.png" alt="image-20220302120047817" style="zoom: 33%;" /><img src=".\图片资源\图2.png" alt="image-20220302120244816" style="zoom: 33%;" />

# **图书推荐系统**

## 一、**赛题简介**

该赛题为DataFoutain中的一道训练赛题目，赛题任务是依据真实世界中的用户-图书交互记录，利用机器学习相关技术，建立一个精确稳定的图书推荐系统，预测用户可能会进行阅读的书籍。

## 二、**训练集和测试集介绍**

数据集来自公开数据集Goodbooks-10k，包含网站Goodreads中对10,000本书共约6,000,000条评分。为了预测用户下一个可能的交互对象，数据集已经处理为隐式交互数据集。该数据集广泛的应用于推荐系统中。

### **2.1****测试集**

测试集为用户-图书交互记录，共分为两列，表头为User_id和Item_id，示例如下：

| 字段名  | 类型 | 字段解释             |
| ------- | ---- | -------------------- |
| User_id | Int  | 用户ID               |
| Item_id | Int  | 用户可能交互的物品ID |

### **2****.2训练集**

测试集只有一列，为即将预测的User_id，示例：

| 字段名  | 类型 | 字段解释         |
| ------- | ---- | ---------------- |
| User_id | Int  | 需要预测的用户ID |

## 三、**EDA-数据探索性分析**

### **3.1数据集特点**

比赛方将数据集解释成用户产品的隐式交互记录，数据集本身结构较为简单。

3.2 **初步分析结果**

大约5万个用户，一万本图书，共6M条记录。数据集较大，且构造出来的矩阵十分稀疏。

**3****.3初步推荐方法简述**

首先加载数据，并划分训练集和验证集。搭建出一个隐式推荐模型，并构建负样本，最终按照模型输出的评分进行排序，做出最终的推荐，最终评测标准为F1值得分。

## 四、**模型对比分析**

**4.1 LightGCN**

传统GCN中最常见的两种设计-特征转换和非线性激活，对于协同过滤并没有那么有用。在协同过滤中，用户-商品交互图的每个节点只有一个ID作为输入，没有具体的语义，如图1。

<img src=".\图片资源\图3.jpg" alt="img" style="zoom: 67%;" /> 

在这种情况下，执行多个非线性转换不会有助于学习更好的特性，而且它们会增加训练的难度，降低推荐性能。为了使其更简洁、更适合推荐，提出了一种新的协同过滤模型LightGCN。该模型只包含了GCN中最基本的部分-邻域聚集。LightGCN通过在用户-商品交互图上线性传播用户和商品的embedding，并使用在所有层学习的embedding的加权和作为最终embedding。架构如图2：

<img src=".\图片资源\图4.jpg" alt="img" style="zoom: 67%;" /> 

GCN的基本思想是通过平滑图上的特征来学习节点的表示。在LightGCN中，只对下一层进行规范化的邻域嵌入和，去除了自连接、特征变换、非线性激活等操作，极大地简化了GCN。在层组合中，我们对每个层的嵌入进行求和，以获得最终的表示，如图3。

<img src=".\图片资源\图5.jpg" alt="img" style="zoom:80%;" /> 

**4****.2 PEAGNN**

PEAGNN是基于融合元路径和实体感知的图神经网络协同过滤算法，在图神经网络协同过滤算法中引入元路径（如图4），基于元路径的随机游走生成用户和商品的潜在特征，并引入融合函数及权重对多条元路径的表征融合。

<img src=".\图片资源\图6.jpg" alt="img" style="zoom:67%;" /> 

为了充分利用协同子图的局部结构，提出了一种基于实体感知的协同子图局部结构。PEAGNN架构主要分为三层：1.元路径聚合层，它显式地聚合元路径，来感知子图的信息。2.元路径融合层，它使用注意力机制融合来自多个元路径感知子图的聚合节点表示。3.预测层，提取匹配分数预测的图级表示，估计潜在用户-商品交互的可能性

<img src=".\图片资源\图7.jpg" alt="img" style="zoom:80%;" /> 

**4****.3 LightGCN & PEAGNN**

LightGCN和PEAGNN都是基于NGCF模型，在使用中各有优缺点。LightGCN的特点有：1.在LightGCN中，唯一可训练的模型参数就是第0层的embedding。2.简单、线性和简洁的模型更易于实现和训练。PEAGNN的特点有：1.基于元路径的随机游走, 能更好的涵盖异质网中丰富的语义信息。2.有效地避免了冷启动及数据稀疏。

这两个模型相比图神经协作过滤显示出显著的改进，但相应的缺点也很明显，训练时间特别长，大量embedding很慢。

**4.4** **NCF**

很多应用场景，并没有显性反馈的存在。因为大部分用户是沉默的用户，并不会明确给系统反馈“我对这个物品的偏好值是多少”。因此，推荐系统可以根据大量的隐性反馈来推断用户的偏好值。

为此NCF提出如下框架：

<img src=".\图片资源\图8.jpg" alt="img" style="zoom:80%;" /> 

Input layer: V_u和V_i表示用户u和物品I

Embedding Layer这一层就将: input layer[稀疏向量](https://www.zhihu.com/search?q=稀疏向量&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={)变稠密

Neural CF Layers: 多个隐藏层实现非线性转化，文中使用的3层，ReLU激活函数

Output Layer :最后输出一个分数，最小化预测值和真实值的pointwise loss求解模型（也有pairwise loss求解的）

**4****.****5** **LT-OCF**

<img src=".\图片资源\图9.jpg" alt="img" style="zoom:67%;" /> 

为了解决时间模式的建模，该文提出了一个将时间信息加入模型建模中，提出learnable-time architecture，并在ODE中加入了残差结构（red）。如上图为提出方法的pipeline，下式为具体做法：

<img src=".\图片资源\图10.jpg" alt="img" style="zoom:80%;" /> 

**4****.****6** **NCF & LT-OCF**

两者都是协同过滤的方法，NCF将传统的矩阵分解改为通过Neural Network实行协同过滤，而LTOCF在传统NCF的基础上，进一步设计了网络结构与模型建模定义，具体来说提出ODE结果内的残差与可以学习的时间特征。

NCF：

Pros：模型简洁易与实现。

Cons：基于协同过滤的思想进行改造，所以NeuralCF模型并没有引入更多其他类型的特征，浪费其他有价值的信息，因此性能可能不如LTOCF

LTOCF：

Pros: 提出的learnable-time 与ODE内部的残差链接能够将时t融入为层级变量。

Cons: 结构较为复杂，训练较慢，复现结果表现一般。

**4****.****6** **SASRec****（实验结果最优）**

<img src=".\图片资源\图11.jpg" alt="img" style="zoom:67%;" /> 

l Input: 定义一个用户的行为序列             ，用于预测下一个用户可能发生交的物品但需要依赖之前用户的交互历史。

l Embedding Layer: 添加Positional Embedding表示序列中的先后关系，再与行为序列相加。

l Self-Attention: Stacking(Self-Attention Block+Feed Forward)并通过加入残差连接、layer normalization和dropout解决过拟合、梯度消失、训练时间长的问题。

l Output Layer :通过对用户行为序列和物品Embedding矩阵作内积得出user-item相关性矩阵，之后将分数排序筛选完成推荐

**4****.****7** **Pros in SASRec**

1.数据集相较dense，SASRec采用了Self-Attention能很好的描述用户多兴趣的行为序列。

2.相较于CF和GCN等方法来说，SASRec模型可以自适应的处理稀疏和密集数据，而且时间复杂度主要取决于Self-Attention Layer，可以完全并行化。训练速度优于GCN等模型十倍。（1.7s/epoch）

3.采用了Stacking Self-Attention，对于密集数据集，用户平均行为序列越长，越适合采用SASRec模型。

**4****.****8** **SASRec****模型调参**

1.Positional Embedding: 在稠密的数据集上,添加PE效果提升很大；

2.共享IE(Item Embedding): 使用共享的item embedding比不使用要好很多；

3.RC(Residual Connection):不实用残差连接,性能会变差非常多;

4.Dropout: Dropout可以帮助模型避免过拟合,由于赛题数据集较为稠密且测试数据集和训练数据集分布相近，不易设置过大的Dropout;

5.blocks的个数: 没有block的时候,效果最差,在赛题数据集上,block越多效果越好;

## 五、**模型改进和提高**

**5.1** **思考分析**

***\*数据集特点：\****

比赛方将数据集解释成用户产品的隐式交互记录，但实际上在开源数据集中，完整的数据是包含着用户-产品-评分三元组以及交互时间戳、产品类别和用户评论等内容的显示反馈数据。

***\*模型结构和表现：\****

SASRec模型用四层卷积+两层自注意力+50维输入向量在论文数据集上得到了很好的结果。但是在比赛数据上效果相对一般，简单的调试参数没有办法得到进一步的提升。

***\*结论：\****

我们认为关键的问题在于论文中的数据是隐式交互数据而比赛数据是用户在交互之后对认可的产品采取进一步行为的显示反馈，这种用户-产品交互理应包含着更加重要更加丰富的信息。应该用更大的函数空间来容纳用户和产品特征，这样可能会有好的效果。

**5.2** **加深层数**

我们尝试了多种结构改进如下：

| ***\*S\*******\*ASR\*******\*ec\**** | ***\*Att=2,Conv=4\*******\*Emb=100\**** | ***\*Att=4,Conv=4\*******\*Emb=100\**** | ***\*Att=8,Conv=8\*******\*Emb=200\**** |
| ------------------------------------ | --------------------------------------- | --------------------------------------- | --------------------------------------- |
| Average loss                         | 0.35-0.37                               | 0.34-0.37                               | 0.35-0.38                               |
| HR@10                                | 0.591                                   | 0.590                                   | 0.588                                   |
| nDCG@10                              | 0.732                                   | 0.732                                   | 0.731                                   |

但是这种做法也没有什么好的突破，我们转而寻求更强力的特征提取模型。

**5.3 Transformer**

Transformer 是 Google 的团队在 2017 年提出的一种 NLP 经典模型，现在比较火热的 Bert 也是基于 Transformer。Transformer 模型使用了 Self-Attention 机制，不采用 RNN 的顺序结构，使得模型可以并行化训练，而且能够拥有全局信息。

我们考虑用Transformer中的多头注意力+前反馈神经网络作为特征提取器，配合更大维度空间实现对用户和产品向量的更好拟合。

<img src=".\图片资源\图12.jpg" alt="img" style="zoom:50%;" /> 

**5.4** **实现效果**

| ***\*SASRec\****                      | ***\*Att=2,Conv=4\*******\*Emb=100\****        | ***\*Att=4,Conv=4\*******\*Emb=100\****        | ***\*Att=8,Conv=8\*******\*Emb=200\****        |
| ------------------------------------- | ---------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| Average loss                          | 0.35-0.37                                      | 0.34-0.37                                      | 0.35-0.38                                      |
| HR@10                                 | 0.591                                          | 0.590                                          | 0.588                                          |
| nDCG@10                               | 0.732                                          | 0.732                                          | 0.731                                          |
| 比赛测试                              | 0.1196                                         |                                                |                                                |
| ***\*SASRec+\*******\*Multi-Head\**** | ***\*Att=2,Conv=4\*******\*Head=2,Emb=100\**** | ***\*Att=4,Conv=4\*******\*Head=4,Emb=200\**** | ***\*Att=4,Conv=8\*******\*Head=4,Emb=200\**** |
| Average loss                          | 0.30-0.35                                      | 0.27-0.29                                      | 0.24-0.26                                      |
| HR@10                                 | 0.611                                          | 0.632                                          | 0.688                                          |
| nDCG@10                               | 0.732                                          | 0.782                                          | 0.801                                          |
| 比赛测试                              |                                                |                                                | 0.1410                                         |

## 六、**比赛结果**

我们的模型在图书推荐比赛中拿到了***\*第二名\****

<img src=".\图片资源\图13.jpg" alt="img" style="zoom:80%;" /> 

## 七、**人员分工**

***\*组长：\****

刘佳迎：NCF、LT-OCF、LightGCN等模型改进，文档撰写、PPT制作。

***\*组员：\****

郑翊风：NCF、PEAGNN、SASRec等模型改进，文档撰写、PPT制作。

赵唯松：NCF、LT-OCF、PEAGNN等模型改进，文档撰写、PPT制作。

马意：LT-OCF、PEAGNN、LightGCN等模型改进，文档撰写、PPT制作。

李航程：LT-OCF、PEAGNN、LightGCN等模型改进，文档撰写、PPT制作。

## 八、**参考文献**

[1] He X, Liao L, Zhang H, et al. Neural collaborative filtering[C]//Proceedings of the 26th international conference on world wide web. 2017: 173-182.

[2] Choi J, Jeon J, Park N. LT-OCF: Learnable-Time ODE-based Collaborative Filtering[C]//Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021: 251-260.

[3] He X, Deng K, Wang X, et al. Lightgcn: Simplifying and powering graph convolution network for recommendation[C]//Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 2020: 639-648.

[4] Han Z, Anwaar M U, Arumugaswamy S, et al. Metapath-and Entity-aware Graph Neural Network for Recommendation[J]. arXiv e-prints, 2020: arXiv: 2010.11793.

[5] Kang W C, McAuley J. Self-attentive sequential recommendation[C]//2018 IEEE International Conference on Data Mining (ICDM). IEEE, 2018: 197-206.

 