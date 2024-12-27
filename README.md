# thyroid_classification
2024 BUPT 智能医学影像技术实验 实验五 甲状腺结节超声图像分类实验

# 实验五 甲状腺结节超声图像分类实验

## 一、实验目的
- 利用机器算法对甲状腺结节超声图像进行分类，通过提取图像特征并训练机器学习模型，实现对甲状腺结节良恶性的自动判别，为临床诊断提供辅助决策依据，提高诊断效率和准确性。

## 二、实验设备与环境
- **甲状腺结节超声图像数据集**：如图<img width="150" alt="dataset" src="https://github.com/user-attachments/assets/ba89482f-22f1-4ce3-9512-337a624d4516" />
- **计算机设备**：一台可支持图像数据处理和模型训练的计算机。
- **编程软件与库（以 Python 为例）**：
  - Python 编程语言及其相关科学计算库（如 NumPy、Pandas）用于数据处理和分析。
  - OpenCV 库用于图像读取、预处理和基本特征提取操作。
  - Scikit-learn 机器学习库用于构建模型并进行模型评估。

## 三、实验内容与步骤

### 1. 图像预处理 (10分)
1. **图像读取与灰度转换**：使用 OpenCV 库读取超声图像数据，并将彩色图像转换为灰度图像，降低数据维度，简化后续处理。
2. **图像裁剪与归一化**：根据结节在图像中的位置裁剪出感兴趣区域（ROI），去除无关背景信息，并对裁剪后的图像进行归一化处理。
3. **图像尺寸调整**：将所有图像调整为统一大小（如 128x128 像素），便于特征提取和模型输入。
4. **数据增强（可选）**：采用图像翻转、旋转、平移等数据增强技术，生成新的图像样本，确保与对应病理标签的一致性。

### 2. 特征提取 (20分)
1. **灰度特征**：计算图像的灰度均值、方差、灰度直方图统计信息等，这些特征可以反映结节内部回声的整体分布情况和均匀性。
2. **纹理特征**：运用灰度共生矩阵（GLCM）提取对比度、相关性、能量、熵等纹理特征。GLCM 描述了图像中不同灰度值像素对的空间分布关系，对于甲状腺结节的纹理分析具有重要意义。通过在不同方向（如 0°、45°、90°、135°）和不同距离上计算 GLCM，可以获取更全面的纹理信息。
3. **形态特征**：提取结节的周长、面积、圆形度、长轴短轴比等形态特征。这些特征有助于判断结节的形状规则性，恶性结节往往具有不规则的形态特征，与良性结节存在差异。

### 3. 特征选择与降维 (10分)
- **主成分分析（PCA）**：通过线性变换将原始特征投影到新的低维空间，保留主要特征信息。
- **递归特征消除（RFE）**：基于特征重要性评估逐步去除不重要的特征，使用 SVM 模型（建议使用RandomForestClassifier，比较快）作为评估器，每次迭代删除对模型贡献最小的特征，直到达到指定的特征数量或满足其他停止条件。通过 RFE 可以选择出与甲状腺结节分类最相关的特征子集。


### 4. 机器学习模型训练与优化（以 SVM 为例） (30分)
1. **数据划分**：将数据划分为训练集、验证集和测试集，通常按 70%、15%、15% 的比例，确保训练集用于模型训练，验证集用于模型参数调整和性能评估，测试集用于最终模型性能的独立测试。
2. **机器学习模型初始化**：创建 SVM 分类器对象，选择合适的核函数，并设置相关参数。
3. **模型训练**：使用训练集数据对 SVM 模型进行训练，调整模型参数以最小化损失函数。将提取的特征作为输入，对应的病理标签作为输出，通过优化算法（如梯度下降法或 SMO 算法）调整 SVM 模型的参数，使得模型在训练集上的损失函数最小化，从而学习到甲状腺结节特征与良恶性之间的关系。
4. **模型验证与参数调整**：使用验证集评估模型，调整参数以提高性能。采用交叉验证等技术寻找最优的参数组合，以提高模型在验证集上的性能。重复训练和验证过程，直到模型性能不再提升或达到预设的迭代次数。


### 5. 模型评估与可视化 (30分)
1. **性能指标计算**：计算准确率、精确率、召回率、F1 值、ROC 曲线下面积（AUC）等。
2. **混淆矩阵分析**：绘制混淆矩阵，展示分类结果分布情况。
3. **可视化图形**：绘制 ROC 曲线和准确率—召回率曲线，展示模型性能。

## 四、实验总结
- 对模型评估的各项指标结果进行统计分析，讨论本实验模型的优势和不足之处。分析实验过程中可能存在的问题，如数据质量、特征提取方法的有效性、模型选择与优化等方面，并提出改进建议和未来研究方向。同时，结合临床实际应用场景，探讨机器学习模型在辅助甲状腺结节诊断中的可行性和潜在价值。

##### 参考代码：https://github.com/ruchita-2/thyroid-cancer-classification 侵权删
