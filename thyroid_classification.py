import os
import cv2
import numpy as np
from skimage import measure, feature
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
def preprocess_image(image_path, new_width, new_height):
    original_image = cv2.imread(image_path)
    resized_image = cv2.resize(original_image, (new_width, new_height))
    right_pixels_to_remove = 30
    left_pixels_to_remove = 28
    bottom_pixels_to_remove = 23
    # 裁剪图像，去除左右和底部的部分像素
    cropped_image = resized_image[:, left_pixels_to_remove:-right_pixels_to_remove, :]
    cropped_image = cropped_image[:-bottom_pixels_to_remove, :]
    # 对裁剪后的图像进行中值滤波
    median_filtered_image = cv2.medianBlur(cropped_image, 3)
    return median_filtered_image


def calculate_lbp_features(image):
# 检查图像的通道数
    if len(image.shape) == 2:  # 图像是灰度图
        gray_image = image
    else:  # 图像是彩色图
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_image = feature.local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 10), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def calculate_glcm_features(gray_image):
    glcm = feature.graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = feature.graycoprops(glcm, 'contrast')[0][0]
    correlation = feature.graycoprops(glcm, 'correlation')[0][0]
    energy = feature.graycoprops(glcm, 'energy')[0][0]
    entropy = -np.sum(glcm * np.log(glcm + 1e-7))  # Avoid log(0)
    return contrast, correlation, energy, entropy

def extract_features(sample):
    image_path = sample["image_path"]
    final_image = preprocess_image(image_path, new_width, new_height)
    
    # 计算灰度特征
    gray_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray_image)
    std_intensity = np.std(gray_image)
    hist, _ = np.histogram(gray_image.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    # 计算 LBP 特征
    lbp_features = calculate_lbp_features(gray_image)

    # 计算 GLCM 特征
    contrast, correlation, energy, entropy = calculate_glcm_features(gray_image)

    # 计算形态特征
    labeled_image = measure.label(gray_image > 0.5)
    regions = measure.regionprops(labeled_image)
    
    if regions:
        area = regions[0].area
        perimeter = regions[0].perimeter
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        major_axis_length = regions[0].major_axis_length
        minor_axis_length = regions[0].minor_axis_length
        aspect_ratio = major_axis_length / minor_axis_length if minor_axis_length > 0 else 0
    else:
        area = perimeter = circularity = major_axis_length = minor_axis_length = aspect_ratio = 0

    # Combine all features into a single array
    features = np.concatenate([
        lbp_features,
        [mean_intensity, std_intensity],
        [contrast, correlation, energy, entropy],
        [area, perimeter, circularity, aspect_ratio]
    ])
    
    return features

def load_images(folder):
    data = []
    for filename in os.listdir(folder):
        if filename.endswith(".bmp"):
            img_path = os.path.join(folder, filename) 
            data.append({"image_path": img_path})
    return data

benign_folder = r"classify\thyroid-nodule\0"
malignant_folder = r"classify\thyroid-nodule\1"

benign_data = load_images(benign_folder)
malignant_data = load_images(malignant_folder)

# 统一像素
new_width = 256
new_height = 256

# Create a list to store features and labels
X = []
y = []

# Organizing data and labels
for sample in benign_data:
    features = extract_features(sample)
    X.append(features)
    y.append(0)  # Label 0 for benign

for sample in malignant_data:
    features = extract_features(sample)
    X.append(features)
    y.append(1)  # Label 1 for malignant

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# 首先将数据划分为训练集（70%）和临时集（30%）
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 然后从临时集中划分出验证集（15%）和训练集（85%），即从 30% 的临时集中划分出 50% 作为验证集
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 使用 SMOTE 进行数据平衡
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# 特征选择与降维
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_balanced)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# RFE 特征选择
#model = RandomForestClassifier() #faster
model = SVC()
rfe = RFE(model, n_features_to_select=5)
X_train_rfe = rfe.fit_transform(X_train_pca, y_train_balanced)
X_val_rfe = rfe.transform(X_val_pca)
X_test_rfe = rfe.transform(X_test_pca)

# 创建 SVM 分类器，启用概率估计
svm_classifier = SVC(kernel='linear', C=1.0, probability=True)

# 训练 SVM 模型
svm_classifier.fit(X_train_rfe, y_train_balanced)

# 验证模型
y_val_pred = svm_classifier.predict(X_val_rfe)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_report = classification_report(y_val, y_val_pred, zero_division=1)
print("Validation Accuracy:", val_accuracy)
print("Validation Classification Report:\n", val_report)

# 预测测试集
y_pred = svm_classifier.predict(X_test_rfe)
# 评估模型
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Test Accuracy:", accuracy)
print("Test Classification Report:\n", report)

# 混淆矩阵分析
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# ROC 曲线
y_prob = svm_classifier.predict_proba(X_test_rfe)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 准确率—召回率曲线
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.show()