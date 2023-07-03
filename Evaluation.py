import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def Evaluation(y_val, y_pred):
    """
    评估模型得分
    :param y_val:
    :param y_pred:
    :return:
    """
    con_matrix = confusion_matrix(y_val, y_pred)
    Draw_heatmap_matrix(con_matrix)
    Draw_heatmap__normalized_matrix(con_matrix)

    TP = con_matrix[0][0]
    FN = con_matrix[0][1]
    FP = con_matrix[1][0]
    TN = con_matrix[1][1]

    Human_accuracy = round(TP / (TP + FN), 6) * 100
    Machine_accuracy = round(TN / (FP + TN), 6) * 100
    False_Human_accuracy = round(FN / (TP + FN), 6) * 100
    False_Machine_accuracy = round(FP / (FP + TN), 6) * 100
    auc = round(accuracy_score(y_val, y_pred), 6) * 100

    print("Confusion Matrix:")
    print(con_matrix)
    print("对人/ 机器准确率:")
    print([Human_accuracy, Machine_accuracy])
    print("对人/ 机器误伤率:")
    print([False_Human_accuracy, False_Machine_accuracy])
    print("AUC:")
    print(auc)

def Draw_heatmap_matrix(matrix):
    """
    绘制混淆矩阵-热力图
    :param matrix:
    Confusion_Matrix.png
    :return:
    """
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(matrix, annot=True, cmap="Blues", fmt=".0f")
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.set_ylabel('True label', fontsize=11)
    ax.set_xlabel('Predicted label', fontsize=11)
    plt.savefig('Confusion_Matrix.png')
    plt.show()

def Draw_heatmap__normalized_matrix(matrix):
    """
    绘制标准化混淆矩阵-热力图
    :param matrix:
    :return: Normalized_Confusion_Matrix.png
    """
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(5, 5))
    matrix = matrix.tolist()
    Normalized_Confusion_Matrix = [[i/sum(matrix[0]) for i in matrix[0]], [j/sum(matrix[1]) for j in matrix[1]]]
    sns.heatmap(Normalized_Confusion_Matrix, annot=True, cmap="Blues", fmt=".4f")
    ax.set_title('Normalized Confusion Matrix', fontsize=14)
    ax.set_ylabel('True label', fontsize=11)
    ax.set_xlabel('Predicted label', fontsize=11)
    plt.savefig('Normalized_Confusion_Matrix.png')
    plt.show()
