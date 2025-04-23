import numpy as np

# 设置输出格式，避免使用科学计数法
np.set_printoptions(suppress=True)

def a(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    if np.linalg.matrix_rank(eigenvectors) == A.shape[0]:
        # Step 3: 构造对角化矩阵D
        D = np.diag(eigenvalues)
        
        # Step 4: 构造可逆矩阵P
        P = eigenvectors
        
        # Step 5: 计算P的逆矩阵
        P_inv = np.linalg.inv(P)
        
        # Step 6: 计算对角化后的矩阵
        A_diagonalized = P_inv @ A @ P
        
        # 输出结果
        print("矩阵A可对角化!")
        print("对角矩阵D：")
        print(D)
        print("可逆矩阵P：")
        print(P)
        print("P的逆矩阵P_inv：")
        print(P_inv)
        print("对角化后的矩阵A_diagonalized：")
        print(A_diagonalized)
        
        return True, D, P, P_inv, A_diagonalized
    else:
        print("矩阵A不可对角化!")
        return False, None, None, None, None

# 示例：使用一个矩阵进行测试
A = np.array([[0, 1, 1], [1, 2, 1], [1, 1, 0]])
a(A)
