import pandas as pd
import numpy as np

# 定义 safe_eval 函数
def safe_eval(val):
    if isinstance(val, str):
        return eval(val)
    return val

# 定义 LCS 函数
def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:  # 只考虑电影ID
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    
    return L[m][n]

# 定义 Sequence Order Preservation 函数
def sequence_order_preservation(seq, most_similar_seq):
    # 提取电影ID
    seq_ids = [item[0] for item in seq]
    most_similar_seq_ids = [item[0] for item in most_similar_seq]
    
    lcs_length = lcs(seq_ids, most_similar_seq_ids)
    if min(len(seq_ids), len(most_similar_seq_ids)) == 0:
        return 0
    return lcs_length / min(len(seq_ids), len(most_similar_seq_ids))

# 定义 DCG 和 nDCG 函数
def dcg_at_k(r, k):
    """计算前k个位置的DCG值"""
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.0

def ndcg_at_k(r, k):
    """计算前k个位置的nDCG值"""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max

# 解析 seq 和 most_similar_seq 列
def parse_sequences(df):
    df['seq'] = df['seq'].apply(safe_eval)
    df['most_similar_seq'] = df['most_similar_seq'].apply(safe_eval)
    return df

# 计算 Sequence Order Preservation
def calculate_sequence_order_preservation(df):
    df['sequence_order_preservation'] = df.apply(
        lambda row: sequence_order_preservation(row['seq'], row['most_similar_seq']), axis=1)
    return df

# 计算 nDCG@1, nDCG@2, nDCG@3
def calculate_ndcg(df):
    k_list = [1, 2, 3]
    for k in k_list:
        df[f'ndcg@{k}'] = df.apply(lambda row: ndcg_at_k([rating for _, rating in row['most_similar_seq']], k), axis=1)
    return df

# 主函数
def main():
    # 加载数据
    df = pd.read_pickle('/workspace/LLaRA/data/ref/movielens/similar_test_data.df')
    
    # 解析序列
    df = parse_sequences(df)
    
    # 计算 Sequence Order Preservation
    df = calculate_sequence_order_preservation(df)
    
    # 计算 nDCG@1, nDCG@2, nDCG@3
    df = calculate_ndcg(df)
    
    # 输出结果
    print(df[['sequence_order_preservation', 'ndcg@1', 'ndcg@2', 'ndcg@3']])
    
    # 保存更新后的 DataFrame 到 CSV 文件
    df.to_csv('/workspace/LLaRA/data/ref/movielens/similar_test_data_updated.csv', index=False)

# 运行主函数
if __name__ == "__main__":
    main()
