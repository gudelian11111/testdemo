from sklearn.model_selection import StratifiedKFold

# 假设 df 是包含特征和目标变量的 DataFrame
skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 使用 split 方法获取训练集和测试集的索引
for train_index, test_index in skfolds.split(df[['feature1', 'feature2']], df['target']):
    # 在这里执行你的交叉验证逻辑
    X_train, X_test = df[['feature1', 'feature2']].iloc[train_index], df[['feature1', 'feature2']].iloc[test_index]
    y_train, y_test = df['target'].iloc[train_index], df['target'].iloc[test_index]
    # ... 其他交叉验证步骤