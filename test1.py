import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


X = pd.read_csv("X.csv")
#X = pd.read_csv("C:/Users/Chen Li/Desktop/RugPullDetection-main/RugPullDetection-main/ML/X.csv")
X = X.set_index("token_address")
labels = pd.read_csv("labeled_list.csv", index_col="token_address")
#labels = pd.read_csv("C:/Users/Chen Li/Desktop/RugPullDetection-main/RugPullDetection-main/ML/labeled_list.csv", index_col="token_address")
X = X.merge(labels['label'], left_index=True, right_index=True)
X = X.reset_index()
df = X.drop_duplicates(subset=['token_address'])
X = X.set_index("token_address")
lock_features = pd.read_csv("token_lock_features.csv", index_col="token_address")

#lock_features = pd.read_csv("C:/Users/Chen Li/Desktop/RugPullDetection-main/RugPullDetection-matoken_lock_features.csv", index_col="token_address")
X = X.merge(lock_features, how='left', left_index=True, right_index=True)
#inf_mask = np.isinf(X==True)


ids = []
total_probs = []
total_targets = []

skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
for fold, (t, v) in enumerate(skfolds.split(df['token_address'], df['label'])):

    ids_train = df['token_address'].iloc[t]
    df_train = X.loc[ids_train]
    ids_test = df['token_address'].iloc[v]
    df_test = X.loc[ids_test]

    X_train, y_train = df_train.drop(["label", "eval_block"], axis=1), df_train['label']
    X_test, y_test = df_test.drop(["label", "eval_block"], axis=1), df_test['label']

    columns = X_train.columns

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    feature_importances = model.feature_importances_
    print("Feature Importances:")
    print(feature_importances)
    preds_scorings = model.predict_proba(X_test)[:, 1]
    ## preds_scorings 是一个二维数组，每一行代表一个样本，每一列代表一个类别的概率
    
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    #y_test 是测试集的真实标签。
    #preds 是模型对测试集的预测标签。
    
    sensibilitat = recall_score(y_test, preds)
    precisio = precision_score(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    print("{},{},{},{},{}".format(accuracy, sensibilitat, precisio, f1, fold))
    ids += X_test.index.tolist()
    total_probs += preds.tolist()
    total_targets += y_test.tolist()

final_df = pd.DataFrame({'ids': ids, 'Pred': total_probs, 'Label': total_targets})\
    .to_csv("Results_XGBoost.csv", index=False)