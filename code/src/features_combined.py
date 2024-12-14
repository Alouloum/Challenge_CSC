import pandas as pd

train_features_fadi= pd.read_csv('results/train_features_fadi.csv')
eval_features_fadi= pd.read_csv('results/eval_features_fadi.csv')
train_features_elie= pd.read_csv('results/train_features_elie.csv')
eval_features_elie= pd.read_csv('results/eval_features_elie.csv')

#merge on ID 
train_features_combined = pd.merge(train_features_fadi, train_features_elie, on=['ID','EventType'])
eval_features_combined = pd.merge(eval_features_fadi, eval_features_elie, on=['ID'])


#print columns  of the combined features
print(train_features_combined.columns)

train_features_combined.drop(columns=['Avg_Prob','Var_Prob','AvgRepeatedLetters','NoEventTerms_mean'], inplace=True)
eval_features_combined.drop(columns=['Avg_Prob','Var_Prob','AvgRepeatedLetters','NoEventTerms_mean'], inplace=True)

#save the combined features
train_features_combined.to_csv('results/train_features_combined.csv', index=False)
eval_features_combined.to_csv('results/eval_features_combined.csv', index=False)