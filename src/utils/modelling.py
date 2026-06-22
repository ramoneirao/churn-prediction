import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, auc
from lightgbm import LGBMClassifier
import time
from warnings import filterwarnings

filterwarnings('ignore')

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, to_drop):
        self.to_drop = to_drop

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self.to_drop = [col for col in self.to_drop if col in X.columns]
        return X.drop(columns=self.to_drop)
    
class OneHotFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, to_encode):
        self.to_encode = to_encode
        self.encoder = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int8, handle_unknown='ignore')

    def fit(self, X, y=None):
        self.encoder.fit(X[self.to_encode])
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X_one_hot = self.encoder.transform(X[self.to_encode])
        one_hot_df = pd.DataFrame(X_one_hot, columns=self.encoder.get_feature_names_out(self.to_encode))
        X_reset = X.reset_index(drop=True)
        return pd.concat([X_reset.drop(columns=self.to_encode), one_hot_df], axis=1)
    
class StandardFeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, to_scale):
        self.to_scale = to_scale
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        self.scaler.fit(X[self.to_scale])
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X[self.to_scale])
        scaled_df = pd.DataFrame(X_scaled, columns=self.scaler.get_feature_names_out(self.to_scale))
        X_reset = X.reset_index(drop=True)
        return pd.concat([X_reset.drop(columns=self.to_scale), scaled_df], axis=1)
    
class OrdinalFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, to_encode):
        self.to_encode = to_encode
        self.encoder = OrdinalEncoder(dtype=np.int8, categories=[to_encode[col] for col in to_encode])

    def fit(self, X, y=None):
        self.encoder.fit(X[list(self.to_encode.keys())])
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X_ordinal = self.encoder.transform(X[list(self.to_encode.keys())])
        ordinal_encoded_df = pd.DataFrame(X_ordinal, columns=self.encoder.get_feature_names_out(list(self.to_encode.keys())))
        X_reset = X.reset_index(drop=True)
        return pd.concat([X_reset.drop(columns=list(self.to_encode.keys())), ordinal_encoded_df], axis=1)
    
class TargetFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, to_encode):
        self.to_encode = to_encode
        self.encoder = TargetEncoder()

    def fit(self, X, y):
        self.encoder.fit(X[self.to_encode], y)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X_target = self.encoder.transform(X[self.to_encode])
        target_df = pd.DataFrame(X_target, columns=self.encoder.get_feature_names_out(self.to_encode))
        X_reset = X.reset_index(drop=True)
        return pd.concat([X_reset.drop(columns=self.to_encode), target_df], axis=1)
    
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X_copy = X.copy()
        discrete_features = ['customer_age', 'dependent_count', 'months_on_book', 'total_relationship_count', 'months_inactive_12_mon', 'contacts_count_12_mon', 'total_trans_ct']
        continuous_features = ['credit_limit', 'total_revolving_bal', 'total_amt_chng_q4_q1', 'total_trans_amt', 'total_ct_chng_q4_q1']
        
        X_copy[discrete_features] = X_copy[discrete_features].astype('int32')
        X_copy[continuous_features] = X_copy[continuous_features].astype('float32')
        
        X_copy['products_per_dependent'] = (X_copy['total_relationship_count'] / X_copy['dependent_count']).astype('float32')
        X_copy['trans_amt_per_dependent'] = (X_copy['total_trans_amt'] / X_copy['dependent_count']).astype('float32')
        X_copy['trans_ct_per_dependent'] = (X_copy['total_trans_ct'] / X_copy['dependent_count']).astype('float32')
        X_copy['trans_amt_per_products'] = (X_copy['total_trans_amt'] / X_copy['total_relationship_count']).astype('float32')
        X_copy['trans_ct_per_products'] = (X_copy['total_trans_ct'] / X_copy['total_relationship_count']).astype('float32')
        X_copy['avg_trans_amt'] = (X_copy['total_trans_amt'] / X_copy['total_trans_ct']).astype('float32')
        X_copy['credit_util_rate'] = (X_copy['total_revolving_bal'] / X_copy['credit_limit']).astype('float32')
        X_copy['proportion_inactive_months'] = (X_copy['months_inactive_12_mon'] / X_copy['months_on_book']).astype('float32')
        X_copy['products_per_tenure'] = (X_copy['total_relationship_count'] / X_copy['months_on_book']).astype('float32')
        X_copy['products_per_contacts'] = (X_copy['total_relationship_count'] / X_copy['contacts_count_12_mon']).astype('float32')
        X_copy['dependents_per_contacts'] = (X_copy['dependent_count'] / X_copy['contacts_count_12_mon']).astype('float32')
        X_copy['trans_ct_per_contacts'] = (X_copy['total_trans_ct'] / X_copy['contacts_count_12_mon']).astype('float32')
        X_copy['products_per_inactivity'] = (X_copy['total_relationship_count'] / X_copy['months_inactive_12_mon']).astype('float32')
        X_copy['dependents_per_inactivity'] = (X_copy['dependent_count'] / X_copy['months_inactive_12_mon']).astype('float32')
        X_copy['trans_ct_per_inactivity'] = (X_copy['total_trans_ct'] / X_copy['months_inactive_12_mon']).astype('float32')
        X_copy['trans_amt_per_credit_limit'] = (X_copy['total_trans_amt'] / X_copy['credit_limit']).astype('float32')
        X_copy['age_per_tenure'] = (X_copy['customer_age'] / X_copy['months_on_book']).astype('float32')
        X_copy['trans_ct_per_tenure'] = (X_copy['total_trans_ct'] / X_copy['months_on_book']).astype('float32')
        X_copy['trans_amt_per_tenure'] = (X_copy['total_trans_amt'] / X_copy['months_on_book']).astype('float32')
        
        X_copy = X_copy.replace({np.inf: 0, np.nan: 0})
        
        X_copy['total_spending'] = (X_copy['total_trans_amt'] + X_copy['total_revolving_bal']).astype('int32')
        X_copy['inactivity_contacts'] = (X_copy['contacts_count_12_mon'] + X_copy['months_inactive_12_mon']).astype('int32')
        
        education_mapping = {'Uneducated': 0, 'High School': 1, 'College': 2, 'Graduate': 3, 'Post-Graduate': 4, 'Doctorate': 5, 'Unknown': 0}
        income_mapping = {'Less than $40K': 0, '$40K - $60K': 1, '$60K - $80K': 2, '$80K - $120K': 3, '$120K +': 4, 'Unknown': 0}
        
        X_copy['education_numeric'] = X_copy['education_level'].map(education_mapping).astype('int32')
        X_copy['income_numeric'] = X_copy['income_category'].map(income_mapping).astype('int32')
        X_copy['education_income_levels'] = (X_copy['education_numeric'] + X_copy['income_numeric']).astype('int32')
        X_copy = X_copy.drop(columns=['education_numeric', 'income_numeric'])
        
        return X_copy
    
class RecursiveFeatureEliminator(BaseEstimator, TransformerMixin):
    def __init__(self, estimator=LGBMClassifier(verbose=-1), scoring='roc_auc', n_folds=5):
        stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        self.rfe = RFECV(estimator=estimator, cv=stratified_kfold, scoring=scoring)

    def fit(self, X, y):
        self.rfe.fit(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X_selected = self.rfe.transform(X)
        selected_df = pd.DataFrame(X_selected, columns=self.rfe.get_feature_names_out())
        return selected_df

def classification_kfold_cv(models, X_train, y_train, n_folds=5):
    stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    models_val_scores = dict()
    models_train_scores = dict()

    for model in models:
        model_instance = models[model]
        start_time = time.time()
        model_instance.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time

        y_train_pred = model_instance.predict(X_train.values)
        train_score = roc_auc_score(y_train, y_train_pred)

        val_scores = cross_val_score(model_instance, X_train.values, y_train, scoring='roc_auc', cv=stratified_kfold)
        avg_val_score = val_scores.mean()
        val_score_std = val_scores.std()

        models_val_scores[model] = avg_val_score
        models_train_scores[model] = train_score

        print(f'{model} results: ')
        print('-'*50)
        print(f'Training score: {train_score}')
        print(f'Average validation score: {avg_val_score}')
        print(f'Standard deviation: {val_score_std}')
        print(f'Training time: {round(training_time, 5)} seconds\n')

    val_df = pd.DataFrame(list(models_val_scores.items()), columns=['model', 'avg_val_score'])
    train_df = pd.DataFrame(list(models_train_scores.items()), columns=['model', 'train_score'])
    eval_df = val_df.merge(train_df, on='model')
    eval_df = eval_df.sort_values(['avg_val_score'], ascending=False).reset_index(drop=True)
    return eval_df
    
def plot_classification_kfold_cv(eval_df, figsize=(20, 7), bar_width=0.35, title_size=15, title_pad=30, label_size=11, labelpad=20, legend_x=0.08, legend_y=1.08):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(eval_df['model']))
    val_bars = ax.bar(x - bar_width/2, eval_df['avg_val_score'], bar_width, label='Val score', color='#023047')
    train_bars = ax.bar(x + bar_width/2, eval_df['train_score'], bar_width, label='Train score', color='#0077b6')

    ax.set_xlabel('Model', labelpad=labelpad, fontsize=label_size, loc='left')
    ax.set_ylabel('ROC-AUC', labelpad=labelpad, fontsize=label_size, loc='top')
    ax.set_title("Models' performances", fontweight='bold', fontsize=title_size, pad=title_pad, loc='left')
    ax.set_xticks(x, eval_df['model'], rotation=0, fontsize=10.8)
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.tick_params(axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    
    for bar in val_bars + train_bars:
        height = bar.get_height()
        plt.annotate(f'{round(height, 2)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    handles = [plt.Rectangle((0,0), 0.1, 0.1, fc='#023047', edgecolor='none'), plt.Rectangle((0,0), 0.1, 0.1, fc='#0077b6', edgecolor='none')]
    labels = ['Val score', 'Train score']
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(legend_x, legend_y), frameon=False, ncol=2, fontsize=10)

def evaluate_classifier(y_true, y_pred, probas, y_test):
    print(classification_report(y_true, y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    brier_score = brier_score_loss(y_true, probas)
    print(f'Brier Score: {round(brier_score, 2)}')
    
    fpr, tpr, thresholds = roc_curve(y_true, probas)
    roc_auc = roc_auc_score(y_true, probas)
    gini = 2 * roc_auc - 1
    print(f'Gini: {round(gini, 2)}')
    
    scores = pd.DataFrame()
    scores['actual'] = y_test.reset_index(drop=True)
    scores['churn_probability'] = probas
    sorted_scores = scores.sort_values(by=['churn_probability'], ascending=False)
    sorted_scores['cum_negative'] = (1 - sorted_scores['actual']).cumsum() / (1 - sorted_scores['actual']).sum()
    sorted_scores['cum_positive'] = sorted_scores['actual'].cumsum() / sorted_scores['actual'].sum()
    sorted_scores['ks'] = np.abs(sorted_scores['cum_positive'] - sorted_scores['cum_negative'])
    ks = sorted_scores['ks'].max()
    
    print(f'KS: {round(ks, 2)}')
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Values')
    plt.ylabel('Real Values')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}', color='#023047')
    ax.plot([0, 1], [0, 1], linestyle='--', color='#e85d04') 
    ax.set_xlabel('False Positive Rate', fontsize=10.8, labelpad=20, loc='left')
    ax.set_ylabel('True Positive Rate', fontsize=10.8, labelpad=20, loc='top')
    ax.set_title('Receiver operating characteristic (ROC) curve', fontweight='bold', fontsize=12, pad=20, loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()

    p, r, _ = precision_recall_curve(y_true, probas)
    pr_auc = auc(r, p)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r, p, marker='.', label=f'PR AUC = {pr_auc:.2f}', color='#023047')
    ax.set_xlabel('Recall', fontsize=10.8, labelpad=20, loc='left')
    ax.set_ylabel('Precision', fontsize=10.8, labelpad=20, loc='top')
    ax.set_title('Precision-recall (PR) curve', fontweight='bold', fontsize=12, pad=20, loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()

    model_metrics = pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'KS', 'Gini', 'PR-AUC', 'Brier'],
                                  'Value': [accuracy, precision, recall, f1, roc_auc, ks, gini, pr_auc, brier_score]})
    return model_metrics

def plot_feature_importances(model, data):
    importances = model.feature_importances_
    feature_names = data.columns 
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(12, 3))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), sorted_importances, tick_label=sorted_feature_names, color='#023047')
    plt.xticks(rotation=90)
    plt.show()
    
def precision_vs_recall_curve(y_true, predicted_probas):
    precision, recall, threshold = precision_recall_curve(y_true, predicted_probas[:, 1])
    plt.title('Precision vs Recall Curve')
    plt.plot(threshold, precision[:-1], 'b--', label='Precision')
    plt.plot(threshold, recall[:-1], 'r--', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='lower left')
    plt.ylim([0,1])
    return precision, recall, threshold

def get_threshold_metrics(precision, recall, threshold, target_metric, target_metric_value):
    if target_metric == 'recall':
        recall_array = np.asarray(recall)
        target_value_index = np.where(recall_array[:-1] == target_metric_value)[0][0]
    else:
        precision_array = np.asarray(precision)
        target_value_index = np.where(precision_array[:-1] == target_metric_value)[0][0]
    
    threshold_precision = precision[target_value_index]
    threshold_recall = recall[target_value_index]
    threshold_selected = threshold[target_value_index]

    print(f'For a threshold of {threshold_selected}:')
    print(f'Recall: {threshold_recall}')
    print(f'Precision: {threshold_precision}')
    return threshold_precision, threshold_recall, threshold_selected
    
def plot_probability_distributions(y_true, probas):
    probas_df = pd.DataFrame({'churn_probability': probas, 'churn': y_true})
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.kdeplot(data=probas_df, x='churn_probability', hue='churn', fill=True, ax=ax, palette=['#023047', '#e85d04'])
    ax.set_title('Predicted probabilities distribution - churners and non-churners', fontweight='bold', fontsize=12, pad=45, loc='left')
    ax.set_xlabel('Predicted probabilities', fontsize=10.8, labelpad=20, loc='left')
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    handles = [plt.Rectangle((0,0), 0.1, 0.1, fc='#e85d04', edgecolor='none'), plt.Rectangle((0,0), 0.1, 0.1, fc='#023047', edgecolor='none')]
    labels = ['Churn', 'Not churn']
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.14, 1.15), frameon=False, ncol=2, fontsize=10)

def probability_scores_ordering(y_true, probas):
    noise = np.random.uniform(0, 0.0001, size=probas.shape)
    probas += noise
    probas_actual_df = pd.DataFrame({'probabilities': probas, 'actual': y_true.reset_index(drop=True)})
    probas_actual_df = probas_actual_df.sort_values(by='probabilities', ascending=True)
    probas_actual_df['deciles'] = pd.qcut(probas_actual_df['probabilities'], q=10, labels=False, duplicates='drop')
    decile_df = probas_actual_df.groupby(['deciles'])['actual'].mean().reset_index().rename(columns={'actual': 'churn_rate'})
    
    fig, ax = plt.subplots(figsize=(12, 3))
    bars = ax.bar(decile_df['deciles'], decile_df['churn_rate'], color='#023047')
    ax.set_title('Probability scores ordering - Churn rate per decile', loc='left', fontweight='bold', fontsize=14)
    ax.set_xticks(range(10), ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_xlabel('Decil', labelpad=25, loc='center')
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(False)
    
    for bar, absent_rate in zip(bars, decile_df['churn_rate']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.08, f'{absent_rate*100:.1f}%', ha='center', va='top', color='white', fontsize=10.4)
