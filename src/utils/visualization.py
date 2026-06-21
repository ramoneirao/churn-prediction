import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ignorar avisos.
from warnings import filterwarnings
filterwarnings('ignore')

def analysis_plots(data, features, histplot=True, barplot=False, mean=None, text_y=0.5,    
                   outliers=False, boxplot=False, boxplot_x=None, kde=False, hue=None, 
                   nominal=False, figsize=(24, 12)):
    '''
    Gera gráficos para análise univariada e bivariada.

    Esta função gera histogramas, gráficos de barras horizontais
    e boxplots com base nos dados e variáveis fornecidos.
    '''
    
    # Obtém o número de features e o número de linhas para iterar sobre as dimensões do subplot.
    num_features = len(features)
    num_rows = num_features // 3 + (num_features % 3 > 0) 
    
    fig, axes = plt.subplots(num_rows, 3, figsize=figsize)  

    for i, feature in enumerate(features):
        row = i // 3  
        col = i % 3  

        ax = axes[row, col] if num_rows > 1 else axes[col] 
        
        if barplot:
            if mean:
                data_grouped = data.groupby([feature])[[mean]].mean().reset_index()
                data_grouped[mean] = round(data_grouped[mean], 2)
                ax.barh(y=data_grouped[feature], width=data_grouped[mean])
                for index, value in enumerate(data_grouped[mean]):
                    # Ajusta a posição do texto com base na largura das barras
                    ax.text(value + text_y, index, f'{value:.1f}', va='center', fontsize=15)
            else:
                if hue:
                    data_grouped = data.groupby([feature])[[hue]].mean().reset_index().rename(columns={hue: 'pct'})
                    data_grouped['pct'] *= 100
                else:
                    data_grouped = data.groupby([feature])[[feature]].count().rename(columns={feature: 'count'}).reset_index()
                    data_grouped['pct'] = data_grouped['count'] / data_grouped['count'].sum() * 100
    
                ax.barh(y=data_grouped[feature], width=data_grouped['pct'])
                
                if pd.api.types.is_numeric_dtype(data_grouped[feature]):
                    ax.invert_yaxis()
                    
                for index, value in enumerate(data_grouped['pct']):
                    # Ajusta a posição do texto com base na largura das barras
                    ax.text(value + text_y, index, f'{value:.1f}%', va='center', fontsize=15)
            
            ax.set_yticks(ticks=range(data_grouped[feature].nunique()), labels=data_grouped[feature].tolist(), fontsize=15)
            ax.get_xaxis().set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(False)
    
        elif outliers:
            # Plota boxplot univariado.
            sns.boxplot(data=data, x=feature, ax=ax)
        
        elif boxplot:
            # Plota boxplot multivariado.
            sns.boxplot(data=data, x=boxplot_x, y=feature, showfliers=outliers, ax=ax)

        else:
            # Plota histograma.
            sns.histplot(data=data, x=feature, kde=kde, ax=ax, stat='proportion', hue=hue)

        ax.set_title(feature)  
        ax.set_xlabel('')  
    
    # Remove os eixos (axes) não utilizados.
    if num_features < len(axes.flat):
        for j in range(num_features, len(axes.flat)):
            fig.delaxes(axes.flat[j])

    plt.tight_layout()


def check_outliers(data, features):
    '''
    Verifica a presença de outliers nas colunas (features) de um dataset.

    Esta função calcula e identifica outliers nas features especificadas
    utilizando o método de Intervalo Interquartil (IQR).
    '''
    
    outlier_counts = {}
    outlier_indexes = {}
    total_outliers = 0
    
    for feature in features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        feature_outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
        outlier_indexes[feature] = feature_outliers.index.tolist()
        outlier_count = len(feature_outliers)
        outlier_counts[feature] = outlier_count
        total_outliers += outlier_count
    
    print(f'Existem {total_outliers} outliers no dataset.')
    print()
    print(f'Número (porcentagem) de outliers por feature: ')
    print()
    for feature, count in outlier_counts.items():
        print(f'{feature}: {count} ({round(count/len(data)*100, 2)})%')

    return outlier_indexes, outlier_counts, total_outliers
