from IPython.display import display, HTML
import pandas as pd
import numpy as np

def display_side_by_side(*dfs, row_ammount=5, names=None):
    html_str = ''
    if names is None:
        names = [f'Object {i+1}' for i in range(len(dfs))]
        
    for df, name in zip(dfs, names):
        df_html = df.to_frame().head(row_ammount).to_html() if isinstance(df, pd.Series) else df.head(row_ammount).to_html()
        html_str += f'<div style="display:inline-block; vertical-align:top; margin-right: 20px;">'
        html_str += f'<h3>{name}</h3>'
        html_str += df_html
        html_str += '</div>'
        
    display(HTML(html_str))

def print_metrics_report(metrics: dict):
    print("=" * 45)
    print("         Classification Report")
    print("=" * 45)

    if 'accuracy' in metrics:
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")

    class_labels = sorted(list(set([k.split('_')[0] for k in metrics if '_' in k and 'macro' not in k])))
    
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 45)

    for label in class_labels:
        precision = metrics.get(f'{label}_precision', 0)
        recall = metrics.get(f'{label}_recall', 0)
        f1 = metrics.get(f'{label}_f1_score', 0)
        print(f"{label:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    print("-" * 45)

    macro_p = metrics.get('macro_precision', 0)
    macro_r = metrics.get('macro_recall', 0)
    macro_f1 = metrics.get('macro_f1_score', 0)
    print(f"{'Macro Average':<15} {macro_p:<12.4f} {macro_r:<12.4f} {macro_f1:<12.4f}")
    print("=" * 45)