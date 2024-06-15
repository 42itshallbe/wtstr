import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

sns.set_style("darkgrid")

def plot_results(df_predictions: pd.DataFrame, quantity_name: str, title: str) -> None:
    """
    
    """
    q_true = f'{quantity_name}_true'
    q_predict = f'{quantity_name}_predicted'
    q_residuals = f'{quantity_name}_residuals'
    
    df_predict_pivot = df_predictions.loc[:, ['period_of_day', q_residuals, 'date']].pivot_table(index='date', columns='period_of_day')
    stack_array = np.array(df_predict_pivot.values)
    if np.any(np.isnan(stack_array[0, :])):
        stack_array = stack_array[1:, :]
    if np.any(np.isnan(stack_array[-1, :])):
        stack_array = stack_array[:-1, :]
    
    fig = plt.figure(figsize=(30, 5))
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1])
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_predictions.index, df_predictions.loc[:, q_true], label='true')
    ax1.plot(df_predictions.index, df_predictions.loc[:, q_predict], label='predicted')
    ax1.legend(ncol=2)
    ax1.title.set_text(
        f'{quantity_name}:  '
        f'mae = {np.round(mean_absolute_error(df_predictions.loc[:, q_true], df_predictions.loc[:, q_predict]), 2)}, '
        f'rmse = {np.round(root_mean_squared_error(df_predictions.loc[:, q_true], df_predictions.loc[:, q_predict]), 2)}'
        )  
    
    ax2 = fig.add_subplot(gs[1, 0]) 
    ax2.boxplot(stack_array, flierprops={'markersize': 3.})
    ax2.set_xticks([*range(0, 50, 2)])
    ax2.set_xticklabels([*range(0, 25, 1)])
    ax2.set_xlabel("hour")
    ax2.set_ylabel("residuals")
    ax2.set_title(f"Residuals per period of day for: {quantity_name}") 
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(df_predictions.loc[:, q_true], df_predictions.loc[:, q_predict], s=3, label="data")
    line45 = np.linspace(min(df_predictions.loc[:, q_true].values), max(df_predictions.loc[:, q_true].values))
    ax3.plot(line45, line45, color='red', linestyle='--', lw=2, label='ideal line')
    ax3.set_xlabel("True data")
    ax3.set_ylabel("Predicted")
    ax3.legend()   
         
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.scatter(df_predictions.loc[:, q_predict], df_predictions.loc[:, q_residuals], s=3, label="data")
    line = np.linspace(min(df_predictions.loc[:, q_predict].values), max(df_predictions.loc[:, q_predict].values))
    line0 = [0] * len(line)
    ax4.plot(line, line0, color='red', linestyle='--', lw=2, label='ideal line')
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Residuals")
    ax4.legend()

    plt.suptitle(f"{title}")
    plt.subplots_adjust(hspace=0.3)
    plt.show()