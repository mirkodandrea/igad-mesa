#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

sensitivity_params = pd.read_excel('sensitivity/X.xlsx', index_col=0)
sensitivity_params = sensitivity_params.rename(columns={
    'RP': 'RISK_PERCEPTION_THRESHOLD',
    'Low_D': 'LOW_DAMAGE_THRESHOLD',
    'High_D': 'HIGH_DAMAGE_THRESHOLD',
    'Trust': 'TRUST_CHANGE',
    'Fear': 'FEAR_CHANGE',
    'Awareness_inc': 'AWARENESS_INCREASE',
    'Awareness_dec': 'AWARENESS_DECREASE',
    'Fix_D_Neighbours': 'FIX_DAMAGE_NEIGHBOURS',
    'Fix_D_Conc': 'FIX_DAMAGE_CONCRETE',
    'Fix_D_Mud': 'FIX_DAMAGE_MUDBRICK',
    'Informal_settlement': 'FIX_DAMAGE_INFORMAL_SETTLEMENT',
})


#%%
dfs = []
references = []
for batch in range(5000):
    df = pd.read_csv(f'./sensitivity/analysis_batch_{batch+1}.csv')
    dfs.append(df)
    
analysis_df = pd.concat(dfs, ignore_index=True)

#%%
metrics = ['n_trapped', 'n_evacuations', 'n_displacements', 'displaced_at_last_step', 'displacement_time<=2', 'displacement_time<=5', 'displacement_time>5']
import json
model_parameters = analysis_df.model_parameters.apply(lambda x: list(json.loads(x.replace("'", '"')).keys())).explode().unique()
#%%

metric = metrics[0]
columns = ['-20', '-10', '10', '20']
actual_columns = ['-20mean', '-20std', '-10mean', '-10std', '10mean', '10std', '20mean', '20std']
for metric in metrics:
    df = pd.DataFrame(
        index=model_parameters,
        columns=actual_columns
    )

    for model_parameter in model_parameters:
        param_values = np.sort(sensitivity_params[model_parameter].unique())
        for value, column in zip(param_values, columns):
            idx = analysis_df.model_parameters.str.contains(f"\'{model_parameter}\': {value}")
            selected_rows = analysis_df[idx]

            df.loc[model_parameter, column+'mean'] = selected_rows[metric].mean()
            df.loc[model_parameter, column+'std'] = selected_rows[metric].std()

    df.to_csv(f'./sensitivity/results/{metric}.csv')
#%%
for model_parameter in model_parameters:
    param_values = np.sort(sensitivity_params[model_parameter].unique())
    columns = [f'{value}' for value in param_values]
    actual_columns = [f'{value}mean' for value in param_values] + [f'{value}std' for value in param_values]
    df = pd.DataFrame(
        index=metrics,
        columns=actual_columns
    )
    for metric in metrics:        
        for value, column in zip(param_values, columns):
            idx = analysis_df.model_parameters.str.contains(f"\'{model_parameter}\': {value}")
            selected_rows = analysis_df[idx]

            df.loc[metric, column+'mean'] = selected_rows[metric].mean()
            df.loc[metric, column+'std'] = selected_rows[metric].std()

    df.to_csv(f'./sensitivity/results/{model_parameter}.csv')


    #print(df)

# %%
