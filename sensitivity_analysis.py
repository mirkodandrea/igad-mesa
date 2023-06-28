#%%

import time

import mesa
import numpy as np
import pandas as pd

from analysis import do_analysis
from graphs import plot_graphs
from model import EWS_MODES, HOUSE_REPAIR_PROGRAMS_LEVELS, IGAD
from utils import SCENARIOS
#%%
# record start time
start_time = time.time()
#%%
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

model_parameters = [r.to_dict() for idx, r in sensitivity_params.iterrows()]

#%%

ews_modes = list(EWS_MODES.keys())
hrp_levels = list(HOUSE_REPAIR_PROGRAMS_LEVELS.keys())

params = dict(
    ews_mode='bad_ews',
    hrp_level='hrp_00',
    basic_income_program=False,
    awareness_program=False,
    scenario=SCENARIOS[9],
    village_0=True,
    village_1=False,
    village_2=False,
    village_3=False,
    village_4=False,
    village_5=False,
    village_6=False,
    model_parameters=model_parameters
)
results = mesa.batch_run(
    IGAD,
    parameters=params,
    iterations=1,
    max_steps=30,
    number_processes=1,
    data_collection_period=1,
    display_progress=True,
)


results_df = pd.DataFrame(results)
# %%
def get_displacements_count(agent_story):
    """
    Returns the number of displacements for a given agent.
    It will count as one displacement a continuous sequence of displaced steps.
    """
    is_displaced = agent_story.status == 'displaced'
    number_of_displacements = ((is_displaced - is_displaced.shift(1)) == 1).fillna(0).sum()
    return number_of_displacements

def get_trapped_count(agent_story):
    """
    Returns the number of trapped for a given agent.
    """    
    is_trapped = agent_story.status == 'trapped'
    number_of_trapped = ((is_trapped - is_trapped.shift(1)) == 1).fillna(0).sum()
    return number_of_trapped


def get_evacuated_count(agent_story):
    """
    Returns the number of evacuations for a given agent.
    It will count as one evacuation a continuous sequence of evacuated steps.
    """
    is_evacuated = (agent_story.status == 'evacuated') & agent_story.flooded
    number_of_evacuations = is_evacuated.sum()
    return number_of_evacuations


def get_max_displacement_time_class(agent_story):
    """
    Get the maximum number of consecutive steps of displacement for a given agent.

    """
    max_time = agent_story.displacement_time.fillna(0).max()
    if max_time == 0:
        return '=0'
    if max_time <= 2:
        return '<=2'
    elif max_time <= 5:
        return '<=5'
    elif max_time > 5:
        return '>5'
    
def get_displaced_at_last_step(run_df):
    """
    Return the number of displaced agents at the last step of the simulation.
    """
    last_step = run_df.Step.max()
    last_step_df = run_df[run_df.Step == last_step]
    return last_step_df[last_step_df.status == 'displaced'].AgentID.count()


sim_data = results_df.copy()
# %%
batch_parameters = [
    'model_parameters',
]

analysis_df = sim_data.groupby('RunId').first()[batch_parameters]

# group by run id and agent id and apply the function to each group
analysis_df['n_displacements'] = \
    sim_data.groupby('RunId')\
        .apply(lambda group: 
            group.groupby('AgentID')\
                    .apply(lambda agent: get_displacements_count(agent))\
                    .sum(axis=0)
        )
analysis_df['n_evacuations'] = \
    sim_data.groupby('RunId')\
        .apply(lambda group:
                group.groupby('AgentID')\
                    .apply(lambda agent: get_evacuated_count(agent))\
                    .sum(axis=0)
        )

analysis_df['n_trapped'] = \
    sim_data.groupby('RunId')\
        .apply(lambda group:
                group.groupby('AgentID')\
                    .apply(lambda agent: get_trapped_count(agent))\
                    .sum(axis=0)
        )

for _class in ['=0', '<=2', '<=5', '>5']:
    analysis_df[f'displacement_time{_class}'] = \
        sim_data.groupby('RunId')\
            .apply(lambda group: (
                    group.groupby('AgentID')\
                        .apply(lambda agent: get_max_displacement_time_class(agent)) == _class)\
                        .sum(axis=0)
            )

analysis_df['displaced_at_last_step'] = \
    sim_data.groupby('RunId')\
        .apply(lambda group: get_displaced_at_last_step(group))
#%%
reference = analysis_df.iloc[0, :]
analysis_df.drop(0, inplace=True)

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

from constants import (
    RISK_PERCEPTION_THRESHOLD,
    LOW_DAMAGE_THRESHOLD,
    HIGH_DAMAGE_THRESHOLD,
    TRUST_CHANGE,
    FEAR_CHANGE,
    AWARENESS_INCREASE,
    AWARENESS_DECREASE,
)

reference_values = dict(
    RISK_PERCEPTION_THRESHOLD=RISK_PERCEPTION_THRESHOLD,
    LOW_DAMAGE_THRESHOLD=LOW_DAMAGE_THRESHOLD,
    HIGH_DAMAGE_THRESHOLD=HIGH_DAMAGE_THRESHOLD,
    TRUST_CHANGE=TRUST_CHANGE,
    FEAR_CHANGE=FEAR_CHANGE,
    AWARENESS_INCREASE=AWARENESS_INCREASE,
    AWARENESS_DECREASE=AWARENESS_DECREASE,
)



metrics = ['n_trapped', 'n_evacuations', 'n_displacements', 'displaced_at_last_step', 'displacement_time<=2', 'displacement_time<=5', 'displacement_time>5']

model_parameters = analysis_df.model_parameters.apply(lambda x: list(x.keys())).explode().unique()
metric = metrics[0]
#for model_parameter in model_parameters:
for metric in metrics:
    df = pd.DataFrame(
        index=model_parameters,
        columns=['-20%', 'default', '+20%'],
    )
    for model_parameter in model_parameters:
        model_params_rows = analysis_df.model_parameters.apply(lambda x: model_parameter in x)
        selected_rows = analysis_df[model_params_rows]
        df.loc[model_parameter, '-20%'] = selected_rows[metric].iloc[0]
        df.loc[model_parameter, 'default'] = reference[metric]
        df.loc[model_parameter, '+20%'] = selected_rows[metric].iloc[1]

    df.to_csv(f'./sensitivity/{metric}.csv')
# %%
