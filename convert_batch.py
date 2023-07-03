#%%

import time

import mesa
import numpy as np
import pandas as pd
import os

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
    run_df_index = run_df.index.to_frame()
    last_step = run_df_index['Step'].max()
    idx = pd.IndexSlice
    last_step_df = run_df.loc[idx[:,:,last_step], 'status']
    displaced_df = last_step_df[last_step_df == 'displaced']
    count = displaced_df.count()
    return count

#%%

def process_batch(sim_data):
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

    return analysis_df

if __name__ == '__main__':
    for batch in range(0, 318):
        batch_file = f'sensitivity/model_parameters_batch_{batch+1}.csv'
        print(f'Processing batch {batch+1}...')
        sim_data = pd.read_csv(batch_file, index_col=['RunId', 'iteration', 'Step', 'AgentID'])
        analysis_df = process_batch(sim_data)
        analysis_df.to_csv(f'sensitivity/analysis_batch_{batch+1}.csv')    
        # remove the batch file
        os.remove(batch_file)