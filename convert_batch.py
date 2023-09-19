#%%

import time

import mesa
import numpy as np
import pandas as pd
import os
import sys
#
from functools import partial

def execute(args):
    f = args[0]
    par = args[1]
    return f(par)


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


def n_displacements_func(sim_data):
    start_time = time.time()    
    # group by run id and agent id and apply the function to each group
    n_displacements = \
        sim_data.groupby('RunId')\
            .apply(lambda group: 
                group.groupby('AgentID')\
                        .apply(lambda agent: get_displacements_count(agent))\
                        .sum(axis=0)
            )
    print(f'n_displacements Elapsed time: {time.time() - start_time} seconds'); sys.stdout.flush()
    return n_displacements

def n_evacuations_func(sim_data):
    start_time = time.time()
    n_evacuations = \
        sim_data.groupby('RunId')\
            .apply(lambda group:
                    group.groupby('AgentID')\
                        .apply(lambda agent: get_evacuated_count(agent))\
                        .sum(axis=0)
            )
    
    print(f'n_evacuations Elapsed time: {time.time() - start_time} seconds'); sys.stdout.flush()
    return n_evacuations

def n_trapped_func(sim_data):
    start_time = time.time()
    n_trapped = \
        sim_data.groupby('RunId')\
            .apply(lambda group:
                    group.groupby('AgentID')\
                        .apply(lambda agent: get_trapped_count(agent))\
                        .sum(axis=0)
            )
    print(f'n_trapped Elapsed time: {time.time() - start_time} seconds'); sys.stdout.flush()
    return n_trapped

def displacement_time_func(_class, sim_data):
    print(f'Processing displacement_time{_class}...'); sys.stdout.flush()
    start_time = time.time()
    displacement_time = \
        sim_data.groupby('RunId')\
            .apply(lambda group: (
                    group.groupby('AgentID')\
                        .apply(lambda agent: get_max_displacement_time_class(agent)) == _class)\
                        .sum(axis=0)
            )
    print(f'displacement_time{_class} Elapsed time: {time.time() - start_time} seconds'); sys.stdout.flush()
    return displacement_time

# partial apply 
displacement_time_0_func = partial(displacement_time_func, '=0')
displacement_time_2_func = partial(displacement_time_func, '<=2')
displacement_time_5_func = partial(displacement_time_func, '<=5')
displacement_time_5_plus_func = partial(displacement_time_func, '>5')


def n_displaced_at_last_step_func(sim_data):
    start_time = time.time()
    print('Processing displaced_at_last_step...'); sys.stdout.flush()
    displaced_at_last_step = \
    sim_data.groupby('RunId')\
        .apply(lambda group: get_displaced_at_last_step(group))
    
    print(f'displaced_at_last_step Elapsed time: {time.time() - start_time} seconds'); sys.stdout.flush()
    return displaced_at_last_step


#%%
def process_batch(executor, sim_data):
    batch_parameters = [
        'model_parameters',
    ]
    start_time = time.time()

    
    # use map method of executor to get the results
    functions = [
        (n_displacements_func, sim_data),
        (n_evacuations_func, sim_data),
        (n_trapped_func, sim_data),
        (displacement_time_0_func, sim_data),
        (displacement_time_2_func, sim_data),
        (displacement_time_5_func, sim_data),
        (displacement_time_5_plus_func, sim_data),
        (n_displaced_at_last_step_func, sim_data),
    ]

    results = executor.map(execute, functions)
    
    n_displacements, n_evacuations, n_trapped, displacement_time_0, displacement_time_2, displacement_time_5, displacement_time_5_plus, displaced_at_last_step = results
    elapsed_time = time.time() - start_time


    analysis_df = sim_data.groupby('RunId').first()[batch_parameters]

    analysis_df['n_displacements'] = n_displacements
    analysis_df['n_evacuations'] = n_evacuations
    analysis_df['n_trapped'] = n_trapped
    analysis_df['displaced_at_last_step'] = displaced_at_last_step
    analysis_df['displacement_time=0'] = displacement_time_0
    analysis_df['displacement_time<=2'] = displacement_time_2
    analysis_df['displacement_time<=5'] = displacement_time_5
    analysis_df['displacement_time>5'] = displacement_time_5_plus

    print(f'Elapsed time: {elapsed_time} seconds')
    return analysis_df

if __name__ == '__main__':
    batch = 0
    batch_file = f'sensitivity/analysis_batch_{batch+1}_raw.csv'
    sim_data = pd.read_csv(batch_file, index_col=['RunId', 'iteration', 'Step', 'AgentID'])
    analysis_df = process_batch(sim_data)
    
    analysis_df.to_csv(f'sensitivity/analysis_batch_{batch+1}.csv')    
    # remove the batch file
    