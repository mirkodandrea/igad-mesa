#%%
# record start time
import time

import mesa
import numpy as np
import pandas as pd

from analysis import do_analysis
from graphs import plot_graphs
from model import EWS_MODES, HOUSE_REPAIR_PROGRAMS_LEVELS, IGAD
from utils import SCENARIOS

import matplotlib.pyplot as plt
#%%
ITERATIONS = 100
N_REPETITIONS = 300
MAX_STEPS = 23
for i in range(N_REPETITIONS):
    start_time = time.time()

    ews_modes = list(EWS_MODES.keys())
    hrp_levels = list(HOUSE_REPAIR_PROGRAMS_LEVELS.keys())

    params = dict(
        ews_mode=ews_modes[1],
        hrp_level=hrp_levels[0],
        
        basic_income_program=False,
        awareness_program=False,
        
        scenario=SCENARIOS[9],
        village_0=True,
        village_1=True,
        village_2=True,
        village_3=True,
        village_4=True,
        village_5=True,
        village_6=True,    
    )
    results = mesa.batch_run(
        IGAD,
        parameters=params,
        iterations=ITERATIONS,
        max_steps=MAX_STEPS,
        number_processes=16,
        #data_collection_period=1,
        display_progress=True,
    )


    results_df = pd.DataFrame(results)
    results_df = results_df.query(f'Step == {MAX_STEPS}')\
                    .query('AgentID == "H0"')
    results_df = results_df[['n_displaced', 'n_evacuated', 'n_trapped', 'displaced_lte_2', 'displaced_lte_5', 'displaced_gt_5']]

    results_df.to_csv(f'sensitivity/simulation_repetition_{i}.csv')

    #df.rolling(100, min_periods=1, axis=0).apply(lambda s : 100 * (s.std(axis=0) / s.mean(axis=0)) ).iloc[::10, :].T

    # record end time
    end_time = time.time()

# print(f'Elapsed time: {end_time - start_time} seconds')
#%%
df = pd.DataFrame()
for i in range(N_REPETITIONS):
    _df = pd.read_csv(f'sensitivity/simulation_repetition_{i}.csv')
    df = pd.concat([df, _df], axis=0)

# replace index with 0..n
df.index = range(len(df))
#%%


stability = df.rolling(len(df), min_periods=1, axis=0)\
        .apply(lambda s : 100 * (s.std(axis=0) / s.mean(axis=0)))
        #.iloc[100::100, :]

stability = stability.iloc[:, 1:]        
# %%

stability.to_csv('sensitivity/simulation_repetition_stability.csv')
# %%
for c in stability.columns:

    plt.figure()
    stability.loc[:, c].plot()
    plt.title(c)
    # save figure
    plt.savefig(f'sensitivity/simulation_repetition_stability_{c}.png')

# %%
