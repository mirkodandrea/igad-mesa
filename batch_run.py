#%%
# record start time
import time

import sys
import mesa
import numpy as np
import pandas as pd

from analysis import do_analysis
from graphs import plot_graphs
from model import EWS_MODES, HOUSE_REPAIR_PROGRAMS_LEVELS, IGAD
from utils import SCENARIOS

start_time = time.time()

# ews_modes = list(EWS_MODES.keys())
# hrp_levels = list(HOUSE_REPAIR_PROGRAMS_LEVELS.keys())

ews_mode = sys.argv[1]
hrp_level = sys.argv[2]
scenario = sys.argv[3]
basic_income_program = sys.argv[4]
awareness_program = sys.argv[5]
N = int(sys.argv[6])
print("Running with parameters:")
print("ews mode ", ews_mode)
print("hrp_level ", hrp_level)
print("Scenario ", scenario)
print("Basic income program", basic_income_program)
print("Awareness program", awareness_program)
print("")

N_PROCESSES = 48
N_BATCHES = 50

params = dict(
    ews_mode=[ews_mode],
    hrp_level=[hrp_level],

    basic_income_program=[basic_income_program],
    awareness_program=[awareness_program],

    scenario=[scenario],
    village_0=True,
    village_1=True,
    village_2=True,
    village_3=True,
    village_4=True,
    village_5=True,
    village_6=True,
)

for batch in range(0, N_BATCHES):
    print(f"RUNNING batch{batch+1}")
    results = mesa.batch_run(
        IGAD,
        parameters=params,
        iterations=100,
        max_steps=30,
        number_processes=N_PROCESSES,
        data_collection_period=1,
        display_progress=True,
    )

    results_df = pd.DataFrame(results)
    print('Analysis')

    analysis_df = do_analysis(results_df)
    analysis_df.to_csv(f'results/batch/scenario_{N+1}/analysis_{batch+1}.csv')
