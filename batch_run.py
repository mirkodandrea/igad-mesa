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

start_time = time.time()

ews_modes = list(EWS_MODES.keys())
hrp_levels = list(HOUSE_REPAIR_PROGRAMS_LEVELS.keys())

params = dict(
    ews_mode=[ews_modes[0]],
    hrp_level=[hrp_levels[0]],
    
    basic_income_program=[False, True],
    awareness_program=[False, True],
    
    scenario=[SCENARIOS[0]],
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
    iterations=1,
    max_steps=30,
    number_processes=1,
    data_collection_period=1,
    display_progress=True,
)


results_df = pd.DataFrame(results)
results_df.to_csv('results_new.csv')

# record end time
end_time = time.time()

print(f'Elapsed time: {end_time - start_time} seconds')
analysis_df = do_analysis(results_df)
plot_graphs(analysis_df)