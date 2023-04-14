#%%
import mesa
from model import IGAD, VILLAGES
from utils import SCENARIOS
import numpy as np

# record start time
import time
start_time = time.time()

#%%
params = dict(
    save_to_csv=False,
    do_early_warning=[False, True],
    false_alarm_rate=0.3,
    false_negative_rate=0.1,
    trust=0.75,
    house_repair_program=[0.1, 0.2, 0.3, 0.4, 0.5],
    house_improvement_program=[False, True],
    basic_income_program=[False, True],
    awareness_program=[False, True],
    
    scenario=[SCENARIOS[0],SCENARIOS[2],SCENARIOS[4]],
    village_0=True,
    village_1=True,
    village_2=True,
    village_3=True,
    village_4=True,
    village_5=True,
    village_6=True,    
)
#%%
results = mesa.batch_run(
    IGAD,
    parameters=params,
    iterations=1,
    max_steps=100,
    number_processes=16,
    data_collection_period=-1,
    display_progress=True,
)

# # %%
# import pandas as pd
# results_df = pd.DataFrame(results)
# print(results_df)
# %%

# record end time
end_time = time.time()

print(f'Elapsed time: {end_time - start_time} seconds')