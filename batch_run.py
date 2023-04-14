#%%
import mesa
from model import IGAD, VILLAGES
from utils import SCENARIOS
import numpy as np
#%%
params = dict(
    save_to_csv=False,
    do_early_warning=True,
    false_alarm_rate=0.3,
    false_negative_rate=0.1,
    trust=0.75,
    house_repair_program=0.0,
    house_improvement_program=False,
    basic_income_program=[False, True],
    awareness_program=[False, True],
    
    scenario=SCENARIOS[0],
    village_0=True,
    village_1=False,
    village_2=False,
    village_3=False,
    village_4=False,
    village_5=False,
    village_6=False,
    
)
#%%
results = mesa.batch_run(
    IGAD,
    parameters=params,
    iterations=1,
    max_steps=100,
    number_processes=1,
    data_collection_period=-1,
    display_progress=False,
)

# %%
import pandas as pd
results_df = pd.DataFrame(results)
print(results_df)
# %%
