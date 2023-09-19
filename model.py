
print("Loading model.py")


from datetime import datetime
from typing import List

import geopandas as gpd
import mesa
import mesa_geo as mg
import numpy as np
import pandas as pd
from numpy.random import random
from shapely.geometry import Point

from agents import (STATUS_DISPLACED, STATUS_EVACUATED, STATUS_NORMAL,
                    STATUS_TRAPPED, HouseholdAgent)
from spaces import IGADSpace
from utils import (DF_SCENARIOS, MAPS_BASENAME, MAX_YEARS,
                   SimulationData, get_events)

from constants import (
    RISK_PERCEPTION_THRESHOLD, LOW_DAMAGE_THRESHOLD, 
    HIGH_DAMAGE_THRESHOLD, TRUST_THRESHOLD,
    TRUST_CHANGE,
    FEAR_CHANGE,
    AWARENESS_DECREASE,
    AWARENESS_INCREASE,
    NEIGHBOURS_HIGH_DAMAGE_FRACTION,
    FIX_DAMAGE_NEIGHBOURS,
    FIX_DAMAGE_CONCRETE,
    FIX_DAMAGE_MUDBRICK,
    FIX_DAMAGE_INFORMAL_SETTLEMENT,
)


SAVE_TO_CSV = False 
RAND_POSITION = False

VILLAGES = [
    'Al-Gaili', 
    'Wawise Garb', 
    'Wad Ramli Camp', 
    'Eltomaniat', 
    'Al-Shuhada', 
    'Wawise Oum Ojaija', 
    'Wad Ramli'
]

STAGE_LIST = [
    'update_trapped_probability',
    'return_decision',
    'reset_flags',
    'check_for_early_warning', 
    # 'check_neighbours_for_evacuation',
    'react_to_flood',
    'displacement_decision',
    # 'check_neighbours_for_displacement',
    'update_sentiments',
    'fix_damage',
    'fix_neighbours_damage',
    
]

EWS_MODES = {
    'no_ews': dict(do_early_warning=False, false_alarm_rate=None, false_negative_rate=None, trust=0.5),
    'bad_ews': dict(do_early_warning=True, false_alarm_rate=0.5, false_negative_rate=0.3, trust=0.3),
    'good_ews': dict(do_early_warning=True, false_alarm_rate=0.3, false_negative_rate=0.1, trust=0.6),
    'perfect_ews': dict(do_early_warning=True, false_alarm_rate=0.0, false_negative_rate=0.0, trust=0.8),
}

HOUSE_REPAIR_PROGRAMS_LEVELS = {
    'hrp_00': dict(house_improvement_program=False, house_repair_program=0.0),
    'hrp_30': dict(house_improvement_program=False, house_repair_program=0.3),
    'hrp_60': dict(house_improvement_program=False, house_repair_program=0.6),
    'hrp_90': dict(house_improvement_program=False, house_repair_program=0.9),
    'hrp_30_hi': dict(house_improvement_program=True, house_repair_program=0.3),
    'hrp_60_hi': dict(house_improvement_program=True, house_repair_program=0.6),    
    'hrp_90_hi': dict(house_improvement_program=True, house_repair_program=0.9),
}
       
class IGAD(mesa.Model):
    simulation_data = SimulationData()

    """Model class for the IGAD model."""
    def __init__(
        self, 
        ews_mode=None,
        hrp_level=None,
        basic_income_program=None,
        awareness_program=None,
        scenario=None,
        **kwargs
    ):
        """
        Create a new IGAD model.
        :param early_warning_mode: Early warning mode
        :param house_repair_program: House repair program level
        :param basic_income_program: Whether the government provides a basic income or not
        :param awareness_program: Whether the government provides awareness programs or not
        :param scenario:    Scenario to run
        :param **kwargs:   Additional keyword arguments
        """
        super().__init__()
        self.running = True

        self.save_to_csv = SAVE_TO_CSV

        # Set random seed to reset random sequence
        #np.random.seed(0)

        self.scenario = scenario
        self.schedule = mesa.time.StagedActivation(self, 
            stage_list=STAGE_LIST, 
            shuffle_between_stages=True
        )
        
        self.space = IGADSpace(crs='epsg:4326', 
            warn_crs_conversion=False, 
            reference=f'{MAPS_BASENAME}_0001_cut.tif'
        )
        
        self.steps = 0
        self.emitted_early_warning = False
        self.flood_event = False

        # early warning mode
        ews_mode_dict = EWS_MODES[ews_mode]
        self.do_early_warning = ews_mode_dict['do_early_warning']
        self.false_alarm_rate = ews_mode_dict['false_alarm_rate']
        self.false_negative_rate = ews_mode_dict['false_negative_rate']
        self.trust = ews_mode_dict['trust']

        # active government programs
        hrp_dict = HOUSE_REPAIR_PROGRAMS_LEVELS[hrp_level]

        self.house_repair_program = hrp_dict['house_repair_program']
        self.house_improvement_program = hrp_dict['house_improvement_program']

        self.basic_income_program = basic_income_program
        self.awareness_program = awareness_program        

        # Set model parameters
        if 'model_parameters' in kwargs:
            model_parameters = kwargs['model_parameters']
        else:
            model_parameters = {}

        self.RISK_PERCEPTION_THRESHOLD =  model_parameters.get('RISK_PERCEPTION_THRESHOLD', RISK_PERCEPTION_THRESHOLD)
        self.LOW_DAMAGE_THRESHOLD = model_parameters.get('LOW_DAMAGE_THRESHOLD', LOW_DAMAGE_THRESHOLD)
        self.HIGH_DAMAGE_THRESHOLD = model_parameters.get('HIGH_DAMAGE_THRESHOLD', HIGH_DAMAGE_THRESHOLD)
        self.TRUST_THRESHOLD = model_parameters.get('TRUST_THRESHOLD', TRUST_THRESHOLD)
        self.TRUST_CHANGE =  model_parameters.get('TRUST_CHANGE', TRUST_CHANGE)
        self.FEAR_CHANGE = model_parameters.get('FEAR_CHANGE', FEAR_CHANGE)
        self.AWARENESS_DECREASE = model_parameters.get('AWARENESS_DECREASE', AWARENESS_DECREASE)
        self.AWARENESS_INCREASE = model_parameters.get('AWARENESS_INCREASE', AWARENESS_INCREASE)
        self.NEIGHBOURS_HIGH_DAMAGE_FRACTION = model_parameters.get('NEIGHBOURS_HIGH_DAMAGE_FRACTION', NEIGHBOURS_HIGH_DAMAGE_FRACTION)
        
        self.FIX_DAMAGE_NEIGHBOURS  = model_parameters.get('FIX_DAMAGE_NEIGHBOURS', FIX_DAMAGE_NEIGHBOURS)
        self.FIX_DAMAGE_CONCRETE = model_parameters.get('FIX_DAMAGE_CONCRETE', FIX_DAMAGE_CONCRETE)
        self.FIX_DAMAGE_MUDBRICK = model_parameters.get('FIX_DAMAGE_MUDBRICK', FIX_DAMAGE_MUDBRICK)
        self.FIX_DAMAGE_INFORMAL_SETTLEMENT = model_parameters.get('FIX_DAMAGE_INFORMAL_SETTLEMENT', FIX_DAMAGE_INFORMAL_SETTLEMENT)

        self.create_datacollector()
        

        # extract villages from kwargs
        active_villages = [
            village 
            for n, village in enumerate(VILLAGES)
            if 
            f'village_{n}' in kwargs and
            kwargs[f'village_{n}'] == True
        ]

        self.agents = []

        # Generate HouseHold Agents
        ac_population = mg.AgentCreator(
            HouseholdAgent,
            model=self,
            crs=self.space.crs,
            agent_kwargs={},
        )       

        data = IGAD.simulation_data
        for i, village in enumerate(data.villages):
            if village not in active_villages:
                continue

            x, y = data.positions[i]
            household = ac_population.create_agent(
                Point(x, y), 
                "H" + str(i)
            )
            
            # Assign attributes            
            household.base_income = data.incomes[i]
            household.flood_prone = bool(data.flood_prones[i])
            household.awareness = data.awarenesses[i]
            household.fear = data.fears[i]
            #household.trust = trusts[i]
            household.trust = self.trust
            household.household_size = data.households_size[i]
            household.house_materials = data.house_materials[i]
            household.village = data.villages[i]
            
            household.number_of_floods = data.number_of_floods[i]
            household.health_issues = data.health_issues[i]
            household.livestock = data.livestock[i]
            household.house = data.house[i]
            household.cropland = data.cropland[i]

            self.space.add_agents(household)
            self.schedule.add(household)
            self.agents.append(household)

        # load scenarios
        start_year, end_year = DF_SCENARIOS.loc[self.scenario, ['start_year', 'end_year']]
        self.events = get_events(start_year=start_year, end_year=end_year)

        self.datacollector.collect(self)



    def create_datacollector(self):
        """Create the datacollector."""
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "n_displaced": lambda this: len([a for a in this.agents if a.status == STATUS_DISPLACED]),
                "n_normal": lambda this: len([a for a in this.agents if a.status == STATUS_NORMAL]),
                "n_evacuated": lambda this: len([a for a in this.agents if a.status == STATUS_EVACUATED]),
                "n_trapped": lambda this: len([a for a in this.agents if a.status == STATUS_TRAPPED]),
                
                "mean_house_damage": lambda this: np.mean([a.house_damage for a in this.agents]) * 100,
                "mean_livelihood_damage": lambda this: np.mean([a.livelihood_damage for a in this.agents]) * 100,
                "mean_trust": lambda this: np.mean([a.trust for a in this.agents]) * 100,
                "mean_perception": lambda this: np.mean([a.perception for a in this.agents]) * 100,
                "mean_income": lambda this: np.mean([a.income for a in this.agents]),
                "mean_awareness": lambda this: np.mean([a.awareness for a in this.agents]) * 100,
                "mean_fear": lambda this: np.mean([a.fear for a in this.agents]) * 100,
                "displaced_lte_2": lambda this: sum([1 <= a.displacement_time <= 2  for a in this.agents]),
                "displaced_lte_5": lambda this: sum([2 < a.displacement_time <= 5 for a in this.agents]),
                "displaced_gt_5": lambda this: sum([ a.displacement_time > 5 for a in this.agents]),

                "n_flooded": lambda this: sum([a.household_size for a in this.agents if a.received_flood]),
                "affected_population": lambda this: sum([a.household_size for a in this.agents if a.received_flood and a.status in [STATUS_NORMAL, STATUS_TRAPPED]]),

            },
            agent_reporters={
                "status": lambda agent: agent.status,
                "flooded": lambda agent: agent.received_flood,
                "alerted": lambda agent: agent.alerted,
                "house_damage": lambda agent: agent.house_damage,
                "livelihood_damage": lambda agent: agent.livelihood_damage,
                "trust": lambda agent: agent.trust,
                "perception": lambda agent: agent.perception,
                "income": lambda agent: agent.income,
                "displacement_time": lambda agent: agent.displacement_time,
                "village": lambda agent: agent.village,    
            },
        )

    def __has_floods(self):
        """
        Check if there is a flood event in current time step
        """
        return self.steps in self.events
    

    def maybe_emit_early_warning(self):
        """ 
        fuzzy emit early warning
        If there is a flood event in time t, emit early warning with probability 1 - false_negative_rate
        If there is no flood event in time t, emit early warning with probability false_alarm_rate        
        """
        if not self.do_early_warning:
            self.emitted_early_warning = False
            return

        emit = False
        if self.__has_floods():    
            emit = not self.random.random() <= self.false_negative_rate 
        else:
            emit = self.random.random() <= self.false_alarm_rate
                   
        self.emitted_early_warning = emit            

            
    def update_flood(self):
        """ 
        Apply flood to all agents
        """
        if self.__has_floods():
            events = self.events[self.steps]
            event_filenames = [event['filename'] for event in events]
            self.space.update_water_level(event_filenames)  
            self.flood_event = True
        else:
            self.space.reset_water_level()
            self.flood_event = False


    def step(self):
        """Run one step of the model."""
        self.steps += 1
        self.maybe_emit_early_warning()
        self.update_flood()
        self.datacollector.collect(self)
        self.schedule.step()
        
        if self.steps >= MAX_YEARS:
            self.running = False
            if self.save_to_csv:
                # current date
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                df = self.datacollector.get_agent_vars_dataframe()
                df.to_csv(f'output/data_{now}.csv')


