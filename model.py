from typing import List
from datetime import datetime
import geopandas as gpd
import mesa_geo as mg
import numpy as np
from numpy.random import random
import pandas as pd
from shapely.geometry import Point

import mesa
from spaces import IGADSpace
from agents import (STATUS_DISPLACED, STATUS_EVACUATED, STATUS_NORMAL,
                    STATUS_TRAPPED, HouseholdAgent)
from utils import get_events, load_population_data, MAPS_BASENAME, DF_SCENARIOS, DF_EVENTS, MAX_YEARS


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
    'init_step', 
    'return_decision',
    'check_for_early_warning', 
    'check_neighbours_for_evacuation',
    'react_to_flood',
    'displacement_decision',
    'check_neighbours_for_displacement',
    'update_sentiments',
    'fix_damage',
    'fix_neighbours_damage',
]
        

ALL_SETTLEMENTS = gpd.read_file('IGAD/settlements_grid_wdst_sampled.gpkg').to_crs(epsg=4326)
BOUNDING_BOXES = gpd.read_file('IGAD/BoundingBox20022023/BoundingBox_20022023.shp').to_crs(epsg=4326)
# select only the bounding box of the village
ALL_POPULATION_DATA = load_population_data()

class IGAD(mesa.Model):
    """Model class for the IGAD model."""
    def __init__(
        self, 
        save_to_csv=None,
        false_alarm_rate=None,
        false_negative_rate=None,
        trust=None,
        do_early_warning=None,
        house_repair_program=None,
        house_improvement_program=None,
        basic_income_program=None,
        awareness_program=None,
        scenario=None,
        **kwargs
    ):
        """
        Create a new IGAD model.
        :param positions:   List of tuples with the x and y coordinates of each agent
        :param false_alarm_rate:    False alarm rate for the model
        :param false_negative_rate: False negative rate for the model
        :param trust:   Trust value for the model
        :param house_repair_program: % of repaired houses 
        :param house_improvement_program: whether the government provides house improvement (e.g. change materials)
        :param basic_income_program: Whether the government provides a basic income or not
        :param awareness_program: Whether the government provides awareness programs or not
        :param scenario:    Scenario to run
        :param **kwargs:   Additional keyword arguments
        """
        super().__init__()

        self.save_to_csv = save_to_csv

        # Set random seed to reset random sequence
        np.random.seed(0)

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

        # active government programs
        self.do_early_warning = do_early_warning
        self.house_repair_program = house_repair_program
        self.basic_income_program = basic_income_program
        self.awareness_program = awareness_program
        self.house_improvement_program = house_improvement_program  
        
        # IGAD MODEL PARAMETERS
        self.false_alarm_rate = false_alarm_rate
        self.false_negative_rate = false_negative_rate
        
        self.running = False
        self.create_datacollector()
        

        # extract villages from kwargs
        active_villages = [
            village 
            for n, village in enumerate(VILLAGES)
            if 
            f'village_{n}' in kwargs and
            kwargs[f'village_{n}'] == True
        ]
        self.load_data(villages=active_villages)

        
        self.agents = []      
        # Generate HouseHold Agents
        ac_population = mg.AgentCreator(
            HouseholdAgent,
            model=self,
            crs=self.space.crs,
            agent_kwargs={},
        )       
        n_agents = len(self.positions)

        for i in range(n_agents):
            x, y = self.positions[i]
            household = ac_population.create_agent(
                Point(x, y), 
                "H" + str(i)
            )
            
            # Assign attributes            
            household.base_income = self.incomes[i]
            household.flood_prone = bool(self.flood_prones[i])
            household.awareness = self.awarenesses[i]
            household.fear = self.fears[i]
            #household.trust = trusts[i]
            household.trust = trust
            household.household_size = self.households_size[i]
            household.house_materials = self.house_materials[i]
            household.obstacles_to_movement = bool(self.obstacles_to_movement[i])

            self.space.add_agents(household)
            self.schedule.add(household)
            self.agents.append(household)

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
                "mean_income": lambda this: np.mean([a.income for a in this.agents]) * 100,
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
            },
        )

    def load_data(self, villages: List[str]):
        """
        Load data from population, settlements and flood events.
        """
        start_year, end_year = DF_SCENARIOS.loc[self.scenario, ['start_year', 'end_year']]
        self.events = get_events(start_year=start_year, end_year=end_year)

        self.incomes = []
        self.flood_prones = []
        self.awarenesses = []
        self.house_materials = []
        self.households_size = []
        self.obstacles_to_movement = []
        self.fears = []
        self.positions = []

        for village in villages:
            print('Loading data for village', village)

            bounding_boxes = BOUNDING_BOXES.query('village == @village')            
            for id, bounding_box in bounding_boxes.iterrows():
                flood_prone = bounding_box.floodprone == 1
                settlements = ALL_SETTLEMENTS[ALL_SETTLEMENTS.geometry.within(bounding_box.geometry)]

                n_households = len(settlements)

                village_lons = settlements.geometry.centroid.x
                village_lats = settlements.geometry.centroid.y

                village_flood_prones = [flood_prone] * n_households
                village_positions = list(zip(village_lons, village_lats))
                village_data = ALL_POPULATION_DATA\
                        .query('village == @village')\
                        .sample(n_households, replace=True)

                self.positions += village_positions
                self.incomes += village_data['income'].values.tolist()
                self.flood_prones += village_flood_prones #.tolist()
                self.awarenesses += village_data['awareness'].values.tolist()
                self.house_materials += village_data['walls_materials'].values.tolist()
                self.households_size += village_data['household_size'].values.tolist()
                self.obstacles_to_movement += village_data['obstacles_to_movement'].values.tolist()
                self.fears += village_data['fear_of_flood'].tolist()


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
        
        if not emit:
            return 

        print('Early warning at time step', self.steps)
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
        self.schedule.step()
        self.datacollector.collect(self)
        
        if self.steps >= MAX_YEARS:
            self.running = False
            if self.save_to_csv:
                # current date
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                df = self.datacollector.get_agent_vars_dataframe()
                df.to_csv(f'output/data_{now}.csv')


