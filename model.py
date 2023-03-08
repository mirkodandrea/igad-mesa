from typing import List

import geopandas as gpd
import mesa_geo as mg
import numpy as np
from numpy.random import random
import pandas as pd
from shapely.geometry import Point

import mesa
from agents import (STATUS_DISPLACED, STATUS_EVACUATED, STATUS_NORMAL,
                    STATUS_TRAPPED, HouseholdAgent)
from utils import get_events, load_population_data

EXPORT_TO_CSV = True
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

ALL_SETTLEMENTS = gpd.read_file('IGAD/settlements_grid_wdst_sampled.gpkg').to_crs(epsg=4326)
BOUNDING_BOXES = gpd.read_file('IGAD/BoundingBox20022023/BoundingBox_20022023.shp').to_crs(epsg=4326)
# select only the bounding box of the village
ALL_POPULATION_DATA = load_population_data()


class IGAD(mesa.Model):
    """Model class for the IGAD model."""
    def __init__(
        self, 
        false_alarm_rate=None,
        false_negative_rate=None,
        trust=None,
        house_repair_program=None,
        house_improvement_program=None,
        basic_income_program=None,
        awareness_program=None,
        start_year=None,
        duration=None,
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
        :param start_year:  Start year of the model
        :param duration:    Duration of the flood event
        :param **kwargs:   Additional keyword arguments
        """

        # Set random seed to reset random sequence
        np.random.seed(0)

        self.schedule = mesa.time.BaseScheduler(self)
        self.space = mg.GeoSpace(crs='epsg:4326', warn_crs_conversion=False)
        self.steps = 0
        self.counts = None

        # active government programs
        self.house_repair_program = house_repair_program
        self.basic_income_program = basic_income_program
        self.awareness_program = awareness_program
        self.house_improvement_program = house_improvement_program

        # extract villages from kwargs
        active_villages = [
            village 
            for n, village in enumerate(VILLAGES)
            if 
            f'village_{n}' in kwargs and
            kwargs[f'village_{n}'] == True
        ]
        self.load_data(start_year=start_year, duration=duration, villages=active_villages)

        # IGAD MODEL PARAMETERS
        self.false_alarm_rate = false_alarm_rate
        self.false_negative_rate = false_negative_rate
        
        self.duration = duration
        
        self.running = True
        self.create_datacollector()
        self.agents = []

        # Generate PersonAgent population
        ac_population = mg.AgentCreator(
            HouseholdAgent,
            model=self,
            crs=self.space.crs,
            agent_kwargs={},
        )

        # Create agents and assign them to the space, with slight randomization of the position
        n_agents = len(self.positions)

        for i in range(n_agents):
            x, y = self.positions[i]
            if RAND_POSITION:
                x = x + np.random.normal(0, 0.001)
                y = y + np.random.normal(0, 0.001)
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

    def load_data(self, start_year: int, duration: int, villages: List[str]):
        """
        Load data from population, settlements and flood events.
        """
        self.events = get_events(initial_year=start_year, stride=duration)

        self.incomes = []
        self.flood_prones = []
        self.awarenesses = []
        self.house_materials = []
        self.households_size = []
        self.obstacles_to_movement = []
        self.fears = []
        self.positions = []

        for village in villages:
            bounding_box = BOUNDING_BOXES.query('village == @village').geometry
            settlements = ALL_SETTLEMENTS[ALL_SETTLEMENTS.geometry.within(bounding_box.unary_union)]
            # resample settlements to 1/10 of the original
            
            n_households = len(settlements)

            village_lons = settlements.geometry.centroid.x
            village_lats = settlements.geometry.centroid.y

            # village_flood_prones = (village_lons - village_lons.min()) / (village_lons.max() - village_lons.min()) < 0.3
            # village_flood_prones = village_flood_prones.values
            village_flood_prones = [True] * n_households
            village_positions = list(zip(village_lons, village_lats))

            population_data = ALL_POPULATION_DATA\
                    .query('village == @village')\
                    .sample(n_households, replace=True)
                
            village_incomes = population_data['income'].apply(lambda x: (x + random())**1.3).values
            village_house_materials = population_data['walls_materials'].values
            village_fears = population_data['fear_of_flood'].values / 3

            village_household_size = population_data['household_size'].values
            
            village_obstacles_to_movement = \
                (population_data[['vulnerabilities', 'properties']].sum(axis=1) > 4).values | \
                population_data['household_size'].values > 5
            
            village_awarenesses = 0.75 + random(n_households) * 0.25
            

            self.positions += village_positions
            #trusts += village_trusts.tolist()
            self.incomes += village_incomes.tolist()
            self.flood_prones += village_flood_prones #.tolist()
            self.awarenesses += village_awarenesses.tolist()
            self.house_materials += village_house_materials.tolist()
            self.households_size += village_household_size.tolist()
            self.obstacles_to_movement += village_obstacles_to_movement.tolist()
            self.fears += village_fears.tolist()


    def __has_floods(self):
        """
        Check if there is a flood event in current time step
        """
        return self.steps in self.events
    

    def init_step(self):
        """ execute init step for all agents
        """
        for household in self.agents:
            household.init_step()

    def maybe_emit_early_warning(self):
        """ 
        fuzzy emit early warning
        If there is a flood event in time t, emit early warning with probability 1 - false_negative_rate
        If there is no flood event in time t, emit early warning with probability false_alarm_rate        
        """
        emit = False
        if self.__has_floods():    
            emit = not self.random.random() <= self.false_negative_rate 
        else:
            emit = self.random.random() <= self.false_alarm_rate
        
        if not emit:
            return 

        print('Early warning at time step', self.steps)
        for agent in self.agents:
            agent.receive_early_warning()
                
        for agent in self.agents:                
            agent.check_neighbours_for_evacuation()
            
    def do_flood(self, events):
        """ 
        Apply flood to all agents
        """
        for household in self.agents:
            flood_value = 0
            for event in events:
                r = event['rio_object']
                flood_data = event['data']
                row, col = r.index(*household.geometry.xy)
                flood_value = np.nanmax(
                    flood_data[row, col]+
                    [flood_value]
                )
            household.receive_flood(flood_value)

    def fix_damages(self):
        """
        Fix damages for all agents
        """        

        for agent in self.agents:
            agent.fix_damage(self.house_repair_program)

        for agent in self.agents:
            agent.fix_neighbours_damage()

    def step(self):
        """Run one step of the model."""
        self.steps += 1      

        self.init_step()
        self.maybe_emit_early_warning()

        if self.__has_floods():
            events = self.events[self.steps]
            self.do_flood(events)

        self.fix_damages()

        # execute remaining steps for all agents
        self.schedule.step()
        self.datacollector.collect(self)
        
        if EXPORT_TO_CSV:            
            df = self.datacollector.get_agent_vars_dataframe()
            df.to_csv('output/data.csv')

        if self.steps >= self.duration:
            self.running = False

