import mesa_geo as mg
import numpy as np
from numpy.random import normal, random
from shapely.geometry import Point

from constants import (DISPLACE_DAMAGE_THRESHOLD, EVENT_EARLY_WARNING,
                       EVENT_FLOOD, FLOOD_DAMAGE_MAX, FLOOD_DAMAGE_THRESHOLD,
                       FLOOD_FEAR_MAX, MAX_DISTANCE, POVERTY_LINE)

STATUS_NORMAL = 'normal'
STATUS_EVACUATED = 'evacuated'
STATUS_DISPLACED = 'displaced'

LOW_DAMAGE_THRESHOLD = 0.4
MEDIUM_DAMAGE_THRESHOLD = 0.65



class HouseholdAgent(mg.GeoAgent):
    """Household Agent."""

    def __init__(
        self,
        unique_id,
        model,
        geometry,
        crs,
    ):
        """
        Create a new household agent.
        :param unique_id:   Unique identifier for the agent
        :param model:       Model in which the agent runs
        :param geometry:    Shape object for the agent
        :param agent_type:  Indicator if agent is infected ("infected", "susceptible", "recovered" or "dead")
        :param mobility_range:  Range of distance to move in one step
        """
        super().__init__(unique_id, model, geometry, crs)
        # Agent parameters

        # self.price = price
        # self.vulnerability = vulnerability
        self.obstacles_to_movement = False

        self.house_damage = 0
        self.livelihood_damage = 0
        self.status = STATUS_NORMAL
        
        # received early warning
        self.alerted = False
        # received flood on last step
        self.received_flood = False
        # prepared to flood
        self.prepared = False

    def __repr__(self):
        return "Household " + str(self.unique_id)


    def init_step(self):
        """
        set household status for each step to initial values        
        """
        # set all flags back to False
        self.alerted = False
        self.prepared = False
        self.received_flood = False

        self.return_decision()
    
    def return_decision(self):
        if self.house_damage < LOW_DAMAGE_THRESHOLD:
            self.status = STATUS_NORMAL
            return

    def displacement_decision(self):
        """ 
        check household damage and decide status
        """
        if self.status == STATUS_DISPLACED:
            return

        if self.house_damage > MEDIUM_DAMAGE_THRESHOLD or \
            self.livelihood_damage > MEDIUM_DAMAGE_THRESHOLD:
            self.status = STATUS_DISPLACED
            return

        if self.house_damage < LOW_DAMAGE_THRESHOLD and \
            self.livelihood_damage < LOW_DAMAGE_THRESHOLD:
            return      
    
        # in case of medium house damage or medium livelihood damage, check against perception
        if self.perception < random():
            return
    
        if self.income > POVERTY_LINE and \
            not self.obstacles_to_movement:
                self.status = STATUS_DISPLACED

    def update_displacement_decision(self):
        """
        update displacement decision based on neighbours
        do this only if household is not already displaced and
        has income above poverty line and no obstacles to movement
        """    
        if self.status != STATUS_NORMAL or\
            self.income < POVERTY_LINE or \
            self.obstacles_to_movement:
            return
        
        if self.perception < 0.5:
            return 
        
        neighbours = list(self.model.space.get_neighbors_within_distance(
               self, MAX_DISTANCE
        ))
        
        other_statuses = [neighbour.status == STATUS_DISPLACED for neighbour in neighbours]
        if sum(other_statuses) > 0.75 * len(neighbours):
            self.status = STATUS_DISPLACED


    def receive_early_warning(self):
        """receive early warning from government
        - if household is already evacuated, do nothing
        - if household doesn't trust the government, then do nothing
        - if household is not aware of risk, prepare
        - if household is aware of risk, evacuate without preparing
        """

        if self.status != STATUS_NORMAL:
            # already displaced or evacuated
            # don't receive early warning
            return

        self.alerted = True

        if self.trust < 0.5:
            # distrust the government
            return


        # prepare for flood anyway        
        self.prepared = True
        
        if self.income < POVERTY_LINE:
            # poor household, cannot afford to move
            return 
        
        # trust the government
        if self.perception >= 0.5:
            # aware of risk, move before flood
            self.status = STATUS_EVACUATED



    def check_neighbours_for_evacuation(self):
        """check neighbours for early warning reaction
        - if enough neighbours are evacuated, then evacuate
        - if enough neighbours are prepared, then prepare
        """
        
        if self.status != STATUS_NORMAL:
            # already evacuated
            return
        
        # collect neighbours
        neighbours = list(self.model.space.get_neighbors_within_distance(
               self, MAX_DISTANCE
        ))

        # check if other households are evacuated
        other_status = [neighbour.status == STATUS_EVACUATED for neighbour in neighbours]
        if sum(other_status) > 0.25 * len(neighbours):
            # enough neighbours are evacuated, evacuate myself if income is high enough
            if self.income >= POVERTY_LINE:
                self.status = STATUS_EVACUATED
 

        other_prepared = [neighbour.prepared for neighbour in neighbours]
        if sum(other_prepared) > 0.25 * len(neighbours):
            # enough neighbours are prepared, prepare myself
            self.prepared = True


            
    def receive_flood(self, flood_value):
        """receive flood for current household
        increment damage if household is flooded
        - damage is in percentage
        - damage occurs if flood value is > 10mm
        - damage is proportional to flood value: 1m -> 100% damage
        """

        if flood_value == 0:
            """ nothing happened to this household """
            self.received_flood = False
            self.fix_damage()
        

        self.received_flood = True
        if self.prepared and flood_value < FLOOD_DAMAGE_THRESHOLD:
            return
        
        new_damage = (flood_value / FLOOD_DAMAGE_MAX)
        if self.prepared:
            new_damage = new_damage * 0.5

        if self.house_materials in ['Concrete', 'Stone bricks']:
            new_damage *= 1.0 
        elif self.house_materials in ['Mud bricks','Wood']:
            new_damage *= 1.5
        elif self.house_materials == 'Informal settlement ':
            new_damage *= 2.0

        self.house_damage = np.clip(self.house_damage + new_damage, 0, 1)

        # livelihood damage isn't affected by preparedness
        new_damage = 0.1 * (flood_value / FLOOD_DAMAGE_MAX)
        self.livelihood_damage = np.clip(self.livelihood_damage + new_damage, 0, 1)

        self.displacement_decision()
           

    def step(self):
        self.update_displacement_decision()
        self.update_sentiments()


    def update_sentiments(self):
        """
        update trust for current household
        - if household is alerted and at least a neighbour received flood, trust -> 1
        - if household is alerted and no neighbour received flood, trust -> trust - 50%
        - if household is not alerted and at least a neighbour received flood, fear -> fear + 20%
        - if household is not alerted and noone received flood, 
        """
        if self.status == STATUS_DISPLACED:
            self.awareness = 1.0
            self.fear = 1.0
            return

        neighbours = self.model.space.get_neighbors_within_distance(
               self, MAX_DISTANCE
        )
        neighbours_flooded = [neighbour.received_flood for neighbour in neighbours]
        anyone_flooded = \
            self.received_flood or \
            any(neighbours_flooded)
        
        if self.received_flood:
            self.awareness = 1.0

        elif len(neighbours_flooded) > 0 \
            and sum(neighbours_flooded) > 0.25 * len(neighbours_flooded):
            self.awareness = 1.0

        if self.alerted and anyone_flooded:
            self.trust = 1.0
        
        elif not self.alerted and anyone_flooded:
            self.fear = np.clip(self.fear * 1.2, 0, 1)

        elif not anyone_flooded:
            if self.alerted:
                self.trust = np.clip(self.trust * 0.5, 0, 1)
                self.fear = np.clip(self.fear * 0.8, 0, 1)

            self.awareness = np.clip(self.awareness * 0.8, 0, 1)
        

    def fix_damage(self):
        """
        fix damage for current household
        """
        if self.income <= POVERTY_LINE:
            return

        recovery = 0.3
        # every unit of income above poverty line increases recovery by 10%
        recovery = recovery * (1 + (self.income - POVERTY_LINE) / 10)

        if self.house_materials in ['Concrete', 'Stone bricks']:
            recovery *= 1.0 
        elif self.house_materials in ['Mud bricks','Wood']:
            recovery *= 1.5
        elif self.house_materials == 'Informal settlement ':
            recovery *= 2.0            

        self.house_damage = np.clip(self.house_damage - recovery, 0, 1)
        self.livelihood_damage = np.clip(self.livelihood_damage - recovery, 0, 1)

        neighbours = self.model.space.get_neighbors_within_distance(
               self, MAX_DISTANCE
        )

        # help other household to fix damage
        for neighbour in neighbours:
            if neighbour.house_damage > 0:
                neighbour.house_damage = np.clip(neighbour.house_damage - 0.05, 0, 1)

    
    @property
    def perception(self):
        return self.awareness * self.fear
    
    @property
    def income(self):
        return self.base_income * (1 - self.livelihood_damage)
   
