from constants import (
    POVERTY_LINE, MAX_DISTANCE, FLOOD_DAMAGE_THRESHOLD, 
    FLOOD_DAMAGE_MAX, FLOOD_FEAR_MAX,
    DISPLACE_DAMAGE_THRESHOLD,
    
    EVENT_FLOOD, EVENT_EARLY_WARNING
)

STATUS_NORMAL = 'normal'
STATUS_EVACUATED = 'evacuated'
STATUS_DISPLACED = 'displaced'

LOW_DAMAGE_THRESHOLD = 0.4
MEDIUM_DAMAGE_THRESHOLD = 0.65

import numpy as np
from numpy.random import normal, random

from shapely.geometry import Point

import mesa_geo as mg


class HouseholdAgent(mg.GeoAgent):
    """Person Agent."""

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
        # self.family_members = family_members

        self.damage = 0
        self.displaced = False
        
        self.status = STATUS_NORMAL
        
        # prepared to flood
        self.alerted = False
        self.received_flood = False
        self.prepared = False

    def __repr__(self):
        return "Household " + str(self.unique_id)


    def init_step(self):
        """init household status for each step
        decide wether or not displace or stay based on house damage
        """
        # set all flags back to False
        self.alerted = False
        self.prepared = False
        self.received_flood = False

        # check household damage and decide status
        if self.damage < LOW_DAMAGE_THRESHOLD:
            # low damage, return to normal
            self.status = STATUS_NORMAL

        elif self.damage < MEDIUM_DAMAGE_THRESHOLD:
            if self.status == STATUS_NORMAL:
                # medium damage, if income is high enough, displace
                if random() < 0.5 and self.income >= POVERTY_LINE:
                    self.status = STATUS_DISPLACED
        else:
            # high damage, displace
            self.status = STATUS_DISPLACED

        

    def receive_early_warning(self):
        """receive early warning from government
        - if household is already evacuated, do nothing
        - if household doesn't trust the government, then do nothing
        - if household is not aware of risk, prepare
        - if household is aware of risk, evacuate without preparing
        """
        self.alerted = True

        if self.status != STATUS_NORMAL:
            # already evacuated
            return
        
        if self.income < POVERTY_LINE:
            # poor household
            return
        
        if self.trust >= 0.5:
            # trust the government
            if self.perception >= 0.5:
                # aware of risk
                self.status = STATUS_EVACUATED
                return

            self.prepared = True
        
            

    def check_neighbours(self):
        """check neighbours for early warning reaction
        - if enough neighbours are evacuated, then evacuate
        - if enough neighbours are prepared, then prepare
        """
        
        if self.status != STATUS_NORMAL:
            # already evacuated
            return
        
        neighbours = list(self.model.space.get_neighbors_within_distance(
               self, MAX_DISTANCE
        ))

        other_prepared = [neighbour.prepared for neighbour in neighbours]
        if sum(other_prepared) > 0.5 * len(neighbours):
            self.prepared = True

        other_status = [neighbour.status == STATUS_EVACUATED for neighbour in neighbours]
        if sum(other_status) > 0.5 * len(neighbours):
            self.status = STATUS_EVACUATED

            
    def receive_flood(self, flood_value):
        """receive flood for current household
        increment damage if household is flooded
        - damage is in percentage
        - damage occurs if flood value is > 10mm
        - damage is proportional to flood value: 1m -> 100% damage
        """

        if flood_value == 0:
            """ nothing happened to this household """
            # [TODO] update trust if alerted
            self.received_flood = False
            return

        self.received_flood = True

        if self.prepared and flood_value < FLOOD_DAMAGE_THRESHOLD:
            return
        
        new_damage = (flood_value / FLOOD_DAMAGE_MAX)
        if self.prepared:
            new_damage = new_damage * 0.5

        self.damage = np.clip(self.damage + new_damage, 0, 1)        


    def step(self):
        if self.status != STATUS_DISPLACED \
            and not self.received_flood:
            self.fix_damage()

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
        # every unit of income less than 10 decreases recovery capacity by 1%
        if self.income < 10:
            recovery = recovery * (1 - (10 - self.income) * 0.01)

        self.damage = np.clip(self.damage - recovery, 0, 1)

        neighbours = self.model.space.get_neighbors_within_distance(
               self, MAX_DISTANCE
        )

        # help other household to fix damage
        for neighbour in neighbours:
            if neighbour.damage > 0:
                neighbour.damage = np.clip(neighbour.damage - 0.05, 0, 1)

    
    @property
    def perception(self):
        return self.awareness * self.fear