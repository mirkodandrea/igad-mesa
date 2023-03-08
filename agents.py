import mesa_geo as mg
import numpy as np
from numpy.random import random
from shapely.geometry import Point

from constants import (FLOOD_DAMAGE_MAX, FLOOD_DAMAGE_THRESHOLD, MAX_DISTANCE,
                       POVERTY_LINE)

STATUS_NORMAL = 'normal'
STATUS_EVACUATED = 'evacuated'
STATUS_DISPLACED = 'displaced'
STATUS_TRAPPED = 'trapped'

LOW_DAMAGE_THRESHOLD = 0.25
MEDIUM_DAMAGE_THRESHOLD = 0.7

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
        :param crs:         Coordinate reference system for the agent    
        """
        super().__init__(unique_id, model, geometry, crs)
        # Agent parameters
        self.obstacles_to_movement = False

        self.house_damage = 0
        self.livelihood_damage = 0

        # set initial status
        self._status = STATUS_NORMAL
        # keep track of status changes from normal
        self.status_changed = False
        
        # received early warning
        self.alerted = False
        # received flood on last step
        self.received_flood = False
        self.last_house_damage = 0
        self.last_livelihood_damage = 0
        # prepared to flood
        self.prepared = False
        self.displacement_time = 0

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
        self.status_changed = False
        self.last_house_damage = 0
        self.last_livelihood_damage = 0

        self.return_decision()

    
    def return_decision(self):
        """
        check household damage and decide status
        """
        if self.status == STATUS_NORMAL:
            return
        
        if self.status == STATUS_EVACUATED:
            if self.house_damage < MEDIUM_DAMAGE_THRESHOLD:
                self.status = STATUS_NORMAL

            elif self.house_damage >= MEDIUM_DAMAGE_THRESHOLD:
                self.status = STATUS_DISPLACED

        if self.house_damage < LOW_DAMAGE_THRESHOLD:
            self.status = STATUS_NORMAL

        if self.status == STATUS_DISPLACED:
            self.displacement_time += 1
        else:
            self.displacement_time = 0
        

    def displacement_decision(self):
        """ 
        check household damage and decide status
        """
        if self.status not in [STATUS_NORMAL, STATUS_TRAPPED]:
            # already displaced or evacuated
            return

        if self.house_damage > MEDIUM_DAMAGE_THRESHOLD or \
            self.livelihood_damage > MEDIUM_DAMAGE_THRESHOLD:
            self.status = STATUS_DISPLACED
            return

        if self.house_damage < LOW_DAMAGE_THRESHOLD and \
            self.livelihood_damage < LOW_DAMAGE_THRESHOLD:
            return      
    
        # in case of medium house damage or medium livelihood damage, check against perception
        if self.perception < 0.5:
            return
    
        if self.income > POVERTY_LINE and \
            not self.obstacles_to_movement:
                self.status = STATUS_DISPLACED
        else: # poor household or obstacles to movement
            self.status = STATUS_TRAPPED


    def update_displacement_decision(self):
        """
        update displacement decision based on neighbours
        do this only if household is not already displaced or evacuated.
        Will consider displace if enough neighbours are displaced and can move.
        """    
        if self.status not in [STATUS_NORMAL, STATUS_TRAPPED]:
            return
        
        if self.perception < 0.5:
            return 
        
        neighbours = list(
            self.model.space.get_neighbors_within_distance(
                self, MAX_DISTANCE
        ))
        
        other_statuses = [neighbour.status == STATUS_DISPLACED for neighbour in neighbours]
        if sum(other_statuses) > 0.75 * len(neighbours):
            if self.income < POVERTY_LINE or \
                self.obstacles_to_movement:
                self.status = STATUS_TRAPPED
            else:
                self.status = STATUS_DISPLACED


    def receive_early_warning(self):
        """receive early warning from government
        - if household is already evacuated, do nothing
        - if household doesn't trust the government, then do nothing
        - if household has not perception of risk, prepare
        - if household has perception of risk, evacuate and prepare
        """
        if not self.flood_prone:
            # not flood prone, don't receive early warning
            return
        
        if not self.status in [STATUS_NORMAL, STATUS_TRAPPED]:
            # already displaced or evacuated
            # don't receive early warning
            return

        self.alerted = True

        if self.trust < 0.5:
            # distrust the government
            # don't prepare
            return


        # prepare for flood anyway        
        self.prepared = True
        
        if self.status == STATUS_TRAPPED:
            # can't evcauate anyway
            return
        
        if self.income < POVERTY_LINE \
            or self.obstacles_to_movement:
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
        
        if self.status not in [STATUS_NORMAL, STATUS_TRAPPED]:
            # already evacuated
            return
        
        # collect neighbours
        neighbours = list(self.model.space.get_neighbors_within_distance(
               self, MAX_DISTANCE
        ))

        if self.status != STATUS_TRAPPED:
            # check if other households are evacuated
            other_status = [neighbour.status == STATUS_EVACUATED for neighbour in neighbours]
            if sum(other_status) > 0.25 * len(neighbours):
                # enough neighbours are evacuated, evacuate myself if income is high enough
                if self.income >= POVERTY_LINE and not self.obstacles_to_movement:
                    self.status = STATUS_EVACUATED
    

        other_prepared = [neighbour.prepared for neighbour in neighbours]
        if sum(other_prepared) > 0.25 * len(neighbours):
            # enough neighbours are prepared, prepare myself
            self.prepared = True


            
    def receive_flood(self, flood_value):
        """receive flood for current household
        increment damage if household is flooded
        - damage is in percentage
        - damage occurs if flood value is > 100mm
        - damage is proportional to flood value: 1m -> 100% damage
        - damage is reduced by 50% if household is prepared
        - damage is increased if household is made of mud bricks or wood by 50%
        - damage is increased if household is an informal settlement by 100%
        """

        if flood_value < FLOOD_DAMAGE_THRESHOLD:
            """ nothing happened to this household """
            self.received_flood = False
            return

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

        self.last_house_damage = new_damage
        self.house_damage = np.clip(self.house_damage + new_damage, 0, 1)

        # livelihood damage isn't affected by preparedness
        new_damage = flood_value / FLOOD_DAMAGE_MAX
        self.last_livelihood_damage = new_damage
        self.livelihood_damage = np.clip(self.livelihood_damage + new_damage, 0, 1)

        self.displacement_decision()
           

    def step(self):
        self.update_displacement_decision()
        self.update_sentiments()


    def update_sentiments(self):
        """
        update trust for current household
        - if household is alerted and at least a neighbour received flood, trust -> 1, fear = __
        - if household is alerted and no neighbour received flood, trust -> trust - 50%, fear -10%
        - if household is not alerted and at least a neighbour received flood, fear -> fear + 20%, trust -70%
        - if household is not alerted and no one received flood, 
        """
       
        if self.status not in [STATUS_NORMAL, STATUS_TRAPPED] \
            and not self.status_changed:
            return
        
        neighbours = list(self.model.space\
            .get_neighbors_within_distance(
            self, MAX_DISTANCE
        ))
        neighbours_flooded = [neighbour.received_flood for neighbour in neighbours]
        anyone_flooded = \
            self.received_flood or \
            any(neighbours_flooded)

        MIN_AWARENESS = 0.5 if self.model.awareness_program else 0.3
        if not anyone_flooded:
            # awareness is reduced by 20% if no one is flooded
            self.awareness = np.clip(self.awareness - 0.1, MIN_AWARENESS, 1)

            # fear is reduced by 10% if no one is flooded
            self.fear = np.clip(self.fear - 0.1, 0.3, 1)

            if self.alerted:
                # trust is reduced by 50% if no one is flooded but household is alerted
                self.trust = np.clip(self.trust - 0.1, 0, 1)


        else: # anyone flooded
            # [TODO] check damage of house and livelihood
            max_damage = max(self.last_house_damage, self.last_livelihood_damage)
            if max_damage > LOW_DAMAGE_THRESHOLD:
                # increase awareness if my damage is over LOW_DAMAGE_THRESHOLD
                self.awareness = np.clip(self.awareness + 0.4, MIN_AWARENESS, 1)
            else:
                # increase awareness if at least 25% of neighbours have damage over LOW_DAMAGE_THRESHOLD
                neighbours_high_damage = [neighbour.last_house_damage > LOW_DAMAGE_THRESHOLD for neighbour in neighbours]
                if sum(neighbours_high_damage) > 0.25 * len(neighbours):
                    # take into account the near-miss-event effect
                    # [TODO] think about enabling this only if the household is not flooded

                    if random() < self.awareness:  # actually increase awareness with probability higher if already aware
                        self.awareness = np.clip(self.awareness + 0.4, MIN_AWARENESS, 1)
                    else: # not aware, decrease awareness because of near-miss-event effect
                        self.awareness = np.clip(self.awareness - 0.1, MIN_AWARENESS, 1)

            if self.alerted: # flooded and alerted
                self.trust = 1.0
                #[TODO] modulate fear increase using damage
                self.fear = np.clip(self.fear + 0.1, 0, 1)
            else: # flooded but not alerted
                #[TODO] modulate fear increase using damage
                self.fear = np.clip(self.fear + 0.2, 0, 1)
                
                #[TODO] modulate trust decrease using damage
                #[TODO] talk about this!
                self.trust = np.clip(self.trust - 0.1, 0, 1)


    def fix_damage(self, government_help):
        """
        fix damage for current household
        """
        self.livelihood_damage = np.clip(self.livelihood_damage - 0.3, 0, 1)

        if government_help <= 0:
            return

        # if government help is available, try to use it to fix damage if damage is above MEDIUM_DAMAGE_THRESHOLD
        if self.house_damage > MEDIUM_DAMAGE_THRESHOLD:
            if random() < government_help:
                # government help is used to fix damage 100%
                self.house_damage = 0

                if self.model.house_improvement_program:
                    # if house improvement program is active, house materials are improved
                    self.house_materials = 'Concrete'
                    
                return

                

        if self.income <= POVERTY_LINE\
            or self.status not in [STATUS_NORMAL, STATUS_TRAPPED]:
            # recover only if household is not displaced or evacuated
            # and if household has income above poverty line
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

        

    def fix_neighbours_damage(self):
        """
        fix damage for neighbours 
        only if household is not flooded and has income above poverty line and has low damage
        household should not be displaced or evacuated
        """
        if  self.income <= POVERTY_LINE or \
            self.house_damage > LOW_DAMAGE_THRESHOLD or \
            self.received_flood or \
            self.status not in [STATUS_NORMAL, STATUS_TRAPPED]:
            return

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
        """
        household income:
        livelihood and basic income if applicable
        """
        basic_income = 0
        if self.model.basic_income_program:
            basic_income = POVERTY_LINE

        return self.base_income * (1 - self.livelihood_damage) + basic_income
    
    @property 
    def status(self):
        return self._status
    
    @status.setter
    def status(self, value):
        if self.status == STATUS_NORMAL and value in [STATUS_EVACUATED, STATUS_DISPLACED]:
            self.status_changed = True
        else:
            self.status_changed = False 

        self._status = value
    

    def get_description(self):
        return {
            'id': self.unique_id,
            'damage': f"h: {int(100 * self.house_damage)}% - l: {int(100 * self.livelihood_damage)}%",
            'status': self.status, 
            'income': f"{self.income:.2f}", 
            'awareness': f"{int(100 * self.awareness)}%", 
            'fear': f"{int(100 * self.fear)}%", 
            'perception': f"{int(100 * self.perception)}%",
            'trust': f"{int(100 * self.trust)}%", 
            'received_flood': self.received_flood,
            'house_materials': self.house_materials,
            'displacement_time': self.displacement_time,
            'obstacles_to_movement': self.obstacles_to_movement,
            'last_house_damage': f"{int(100 * self.last_house_damage)}%",
            'last_livelihood_damage': f"{int(self.last_livelihood_damage)}%",
        }
   
