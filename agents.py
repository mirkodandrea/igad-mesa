import mesa_geo as mg
import numpy as np
from numpy.random import random
from shapely.geometry import Point

from constants import (AL_GAILI_COEFFIECIENT, AL_SHUHADA_COEFFIECIENT,
                       CHECK_TRUST, CROPLAND_COEFFIECIENT,
                       ELTOMANIAT_COEFFIECIENT, FLOOD_COEFFIECIENT,
                       HOUSE_COEFFIECIENT, INCOME_COEFFIECIENT,
                       LIVESTOCK_COEFFIECIENT, MATERIAL_CONCRETE,
                       MATERIAL_INFORMAL_SETTLEMENTS, MATERIAL_MUD_BRICKS,
                       MATERIAL_STONE_BRICKS, MATERIAL_WOOD, MAX_DISTANCE,
                       POVERTY_LINE, SQUARE_INCOME_COEFFIECIENT,
                       STATUS_DISPLACED, STATUS_EVACUATED, STATUS_NORMAL,
                       STATUS_TRAPPED, VULNERABILITY_COEFFIECIENT,
                       WAD_RAMLI_COEFFIECIENT, WAWISE_GARB_COEFFIECIENT,
                       WAWISE_OUM_OJAIJA_COEFFIECIENT,
                       BASE_RECOVERY)
from utils import get_damage, get_livelihood_damage


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

        self.displacement_time = 0

        self._neighbours = None

        self.trapped_probability = 0.0

    def get_neighbours(self):
        """
        get all neighbors within a certain distance
        """
        if self._neighbours is None:
            self._neighbours = list(
                self.model.space.get_neighbors_within_distance(
                self, MAX_DISTANCE
            ))
        return self._neighbours

    @property
    def perception(self):
        """
        perception of flood risk
        """
        return (self.awareness + self.fear)/2.0
    
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
   

    def __repr__(self):
        return "Household " + str(self.unique_id)


    def get_description(self):        
        return {
            'id': self.unique_id,
            'village': self.village,
            'damage': f"h: {int(100 * self.house_damage)}% - l: {int(100 * self.livelihood_damage)}%",
            'status': self.status, 
            'flood_prone': self.flood_prone,
            'income': f"{self.income:.2f}", 
            'awareness': f"{int(100 * self.awareness)}%", 
            'fear': f"{int(100 * self.fear)}%", 
            'perception': f"{int(100 * self.perception)}%",
            'trust': f"{int(100 * self.trust)}%", 
            'received_flood': self.received_flood,
            'house_materials': self.house_materials,
            'displacement_time': self.displacement_time,
            'last_house_damage': f"{int(100 * self.last_house_damage)}%",
            'last_livelihood_damage': f"{int(self.last_livelihood_damage)}%",
        }
   

    def reset_flags(self):
        """
        set household status for each step to initial values        
        """
        # set all flags back to False
        self.alerted = False
        self.received_flood = False
        self.status_changed = False
        self.last_house_damage = 0
        self.last_livelihood_damage = 0
        self.livelihood_damage = 0

    
    def return_decision(self):
        """
        check household damage and decide status
        """
        if self.status in [STATUS_NORMAL, STATUS_TRAPPED]:
            return


        if self.status == STATUS_EVACUATED:
            selected_threshold = self.model.LOW_DAMAGE_THRESHOLD \
                                    if self.income < POVERTY_LINE \
                                    else self.model.HIGH_DAMAGE_THRESHOLD

            if self.house_damage < selected_threshold:
                self.status = STATUS_NORMAL
            else:
                if random() < self.trapped_probability:
                    self.status = STATUS_TRAPPED
                else:
                    self.status = STATUS_DISPLACED
            
        elif self.status == STATUS_DISPLACED:
            if self.house_damage < self.model.HIGH_DAMAGE_THRESHOLD:
                self.status = STATUS_NORMAL

        if self.status == STATUS_DISPLACED:
            self.displacement_time += 1
        else:
            self.displacement_time = 0
        
    def update_trapped_probability(self):
        """
        """        
        x = self.income * INCOME_COEFFIECIENT + \
            self.income * self.income * SQUARE_INCOME_COEFFIECIENT + \
            min(self.number_of_floods, 2) * FLOOD_COEFFIECIENT + \
            self.vulnerability * VULNERABILITY_COEFFIECIENT + \
            self.livestock * LIVESTOCK_COEFFIECIENT + \
            self.house * HOUSE_COEFFIECIENT + \
            self.cropland * CROPLAND_COEFFIECIENT
        
        if self.village == 'Al-Gaili':
            x += AL_GAILI_COEFFIECIENT 
        elif self.village == 'Al-Shuhada':
            x += AL_SHUHADA_COEFFIECIENT
        elif self.village == 'Eltomaniat':
            x += ELTOMANIAT_COEFFIECIENT
        elif self.village == 'Wad Ramli Camp':
            x += WAD_RAMLI_COEFFIECIENT
        elif self.village == 'Wawise Garb':
            x += WAWISE_GARB_COEFFIECIENT
        elif self.village == 'Wawise Oum Ojaija':
            x += WAWISE_OUM_OJAIJA_COEFFIECIENT
        else:
            pass

        self.trapped_probability = np.exp(x) / (1 + np.exp(x))
        

    def displacement_decision(self):
        """ 
        check household damage and decide status
        """
        if self.status not in [STATUS_NORMAL, STATUS_TRAPPED]:
            # already displaced or evacuated
            return

        if self.house_damage < self.model.LOW_DAMAGE_THRESHOLD:
            # low damage to the house -> return without changing status
            return 
        
        if self.house_damage < self.model.HIGH_DAMAGE_THRESHOLD:
            # medium house damage -> check against perception
            if self.perception < self.model.RISK_PERCEPTION_THRESHOLD:
                return
        
        # high house damage -> calculate trapped probability
        
        if random() < self.trapped_probability:
            # household is trapped, cannot move
            self.status = STATUS_TRAPPED
        else:
            # household displace
            self.status = STATUS_DISPLACED


    def check_neighbours_for_displacement(self):
        """
        update displacement decision based on neighbours
        do this only if household is not already displaced or evacuated.
        Will consider displace if enough neighbours are displaced and can move.
        """    
        if self.status != STATUS_NORMAL:
            return
        
        if self.perception < self.model.RISK_PERCEPTION_THRESHOLD:
            return 
        
        neighbours = self.get_neighbours()
        
        other_statuses = [neighbour.status == STATUS_DISPLACED for neighbour in neighbours]
        if sum(other_statuses) > 0.75 * len(neighbours):
            if random() < self.trapped_probability:
                self.status = STATUS_TRAPPED
            else:
                self.status = STATUS_DISPLACED


    def check_for_early_warning(self):
        """
        check if governemnt has issued early warning
        - if household is already evacuated, do nothing
        - if household doesn't trust the government, then do nothing
        """
        if not self.model.emitted_early_warning:
            return 

        if not self.flood_prone:
            # not flood prone, don't receive early warning
            return
        
        if not self.status in [STATUS_NORMAL, STATUS_TRAPPED]:
            # already displaced or evacuated
            # don't receive early warning
            return

        self.alerted = True

        if self.trust < self.model.TRUST_THRESHOLD and CHECK_TRUST:
            # distrust the government
            return

        
        if self.status == STATUS_TRAPPED:
            # can't evcauate anyway
            return
        
        if self.income < POVERTY_LINE:
            # poor household, cannot afford to move
            return
        
        # trust the government
        if self.perception >= self.model.RISK_PERCEPTION_THRESHOLD:
            # aware of risk, move before flood
            if random() < self.trapped_probability:
                self.status = STATUS_TRAPPED
            else:
                self.status = STATUS_EVACUATED


    # [UNUSED]
    def check_neighbours_for_evacuation(self):
        """check neighbours for early warning reaction
        - if enough neighbours are evacuated, then evacuate
        """
        if not self.model.emitted_early_warning:
            return 
        
        if self.status not in [STATUS_NORMAL, STATUS_TRAPPED]:
            # already evacuated
            return
        
        neighbours = self.get_neighbours()
        
        if self.status != STATUS_TRAPPED:
            # check if other households are evacuated
            other_status = [neighbour.status == STATUS_EVACUATED for neighbour in neighbours]
            if sum(other_status) > 0.5 * len(neighbours):
                # enough neighbours are evacuated, evacuate myself if income is high enough
                if self.income >= POVERTY_LINE:
                    self.status = STATUS_EVACUATED
    

    def react_to_flood(self):
        """react to flood
        increment damage if household is flooded
        - damage is in percentage
        - damage occurs if flood value is > 100mm
        - damage is calculated using damage curve defined by house material
        """
        if not self.model.flood_event:
            return
        
        flood_value = self.model.space.get_water_level(self)

        if flood_value > 0:
            self.received_flood = True
            self.number_of_floods += 1

        # house damage using curve        
        new_damage = get_damage(flood_value, self.house_materials)
        self.last_house_damage = new_damage
        self.house_damage = max(self.house_damage, new_damage)

        # livelihood damage using curve
        new_damage = get_livelihood_damage(flood_value, 'crops')


        self.last_livelihood_damage = new_damage
        self.livelihood_damage = np.clip(new_damage, 0, 1)
           

    def update_sentiments(self):
        """
        update sentiments based on previous events
        - if household is alerted and at least a neighbour received flood, trust -> 1, fear = __
        - if household is alerted and no neighbour received flood, trust -> trust - 50%, fear -10%
        - if household is not alerted and at least a neighbour received flood, fear -> fear + 20%, trust -70%
        - if household is not alerted and no one received flood, 
        """
       
        if self.status not in [STATUS_NORMAL, STATUS_TRAPPED] \
            and not self.status_changed:
            return
        
        neighbours = self.get_neighbours()
        neighbours_flooded = [neighbour.received_flood for neighbour in neighbours]
        anyone_flooded = \
            self.received_flood or \
            any(neighbours_flooded)

        min_awareness = 0.5 if self.model.awareness_program else 0.0
        if not anyone_flooded:
            # awareness is reduced by 10% if no one is flooded
            self.awareness = np.clip(self.awareness - self.model.AWARENESS_DECREASE, min_awareness, 1)

            # fear is reduced by 10% if no one is flooded
            self.fear = np.clip(self.fear - self.model.FEAR_CHANGE, 0.0, 1)

            if self.alerted:
                # trust is reduced by 10% absolute value
                self.trust = np.clip(self.trust - self.model.TRUST_CHANGE, 0, 1)


        else: # anyone flooded
            # [TODO] check damage of house and livelihood
            max_damage = max(self.last_house_damage, self.last_livelihood_damage)
            if max_damage > self.model.LOW_DAMAGE_THRESHOLD:
                # increase awareness if my damage is over LOW_DAMAGE_THRESHOLD
                self.awareness = np.clip(self.awareness + self.model.AWARENESS_INCREASE, min_awareness, 1)
            else:
                # increase awareness if at least 25% of neighbours have damage over LOW_DAMAGE_THRESHOLD
                neighbours_high_damage = [neighbour.last_house_damage > self.model.LOW_DAMAGE_THRESHOLD for neighbour in neighbours]
                if sum(neighbours_high_damage) > self.model.NEIGHBOURS_HIGH_DAMAGE_FRACTION * len(neighbours):
                    # take into account the near-miss-event effect
                    # [TODO] think about enabling this only if the household is not flooded

                    if random() < self.awareness:  # actually increase awareness with probability higher if already aware
                        self.awareness = np.clip(self.awareness + self.model.AWARENESS_INCREASE, min_awareness, 1)
                    else: # not aware, decrease awareness because of near-miss-event effect
                        self.awareness = np.clip(self.awareness - self.model.AWARENESS_DECREASE, min_awareness, 1)

            if self.alerted: # flooded and alerted
                self.trust = np.clip(self.trust + self.model.TRUST_CHANGE, 0, 1)
                #[TODO] modulate fear increase using damage
                self.fear = np.clip(self.fear + self.model.FEAR_CHANGE, 0, 1)
            else: # flooded but not alerted
                #[TODO] modulate fear increase using damage
                self.fear = np.clip(self.fear + self.model.FEAR_CHANGE, 0, 1)
                
                #[TODO] modulate trust decrease using damage
                #[TODO] talk about this!
                self.trust = np.clip(self.trust - self.model.TRUST_CHANGE, 0, 1)


    def fix_damage(self):
        """
        fix damage for current household
        """
        if self.model.house_repair_program > 0:
            # if government help is available, try to use it to fix damage if damage is above HIGH_DAMAGE_THRESHOLD
            if self.house_damage > self.model.HIGH_DAMAGE_THRESHOLD:
                if random() < self.model.house_repair_program:
                    # government help is used to fix damage 100%
                    self.house_damage = 0

                    if self.model.house_improvement_program:
                        # if house improvement program is active, house materials are improved
                        self.house_materials = MATERIAL_CONCRETE

                    return
        
        if self.income <= POVERTY_LINE:
            # if household is poor, it cannot fix damage
            return

        
        # every unit of income above poverty line increases recovery by +10%
        recovery = BASE_RECOVERY + (self.income - POVERTY_LINE) / 10

        if self.house_materials in [MATERIAL_CONCRETE, MATERIAL_STONE_BRICKS]:
            pass
        elif self.house_materials in [MATERIAL_MUD_BRICKS, MATERIAL_WOOD]:
            recovery *= 1.5
        elif self.house_materials == MATERIAL_INFORMAL_SETTLEMENTS:
            recovery *= 2.0

        self.house_damage = np.clip(self.house_damage - recovery, 0, 1)


    def fix_neighbours_damage(self):
        """
        fix damage for neighbours 
        only if household is not flooded and has income above poverty line and has low damage
        household should not be displaced or evacuated
        """
        if  self.income <= POVERTY_LINE or \
            self.house_damage > self.model.LOW_DAMAGE_THRESHOLD or \
            self.received_flood or \
            self.status not in [STATUS_NORMAL, STATUS_TRAPPED]:
            return

        neighbours = self.get_neighbours()

        # help other household to fix damage
        for neighbour in neighbours:
            if neighbour.house_damage > 0:
                neighbour.house_damage = np.clip(neighbour.house_damage - 0.05, 0, 1)
   


        