
# Poverty line 
POVERTY_LINE = 1

# 1KM
MAX_DISTANCE = 0.002

# minimum flood depth to be considered as a flood
FLOOD_DAMAGE_THRESHOLD = 100
FLOOD_DAMAGE_MAX = 1000
FLOOD_FEAR_MAX = 300

DISPLACE_DAMAGE_THRESHOLD = 0.65


STATUS_NORMAL = 'normal'
STATUS_EVACUATED = 'evacuated'
STATUS_DISPLACED = 'displaced'
STATUS_TRAPPED = 'trapped'

MATERIAL_STONE_BRICKS = 'Stone bricks'
MATERIAL_CONCRETE = 'Concrete'
MATERIAL_WOOD = 'Wood'
MATERIAL_MUD_BRICKS = 'Mud bricks'
MATERIAL_INFORMAL_SETTLEMENTS = 'Informal settlement'

CHECK_TRUST = False
INCOME_COEFFIECIENT = -1
SQUARE_INCOME_COEFFIECIENT = 0.584
FLOOD_COEFFIECIENT = -0.32
VULNERABILITY_COEFFIECIENT = -0.801
LIVESTOCK_COEFFIECIENT = 0.585
HOUSE_COEFFIECIENT = -0.619
CROPLAND_COEFFIECIENT = -0.514
AL_GAILI_COEFFIECIENT = -0.0651
AL_SHUHADA_COEFFIECIENT = 0.859
ELTOMANIAT_COEFFIECIENT = -2
WAD_RAMLI_COEFFIECIENT = -0.82
WAWISE_GARB_COEFFIECIENT = -0.798
WAWISE_OUM_OJAIJA_COEFFIECIENT = 1.906

BASE_RECOVERY = 0.30

# threshold used to decide whether to 
# increase awareness or not when checking for 
# neighbours damage
NEIGHBOURS_HIGH_DAMAGE_FRACTION = 0.25

TRUST_THRESHOLD = 0.5
RISK_PERCEPTION_THRESHOLD = 0.5
LOW_DAMAGE_THRESHOLD = 0.25
HIGH_DAMAGE_THRESHOLD = 0.60
TRUST_CHANGE = 0.1
FEAR_CHANGE = 0.1
AWARENESS_DECREASE = 0.1
AWARENESS_INCREASE = 0.4

FIX_DAMAGE_NEIGHBOURS = 0.05
FIX_DAMAGE_CONCRETE = 0.3
FIX_DAMAGE_MUDBRICK = 0.4
FIX_DAMAGE_INFORMAL_SETTLEMENT = 0.5