from isaaclab.utils import configclass
from dataclasses import MISSING

@configclass
class MirrorCfg:
    policy_obvs_mirror_id_left  : list[int] = MISSING
    policy_obvs_mirror_id_right : list[int] = MISSING
    policy_obvs_opposite_id     : list[int] = MISSING
    action_mirror_id_left       : list[int] = MISSING
    action_mirror_id_right      : list[int] = MISSING
    action_opposite_id          : list[int] = MISSING
    action_opposite_id_a        : list[int] = MISSING
