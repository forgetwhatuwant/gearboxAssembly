# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Sun Gear 6 environment - Stage 2 curriculum (gears 4-6)."""

import gymnasium as gym

from .sun_gear_6_env import SunGear6Env
from .sun_gear_6_env_cfg import SunGear6EnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Template-Galaxea-Sun-Gear-6-v0",
    entry_point="Galaxea_Lab_External.tasks.direct.galaxea_lab_sun_gear_6:SunGear6Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SunGear6EnvCfg,
    },
)
