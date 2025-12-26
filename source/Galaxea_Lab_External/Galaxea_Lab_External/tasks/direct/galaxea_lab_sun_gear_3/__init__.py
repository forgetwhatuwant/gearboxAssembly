# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="Template-Galaxea-Sun-Gear-3-v0",
    entry_point=f"{__name__}.sun_gear_3_env:SunGear3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sun_gear_3_env_cfg:SunGear3EnvCfg",
    },
)
