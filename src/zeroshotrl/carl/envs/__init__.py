# flake8: noqa: F401
# Modular imports
import importlib.util as iutil
import warnings

# Classic control is in gym and thus necessary for the base version to run
from zeroshotrl.carl.envs.gymnasium import *

__all__ = [
    "CARLAcrobot",
    "CARLCartPole",
    "CARLMountainCar",
    "CARLMountainCarContinuous",
    "CARLPendulum",
]


def check_spec(spec_name: str) -> bool:
    """Check if the spec is installed

    Parameters
    ----------
    spec_name : str
        Name of package that is necessary for the environment suite.

    Returns
    -------
    bool
        Whether the spec was found.
    """
    spec = iutil.find_spec(spec_name)
    found = True # spec is not None
    if not found:
        with warnings.catch_warnings():
            warnings.simplefilter("once")
            warnings.warn(
                f"Module {spec_name} not found. If you want to use these environments, please follow the installation guide."
            )
    return found


# Environment loading
found = True # check_spec("Box2D")
if found:
    from zeroshotrl.carl.envs.gymnasium.box2d import *

    __all__ += ["CARLBipedalWalker", "CARLLunarLander", "CARLVehicleRacing"]

# found = True # check_spec("brax")
# if found:
#     from zeroshotrl.carl.envs.brax import *

#     __all__ += [
#         "CARLBraxAnt",
#         "CARLBraxHalfcheetah",
#         "CARLBraxHopper",
#         "CARLBraxHumanoid",
#         "CARLBraxHumanoidStandup",
#         "CARLBraxInvertedDoublePendulum",
#         "CARLBraxInvertedPendulum",
#         "CARLBraxPusher",
#         "CARLBraxReacher",
#         "CARLBraxWalker2d",
#     ]

# found = True # check_spec("py4j")
# if found:
#     from zeroshotrl.carl.envs.mario import *

#     __all__ += ["CARLMarioEnv"]

# found = True # check_spec("dm_control")
# if found:
#     from zeroshotrl.carl.envs.dmc import *

#     __all__ += [
#         "CARLDmcFingerEnv",
#         "CARLDmcFishEnv",
#         "CARLDmcQuadrupedEnv",
#         "CARLDmcWalkerEnv",
#     ]

# found = True # check_spec("distance")
# if found:
#     from zeroshotrl.carl.envs.rna import *

#     __all__ += ["CARLRnaDesignEnv"]
