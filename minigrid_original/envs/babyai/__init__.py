from __future__ import annotations

from minigrid_original.envs.babyai.goto import (
    GoTo,
    GoToDoor,
    GoToImpUnlock,
    GoToLocal,
    GoToObj,
    GoToObjDoor,
    GoToRedBall,
    GoToRedBallGrey,
    GoToRedBallNoDists,
    GoToRedBlueBall,
    GoToSeq,
)
from minigrid_original.envs.babyai.open import (
    Open,
    OpenDoor,
    OpenDoorsOrder,
    OpenRedDoor,
    OpenTwoDoors,
)
from minigrid_original.envs.babyai.other import (
    ActionObjDoor,
    FindObjS5,
    KeyCorridor,
    MoveTwoAcross,
    OneRoomS8,
)
from minigrid_original.envs.babyai.pickup import (
    Pickup,
    PickupAbove,
    PickupDist,
    PickupLoc,
    UnblockPickup,
)
from minigrid_original.envs.babyai.putnext import PutNext, PutNextLocal
from minigrid_original.envs.babyai.synth import (
    BossLevel,
    BossLevelNoUnlock,
    MiniBossLevel,
    Synth,
    SynthLoc,
    SynthSeq,
)
from minigrid_original.envs.babyai.unlock import (
    BlockedUnlockPickup,
    KeyInBox,
    Unlock,
    UnlockLocal,
    UnlockPickup,
    UnlockToUnlock,
)
