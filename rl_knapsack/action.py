from enum import IntEnum


class Action(IntEnum):
    """Choose whether to include the current item."""

    SKIP = 0
    TAKE = 1
