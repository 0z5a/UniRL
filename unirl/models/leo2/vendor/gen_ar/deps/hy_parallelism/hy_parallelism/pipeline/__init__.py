# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

from torch.distributed.pipelining.schedules import get_schedule_class
from .schedules import InferencePipelineScheduleMulti

def hy_get_schedule_class(schedule_name: str):
    """
    Maps a schedule name (case insensitive) to its corresponding class object.

    Args:
        schedule_name (str): The name of the schedule.
    """
    try:
        return get_schedule_class(schedule_name)
    except ValueError as e:
        ...
    schedule_map = {
        'InferencePipelineScheduleMulti': InferencePipelineScheduleMulti,
    }
    lowercase_keys = {k.lower(): k for k in schedule_map.keys()}
    lowercase_schedule_name = schedule_name.lower()
    if lowercase_schedule_name not in lowercase_keys:
        raise ValueError(
            f"Unknown schedule name '{schedule_name}'. The valid options are {list(schedule_map.keys())}"
        ) from e
    return schedule_map[lowercase_keys[lowercase_schedule_name]]
