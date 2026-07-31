from datetime import datetime
from typing import Optional, Any
import random
import pandas as pd

TASK_TYPE_DEFAULTS = {
    "training": {
        "sla_multiplier": 12.0,
        "max_delay_minutes": 24 * 60,
        "migration_allowed": True,
        "defer_allowed": True,
        "latency_sensitive": False,
    },
    "batch": {
        "sla_multiplier": 6.0,
        "max_delay_minutes": 8 * 60,
        "migration_allowed": True,
        "defer_allowed": True,
        "latency_sensitive": False,
    },
    "inference": {
        "sla_multiplier": 1.2,
        "max_delay_minutes": 15,
        "migration_allowed": False,
        "defer_allowed": False,
        "latency_sensitive": True,
    },
}

class Task:
    """
    定义一个数据中心任务对象,用来描述一个计算任务的资源需求、时间约束、SLA 截止时间、调度状态、等待状态以及来源/目标数据中心信息。
    Represents a computing task in a datacenter scheduling system.
    
    Each task is assigned resource requirements, timing attributes, and an SLA deadline.
    Tasks can be deferred if they are not immediately scheduled, and they are tracked until
    they complete execution.

    代表数据中心调度系统中的计算任务。每个任务都被分配了资源需求、时间属性和服务级别协议 (SLA) 截止时间。如果任务未能立即调度，则可以延迟执行，并且系统会跟踪任务直至其执行完成。

    Attributes:
        job_name (str): 任务的唯一标识符。
        arrival_time (datetime): 任务进入系统的时间戳。
        duration (float): 所需执行时间，单位为分钟。
        cores_req (float): 所需的 CPU 核心数量。
        gpu_req (float): 所需的 GPU 单元数量。
        mem_req (float): 所需内存，单位为 GB。
        bandwidth_gb (float): 所需带宽，单位为 GB。
        task_type (str): 工作负载类别，例如 training、inference 或 batch。
        max_delay_minutes (float): 最大端到端等待/延迟容忍时间。
        migration_allowed (bool): 任务是否允许在其来源数据中心之外运行。
        defer_allowed (bool): 调度器是否可以将任务推迟到之后的时间步执行。
        latency_sensitive (bool): 任务是否应被视为延迟敏感型任务。
        start_time (Optional[datetime]): 任务开始执行的时间。
        finish_time (Optional[datetime]): 调度后预计完成时间。
        sla_deadline (datetime): 根据 arrival_time + sla_multiplier * duration 计算出的截止时间。
        sla_met (bool): 表示任务是否满足 SLA。
        wait_intervals (int): 任务已等待时长的时间步计数器。
        origin_dc_id (Optional[int]): 任务来源数据中心的 ID。
        dest_dc_id (Optional[int]): 被分配的目标数据中心 ID。
        dest_dc (Optional[Any]): 目标数据中心的引用。
        temporarily_deferred (bool): 表示任务是否被暂时推迟以便稍后分配。
        sla_multiplier (int): 用于计算 SLA 截止时间的乘数。
    """
    
    def __init__(
        self,
        job_name: str,
        arrival_time: datetime,
        duration: float,
        cores_req: float,
        gpu_req: float,
        mem_req: float,
        bandwidth_gb: float,
        sla_multiplier: float = None,
        task_type: str = "batch",
        max_delay_minutes: float = None,
        migration_allowed: bool = None,
        defer_allowed: bool = None,
        latency_sensitive: bool = None,
    ) -> None:
        # Initialize task properties
        self.job_name = job_name
        self.arrival_time = arrival_time
        self.duration = duration
        self.cores_req = cores_req
        self.gpu_req = gpu_req
        self.mem_req = mem_req
        self.bandwidth_gb = bandwidth_gb

        self.task_type = task_type if task_type in TASK_TYPE_DEFAULTS else "batch"
        defaults = TASK_TYPE_DEFAULTS[self.task_type]
        self.max_delay_minutes = (
            float(max_delay_minutes)
            if max_delay_minutes is not None
            else float(defaults["max_delay_minutes"])
        )
        self.migration_allowed = (
            bool(migration_allowed)
            if migration_allowed is not None
            else bool(defaults["migration_allowed"])
        )
        self.defer_allowed = (
            bool(defer_allowed)
            if defer_allowed is not None
            else bool(defaults["defer_allowed"])
        )
        self.latency_sensitive = (
            bool(latency_sensitive)
            if latency_sensitive is not None
            else bool(defaults["latency_sensitive"])
        )
        
        # Timing properties: to be set upon scheduling by the global scheduler
        self.start_time: Optional[datetime] = None
        self.finish_time: Optional[datetime] = None
        
        # Compute the SLA deadline based on a fixed factor
        self.sla_multiplier = sla_multiplier if sla_multiplier is not None else defaults["sla_multiplier"]
        self.sla_deadline = arrival_time + pd.Timedelta(minutes=self.sla_multiplier * duration)
        self.sla_met = False
        
        # Initialize wait time counter
        self.wait_intervals: int = 0
        
        # Record the origin datacenter
        self.origin_dc_id: Optional[int] = None
        self.origin_dc: Optional[Any] = None
        
        # Destination information will be assigned when the task is routed
        self.dest_dc_id: Optional[int] = None
        self.dest_dc: Optional[Any] = None
        
        # Flag to indicate if the task is deferred for future scheduling
        self.temporarily_deferred = False
        
        # Ensure unique identification by appending a random number
        self.job_name += f"_{random.randint(0, 10000)}"

    def __repr__(self) -> str:
        """
        Returns a string representation of the Task object for debugging.

        Returns:
            str: A formatted string representation of the task.
        """
        return (
            f"Task(job_name='{self.job_name}', arrival_time={self.arrival_time}, "
            f"task_type='{self.task_type}', "
            f"duration={self.duration}, cores_req={self.cores_req}, gpu_req={self.gpu_req}, "
            f"mem_req={self.mem_req}, bandwidth_gb={self.bandwidth_gb}, start_time={self.start_time}, "
            f"finish_time={self.finish_time}, wait_intervals={self.wait_intervals}, origin_dc_id={self.origin_dc_id})"
        )

    def increment_wait_intervals(self) -> None:
        """
        Increments the wait time counter by 1 timestep.
        """
        self.wait_intervals += 1

    def is_scheduled(self) -> bool:
        """
        Checks if the task has been scheduled (i.e., if a start time is defined).

        Returns:
            bool: True if scheduled, False otherwise.
        """
        return self.start_time is not None

    def is_completed(self, current_time: datetime) -> bool:
        """
        Determines whether the task has finished execution.

        Args:
            current_time (datetime): The current timestamp in the system.

        Returns:
            bool: True if the task has finished, False otherwise.
        """
        return self.finish_time is not None and current_time >= self.finish_time
