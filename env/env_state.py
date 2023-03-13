"""Immutable Environment state"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from env.task import Task
    from env.server import Server
    from typing import Dict, List, Optional


class EnvState(NamedTuple):
    """
    The environment state that contains a dictionary of server to list of tasks, the task being auctioned
        and the time step
    """
    # 服务器及其任务列表
    server_tasks: Dict[Server, List[Task]]
    # 拍卖任务（如果存在）
    auction_task: Optional[Task]
    # 当前时间步
    time_step: int

    # 将Env对象转换为字符串表示
    def __str__(self) -> str:
        # 将每个服务器及其任务列表转换为字符串表示
        server_tasks_str = ', '.join([f'{server.name}: [{", ".join([task.name for task in tasks])}]'
                                      for server, tasks in self.server_tasks.items()])
        auction_task_str = str(self.auction_task) if self.auction_task else 'None'
        # 返回描述环境状态的字符串
        return f'Env State ({hex(id(self))}) at time step: {self.time_step}\n' \
               f'\tAuction Task -> {auction_task_str}\n' \
               f'\tServers -> {{{server_tasks_str}}}'

    # 在Jupyter Notebook等环境中打印美观的输出
    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())
