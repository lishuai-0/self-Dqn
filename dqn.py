# 导入必要的模块和类
from __future__ import annotations

import os

from dqn_agents import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from dqn_network import create_lstm_dqn_network
from env.environment import OnlineFlexibleResourceAllocationEnv
from train_helper import setup_tensorboard, generate_eval_envs, run_training

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 如果这个程序是主程序
if __name__ == "__main__":
    # 定义TensorBoard日志的文件夹和记录器
    folder = 'dqn_agents'
    writer, datetime = setup_tensorboard('./results/logs/', folder)

    # 定义保存模型文件的文件夹
    save_folder = f'{folder}_{datetime}'

    # 创建OnlineFlexibleResourceAllocationEnv环境对象和评估环境对象列表
    env = OnlineFlexibleResourceAllocationEnv([
        '../settings/basic.env',
        '../settings/large_tasks_servers.env',
        '../settings/limited_resources.env',
        '../settings/mixture_tasks_servers.env'
    ])
    # 生成20个评估环境
    eval_envs = generate_eval_envs(env, 20, f'settings/eval_envs/algo/')

    # 创建TaskPricingDqnAgent和ResourceWeightingDqnAgent代理对象
    task_pricing_agents = [
        TaskPricingDqnAgent(agent_num, create_lstm_dqn_network(9, 21), save_folder=save_folder)
        for agent_num in range(3)
    ]
    resource_weighting_agents = [
        ResourceWeightingDqnAgent(0, create_lstm_dqn_network(16, 11), save_folder=save_folder)
    ]

    # 使用TensorBoard记录器运行训练
    with writer.as_default():
        # run_training(env, eval_envs, 600, task_pricing_agents, resource_weighting_agents, 10)
        run_training(env, eval_envs, 6, task_pricing_agents, resource_weighting_agents, 10)

    # 保存模型文件
    for agent in task_pricing_agents:
        agent.save()
    for agent in resource_weighting_agents:
        agent.save()
