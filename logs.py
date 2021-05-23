import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
import numpy as np


def log_statistics(T, eval_env, policy, metrics, results_dir, evaluation_episodes, evaluate=False):
    metrics['steps'].append(T)
    T_rewards = []

    # Test performance over several episodes
    done = True
  
    for _ in range(evaluation_episodes):
        while True:
            if done:
                state, reward_sum, done = eval_env.reset(), 0, False

            action = policy.act(state, sample=True)  # Choose an action ε-greedily
            state, reward, done, info = eval_env.step(action)  # Step
            reward_sum += reward

            if done:
                T_rewards.append(reward_sum)
                break

    avg_reward = sum(T_rewards) / len(T_rewards)

    if not evaluate:
        # Append to results and save metrics
        metrics['rewards'].append(T_rewards)
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))
        # Plot
        _plot_line(metrics['steps'], metrics['rewards'], 'Reward', path=results_dir)

    # Return average reward and Q-value
    return avg_reward


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

    ys = torch.tensor(ys_population, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

    plotly.offline.plot({
        'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
        'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)
