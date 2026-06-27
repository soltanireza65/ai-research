# Reinforcement Learning

> **Status:** Stub chapter — Part II placeholder. Complete Part I before expanding this chapter.

## Overview

Reinforcement learning (RL) trains **agents** that take **actions** in an **environment** to maximize cumulative **reward**. Unlike supervised learning (fixed input–output pairs), RL learns from consequences: a chess move is judged by whether it leads to victory many steps later, not by an immediate label.

This chapter will connect probability (policies as distributions over actions), optimization (policy gradient methods), and PyTorch (differentiable simulators and training loops). You will move from multi-armed bandits — the simplest RL problem — toward policy gradients and an introduction to value functions.

## Planned Topics

- Markov decision processes (states, actions, rewards, transitions)
- Exploration vs exploitation in bandit problems
- Policy gradients: REINFORCE and baseline intuition
- Value functions and Q-learning sketch
- When RL beats supervised learning (and when it does not)
- Tiny PyTorch lab: train an agent on CartPole or a grid world

## Prerequisites

Complete Modules 01–03 at minimum. Probability and gradients are essential; transformer material (Module 04) is helpful but not required for introductory RL.

## Further Reading (when ready)

- [Sutton & Barto — Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) (free online)
- [Spinning Up in Deep RL — OpenAI](https://spinningup.openai.com/)

See [Learning Path](../00-intro/03-learning-path.md) for curriculum context.
