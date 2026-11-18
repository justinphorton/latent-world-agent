"""Minds that drive a body.

An agent only ever chooses Actions. It never touches the world state; for now it is handed the
body's last Feedback (senses come later). A human player is just another agent whose choices
come from the keyboard.
"""

from __future__ import annotations

import random

from world.body import ACTIONS, Action, Feedback


class Agent:
    """Interface every mind implements: given what the body last felt, choose the next action."""

    def act(self, feedback: Feedback | None) -> Action:
        raise NotImplementedError


class HumanAgent(Agent):
    """Plays whatever the person last pressed. One action per tick; nothing pressed means wait."""

    def __init__(self):
        self.pending: Action | None = None

    def press(self, action: Action) -> None:
        """Queue the action to be taken on the next tick (a newer press replaces an older one)."""
        self.pending = action

    def act(self, feedback: Feedback | None) -> Action:
        action, self.pending = self.pending or Action("wait"), None
        return action


class RandomAgent(Agent):
    """Picks uniformly among all 23 actions: the baseline every curious agent should beat."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, feedback: Feedback | None) -> Action:
        return self.rng.choice(ACTIONS)
