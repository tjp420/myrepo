"""Planning shim for linting and tests."""


def plan_from_goal(goal: str):
    return []


class PlanningEngine:
    def __init__(self, *args, **kwargs):
        pass

    def plan(self, goal: str):
        return plan_from_goal(goal)
