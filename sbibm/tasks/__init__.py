from pathlib import Path
from typing import Any, List

from sbibm.tasks.task import Task


def get_task(task_name: str, *args: Any, **kwargs: Any) -> Task:
    """Get task

    Args:
        task_name: Name of task

    Returns:
        Task instance
    """
    if task_name == "hierarchical_two_moons":
        from sbibm.tasks.hierarchical_two_moons.task import (
            HierarchicalTwoMoons,
        )

        return HierarchicalTwoMoons(*args, **kwargs)

    elif task_name == "hierarchical_gaussian_mixture":
        from sbibm.tasks.hierarchical_gaussian_mixture.task import (
            HierarchicalGaussianMixture,
        )

        return HierarchicalGaussianMixture(*args, **kwargs)

    elif task_name == "hierarchical_gaussian_linear":
        from sbibm.tasks.hierarchical_gaussian_linear.task import (
            HierarchicalGaussianLinear,
        )

        return HierarchicalGaussianLinear(*args, **kwargs)

    elif task_name == "hierarchical_slcp":
        from sbibm.tasks.hierarchical_slcp.task import HierarchicalSLCP

        return HierarchicalSLCP(*args, **kwargs)

    elif task_name == "hierarchical_gaussian_linear_uniform":
        from sbibm.tasks.hierarchical_gaussian_linear_uniform.task import (
            HierarchicalGaussianLinearUniform,
        )

        return HierarchicalGaussianLinearUniform(*args, **kwargs)

    elif task_name == "hierarchical_sir":
        from sbibm.tasks.hierarchical_sir.task import HierarchicalSIR

        return HierarchicalSIR(*args, **kwargs)

    else:
        raise NotImplementedError()


def get_task_name_display(task_name: str, *args: Any, **kwargs: Any) -> str:
    return get_task(task_name).name_display


def get_available_tasks() -> List[str]:
    """Get available tasks

    Returns:
        List of tasks
    """
    task_dir = Path(__file__).parent.absolute()
    tasks = [f.name for f in task_dir.glob("*") if f.is_dir() and f.name[0] != "_"]
    return tasks
