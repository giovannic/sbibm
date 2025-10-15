# Implementation Plan

## Task Overview

Migrate ODE-based tasks (SIR and Lotka-Volterra) from Julia's `diffeqtorch`
to PyTorch's native `torchdiffeq`. This involves creating PyTorch ODE
functions, updating the solver integration, modifying simulators, updating
dependencies, and ensuring tests pass. The tasks are ordered to minimize
context switching and enable incremental testing.

## Atomic Task Requirements
**Each task must meet these criteria for optimal agent execution:**
- **File Scope**: Touches 1-3 related files maximum
- **Time Boxing**: Completable in 15-30 minutes
- **Single Purpose**: One testable outcome per task
- **Specific Files**: Must specify exact files to create/modify
- **Agent-Friendly**: Clear input/output with minimal context switching

## Task Format Guidelines
- Use checkbox format: `- [ ] Task number. Task description`
- **Specify files**: Always include exact file paths to create/modify
- **Include implementation details** as bullet points
- Reference requirements using: `_Requirements: X.Y, Z.A_`
- Reference existing code to leverage using: `_Leverage: path/to/file_1,
  path/to/file_2`
- Focus only on coding tasks (no deployment, user testing, etc.)
- **Avoid broad terms**: No "system", "integration", "complete" in task
  titles

## Tasks

- [x] 1. Create SIR ODE function in sbibm/tasks/sir/task.py
  - File: sbibm/tasks/sir/task.py
  - Remove `from diffeqtorch import DiffEq` import
  - Add `from torchdiffeq import odeint` import
  - Create `_sir_ode` function with signature `(t, u)` that returns du/dt
  - Implement: `du[0] = -beta * S * I / N`, `du[1] = beta * S * I / N -
    gamma * I`, `du[2] = gamma * I`
  - Store current parameters (beta, gamma) and N as instance attributes
    accessed in ODE function
  - _Requirements: 1.1, 1.2, 2.1_
  - _Leverage: sbibm/tasks/sir/task.py lines 16-96 (existing structure)_

- [x] 2. Replace SIR ODE solver with torchdiffeq in sbibm/tasks/sir/task.py
  - File: sbibm/tasks/sir/task.py
  - Remove `@lazy_property def de` method (lines 97-111)
  - Update `get_simulator` method to call `odeint` directly with
    `_sir_ode`, `u0`, time points
  - Generate time points from tspan and saveat: `t = torch.linspace(0,
    days, int(days/saveat)+1)`
  - Handle return format: `odeint` returns shape `(time_steps, state_dim)`,
    transpose to match expected shape
  - Wrap odeint call in try/except to catch integration failures and return
    NaN tensors
  - _Requirements: 1.2, 2.1, 2.2_
  - _Leverage: sbibm/tasks/sir/task.py lines 123-183 (simulator structure)_

- [x] 3. Create Lotka-Volterra ODE function in
  sbibm/tasks/lotka_volterra/task.py
  - File: sbibm/tasks/lotka_volterra/task.py
  - Remove `from diffeqtorch import DiffEq` import
  - Add `from torchdiffeq import odeint` import
  - Create `_lotka_volterra_ode` function with signature `(t, u)` returning
    du/dt
  - Implement: `du[0] = alpha * x - beta * x * y`, `du[1] = -gamma * y +
    delta * x * y`
  - Store current parameters (alpha, beta, gamma, delta) as instance
    attributes
  - _Requirements: 1.1, 1.2, 2.1_
  - _Leverage: sbibm/tasks/lotka_volterra/task.py lines 18-90 (existing
    structure)_

- [x] 4. Replace Lotka-Volterra ODE solver with torchdiffeq in
  sbibm/tasks/lotka_volterra/task.py
  - File: sbibm/tasks/lotka_volterra/task.py
  - Remove `@lazy_property def de` method (lines 94-107)
  - Update `get_simulator` method to call `odeint` with
    `_lotka_volterra_ode`, `u0`, time points
  - Generate time points: `t = torch.linspace(0, days, int(days/saveat)+1)`
  - Transpose odeint output to match expected shape
  - Remove Julia garbage collection call (line 147)
  - Wrap odeint in try/except for NaN handling
  - _Requirements: 1.2, 2.1, 2.2_
  - _Leverage: sbibm/tasks/lotka_volterra/task.py lines 119-183 (simulator
    structure)_

- [x] 5. Update dependencies in pyproject.toml
  - File: pyproject.toml
  - Remove `"diffeqtorch"` from dependencies list (line 36)
  - Add `"torchdiffeq"` to dependencies list
  - _Requirements: 1.1_
  - _Leverage: pyproject.toml existing structure_

- [x] 6. Update test exclusions in tests/tasks/test_task_interface.py
  - File: tests/tasks/test_task_interface.py
  - Remove `julia_tasks` variable (line 12)
  - Update all test parametrizations to remove `- julia_tasks` filter
    (lines 16, 24, 33, 43, 55, 66, 77, 89, 103, 128)
  - Tests should now run for SIR and Lotka-Volterra
  - _Requirements: 2.3_
  - _Leverage: tests/tasks/test_task_interface.py existing test structure_

- [ ] 7. Update README.md to remove Julia dependency notes
  - File: README.md
  - Remove or update lines 21-22 that mention Julia installation
    requirement
  - Replace with note that torchdiffeq is included as a dependency
  - _Requirements: 1.1_
  - _Leverage: README.md existing installation section_

- [ ] 8. Update CLAUDE.md to reflect torchdiffeq migration
  - File: CLAUDE.md
  - Update line mentioning "ODE tasks require Julia via diffeqtorch"
  - Replace with "ODE tasks (SIR, Lotka-Volterra) now use torchdiffeq"
  - Update relevant sections about installation and dependencies
  - _Requirements: 1.1_
  - _Leverage: CLAUDE.md existing task documentation_

- [ ] 9. Run tests to validate SIR task
  - Command: `pytest tests/tasks/two_moons/test_task.py -v` (sanity check)
  - Command: `pytest tests/tasks/test_task_interface.py::test_task_can_be_obtained[sir] -v`
  - Command: `pytest tests/tasks/test_task_interface.py::test_simulate_from_thetas[sir] -v`
  - Verify no import errors and basic simulation works
  - _Requirements: 2.2, 2.3_

- [ ] 10. Run tests to validate Lotka-Volterra task
  - Command: `pytest
    tests/tasks/test_task_interface.py::test_task_can_be_obtained[lotka_volterra]
    -v`
  - Command: `pytest
    tests/tasks/test_task_interface.py::test_simulate_from_thetas[lotka_volterra]
    -v`
  - Verify no import errors and basic simulation works
  - _Requirements: 2.2, 2.3_

- [ ] 11. Run full test suite for ODE tasks
  - Command: `pytest tests/tasks/test_task_interface.py -k "sir or
    lotka_volterra" -v`
  - All parametrized tests should pass for both tasks
  - Fix any remaining issues with shapes, NaN handling, or numerical
    tolerances
  - _Requirements: 2.2, 2.3, 3.1_
