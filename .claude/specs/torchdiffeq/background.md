# Background

## Introduction

This spec outlines the migration from Julia-based `diffeqtorch` to PyTorch's
native `torchdiffeq` library for ODE simulation tasks. Currently, the SIR and
Lotka-Volterra tasks require Julia installation and bridge to PyTorch via
`diffeqtorch`, which creates installation complexity and maintenance burden.
Migrating to `torchdiffeq` eliminates the Julia dependency, simplifies the
installation process, and keeps the entire stack within the Python/PyTorch
ecosystem.

## Requirements

### Requirement 1

**User Story:** As a user of sbibm, I want to use ODE-based tasks (SIR and
Lotka-Volterra) without installing Julia, so that I can install sbibm with a
simple `pip install` command.

#### Acceptance Criteria

1. WHEN a user installs sbibm THEN they SHALL NOT need to install Julia or
   `diffeqtorch` to use SIR and Lotka-Volterra tasks
2. WHEN a user runs SIR or Lotka-Volterra simulations THEN the results SHALL
   match the existing reference posteriors within numerical tolerance
3. WHEN a user imports SIR or Lotka-Volterra tasks THEN they SHALL load
   without Julia-related errors

### Requirement 2

**User Story:** As a sbibm developer, I want the ODE solver implementation to
be maintainable and performant, so that the codebase is easier to maintain
and benchmark results remain reproducible.

#### Acceptance Criteria

1. WHEN ODE integration fails or produces NaN values THEN the system SHALL
   handle the error gracefully and return NaN-filled tensors (matching
   current behavior)
2. WHEN running batch simulations THEN the implementation SHOULD be
   vectorized where possible to improve performance
3. WHEN tests are run THEN all existing SIR and Lotka-Volterra tests SHALL
   pass without modification

### Requirement 3

**User Story:** As a researcher using sbibm, I want existing results and
reference posteriors to remain valid, so that my published benchmarks are
still reproducible.

#### Acceptance Criteria

1. WHEN comparing new simulations to reference data THEN the statistical
   properties SHALL match (same ODEs, same parameters, same random seeds)
2. WHEN loading existing reference posterior samples THEN they SHALL remain
   unchanged (no regeneration required)
3. WHEN running with the same observation seeds THEN the outputs SHALL be
   numerically equivalent within solver tolerance

## Non-Functional Requirements

### Performance
- ODE solving performance should be comparable to or better than
  `diffeqtorch`
- Should support batch processing where feasible
- Memory usage should not significantly increase

### Usability
- No breaking changes to the task API (`get_prior()`, `get_simulator()`,
  etc.)
- Installation becomes simpler (pure Python/PyTorch)
- Error messages should be clear when ODE integration fails

### Maintainability
- Remove Julia dependency from `pyproject.toml`
- Update documentation to reflect pure Python installation
- Use PyTorch-native ODE definitions (no Julia code strings)
- Follow existing code patterns in the sbibm codebase

## Design

## Code Reuse Analysis

**Existing components to leverage:**
- Task base class (`sbibm/tasks/task.py`) - interface remains unchanged
- Simulator wrapper (`sbibm/tasks/simulator.py`) - no changes needed
- Test infrastructure (`tests/tasks/test_task_interface.py`) - will work
  without modification once Julia tasks are no longer excluded
- Prior distributions (Pyro/PyTorch distributions) - reused as-is
- Reference posterior sampling (`_sample_reference_posterior`) - reused as-is

**Pattern to follow:**
Both SIR and Lotka-Volterra tasks follow nearly identical structure:
- Initialize task with ODE parameters (N, days, saveat, etc.)
- Define prior distributions using Pyro
- Lazy-load ODE solver
- Simulator loops over parameter samples, solves ODE for each
- Handle NaN cases when ODE integration fails
- Apply summary statistics (subsampling + observation noise)

## Components and Interfaces

### Component 1: ODE Function Definitions
- **Purpose:** Define the SIR and Lotka-Volterra ODE right-hand side
  functions in PyTorch
- **Interfaces:** Callable with signature `f(t, u, p)` where `t` is time,
  `u` is state, `p` is parameters
- **Dependencies:** PyTorch tensor operations
- **Reuses:** None (new implementations of existing Julia code)

**SIR ODE:**
```
du[0] = -beta * S * I / N
du[1] = beta * S * I / N - gamma * I
du[2] = gamma * I
```

**Lotka-Volterra ODE:**
```
du[0] = alpha * x - beta * x * y
du[1] = -gamma * y + delta * x * y
```

### Component 2: ODE Solver Integration
- **Purpose:** Replace `DiffEq` wrapper with `torchdiffeq.odeint`
- **Interfaces:** Match existing `self.de(u0, tspan, params)` call
  signature, returning `(u, t)`
- **Dependencies:** `torchdiffeq` library
- **Reuses:** Existing lazy_property pattern, existing error handling for
  failed integrations

**Key changes:**
- Remove: `from diffeqtorch import DiffEq`
- Add: `from torchdiffeq import odeint`
- Replace `@lazy_property def de` with direct function calls or thin wrapper
- Convert Julia f-string ODE definitions to Python callables
- Map solver options (saveat, tolerances)

### Component 3: Simulator Updates
- **Purpose:** Update simulator to use new ODE solver
- **Interfaces:** No changes - maintains existing Simulator interface
- **Dependencies:** New ODE functions and torchdiffeq
- **Reuses:** Existing simulator structure, NaN handling, summary
  statistics, binomial/lognormal sampling

**Required modifications:**
- Update ODE solver call syntax
- Handle return value format from `odeint` (may differ from diffeqtorch)
- Verify shape handling remains correct
- Remove Julia-specific garbage collection calls (Lotka-Volterra line 147)

### Component 4: Dependency Management
- **Purpose:** Update package dependencies
- **Interfaces:** pyproject.toml dependencies
- **Dependencies:** torchdiffeq (new), remove diffeqtorch
- **Reuses:** Existing dependency structure

### Error Scenarios

1. **ODE Integration Failure**
   - **Handling:** Return NaN-filled tensor of expected shape (matches
     existing behavior in lines 145-147 of SIR and 141-143 of Lotka-Volterra)
   - **User Impact:** Same as current - failed simulations excluded from
     analysis via NaN filtering

2. **Numerical Instability**
   - **Handling:** Use appropriate `torchdiffeq` solver (e.g., `dopri5`
     with adaptive stepping) and tolerances
   - **User Impact:** Should be transparent if tolerances set correctly

3. **Import Error (torchdiffeq not installed)**
   - **Handling:** Clear error message directing user to install torchdiffeq
   - **User Impact:** Better than current Julia installation complexity

## Testing Strategy

### Unit Testing
- Test ODE functions directly with known parameters to verify derivatives
- Verify output shapes match expected dimensions
- Test NaN handling when ODE solver fails
- Test with the same observation seeds used for reference data

### Integration Testing
- Run existing test suite (`test_task_interface.py`) with Julia exclusions
  removed
- Verify SIR and Lotka-Volterra can be instantiated
- Test full simulation pipeline (prior → simulator → observations)
- Compare numerical outputs to reference data for validation

### End-to-End Testing
- Run small-scale MCMC sampling for one observation to verify reference
  posterior generation would work
- Test that existing reference posterior samples still load correctly
- Verify metrics (C2ST, etc.) can be computed on algorithm outputs
