# UI Targets:
- if solver is fast: re-run after weights/DVH constraints change
- if solver is slow: hit run to push changes from UI to solver

- display DVHs
- display DVH constraints
- drag DVH constraint to adjust
- interaction to drop in new (arbitrary) DVH constraints (e.g., double click)
- interaction to remove existing DVH constraint (e.g., double click)
- DVH plot user is interacting with should be in focus somehow (e.g., larger)
- display prescription & DVH constraint satisfaction status
- display summarized plan statistics
- clear all DVH constraints
- interaction to add/activate all CLINICALLY SPECIFIED (e.g., in the prescription document) DVH constraints, e.g., a button
- interaction add/activate any CLINICALLY SPECIFIED (e.g., in the prescription document) DVH constraint, e.g. from a list
- objective-mediated interaction vs. DVH constraint-mediated interaction: DVHC interaction involves creating/removing/dragging DVH control points. OBJ interaction involves ... sliders for weights? graphs of penalty function with draggable slopes and rx points?


- display dose distribution slices
- warn user and stop plan execution if infeasible status returned

## Version 0.0.3 (alpha)
- V() syntax for DVH constraints

TODO: allow plotting of 1st pass when 2-pass is enabled
TODO: __str__ method for RunRecord (why?)
TODO: RunRecord compatibility for CaseIO.save_solution()
TODO: DictionaryMapping class for structure-wise clustering 
TODO: cluster-and-bound solution heuristics for unconstrained problems

## Version 0.0.2 
- IO module for saving/loading cases from config files (database compatible)
- Objective class: specify primal, dual eval and expression methods for cvxpy
  and POGS backends
  
TODO: document conrad.visualization.plot module
TODO: document conrad.io module
TODO: Update docs for v0.0.2

## Version 0.0.1 

- Domain-specific language for specifying and solving treatment planning cases
- Convex handling of dose-volume constraints
- Connects to cvxpy (solvers: SCS, ECOS, ... ) and POGS solver backends 
- Plotting module for DVH graphs
