# Test Suite for Multi-Objective Nonlinear Conjugate Gradient Algorithms

## Project Structure

There are 2 standalone Julia Packages in here: `TestProblems` and `NonlinearCG`.
With `Pkg` they can be added/`dev`ed by path.

Both packages make use of `Pluto` notebooks, and that is why they have `Markdown` and `InteractiveUtils` in their dependencies.
Most notebook files end in `*.pluto.jl`.

Manually adding the packages `TestProblems` and `NonlinearCG` is not mandatory for the 
standalone notebook files in the toplevel directory.
We leverage the Pluto package manager and instantiate according to `Project.toml` in every 
standalone notebook.

## Julia Version

I have used version `1.11.1`.
The repository contains a flake that provides a dev-shell with unpatched Julia 1.11.1 for NixOS.
