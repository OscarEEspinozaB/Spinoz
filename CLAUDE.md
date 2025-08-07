# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Spinoz** project - a mathematical method that demonstrates how all division can be represented as a numerical progression using the formula: `Cociente = 1 / (1 - S)` where `S = 1 - Divisor / Dividendo`. The project applies the Spinoz method to various domains including turbulence analysis, F1 aerodynamics, and mathematical demonstrations.

## Code Architecture

### Core Components

1. **Python Analysis Scripts**
   - `spinoz-turbulence-analysis.py` - Main computational analysis applying Spinoz to turbulence with F1 aerodynamics applications
   - `spinoz-turbulence-validation.py` - Experimental validation against real turbulence data with Monte Carlo simulations and ML predictors

2. **HTML Demonstrations**
   - `spinoz-turbulence-calc.html` - Interactive demonstration with real experimental data
   - `spinoz-web-demo.html` - Interactive web calculator for the Spinoz method

3. **Key Classes and Structure**
   - `SpinozTurbulenceAnalyzer` - Main analysis engine for turbulence applications
   - `F1AerodynamicsSpinoz` - Specialized class for Formula 1 aerodynamic analysis
   - `SpinozValidator` - Validation against experimental data
   - `ExperimentalDatabase` - Contains real experimental data from literature
   - `ParallelSpinozProcessor` - High-performance parallel processing for large datasets

## Common Development Commands

### Python Scripts
```bash
# Run the main turbulence analysis (generates visualizations and reports)
python spinoz-turbulence-analysis.py

# Run experimental validation
python spinoz-turbulence-validation.py

# View HTML demonstrations
# Open in browser: spinoz-turbulence-calc.html or spinoz-web-demo.html
```

### Dependencies
The Python scripts use:
- numpy, pandas, matplotlib - Core scientific computing
- scipy - Advanced mathematical functions and optimization
- dataclasses, typing - Modern Python features
- concurrent.futures - Parallel processing
- sklearn (in validation script) - Machine learning capabilities

## Project Context

### Mathematical Foundation
The Spinoz method transforms division into series expansion:
- **Taylor Series** (|s| < 1): `1/(1-s) = 1 + s + s² + s³ + ...`
- **Laurent Series** (|s| > 1): For values outside convergence radius
- **Reversibility Theorem**: `1/c = 1-s` for backscatter modeling

### Applications
1. **Turbulence Analysis**: Relates Kolmogorov scales to cascade energy
2. **F1 Aerodynamics**: Models downforce, ground effect, wake turbulence
3. **Mathematical Validation**: Compares with experimental data from literature

### Key Insights
- Parameter `s` approaches 1 for developed turbulence (intense energy cascade)
- Series convergence relates to physical cascade efficiency
- Method unifies various turbulent flow descriptions under single parameter

## File Organization

- Research/documentation files: `*.md` (various analysis documents)
- Python implementations: `*.py` (analysis and validation)
- Interactive demos: `*.html` (web-based calculators)
- Generated outputs: PNG visualizations and CSV data files

This is primarily a research project demonstrating mathematical concepts through computational analysis and interactive visualizations.