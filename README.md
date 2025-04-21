# Graph Analysis and Visualization

![Python CI](https://github.com/Bernaljp/graph-analysis/workflows/Python%20CI/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)

 This project analyzes weighted directed graphs, computes their equivalence classes, simulates dynamics using ODEs, and visualizes results interactively with PCA/UMAP embeddings.

 ## Features
 - Generate unique graph structures for a given number of nodes.
 - Compute graph invariants and canonical forms.
 - Simulate dynamics using PyTorch and torchode.
 - Visualize graphs and simulations with Plotly and ipywidgets.

 ## Installation
 1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/graph-analysis.git
    cd graph-analysis
    ```
 2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
 3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
 4. Run the example notebook or script in a Jupyter environment.

 ## Usage
 ```python
 from graph_analysis import analyze_all_graphs, visualize_graphs
 graphs, feats, pca2, umap2 = analyze_all_graphs(n=4, num_inits=20)
 fw, ui, run_btn, rand_btn, init_boxes = visualize_graphs(graphs, pca2)
```

## Dependencies
See requirements.txt for a full list.

## License
MIT License. See LICENSE for details.

## Running Tests
To run the unit tests locally:
```bash
python -m unittest discover tests
```