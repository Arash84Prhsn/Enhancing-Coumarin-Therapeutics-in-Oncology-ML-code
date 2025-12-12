from graphviz import Digraph

# Simplified, high-level multi-model workflow

dot = Digraph(
    format="png",
    graph_attr={
        "rankdir": "LR",
        "splines": "ortho",
        "dpi": "300",
        "size": "12,4",
        "ranksep": "0.6",
        "nodesep": "0.45"
    },
    node_attr={
        "style": "solid",
        "color": "black",
        "fontsize": "13",
        "fontname": "Arial",
        "width": "1.8",
        "height": "0.9",
        "shape": "box"
    },
    edge_attr={"fontsize": "10", "fontstyle": "italic"}
)

# === Nodes (high-level only) ===
dot.node("Load", "Load Dataset")
dot.node(
    "Pre-processing",
    "Pre-processing"
)
dot.node("GridSearching", "HyperParameter fine tuning")
dot.node("Evaluation", "5-Fold cross-evaluation")
dot.node("Predict", "Predict viability")

# === Edges ===
dot.edge("Load", "Pre-processing")
dot.edge("Pre-processing", "GridSearching")
dot.edge("GridSearching", "Evaluation")
dot.edge("Evaluation", "Predict")

# === Render ===
output_path = "/home/arashp/Programming_Files/ML_Paper/Plots/workFlow"
dot.render(output_path, cleanup=True)
print(f"Flowchart saved as {output_path}.png")
