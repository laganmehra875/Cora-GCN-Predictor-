from os import name
from starlette.responses import FileResponse
from fastapi import encoders
from fastapi import FastAPI, HTTPException
import os
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import onnxruntime as ort
from fastapi.staticfiles import StaticFiles


app = FastAPI()


"""
The model can predict 7 possible topics for each paper.
their is a dictionary that checks which labels belongs to which topics.
"""
CORA_CLASSES = {
    0: 'Case_Based',
    1: 'Genetic_Algorithms',
    2: 'Neural_Networks',
    3: 'Probabilistic_Method',
    4: 'Reinforcement_Learning',
    5: 'Rule_Learning',
    6: 'Theory'
}


"""
Figure out where this files lives , so we can find the model file
"""

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'simple_gcn_cora.onnx')
DATA_DIR = os.path.join(BASE_DIR, "data", "Planetoid")
STATIC_DIR=os.path.join(BASE_DIR,"static")


"Load the trained model once when the app starts up"

model_session = ort.InferenceSession(
    MODEL_PATH, providers=['CPUExecutionProvider'])


"""
Request Formats:these describes how the data is sent to the API
"""


class GraphPredictRequest(BaseModel):
    # A list of nodes ,where each node is a list of 1433 numbers(its features)
    node_features: List[List[float]]
    # Optional : Which nodes connect to which shape[2,number of edges]
    edge_indices: Optional[List[List[int]]] = None


class CoraNodeRequest(BaseModel):
    # We will add a node id field to be specific
    node_indices: List[int]


"""
Helper Function 
"""


def softmax(scores: np.ndarray):
    # turns raw model scores (logits) into probabilities that comes between 0 to 1.
    # We will get the max probability
    shifted = scores-scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores/exp_scores.sum(axis=1, keepdims=True)


def run_model(node_features: np.ndarray, edge_index: np.ndarray, node_indices_to_return):
    """
    Run OONX  model only on the graph we are given and return probabilities for the requested nodes
    """
    output = model_session.run(
        ["logits"],
        {
            'node_features': node_features.astype(np.float32),
            'edge_indices': edge_index.astype(np.int64)
        },
    )
    logits = output[0]
    Probabilities = softmax(logits)
    predicted_class = logits.argmax(axis=-1)

    results = []
    for i in node_indices_to_return:
        results.append({
            "node_index": i,
            "predicted_class": int(predicted_class[i]),
            "predicted_label": CORA_CLASSES[int(predicted_class[i])],
            "probabilities": Probabilities[i].tolist(),
            "logits": logits[i].tolist(),
        }
        )

    return {
        'num_nodes': node_features.shape[0],
        'num_edges': edge_index.shape[1],
        'predictions': results
    }

# Routes.


@app.get("/")
def home_page():
    """Our UI will be Shown"""
    index_file=os.path.join(STATIC_DIR,"index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    
    return {"service":"Simple GCN Cora API","status":"Running"}




@app.get("/health")
def health_check():
    return {"Status": "healthy", "providers": model_session.get_providers()}


@app.get("/info")
def model_info():
    """Shows Basic details about the model input, output,classes"""
    return {
        "Model Name": "Simple GCN ",
        "Feature_dimension": 1433,
        "num_classes": 7,
        "class_mapping": CORA_CLASSES,

        "inputs": [
            {"name": inp.name, "shape": inp.shape, "type": inp.type}
            for inp in model_session.get_inputs()
        ],
        "outputs": [

            {"name": out.name, "shape": out.shape, "type": out.type}
            for out in model_session.get_outputs()
        ],


    }


@app.post("/predict")
def predict_custom_graph(request: GraphPredictRequest):
    """
    Classify a graph you provide yourself and optionally specify which nodes to return
    """
    if not request.node_features:
        raise HTTPException(
            status_code=400, detail="node_features cannot be empty")

    for feature_vector in request.node_features:
        if len(feature_vector) != 1433:
            raise HTTPException(
                status_code=400, detail="all feature vectors must have 1433 dimensions")

    node_features = np.array(request.node_features, dtype=np.float32)
    num_nodes = len(request.node_features)
    if request.edge_indices:
        edge_index = np.array(request.edge_indices, dtype=np.int64)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise HTTPException(
                422, "edge_index must be a matrix of shape [2, num_edges]")

    else:
        # No edges given
        node_ids = np.arange(num_nodes, dtype=np.int64)
        edge_index = np.vstack([node_ids, node_ids])

    all_nodes_features = list(range(num_nodes))
    return run_model(node_features, edge_index, all_nodes_features)


@app.post("/predict/cora_node")
def predict_real_cora_node(request: CoraNodeRequest):
    """
    classify real papers from the actual cora dataset.
    """
    try:
        from torch_geometric.datasets import Planetoid
        cora_dataset = Planetoid(root=DATA_DIR, name="cora")[0]
    except ImportError as error:
        raise HTTPException(500, detail=f"Failed to import PyTorch libraries. Please ensure PyTorch is installed correctly for your Python version. Error: {error}")
    except Exception as error:
        raise HTTPException(500, detail=f"Error loading cora: {error}")

    largest_valid_index = cora_dataset.num_nodes-1
    invalid_index = [
        i for i in request.node_indices
        if i < 0 or i > largest_valid_index
    ]

    if invalid_index:
        raise HTTPException(
            400, f"Node_index out of bound (must be  0 to): {largest_valid_index}"
            )

    return run_model(
        cora_dataset.x.numpy(),
        cora_dataset.edge_index.numpy(),
        request.node_indices
    )

if os.path.isdir(STATIC_DIR):
    app.mount("/",StaticFiles(directory=STATIC_DIR,html=True),name="static")

# UI is now ready!
