# CORA GCN Predictor

A sleek, interactive Graph Neural Network (GNN) inference API and Web Explorer built with **PyTorch Geometric**, **ONNX Runtime**, and **FastAPI**.

This project provides an elegant frontend interface for exploring the famous **Cora Citation Network** using a Graph Convolutional Network (GCN) model exported to ONNX format for blazing-fast inference.

## ✨ Features
- **Real-Time Graph Inference**: Run GNN predictions on single or multiple nodes simultaneously using ONNX Runtime.
- **Premium UI/UX**: Beautiful glassmorphic design with a dynamic animated neural network background.
- **Custom Graph Predictor**: Input your own custom node features and edges via JSON payloads to run inferences on completely custom graph topologies.
- **Latency Benchmarking**: Includes an automated benchmarking tool to measure end-to-end API response times across the full Cora graph.
- **Prediction History Sidebar**: Keep track of all your past predictions via a persistent sliding sidebar.

---

## 🏗️ Project Architecture & Work Structure

The application is structured into a fast Python backend for model inference and a vanilla web frontend for the interactive user experience.

### Architecture Diagram

```mermaid
graph TD
    subgraph Frontend [Web Interface - static/]
        UI(index.html / CSS / JS)
        UI --> |Batch Inference Request| API_Node[/predict/cora_node]
        UI --> |Custom Graph JSON| API_Custom[/predict]
        UI --> |Model Metadata| API_Info[/info]
    end

    subgraph Backend [FastAPI Server - main.py]
        API_Node --> |Node IDs| ONNX_Runner
        API_Custom --> |Features & Edges| ONNX_Runner
        API_Info --> |Extract Inputs/Outputs| ONNX_Runner
    end

    subgraph Core [ONNX Inference Engine]
        ONNX_Runner(ONNX Runtime Session)
        Model[(simple_gcn_cora.onnx)]
        Dataset[(Planetoid Cora Dataset)]
        
        ONNX_Runner <--> Model
        API_Node --> Dataset
        Dataset --> |Graph Data| ONNX_Runner
    end
    
    ONNX_Runner --> |Logits & Class Probs| UI
```

### Directory Tree
```text
GNN project/
├── README.md                                  # Project documentation
├── requirements.txt                           # Python dependencies
├── Cora_citation_network_classification.ipynb # Original PyTorch model training notebook
├── simple_gcn_cora.onnx                       # Exported ONNX model weights
├── main.py                                    # FastAPI Backend Server
├── data/                                      # Auto-downloaded Cora dataset files
└── static/                                    # Frontend Web Application
    ├── index.html                             # Main dashboard markup
    ├── styles.css                             # Glassmorphic UI styles
    └── script.js                              # App logic, charts, and interactions
```

---

## 🚀 Getting Started

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```
*(Dependencies include FastAPI, Uvicorn, ONNXRuntime, PyTorch Geometric, and Numpy).*

### 2. Run the Server
Launch the FastAPI development server:
```bash
uvicorn main:app --reload
```

### 3. Open the Dashboard
Navigate your web browser to:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🧠 Technologies Used
- **Backend:** FastAPI, Python 3, Uvicorn
- **AI/ML:** PyTorch Geometric (PyG), ONNX Runtime
- **Frontend:** Vanilla HTML5, CSS3, JavaScript (ES6)
- **Styling/Icons:** FontAwesome, Google Fonts (Inter & Outfit)
