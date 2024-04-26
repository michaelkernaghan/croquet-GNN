# Croquet Shot Prediction using Graph Neural Networks

This repository contains a Graph Neural Network (GNN) model designed to predict the success of shots in the game of croquet. The model utilizes PyTorch and PyTorch Geometric to process graph-based data representing the positions of balls and hoops, and the relationships between them.

## Project Overview

The purpose of this project is to apply Graph Neural Networks to predict whether a specific shot in croquet (starting from a baulk) will succeed. The model takes into account the positions of all balls and hoops at the start of the shot, represented as nodes in a graph, with potential shots represented as edges.

## Features

- **Graph Representation**: Nodes represent croquet balls and hoops with features including positions and types.
- **Edge Definition**: Edges define potential interactions such as shot attempts from a ball to a hoop.
- **GNN Model**: The model predicts the probability of shot success using node features and the graph structure.

## Installation

To set up the project environment to run the code, follow these steps:

### Prerequisites

- Python 3.8 or higher
- pip

### Libraries

Install the required Python libraries using pip:

```bash
pip install torch torch-geometric
```

## Usage

To run the prediction model, execute the Python script `predict.py` which includes the model definition, data setup, and prediction logic:

```bash
python predict.py
```

The output will include the predicted probabilities of success for each potential shot from a ball to a hoop.

## Contributing

Contributions to this project are welcome! Here are some ways you can contribute:

- **Improvements**: Suggest changes to the model or data representation.
- **Features**: Add new features, such as the ability to simulate entire games.
- **Bug Fixes**: Report and fix any bugs found in the existing code.

Please open an issue to discuss your ideas or submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or want to reach out to the project maintainers, please open an issue in this repository.

Thank you for checking out our croquet shot prediction model!
```

