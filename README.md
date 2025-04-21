# NIFTI Traffic Simulation

Welcome to the NIFTI Traffic Simulation project! This repository contains the code and resources for simulating the NIFTI (National Individual Floating Transport Infrastructure) system, a proposed MagLev-based autonomous transport system designed to address congestion and inefficiencies in current road networks.

## Project Overview

The NIFTI system aims to provide an alternative to traditional electric cars by utilizing a centrally controlled, autonomous vehicle network that operates on magnetic levitation technology. This project explores the potential of NIFTI to reduce congestion through traffic simulations and routing algorithms.

## Features

- **MagLev Technology**: Simulates the use of magnetic levitation for efficient, zero-emissions travel.
- **Autonomous Pods**: Models autonomous pods that follow predetermined routes to minimize congestion.
- **Traffic Simulations**: Conducts simulations to assess the impact of NIFTI on congestion in urban networks.
- **Routing Algorithms**: Implements various routing algorithms, including Dijkstra's and Yen's algorithms, to optimize pod routes.
- **Graph-Based Network Modeling**: Utilizes graph theory to model road networks and simulate traffic flow.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries: `networkx`, `osmnx`, `numpy`, `matplotlib`

### Project Structure
- **master_file.py**: Main script to run traffic simulations.
- **custom_classes.py**: Defines the pod and fleet class.
- **city_importer.py**: Loads, cleans, and saves a road network from a place name
- **data_analyser.py**: Extracts data insights from fleet objects
- **grapher.py**: Tools for visualizing simulation results.
- **data/**: Directory for storing road networks
- **fleets/**: Directory for outputting python objects
- **graph_data/**: Directory for storing processed data

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Special thanks to Professor Nigel Hussey, the inventor of the nifti technology.
Data and road network information provided by OpenStreetMap and its contributors.