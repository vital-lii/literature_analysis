# Literature Analysis and Knowledge Graph Construction System

## Project Overview

This project is a specialized tool for analyzing literature related to Autophagy and Regulatory T cells (Treg). It constructs knowledge graphs using text mining and machine learning technologies. The system features a modular design that integrates entity recognition, relationship extraction, vector representation, and knowledge graph visualization. It also includes a relationship classification function based on the TinyBERT model to predict relationship types between entities.

## System Architecture

The system consists of the following core modules:

1. **Data Processing and Analysis**: Loads and processes literature data, performs basic statistical analysis
2. **Entity Recognition**: Identifies biomedical entities in text using dictionary and rule-based methods
3. **Vector Representation**: Generates semantic vectors for entities using pre-trained language models
4. **Relationship Extraction and Classification**: Extracts entity relationships from text and predicts relationship types using trained models
5. **Knowledge Graph Construction and Visualization**: Builds biomedical knowledge graphs with multiple visualization options

## File Structure and Functionality

### Core Execution Files

- `simple_analysis.py`: Main entry point, executes the complete literature analysis and knowledge graph construction workflow
- `train_main.py`: Training script for relationship classification models (input files require columns: source_entity, target_entity, relation_type)
- `test_relation_trainer.py`: Helper script for testing TinyBERT model initialization and configuration

### Core Functional Modules

- `knowledge_graph.py`: Knowledge graph builder responsible for entity recognition, relationship extraction, and graph generation
- `embedding_manager.py`: Vector encoding manager that generates vector representations for entities and relationships
- `relation_trainer.py`: Relationship classifier for training and predicting relationship types between entities

### Data and Configuration

- `config/`: Contains system configuration files (entity dictionaries, relationship types, visualization settings)
- `data/`: Stores input data (custom training literature CSV files/search results) and intermediate processing results
- `models/`: Contains pre-trained models and trained relationship classification models
- `output/`: Stores knowledge graphs and visualization results

### Auxiliary Components

- `utils/`: Utility functions and helper classes (e.g., path manager)
- `nltk/`: Natural language processing resources and data
- `logs/`: Runtime logs

## Workflow

### Basic Analysis Workflow

1. **Initialize Configuration**: Load configuration files and set up runtime environment
2. **Data Loading**: Read literature CSV files
3. **Basic Analysis**: Calculate statistical information (year distribution, journal distribution)
4. **Entity and Relationship Extraction**: Identify autophagy and Treg-related entities and extract relationships
5. **Build Knowledge Graph**: Create network graph structure with nodes and edges
6. **Visualization Output**: Generate interactive HTML visualizations

### Relationship Classification Model Training Workflow

1. **Prepare Training Data**: Load from CSV or use predefined relationship examples
2. **Initialize Model**: Create sequence classification model based on TinyBERT
3. **Model Training**: Optimize model parameters through gradient descent
4. **Model Evaluation**: Assess performance using validation sets
5. **Model Saving**: Save trained models to specified directory

## Core Functionality Details

### 1. Entity Recognition

The system identifies the following types of biomedical entities using dictionary and rule-based methods:

- Genes/Proteins (e.g., FOXP3, LC3, mTOR)
- Chemical Substances/Cytokines (e.g., IL-10, TGF-β)
- Diseases/Symptoms (e.g., Inflammation, Autoimmune disorders)
- Cell Types (e.g., Treg, T cells)

### 2. Vector Representation

The system uses the TinyBERT model to generate semantic vectors for identified entities, enabling:

- Calculation of semantic similarity between entities
- Inference of potential relationships based on similarity
- Visualization of entity semantic spaces through dimensionality reduction

### 3. Relationship Classification

The system trains a relationship classification model that predicts six relationship types:

- `positive_regulation`: Positive regulation (e.g., activation, promotion)
- `negative_regulation`: Negative regulation (e.g., inhibition, reduction)
- `regulation`: General regulation (no specific direction)
- `association`: Correlation (no causal relationship)
- `conversion`: Conversion/transformation relationship
- `DEFAULT`: Default relationship type

### 4. Knowledge Graph Construction and Visualization

The system provides multiple knowledge graph formats and visualization options:

- Interactive HTML graphs (using PyVis)
- ECharts interactive visualizations
- GEXF format export (Gephi compatible)

## Usage Examples

### Basic Usage

(Requires pre-trained models)

```bash
# Run basic analysis and knowledge graph construction
python simple_analysis.py -i data/Autophagy_Treg_literature.csv
```

### Relationship Model Training

```bash
# Train model with custom relationship data
python train_main.py -d data/relations_data.csv -e 10 -t
```

## System Features and Advantages
- Modular Design: Independent functional modules for easy extension and maintenance
- Multiple Visualizations: Supports various graph visualization formats
- Vector Semantics: Utilizes pre-trained language models to capture semantic relationships
- Relationship Prediction: Accurate relationship classification through fine-tuned TinyBERT
- Configurability: Most functions adjustable through configuration files

## Technology Stack
- Python: Core programming language
- NetworkX: Network graph data structures
- PyVis/ECharts: Interactive visualization
- NLTK: Natural language processing
- Transformers/PyTorch: Deep learning and vector representation
- Pandas: Data processing

## Acknowledgments
Special thanks to the Huawei team for distilling the BERT model.
Original model available at HuggingFace: huawei-noah/TinyBERT_General_4L_312D
Reference paper: https://doi.org/10.48550/arXiv.1909.10351
