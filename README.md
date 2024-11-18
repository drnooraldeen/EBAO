
# EBOA (Enhanced Binary Aquila Optimizer)

## Project Description
EBOA is a feature selection and optimization algorithm designed for use in binary and continuous optimization tasks. The algorithm leverages various mutation strategies and optimization techniques, including Lévy flights and binary transfer functions, to select optimal feature subsets in high-dimensional data. This implementation uses the KDDCup99 dataset as an example application to demonstrate its performance in feature selection and classification tasks.

## How to Run the Code

1. **Clone the repository**:
   ```bash
   git clone https://github.com/drnooraldeen/EBOA.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd EBOA
   ```
3. **Install required dependencies**:
   Ensure that you have `numpy`, `pandas`, `scikit-learn`, and other necessary Python libraries installed. You can install these using:
   ```bash
   pip install -r requirements.txt
   ```
   *(Create a `requirements.txt` file if you haven't already, listing the dependencies.)*

4. **Run the main file**:
   Execute the main file to run the EBOA algorithm:
   ```bash
   python EBOA.py
   ```
   
## Dependencies and Requirements
The project relies on the following libraries:
- Python 3.x
- NumPy
- Pandas
- scikit-learn

To install all dependencies, you can use:
```bash
pip install numpy pandas scikit-learn
```

## Example Usage and Expected Outputs
- The EBOA algorithm is applied to the KDDCup99 dataset for feature selection and classification.
- After running the code, you can expect output showing the best solution found by the optimizer and its corresponding fitness score, including accuracy and F1-score metrics.

Example output:
```
Best solution found: [array([...]), fitness_value]
```

## Project Structure
```
EBOA/
├── EBOA.py               # Main implementation of the EBOA algorithm
├── KDDCup99.csv          # Example dataset (ensure the data file is in the project directory)
└── README.md             # Project documentation
```
