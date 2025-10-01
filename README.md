# Integrated Decision Support System for Diabetes Prediction

This project is a web-based application that leverages a high-accuracy Bidirectional Long Short-Term Memory (BiLSTM) model to predict the likelihood of diabetes. Following (BiLSTM) model to predict the likelihood of diabetes. Following the prediction, it provides personalized, actionable lifestyle recommendations to the user, bridging the gap between diagnosis and self-management.

This application is the practical implementation of the research paper: **"An Integrated BiLSTM-Based Decision Support System for Diabetes Prediction and Personalized Lifestyle Recommendations."**

## Key Features

-   **High-Accuracy Prediction:** Utilizes a BiLSTM deep learning model with a **97.54% cross-validated accuracy**.
-   **Personalized Recommendations:** A rule-based engine generates tailored advice for diet, exercise, the prediction, it provides personalized, actionable lifestyle recommendations to the user, bridging the gap between diagnosis and self-management.

This application is the practical implementation of the research paper: **"An Integrated BiLSTM-Based Decision Support System for Diabetes Prediction and Personalized Lifestyle Recommendations."**

## Key Features

-   **High-Accuracy Prediction:** Utilizes a BiLSTM deep learning model with a **97.54% cross-validated accuracy**.
-   **Personalized Recommendations:** A rule-based engine generates tailored advice for diet, exercise, and wellness based on the user's specific health data.
-   **User-Friendly Interface:** A clean and intuitive web interface built with Flask for easy data entry and clear presentation of results and wellness based on the user's specific health data.
-   **User-Friendly Interface:** A clean and intuitive web interface built with Flask for easy data entry and clear presentation of results.
-   **Robust Data Handling:** The model is trained on a fused dataset, balanced using the SMOTEENN technique to handle class imbalance effectively.

## Technology Stack

-   **Backend:** Python, Flask
-   **Machine Learning:** TensorFlow, Keras, Scikit-learn
-   **Data Processing:** Pandas, NumPy, Imbalanced-learn
-   **Frontend:** HTML, CSS

## Model Performance

The BiLSTM model was rigorously evaluated using 10-fold cross-validation, achieving state-of-the-art performance across all key metrics.

| Metric              | Average Score (%) |
| ------------------- | .
-   **Robust Data Handling:** The model is trained on a fused dataset, balanced using the SMOTEENN technique to handle class imbalance effectively.

## Technology Stack

-   **Backend:** Python, Flask
-   **Machine Learning:** TensorFlow, Keras, Scikit-learn
-   **Data Processing:** Pandas, NumPy, Imbalanced-learn
-   **Frontend:** HTML, CSS

## Model Performance

The BiLSTM model was rigorously evaluated using 10-fold cross-validation, achieving state-of-the-art performance across all key metrics.

| Metric              | Average Score (%) |
| ------------------- | ----------------- |
| **Accuracy**        | 97.54%            |
| **Precision**       | 98.09%            |
| **Recall**          | 97.56%            |
| **F1-Score**        | 98.7----------------- |
| **Accuracy**        | 97.54%            |
| **Precision**       | 98.09%            |
| **Recall**          | 97.56%            |
| **F1-Score**        | 98.77%            |

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

-   Python 3.8+
-   pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/darainf468-hash/diabetes-prediction-system.git
    ```

2.  **Navigate to the project directory:**
    ```sh
    cd diabetes-prediction-system
    ```

3.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt7%            |

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

-   Python 3.8+
-   pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/darainf468-hash/diabetes-prediction-system.git
    ```

2.  **Navigate to the project directory:**
    ```sh
    cd diabetes-prediction-system
    ```

3.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1.  **Run the Flask application:**
    ```sh
    python app.py
    ```

2.  **Open your web browser** and navigate to the following address:
    ```
    http://127.0.0.1:5000
    ```

### Usage

1.  **Run the Flask application:**
    ```sh
    python app.py
    ```

2.  **Open your web browser** and navigate to the following address:
    ```
    http://127.0.0.1:5000
    ```

3.  Enter the patient's data into the form and click "Predict" to see the results.

## License

This project is distributed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgements

This project is based on the
    ```

3.  Enter the patient's data into the form and click "Predict" to see the results.

## License

This project is distributed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgements

This project is based on the findings and methodology presented in the accompanying research paper. The work aims to provide a practical tool that can aid in clinical decision-making and improve patient outcomes in diabetes care.
 findings and methodology presented in the accompanying research paper. The work aims to provide a practical tool that can aid in clinical decision-making and improve patient outcomes in diabetes care.
