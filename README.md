# Breast Cancer Classification Using Supervised Machine Learning

## Overview
This project aims to classify breast tumors as benign or malignant using supervised machine learning techniques. Leveraging the Breast Cancer Wisconsin (Diagnostic) dataset, we explored various algorithms to determine the best-performing model for accurate and efficient predictions.

---

## Objectives
- **Primary Goal:** Classify tumors as benign or malignant.
- **Significance:** Early and accurate diagnosis can save lives.
- **Dataset:** Breast Cancer Wisconsin Dataset (30 features, 569 samples).

---

## Workflow
1. **Data Exploration:** Examined dataset characteristics and relationships between features.
2. **Data Preprocessing:** Addressed outliers, scaled features, and balanced data using SMOTE.
3. **Modeling:** Implemented Logistic Regression, Decision Tree, Random Forest, and Support Vector Machines (SVM).
4. **Hyperparameter Tuning:** Used GridSearchCV for optimal parameter selection.
5. **Evaluation:** Assessed models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
6. **Results & Insights:** Identified the best-performing model and discussed future improvements.

---

## Key Deliverables
- **Jupyter Notebook:** [View the complete analysis and code](breast_cancer_classification_project.ipynb).
- **Slides Presentation:** [Download the slide deck](Breast_Cancer_Classification_Project_Slides_MehrdadNaderi_2024.pptx).
- **Video Presentation:** [Watch on Vimeo](https://vimeo.com/1037067839?share=copy).

---

## Results
### Performance Comparison:
| Model                 | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression   | 76.0%   | 76.0%     | 76.0%  | 76.0%    | 0.8432  |
| Decision Tree         | 78.0%   | 79.0%     | 79.0%  | 79.0%    | 0.8432  |
| Random Forest         | 81.0%   | 81.5%     | 82.0%  | 81.8%    | 0.8534  |
| Support Vector Machine| 82.0%   | 82.5%     | 83.0%  | 82.8%    | 0.8552  |

---

## Instructions to Reproduce
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/MehrdadNaderi/Breast-Cancer-Classification.git
   cd Breast-Cancer-Classification
2. **Environment Setup:**
   - Install required libraries:
     ```bash
     pip install -r requirements.txt
     ```
   - Recommended Environment: [Google Colab](https://colab.research.google.com/) for ease of use and scalability.
3. **Run the Jupyter Notebook:**
   - Open `breast_cancer_classification_project.ipynb` in your preferred Jupyter environment.
   - Follow the steps in the notebook to reproduce the analysis and results.

---

## Tools and Libraries
- **Programming Language:** Python 3.10
- **Libraries:**
  - `Scikit-learn`: Model building, hyperparameter tuning, and evaluation.
  - `Matplotlib` & `Seaborn`: Data visualization.
  - `NumPy` & `Pandas`: Data manipulation and numerical computations.
  - `SMOTE`: Data balancing to address class imbalance.
- **Environment:** Google Colab for execution and GitHub for collaboration.

---

## Challenges and Future Work
- **Challenges:**
  - Limited dataset size may affect model generalizability.
  - Balancing model complexity and interpretability for real-world applications.
- **Future Directions:**
  - Explore advanced models like Gradient Boosting or Neural Networks.
  - Expand the dataset for improved generalizability.
  - Deploy models in clinical settings for real-world validation.
## References
1. **Dataset**  
   - Breast Cancer Wisconsin (Diagnostic) Dataset:  
     [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

2. **Libraries**  
   - [Scikit-learn](https://scikit-learn.org/stable/): Machine learning model building and evaluation.  
   - [Matplotlib](https://matplotlib.org/stable/): Data visualization.  
   - [Seaborn](https://seaborn.pydata.org/): Advanced visualization and plots.  
   - [NumPy](https://numpy.org/): Numerical computations.  
   - [Pandas](https://pandas.pydata.org/): Data manipulation and analysis.  

3. **Tools**  
   - [Google Colab](https://colab.research.google.com/): Cloud-based environment for reproducible research.  
   - [GitHub](https://github.com/): Repository hosting and collaboration.

4. **Acknowledgments**  
   - UCI Machine Learning Repository for providing the dataset.  
   - Open-source contributors to Python libraries for maintaining critical tools.

---

## Contact
For questions, suggestions, or collaboration opportunities, please feel free to reach out:

**Name:** Mehrdad Naderi  
**Email:** [mail@mehrdadnaderi.com](mail@mehrdadnaderi.com)  
**GitHub Profile:** [MehrdadNaderiCom](https://github.com/MehrdadNaderiCom/)  
**LinkedIn Profile:** [Mehrdad Naderi](https://www.linkedin.com/in/mehrdad-naderi/)
