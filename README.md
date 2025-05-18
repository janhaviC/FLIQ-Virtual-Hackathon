# Quantum Coalition Future Leaders in Quantum (QC-FLIQ) Virtual Hackathon

https://www.quantumcoalition.io/

For any questions, please reach out to Anusha Dandapani (dandapani@unicc.org), Gillian Makamara (gillian.makamara@itu.int), Devyani Rastogi (rastogi@unicc.org), Luke Sebold (lts45@case.edu)

NOTE: The submission for this challenge is under Submissions folder. Details of the submission are in [README.md](submissions/README.md).

---

## Table of Contents
- [Challenge Statement](#challenge-statement)
- [GitHub Repository](#github-repository)
- [Provided Datasets](#provided-datasets)
- [Submission](#submission)
- [Sample Code](#sample-code)
- [Suggested Reading](#suggested-reading)
- [Grading Criteria](#grading-criteria)

---

## Challenge Statement

In this challenge, participants will **enhance an AI/ML classifier or clusterer using Variational Quantum Algorithms (VQAs)**.  
You will be provided with:
- A set of public datasets.
- Starter code in this repository.

Your mission: **Finish and innovate on the starter code** (or create your own approach) to develop quantum-enhanced machine learning models that meet the evaluation standards.

> **Focus Areas:** Performance, explainability, fairness, security, robustness, and innovation.

---

## Submission

Your submission must contain the code you used to train your model, model weights and a demo script if applicable (.pt files or similar is great), a write up describing what you did and how you did it, and optionally a short video demo and presentation.

You are not required to use the sample code provided but **please make a public fork of the repository for your submission.**

Feel free to try any combination of classification, clustering, supervised/unsupervised learning, unique backpropagation techniques, quantum circuit designs, etc.

---

## Sample Code

Our sample code for a VQA classifier will be available in the [GitHub Repository](https://github.com/UN-ICC/FLIQ-Virtual-Hackathon).  

Use them as a foundation or as inspiration but don't be afraid to innovate or try your own unique solution.

Keep in mind you will need to be using an environment capable of running torch or a similar ML library and qiskit. Additionally, as of 2025, qiskit has just moved to 2.x.x, meaning deprecated code is common and documentation/functionality can differ between versions.

---

## GitHub Repository

**Challenge Repo**: [https://github.com/UN-ICC/FLIQ-Virtual-Hackathon](https://github.com/UN-ICC/FLIQ-Virtual-Hackathon)

---

## Provided Datasets

You can use any of the following datasets:
- [Breast Cancer Wisconsin Diagnostic Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- [Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
- [Adult Income Dataset](https://archive.ics.uci.edu/dataset/2/adult)
- [Drug Induced Autoimmunity Prediction Dataset](https://archive.ics.uci.edu/dataset/1104/drug_induced_autoimmunity_prediction)

Feel free to propose additional datasets and justification if so.

---

## Grading Criteria

| Category | Description |
|:---------|:------------|
| **Performance Evaluation** | Measure Precision, Recall, F1-score, Accuracy; balance false positives and negatives. |
| **Fairness & Bias Detection** | Evaluate demographic fairness using metrics like Disparate Impact Ratio. |
| **Adversarial Robustness Testing** | Test resilience against manipulated/adversarial inputs. |
| **Explainability & Interpretability Testing** | Provide understandable model decisions (e.g., SHAP, LIME, Counterfactuals). |
| **Privacy & Security Compliance** | Ensure data encryption, anonymization, GDPR/ISO compliance. |
| **Operational Reliability & Governance** | Maintain model stability, version control, governance for compliance. |
| **Human-in-the-Loop Validation** | Allow human oversight: reviewing, overriding, validating AI outputs. |
| **Innovation & Uniqueness** | Creativity and originality of your quantum-classical hybrid model. |
| **Implementation Quality** | Code quality, structure, and efficiency. |
| **Presentation & Understanding** | Clear explanation of approach, trade-offs, and insights. |

---

## Suggested Resources

- [Introduction to Variational Algorithms - IBM Quantum](https://learning.quantum.ibm.com/course/variational-algorithm-design/variational-algorithms)
- [Quantum Machine Learning Algorithms - arXiv:2012.09265](https://arxiv.org/abs/2012.09265)
- [A Variational Algorithm for Quantum Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-50433-5_45)

---

## Final Notes

- Try to push yourself, demonstrate your understanding, and write clean efficient code. 
- The UN-ICC and Quantum Coalition value safe, responsible, and humanitarian use of AI/ML, and heavily suggest you keep these values in mind as you develop and test your submission. 
- This challenge and all related submissions are intended for the benefit of **all** and must be submitted under the CC0 License or CC BY License. 

---
