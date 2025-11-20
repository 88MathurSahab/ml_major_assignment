MLOps Major Assignment: End-to-End Decision Tree Pipeline
This repository hosts a complete, production-ready MLOps pipeline for a face recognition system based on the Scikit-learn Olivetti Faces dataset. The goal of this project is to demonstrate an automated, scalable workflow for deploying a machine learning model.

Core Pipeline Summary
The application is built on three main branches, representing distinct stages of the MLOps lifecycle:

Development and Testing (dev): This branch contains the core Python logic for training and evaluating the Decision Tree Classifier model. It integrates a GitHub Actions Continuous Integration (CI) workflow that automatically triggers model training and accuracy testing upon every code update, serving as the critical quality gate.

Containerization and Deployment (docker_cicd_final): This final branch packages the model and its Flask web service into a Docker image. The resulting container is then deployed using Kubernetes, which is configured to maintain three replicas of the prediction service for high availability and scalability.

Deployment Model: The pipeline ensures seamless deployment, with the trained model serving predictions via a stable NodePort service accessible to external users. The entire setup validates the principles of automated model testing, reproducible packaging, and orchestrated, scalable infrastructure management.

Technical Details
Model: Scikit-learn Decision Tree Classifier.

CI: GitHub Actions (Automated Training and Testing).

Infrastructure: Docker and Kubernetes (Minikube).

Application: Flask web service for handling image preprocessing and real-time inference.