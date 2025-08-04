# California House Pricing Prediction
This is the DTClub MLOps Zoomcamp 2025 repository. The goal of this project is to apply everything we have learned to an end-to-end machine learning project. The following are some guidelines for running and evaluate it.
## Introduction: Problem description
Let's say that you have a real estate business located in California, USA. One of the main problems in this field is determining a fair market value for a property based on different factors such as the age of the house, location and proximity to the ocean. 
Furthermore, you would need this information as soon as possible for every new property available on the market. This is because you will have to negotiate in every house buying and selling process. 
It would also be useful to have this information organised in a table, so that you can quickly find the data you need without having to do a lot of calculations based on the property information.
This project solves that business logic problem. Using an ML model and following MLOps practices, it provides real estate decision-makers with the necessary information in a convenient format whenever it is requested.
## How to run the project
For references, the reproducibility of this project was tested on a fresh Github Codespace, but you can use the IDE configuration of your preferences. 
It is also recommended to have Anaconda installed to avoid interpreter and enviroment conflicts at managing dependencies.
### Create a new conda enviroment for isolation porpuose (recommended)
Python version 3.12.11 was used to build this project. If no version is specified, Conda will install the latest available Python. 
```
conda create --name <environment_name> python=<python_version>
```
If you are not using Conda, it is equal recommended to create a virtual env from Python to isolate the dependencies installation and avoid future version conflicts.
### Install all the dependencies from requirements
Once you checked that you are using the envioment created in the previous step, install the requirements for this project
```
pip install -r requirements.txt
```
Once this command finish its execution, you will be able to run the project and all the technologies used in it.
### Inicialize the MLFlow server for experiment tracking & model registry
It is important to keep in mind the structure of the project. You have to inicialice the MLFlow server **inside the orchestration directory** for a correct work (be sure of being on the root dir of the project)
```
cd orchestration/
mlflow server
```
Or the equivalent commands in your OS.
