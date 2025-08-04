# California House Pricing Prediction
This is the DTClub MLOps Zoomcamp 2025 repository. The goal of this project is to apply everything we have learned to an end-to-end machine learning project. The following are some guidelines for running and evaluate it.
<img width="450" height="450" alt="imagen" src="https://github.com/user-attachments/assets/4f2de027-ac88-4830-9df3-b02c6a0685c3"/>
<img width="500" height="450" alt="imagen" src="https://github.com/user-attachments/assets/0fb85c8c-c477-4e63-a5f0-4c4c352f0360"/>
## Introduction: Problem description
Let's say that you have a real estate business located in California, USA. One of the main problems in this field is determining a fair market value for a property based on different factors such as the age of the house, location and proximity to the ocean. 
Furthermore, you would need this information as soon as possible for every new property available on the market. This is because you will have to negotiate in every house buying and selling process. 
It would also be useful to have this information organised in a table, so that you can quickly find the data you need without having to do a lot of calculations based on the property information.
This project solves that business logic problem. Using an ML model and following MLOps practices, it provides real estate decision-makers with the necessary information in a convenient format whenever it is requested.
## How to run the project
For reference, this project was tested for reproducibility on a fresh GitHub Codespace, but you can use your preferred IDE configuration.
It is also recommended that you install Anaconda to avoid conflicts with the interpreter and environment when managing dependencies.
### Create a new Conda environment for isolation (recommended).
Python version 3.12.11 was used to build this project. If no version is specified, Conda will install the latest available version of Python. 
```
conda create --name <environment_name> python=<python_version>
```
If you are not using Conda, it is equally recommended that you create a Python virtual environment to isolate dependency installation and avoid future version conflicts.
### Install all the dependencies from the requirements file.
Once you have checked that you are using the environment created in the previous step, install the requirements for this project.
```
pip install -r requirements.txt
```
Once this command has finished executing, you will be able to run the project and all the technologies used in it.
### Initialise the MLFlow server for experiment tracking and model registry.
It is important to bear in mind the structure of the project. You must initialise the MLFlow server inside the orchestration directory for it to work correctly (make sure you are in the root directory of the project).
```
cd orchestration/
mlflow server
```
Or the equivalent commands in your OS.
### Run the main file
Once you have started the MLFlow server in your terminal, open a new terminal window and run the main.py file in the main directory to obtain the predicted values for the properties in California. Make sure you are in the root directory of the project when executing these commands.
```
cd main/
python main.py housing.csv <output_name.csv>
```
The first parameter is obligatory and refers to the input data. You can choose a name for the output file; by default, it is named 'housing_output.csv'. The output will be saved in the main/output folder.
### Monitoring
As well as the metadata files stored locally, you can use the Evidently UI to check the monitoring data once you have run the project at least once. It is important to be inside the main directory for this command; otherwise, you will see an empty result.
```
cd main/
evidently ui
```
Check the forward port on the local service (i.e. 8000) to view the monitoring. Although the Docker image has been implemented for more advanced monitoring, currently only the Grafana and Adminer services are registered with their basic configurations.
### Testing
If you want to run tests on the project, there are currently two unit tests for this purpose. Here are the steps to run the tests:
  1. Make sure you have installed the Python extension from Microsoft.
  2. In Bash, run the command `pipenv install` and ensure you use the interpreter from this venv (use the command `pipenv --venv` to find out which one).
  3. In Bash, run `pipenv shell` to initialise the Pipenv shell.

You can now run the tests using the testing tool from the Python extension.
