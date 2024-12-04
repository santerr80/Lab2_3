from clearml import Task
from clearml import Dataset

# Initialize the task
task = Task.init(project_name="Lab2_3", task_name="dataset")

# Create the dataset
dataset = Dataset.create(dataset_name="diabetes", dataset_project="Lab2_3")

# Add files to the dataset
dataset.add_files(path="diabetes_data.csv")

# Upload the dataset
dataset.upload()

# Finalize the dataset
dataset.finalize()
