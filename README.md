## Setup

Before running any scripts or starting the project, you must configure the `PYTHONPATH` environment variable to include the path to the repository. This ensures that Python can locate the modules and packages within the repository correctly.

Execute the following command in your terminal:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/the/repository"
```

Also download the necessary datasets from Google Drive: https://drive.google.com/drive/folders/1G3WxN1l4DK3Xy9ngXqoqMqpc_HWH3-1M?usp=sharing and place the datasets folder in the code folder.

To run the app, go to the code folder and run the app.py file from it:

```bash
cd code
python app.py train --task task_name --models list of models for training
```