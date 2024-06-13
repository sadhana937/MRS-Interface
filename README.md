# Movie Recommendation System Interface

## Prerequisites

- **Python**: Ensure Python 3.x is installed on your system.
- **Git**: Ensure Git is installed to clone the repository.

## Step-by-Step Instructions

### 1. Clone the Repository
Open a terminal (or command prompt) and run:
```sh
git clone https://github.com/sadhana937/MRS-Interface.git
cd MRS-Interface
```

### 2. Install Required Dependencies
Run the following command to install the necessary dependencies:
```sh
pip install -r requirements.txt
```

### 3. Run the ML Model File
Execute the following command:
```sh
python model.py
```
Ensure the file `nnmf_model.h5` exists in the repository.

### 4. Run the Application
Start the application with:
```sh
python app.py
```

### 5. Access the Application
Open your web browser and navigate to: `http://127.0.0.1:5000`

## Additional Information

### Repository Structure
- `app.py`: Main file to run the Flask application.
- `model.py`: Contains the logic for the movie recommendation model.
- `requirements.txt`: Lists the dependencies needed for the project.
- `templates/`: Directory containing HTML templates.
- `static/`: Directory containing static files (images).

### Modifying the Application
If you need to modify the application (e.g., change the model or update the UI), you can edit the respective files and restart the application by running `python app.py` again.

## Troubleshooting
- If you encounter any issues during installation, ensure all dependencies are correctly installed and compatible with your Python version.
- Check if the `nnmf_model.h5` file is correctly loaded in the `model.py` file.
- Ensure no other application is using port 5000 on your system.

By following these steps, you should be able to set up and run the Movie Recommendation System interface locally.
