import os

def create_project_structure():
    # Define the project directory structure
    project_structure = {
            'data': ['stock_data.csv'],
            'notebooks': ['EDA_and_Preprocessing.ipynb'],
            'models': ['arima_model.py', 'sarima_model.py', 'prophet_model.py', 'lstm_model.py'],
            'utils': ['helpers.py'],
            'evaluation': ['evaluate_models.py'],
            'app': ['streamlit_app.py'],
            'outputs': {
                'plots': [],
                'forecasts': []
            },
            '': ['requirements.txt', 'README.md']
        }
    

    def create_structure(base_path, structure):
        for folder, contents in structure.items():
            # Create folder path
            folder_path = os.path.join(base_path, folder) if folder else base_path
            
            # Create directory if it doesn't exist
            if folder:
                os.makedirs(folder_path, exist_ok=True)
                print(f"Created directory: {folder_path}")
            
            # Create files or subdirectories
            if isinstance(contents, list):
                for file in contents:
                    file_path = os.path.join(folder_path, file)
                    # Create empty file
                    with open(file_path, 'w') as f:
                        f.write('')
                    print(f"Created file: {file_path}")
            elif isinstance(contents, dict):
                # Recursively create subdirectories
                create_structure(folder_path, contents)

    # Create the project structure
    create_structure('', project_structure)

if __name__ == "__main__":
    create_project_structure()
    print("Project structure created successfully!")