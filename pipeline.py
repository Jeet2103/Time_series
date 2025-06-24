import subprocess
import os

def run_script(script_path):
    """Run a Python script as a subprocess."""
    try:
        print(f"\nRunning: {script_path}")
        subprocess.run(["python", script_path], check=True)
        print(f"Completed: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_path}")
        print(e)

def main():
    # Ensure required directories exist
    os.makedirs("outputs/forecasts", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    # Step 1: Run evaluation script
    run_script("evaluation/evaluate_models.py")

    # Step 2: Run plot generation script
    run_script("evaluation/plot_models.py")

    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()
