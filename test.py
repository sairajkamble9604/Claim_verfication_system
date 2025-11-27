import subprocess
import os

print("--- Starting Python test ---")

# Set the environment variable to allow mismatched encodings
os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    # Define the command
    command = [
        "ollama", 
        "run", 
        "llama3:latest", 
        "Hello, who are you? Respond in one sentence."
    ]

    # Run the command
    print(f"Running command: {' '.join(command)}")

    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding='utf-8', # Explicitly set encoding
        timeout=120 # 2 minute timeout
    )

    # Print the results
    print("\n--- Ollama Standard Output ---")
    print(proc.stdout)

    print("\n--- Ollama Standard Error ---")
    print(proc.stderr)

    print("\n--- Test Finished ---")

except FileNotFoundError:
    print("\n*** ERROR: 'ollama' command not found. ***")
    print("This means Ollama is not in your system's PATH for Python.")

except subprocess.TimeoutExpired:
    print("\n*** ERROR: Command timed out after 120 seconds. ***")

except Exception as e:
    print(f"\n*** An unexpected error occurred: {e} ***")