import subprocess
import time
import os
import random
import datetime
import sys

# --- Configuration ---
EXECUTABLE_NAME = "./semettre"
# List of digit sizes to test for multiplication (N x N)
DIGIT_SIZES = [1, 10, 100, 1000, 10000, 100000, 999999]

sys.set_int_max_str_digits(0)

def generate_random_number_str(num_digits):
    """
    Generates a random number as a string with a specific number of digits.
    The first digit will not be zero to ensure the correct length.
    """
    if num_digits <= 0:
        return "0"
    # First digit is from 1-9
    first_digit = str(random.randint(1, 9))
    # Remaining digits are from 0-9
    remaining_digits = "".join(random.choices("0123456789", k=num_digits - 1))
    return first_digit + remaining_digits

def run_benchmark():
    """
    Runs the benchmark against the 'semettre' executable.
    It measures performance, verifies correctness, and saves all data to text files.
    """
    # --- 1. Check if the executable exists ---
    if not os.path.exists(EXECUTABLE_NAME):
        print(f"Error: Executable '{EXECUTABLE_NAME}' not found.")
        print("Please compile your C code first (e.g., 'gcc -O3 your_code.c -o semettre')")
        return

    # --- Create a timestamped directory for the results ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"benchmark_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving results to '{results_dir}/'\n")

    print("--- Starting Multiplication Benchmark ---")
    print(f"Testing executable: {EXECUTABLE_NAME}\n")

    # --- 2. Loop through each specified digit size ---
    for size in DIGIT_SIZES:
        print(f"Testing {size} digits x {size} digits...")

        # Generate two large random numbers as strings.
        num1_str = generate_random_number_str(size)
        num2_str = generate_random_number_str(size)

        # Save the generated input numbers to a file
        with open(os.path.join(results_dir, f"{size}_input.txt"), "w") as f:
            f.write(f"Number 1: {num1_str}\n")
            f.write(f"Number 2: {num2_str}\n")

        # Prepare the input for the C program
        input_data = f"{num1_str}\n{num2_str}\n".encode('utf-8')

        # --- 3. Run the executable and measure time ---
        start_time = time.perf_counter()
        process = subprocess.Popen(
            [EXECUTABLE_NAME],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(input=input_data)
        end_time = time.perf_counter()
        duration = end_time - start_time

        if process.returncode != 0:
            print(f"  -> Error running executable for size {size}:")
            print(stderr.decode('utf-8'))
            continue

        # Decode the result and save it to a file
        executable_result = stdout.decode('utf-8').strip()
        with open(os.path.join(results_dir, f"{size}_algorithm_output.txt"), "w") as f:
            f.write(executable_result)

        # --- 4. Verify the result and save it to a file ---
        try:
            num1_int = int(num1_str)
            num2_int = int(num2_str)
            expected_result = str(num1_int * num2_int)
            verification_status = "Correct" if executable_result == expected_result else "INCORRECT"
            
            with open(os.path.join(results_dir, f"{size}_python_verification.txt"), "w") as f:
                f.write(expected_result)

        except ValueError:
            expected_result = "Error during Python verification."
            verification_status = "Verification Failed"

        # --- 5. Print the results for this run ---
        print(f"  -> Time taken: {duration:.6f} seconds")
        print(f"  -> Verification: {verification_status}\n")

    print(f"--- Benchmark Complete ---")
    print(f"All inputs and outputs have been saved in the '{results_dir}/' directory.")

if __name__ == "__main__":
    run_benchmark()
