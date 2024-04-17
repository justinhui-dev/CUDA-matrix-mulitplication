CUDA Matrix Manipulation

Introduction
This repository contains code for performing matrix multiplication using CUDA (Compute Unified Device Architecture). The code leverages GPU parallelism to accelerate matrix multiplication operations, offering significant performance improvements over traditional CPU-based implementations.

Features
GPU Acceleration: Utilizes CUDA technology to harness the computational power of GPUs for matrix manipulation tasks.
Matrix Multiplication: Implements matrix multiplication functionality both on CPU and GPU, allowing for performance comparison between the two approaches.
Verification: Provides verification mechanism to ensure correctness of GPU-accelerated matrix multiplication results by comparing them with CPU-calculated results.
Performance Metrics: Measures and reports execution times for both CPU and GPU implementations, facilitating performance analysis and benchmarking.
Usage
Ensure you have a CUDA-enabled GPU and the necessary CUDA toolkit installed on your system.

Clone this repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/CUDA-Matrix-Manipulation.git
Navigate to the cloned directory:

bash
Copy code
cd CUDA-Matrix-Manipulation
Compile the CUDA code:

bash
Copy code
nvcc -o matrix_manipulation matrix_manipulation.cu
Execute the compiled binary:

bash
Copy code
./matrix_manipulation
Review the output, which includes execution times for both CPU and GPU implementations, as well as verification results.

Requirements
CUDA-enabled GPU
CUDA toolkit
Contributing
Contributions are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License.
