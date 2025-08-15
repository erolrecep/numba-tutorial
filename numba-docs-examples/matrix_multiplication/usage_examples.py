#!/usr/bin/env python3
"""
Usage Examples for Interactive CUDA Matrix Multiplication

This script demonstrates various ways to use the interactive matrix multiplication
program programmatically and provides examples of different use cases.
"""

import subprocess
import sys
import os

def run_quick_benchmark():
    """Run the quick benchmark mode programmatically."""
    print("🚀 Running Quick Benchmark Mode")
    print("=" * 50)
    
    # Simulate user input: choice 2 (quick benchmark), then 4 (exit)
    input_data = "2\n4\n"
    
    try:
        result = subprocess.run(
            [sys.executable, "interactive_matrix_multiplication.py"],
            input=input_data,
            text=True,
            capture_output=True,
            timeout=60
        )
        
        print("Output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("❌ Benchmark timed out")
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")


def run_custom_benchmark(M, N, K, matrix_type=1, trials=1):
    """Run a custom benchmark with specified parameters."""
    print(f"🔢 Running Custom Benchmark: {M}×{N}×{K}")
    print("=" * 50)
    
    # Simulate user input: choice 1 (custom), then dimensions, then exit
    input_data = f"1\n{M}\n{K}\n{N}\n{matrix_type}\n{trials}\nn\n4\n"
    
    try:
        result = subprocess.run(
            [sys.executable, "interactive_matrix_multiplication.py"],
            input=input_data,
            text=True,
            capture_output=True,
            timeout=120
        )
        
        print("Output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("❌ Benchmark timed out")
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")


def show_help():
    """Display help information programmatically."""
    print("📚 Getting Help Information")
    print("=" * 50)
    
    # Simulate user input: choice 3 (help), then 4 (exit)
    input_data = "3\n4\n"
    
    try:
        result = subprocess.run(
            [sys.executable, "interactive_matrix_multiplication.py"],
            input=input_data,
            text=True,
            capture_output=True,
            timeout=30
        )
        
        print("Output:")
        print(result.stdout)
        
    except Exception as e:
        print(f"❌ Error getting help: {e}")


def demonstrate_scaling():
    """Demonstrate scaling behavior with different matrix sizes."""
    print("📈 Scaling Demonstration")
    print("=" * 50)
    print("Testing how performance scales with matrix size...")
    
    test_cases = [
        (32, 32, 32, "Very small matrices"),
        (64, 64, 64, "Small matrices"),
        (128, 128, 128, "Medium matrices"),
        (256, 256, 256, "Large matrices"),
    ]
    
    for M, N, K, description in test_cases:
        print(f"\n{description}: {M}×{N}×{K}")
        print("-" * 40)
        
        # Run custom benchmark for this size
        input_data = f"1\n{M}\n{K}\n{N}\n1\n1\nn\n4\n"
        
        try:
            result = subprocess.run(
                [sys.executable, "interactive_matrix_multiplication.py"],
                input=input_data,
                text=True,
                capture_output=True,
                timeout=60
            )
            
            # Extract key performance metrics from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CPU (Numba JIT):' in line or 'GPU (Shared Memory):' in line or 'Shared GPU Speedup:' in line:
                    print(f"  {line.strip()}")
                    
        except Exception as e:
            print(f"  ❌ Error: {e}")


def demonstrate_matrix_types():
    """Demonstrate different matrix types."""
    print("🔢 Matrix Type Demonstration")
    print("=" * 50)
    
    matrix_types = [
        (1, "Random values"),
        (2, "Sequential values"),
        (3, "Identity matrices"),
        (4, "All ones")
    ]
    
    M, N, K = 64, 64, 64  # Use small matrices for quick demonstration
    
    for type_num, type_name in matrix_types:
        print(f"\n{type_name}:")
        print("-" * 30)
        
        input_data = f"1\n{M}\n{K}\n{N}\n{type_num}\n1\nn\n4\n"
        
        try:
            result = subprocess.run(
                [sys.executable, "interactive_matrix_multiplication.py"],
                input=input_data,
                text=True,
                capture_output=True,
                timeout=30
            )
            
            # Extract speedup information
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Shared GPU Speedup:' in line:
                    print(f"  {line.strip()}")
                    break
                    
        except Exception as e:
            print(f"  ❌ Error: {e}")


def main():
    """Main demonstration function."""
    print("🎯 Interactive CUDA Matrix Multiplication - Usage Examples")
    print("=" * 80)
    
    # Check if the interactive program exists
    if not os.path.exists("interactive_matrix_multiplication.py"):
        print("❌ Error: interactive_matrix_multiplication.py not found!")
        print("Make sure you're running this from the correct directory.")
        return
    
    print("This script demonstrates various ways to use the interactive matrix multiplication program.")
    print()
    
    try:
        # Example 1: Quick benchmark
        print("Example 1: Quick Benchmark Mode")
        run_quick_benchmark()
        
        print("\n" + "="*80)
        
        # Example 2: Custom benchmark
        print("Example 2: Custom Benchmark (256×256×256)")
        run_custom_benchmark(256, 256, 256, matrix_type=1, trials=1)
        
        print("\n" + "="*80)
        
        # Example 3: Help information
        print("Example 3: Getting Help Information")
        show_help()
        
        print("\n" + "="*80)
        
        # Example 4: Scaling demonstration
        print("Example 4: Scaling Demonstration")
        demonstrate_scaling()
        
        print("\n" + "="*80)
        
        # Example 5: Matrix types
        print("Example 5: Different Matrix Types")
        demonstrate_matrix_types()
        
        print("\n" + "="*80)
        print("✅ All examples completed!")
        print("\n💡 To run the interactive program manually:")
        print("   python interactive_matrix_multiplication.py")
        
    except KeyboardInterrupt:
        print("\n\n👋 Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")


if __name__ == "__main__":
    main()
