# import required libraries
import numpy as np
import time
import warnings
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

# Suppress performance warnings for educational examples
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

# Set the timeout to one hour (in nanoseconds for datetime64[ns] compatibility)
SESSION_TIMEOUT_NS = np.int64(3600 * 1e9)  # 1 hour in nanoseconds


@cuda.jit
def mark_session_boundaries(user_id, timestamp, boundaries):
    """
    First pass: Mark session boundaries based on user changes and timeouts.
    This is embarrassingly parallel - each thread only examines its own element.
    """
    gid = cuda.grid(1)
    size = len(user_id)

    if gid >= size:
        return

    # First element is always a session boundary
    if gid == 0:
        boundaries[gid] = 1
        return

    # Check if this position starts a new session
    new_user = user_id[gid] != user_id[gid - 1]
    time_gap = timestamp[gid] - timestamp[gid - 1]
    timed_out = time_gap > SESSION_TIMEOUT_NS

    if new_user or timed_out:
        boundaries[gid] = 1
    else:
        boundaries[gid] = 0


@cuda.jit
def compute_session_ids(boundaries, session_ids):
    """
    Second pass: Convert boundary markers to session IDs using parallel prefix scan.
    Each thread computes the cumulative sum of boundaries up to its position.
    """
    gid = cuda.grid(1)
    size = len(boundaries)

    if gid >= size:
        return

    # Simple parallel prefix sum (not optimal but correct for educational purposes)
    session_count = 0
    for i in range(gid + 1):
        session_count += boundaries[i]

    session_ids[gid] = session_count - 1  # Convert to 0-based indexing


def cpu_sessionize(user_id, timestamp, session_timeout_ns):
    """
    CPU reference implementation for validation and performance comparison.
    """
    n = len(user_id)
    session_ids = np.zeros(n, dtype=np.int32)
    current_session = 0

    session_ids[0] = current_session

    for i in range(1, n):
        # Check for session boundary
        new_user = user_id[i] != user_id[i-1]
        time_gap = timestamp[i] - timestamp[i-1]
        timed_out = time_gap > session_timeout_ns

        if new_user or timed_out:
            current_session += 1

        session_ids[i] = current_session

    return session_ids


def gpu_sessionize(user_id, timestamp, block_size=256):
    """
    GPU implementation using a two-pass algorithm:
    1. Mark session boundaries (embarrassingly parallel)
    2. Compute session IDs using prefix scan
    """
    n = len(user_id)

    # Calculate grid configuration
    num_blocks = (n + block_size - 1) // block_size

    # Transfer data to GPU
    d_user_id = cuda.to_device(user_id)
    d_timestamp = cuda.to_device(timestamp)
    d_boundaries = cuda.device_array(n, dtype=np.int32)
    d_session_ids = cuda.device_array(n, dtype=np.int32)

    # Pass 1: Mark session boundaries
    mark_session_boundaries[num_blocks, block_size](d_user_id, d_timestamp, d_boundaries)

    # Pass 2: Compute session IDs from boundaries
    compute_session_ids[num_blocks, block_size](d_boundaries, d_session_ids)

    # Copy results back to host
    session_ids = d_session_ids.copy_to_host()
    boundaries = d_boundaries.copy_to_host()

    return session_ids, boundaries


def generate_test_data():
    """
    Generate realistic test data with multiple users and various time patterns.
    """
    # User IDs with different session patterns
    user_ids = np.array([
        1, 1, 1, 1, 1, 1,           # User 1: 2 sessions (timeout after element 2)
        2, 2, 2,                    # User 2: 1 session
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  # User 3: 3 sessions (timeouts)
        4, 4, 4, 4, 4, 4, 4, 4, 4,  # User 4: 2 sessions
    ], dtype=np.int32)

    # Timestamps in nanoseconds (datetime64[ns] compatible)
    # Create realistic patterns with some sessions timing out
    timestamps = np.array([
        # User 1: First 3 clicks close together, then timeout, then 3 more
        1000000000, 1001000000, 1002000000,  # Session 0
        5000000000000, 5000001000000, 5000002000000,  # Session 1 (after timeout)

        # User 2: All clicks within session timeout
        6000000000000, 6000001000000, 6000002000000,  # Session 2

        # User 3: Multiple timeouts creating several sessions
        7000000000000, 7000001000000,  # Session 3
        12000000000000, 12000001000000,  # Session 4 (after timeout)
        17000000000000, 17000001000000, 17000002000000, 17000003000000,  # Session 5
        25000000000000, 25000001000000,  # Session 6 (after timeout)

        # User 4: One timeout in the middle
        30000000000000, 30000001000000, 30000002000000,  # Session 7
        35000000000000, 35000001000000, 35000002000000,  # Session 8 (after timeout)
        35000003000000, 35000004000000, 35000005000000,  # Still session 8
    ], dtype=np.int64)

    return user_ids, timestamps


def validate_results(gpu_results, cpu_results):
    """
    Validate GPU results against CPU reference implementation.
    """
    if not np.array_equal(gpu_results, cpu_results):
        print("❌ Validation failed!")
        print(f"GPU results: {gpu_results}")
        print(f"CPU results: {cpu_results}")
        return False

    print("✅ Validation passed!")
    return True


def analyze_sessions(user_ids, timestamps, session_ids):
    """
    Provide detailed analysis of the sessionization results.
    """
    print("\n📊 Session Analysis:")
    print("-" * 50)

    unique_sessions = np.unique(session_ids)
    print(f"Total sessions found: {len(unique_sessions)}")

    for session_id in unique_sessions:
        mask = session_ids == session_id
        session_users = user_ids[mask]
        session_times = timestamps[mask]

        user_id = session_users[0]  # All should be the same user
        duration_ns = session_times[-1] - session_times[0]
        duration_sec = duration_ns / 1e9
        click_count = len(session_times)

        print(f"  Session {session_id}: User {user_id}, {click_count} clicks, {duration_sec:.1f}s duration")


def benchmark_performance(user_ids, timestamps, num_trials=5):
    """
    Compare CPU vs GPU performance across multiple trials.
    """
    print("\n⚡ Performance Benchmark:")
    print("-" * 50)

    # CPU timing
    cpu_times = []
    for _ in range(num_trials):
        start_time = time.time()
        cpu_results = cpu_sessionize(user_ids, timestamps, SESSION_TIMEOUT_NS)
        cpu_times.append((time.time() - start_time) * 1000)

    cpu_time_avg = np.mean(cpu_times)

    # GPU timing
    gpu_times = []
    for _ in range(num_trials):
        start_time = time.time()
        gpu_results, _ = gpu_sessionize(user_ids, timestamps)
        gpu_times.append((time.time() - start_time) * 1000)

    gpu_time_avg = np.mean(gpu_times)

    # Results
    print(f"CPU time: {cpu_time_avg:.3f} ms")
    print(f"GPU time: {gpu_time_avg:.3f} ms")

    if gpu_time_avg > 0:
        speedup = cpu_time_avg / gpu_time_avg
        print(f"Speedup: {speedup:.2f}x {'(GPU faster)' if speedup > 1 else '(CPU faster)'}")

    # Validate correctness
    validate_results(gpu_results, cpu_results)

    return cpu_results, gpu_results


def main():
    """
    Main function demonstrating the sessionization algorithm.
    """
    print("🔍 CUDA Click Data Sessionization Example")
    print("=" * 60)

    try:
        # Generate test data
        print("\n📊 Generating Test Data...")
        user_ids, timestamps = generate_test_data()

        print(f"Data size: {len(user_ids)} click events")
        print(f"Users: {np.unique(user_ids)}")
        print(f"Session timeout: {SESSION_TIMEOUT_NS / 1e9:.0f} seconds")

        # Show sample data
        print("\n📋 Sample Data (first 10 events):")
        print("Index | User | Timestamp (relative)")
        print("-" * 35)
        base_time = timestamps[0]
        for i in range(min(10, len(user_ids))):
            rel_time = (timestamps[i] - base_time) / 1e9
            print(f"{i:5d} | {user_ids[i]:4d} | {rel_time:10.1f}s")

        # Run CPU implementation
        print("\n🖥️  Running CPU Implementation...")
        start_time = time.time()
        cpu_results = cpu_sessionize(user_ids, timestamps, SESSION_TIMEOUT_NS)
        cpu_time = (time.time() - start_time) * 1000
        print(f"CPU time: {cpu_time:.3f} ms")

        # Run GPU implementation
        print("\n🚀 Running GPU Implementation...")
        start_time = time.time()
        gpu_results, boundaries = gpu_sessionize(user_ids, timestamps)
        gpu_time = (time.time() - start_time) * 1000
        print(f"GPU time: {gpu_time:.3f} ms")

        # Validate results
        print("\n🔍 Validating Results...")
        is_valid = validate_results(gpu_results, cpu_results)

        if is_valid:
            # Analyze sessions
            analyze_sessions(user_ids, timestamps, gpu_results)

            # Show detailed results
            print("\n📋 Detailed Results:")
            print("Index | User | Session | Boundary | Timestamp (relative)")
            print("-" * 55)
            base_time = timestamps[0]
            for i in range(len(user_ids)):
                rel_time = (timestamps[i] - base_time) / 1e9
                boundary_marker = "🔸" if boundaries[i] else " "
                print(f"{i:5d} | {user_ids[i]:4d} | {gpu_results[i]:7d} | {boundary_marker:8s} | {rel_time:10.1f}s")

            # Performance comparison
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"\n⚡ Performance: {speedup:.2f}x {'speedup' if speedup > 1 else 'slowdown'}")

            # Algorithm explanation
            print("\n🧠 Algorithm Explanation:")
            print("-" * 30)
            print("1. Pass 1: Mark session boundaries (parallel)")
            print("   - New user = boundary")
            print("   - Time gap > timeout = boundary")
            print("2. Pass 2: Convert boundaries to session IDs (prefix scan)")
            print("   - Each boundary increments session counter")
            print("   - All events between boundaries get same session ID")

        else:
            print("❌ Validation failed - results don't match!")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("Make sure you have a CUDA-capable GPU and proper CUDA installation.")


if __name__ == "__main__":
    main()
