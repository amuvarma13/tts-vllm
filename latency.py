import torch
import time

def measure_latency(num_iterations=1, tensor_size=(1024, 1)):
    # Lists to store measured latencies for each iteration.
    latencies_gpu0_to_gpu1 = []
    latencies_round_trip = []

    for _ in range(num_iterations):
        # Ensure all previous CUDA work is complete.
        torch.cuda.synchronize()

        # Start timing: create a tensor on GPU 0.
        start = time.perf_counter()
        tensor = torch.rand(tensor_size, device='cuda:0')

        # Transfer tensor from GPU 0 to GPU 1.
        tensor_gpu1 = tensor.to('cpu')
        torch.cuda.synchronize()  # Wait for the transfer to complete.
        mid = time.perf_counter()

        # Transfer back from GPU 1 to GPU 0 for round-trip measurement.
        tensor_back = tensor_gpu1.to('cuda:0')
        torch.cuda.synchronize()  # Ensure round-trip transfer is complete.
        end = time.perf_counter()

        latencies_gpu0_to_gpu1.append(mid - start)
        latencies_round_trip.append(end - start)

    avg_latency = sum(latencies_gpu0_to_gpu1) / num_iterations
    avg_round_trip = sum(latencies_round_trip) / num_iterations
    print(f"Average one-way latency (GPU0 -> GPU1): {avg_latency * 1000:.3f} ms")
    print(f"Average round-trip latency (GPU0 -> GPU1 -> GPU0): {avg_round_trip * 1000:.3f} ms")

if __name__ == "__main__":
    # if torch.cuda.device_count() < 2:
    #     print("This script requires at least 2 GPUs.")
    # else:
    measure_latency()
