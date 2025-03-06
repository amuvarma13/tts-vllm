import torch
import time

def measure_latency(num_iterations=10, tensor_size=(1024, 1024)):
    # Check available GPUs
    if torch.cuda.device_count() < 2:
        print(f"Only {torch.cuda.device_count()} GPU(s) available. Some tests will be skipped.")
    
    # Lists to store measured latencies for each iteration.
    latencies_gpu0_to_cpu = []
    latencies_gpu1_to_cpu = []
    latencies_gpu0_to_gpu1 = []
    latencies_cpu_to_gpu0 = []
    latencies_cpu_to_gpu1 = []
    
    # Print GPU info
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    for i in range(num_iterations):
        # Ensure all previous CUDA work is complete
        torch.cuda.synchronize()
        
        # === GPU 0 to CPU transfer ===
        start = time.perf_counter()
        tensor = torch.rand(tensor_size, device='cuda:0')
        torch.cuda.synchronize()
        tensor = tensor.to('cpu')
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies_gpu0_to_cpu.append(end - start)
        
        # === CPU to GPU 0 transfer ===
        start = time.perf_counter()
        tensor = torch.rand(tensor_size, device='cpu')
        tensor = tensor.to('cuda:0')
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies_cpu_to_gpu0.append(end - start)
        
        if torch.cuda.device_count() >= 2:
            # === GPU 1 to CPU transfer ===
            start = time.perf_counter()
            tensor = torch.rand(tensor_size, device='cuda:1')
            torch.cuda.synchronize()
            tensor = tensor.to('cpu')
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies_gpu1_to_cpu.append(end - start)
            
            # === CPU to GPU 1 transfer ===
            start = time.perf_counter()
            tensor = torch.rand(tensor_size, device='cpu')
            tensor = tensor.to('cuda:1')
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies_cpu_to_gpu1.append(end - start)
            
            # === GPU 0 to GPU 1 transfer (direct) ===
            start = time.perf_counter()
            tensor = torch.rand(tensor_size, device='cuda:0')
            torch.cuda.synchronize()
            tensor = tensor.to('cuda:1')
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies_gpu0_to_gpu1.append(end - start)
    
    # Calculate and print average latencies
    print("\n=== AVERAGE LATENCIES ===")
    print(f"GPU 0 → CPU: {sum(latencies_gpu0_to_cpu) / num_iterations * 1000:.3f} ms")
    print(f"CPU → GPU 0: {sum(latencies_cpu_to_gpu0) / num_iterations * 1000:.3f} ms")
    
    if torch.cuda.device_count() >= 2:
        print(f"GPU 1 → CPU: {sum(latencies_gpu1_to_cpu) / num_iterations * 1000:.3f} ms")
        print(f"CPU → GPU 1: {sum(latencies_cpu_to_gpu1) / num_iterations * 1000:.3f} ms")
        print(f"GPU 0 → GPU 1 (direct): {sum(latencies_gpu0_to_gpu1) / num_iterations * 1000:.3f} ms")
    
    # Print hardware details to help diagnose issues
    print("\n=== HARDWARE DETAILS ===")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  - Compute capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")

if __name__ == "__main__":
    measure_latency()