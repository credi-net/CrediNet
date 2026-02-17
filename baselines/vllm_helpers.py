"""
Helper functions to manage vLLM server instances for CrediNet experiments.
Adapted from StabilityBench's vllm_helpers.py
"""

import socket
import time
import requests
import hashlib
from typing import Optional
import subprocess
import os
import psutil


def _model_to_port(model_name: str, base_port: int = 8000) -> int:
    """
    Deterministically map a model name to a port.
    Prevents collisions when running multiple models.
    """
    h = int(hashlib.sha1(model_name.encode()).hexdigest(), 16)
    return base_port + (h % 1000)


def _is_port_open(host: str, port: int) -> bool:
    """Check if a port is already open/in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _wait_for_vllm_ready(
    port: int,
    timeout: int = 600,
):
    """Wait for vLLM server to become ready by polling its /v1/models endpoint."""
    start = time.time()
    url = f"http://127.0.0.1:{port}/v1/models"
    elapsed = 0

    while elapsed < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"[INFO] vLLM server ready on port {port}")
                return
        except Exception:
            pass
        elapsed = time.time() - start
        if elapsed % 100 < 1: 
            print(f"[INFO] Still waiting for vLLM server ({elapsed:.0f}s/{timeout}s)...")
        time.sleep(1)

    raise RuntimeError(f"vLLM server on port {port} did not become ready in time (timeout={timeout}s)")


def _kill_process_on_port(port: int):
    """Kill any process listening on the specified port."""
    print(f"[INFO] Attempting to kill any process on port {port}")
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and '--port' in cmdline and str(port) in cmdline:
                if 'vllm.entrypoints.openai.api_server' in ' '.join(cmdline):
                    print(f"[INFO] Killing vLLM process (PID: {proc.pid}) on port {port}")
                    try:
                        children = proc.children(recursive=True)
                        for child in children:
                            child.kill()
                        proc.kill()
                        proc.wait(timeout=10)
                    except Exception:
                        pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    time.sleep(2)  # Give system time to release the port


def ensure_vllm_server(
    model_name: str,
    tensor_parallel_size: int = 1,
    base_port: int = 8000,
    cuda_visible_devices: Optional[str] = None,
    seed: int = 1,
    max_retries: int = 3,
) -> int:
    """
    Ensure a vLLM OpenAI-compatible server is running for `model_name`.

    If already running, this is a no-op.
    Otherwise, starts the server and waits until ready with exponential backoff retry.
    
    Args:
        model_name: Name of the model to load
        tensor_parallel_size: Number of GPUs to use
        base_port: Base port number for port calculation
        cuda_visible_devices: Optional CUDA device specification
        seed: Random seed
        max_retries: Maximum number of retry attempts (default: 3)
    
    Returns:
        port: The port number where the server is running
    """

    port = _model_to_port(model_name, base_port)

    if _is_port_open("127.0.0.1", port):
        try:
            _wait_for_vllm_ready(port, timeout=5)
            print(f"[INFO] Using existing vLLM server on port {port}")
            return port
        except RuntimeError:
            pass 

    # Exponential backoff retry logic
    for attempt in range(max_retries):
        try:
            # Check GPU memory before attempting to start
            print("[INFO] Checking GPU memory availability...")
            verify_gpu_memory_available(min_free_gb=5.0)

            print(f"[INFO] Starting vLLM server for {model_name} on port {port} (attempt {attempt + 1}/{max_retries})")
            print(f"[INFO] Using tensor_parallel_size={tensor_parallel_size}")

            cmd = [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model_name,
                "--tensor-parallel-size",
                str(tensor_parallel_size),
                "--dtype",
                "half",
                "--seed",
                str(seed),
                "--port",
                str(port),
            ]

            env = os.environ.copy()
            if cuda_visible_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

            log_file_path = f"logs/vllm_server_{port}_attempt{attempt + 1}.log"  
            os.makedirs("logs", exist_ok=True) 
            with open(log_file_path, "w") as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            
            print(f"[INFO] vLLM server process started (PID: {process.pid})")
            print(f"[INFO] Server logs available at: {log_file_path}")
            print("[INFO] Waiting for vLLM server to become ready...")
            
            try:
                _wait_for_vllm_ready(port, timeout=1200)
                print(f"[INFO] vLLM server ready on port {port}")
                return port
            except RuntimeError as e:
                print(f"[ERROR] {e}")
                print(f"[ERROR] Check server logs at: {log_file_path}")
                
                # If this is not the last attempt, cleanup and retry
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt * 10, 300)  # Cap at 5 minutes
                    print(f"[WARNING] Server failed to start. Cleaning up and retrying in {wait_time}s...")
                    
                    # Try to kill the process
                    try:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=10)
                    except Exception:
                        try:
                            process.kill()
                        except Exception:
                            pass
                    
                    # Additional cleanup: kill any process on this port
                    try:
                        _kill_process_on_port(port)
                    except Exception as cleanup_error:
                        print(f"[WARNING] Cleanup error: {cleanup_error}")
                    
                    # Wait before retrying (exponential backoff)
                    time.sleep(wait_time)
                else:
                    # Last attempt failed, raise the error
                    raise
                
        except RuntimeError:
            # If this was the last attempt, re-raise
            if attempt == max_retries - 1:
                raise
            # Otherwise, continue to next iteration
        except Exception as e:
            print(f"[ERROR] Unexpected error during server launch: {e}")
            if attempt == max_retries - 1:
                raise
            wait_time = min(2 ** attempt * 10, 300)
            print(f"[WARNING] Retrying in {wait_time}s...")
            time.sleep(wait_time)

    # This should not be reached, but just in case
    raise RuntimeError(f"Failed to start vLLM server after {max_retries} attempts")


def shutdown_vllm_server(model_name: str, base_port: int = 8000) -> bool:
    """
    Shutdown a running vLLM server for the given model.
    
    Returns:
        bool: True if server was found and terminated, False otherwise
    """
    port = _model_to_port(model_name, base_port)
    
    print(f"[INFO] Shutting down vLLM server on port {port} for {model_name}")
    
    # Find and kill the process and all its children
    found = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and 'vllm.entrypoints.openai.api_server' in ' '.join(cmdline):
                if '--port' in cmdline and str(port) in cmdline:
                    print(f"[INFO] Terminating vLLM server process (PID: {proc.pid})")
                    
                    # Kill all child processes first
                    try:
                        children = proc.children(recursive=True)
                        for child in children:
                            print(f"[INFO] Terminating child process (PID: {child.pid})")
                            child.terminate()
                        
                        # Wait for children to terminate
                        gone, alive = psutil.wait_procs(children, timeout=5)
                        for p in alive:
                            print(f"[WARNING] Force killing child process (PID: {p.pid})")
                            p.kill()
                    except psutil.NoSuchProcess:
                        pass
                    
                    # Now terminate the parent
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        print(f"[WARNING] Force killing parent process (PID: {proc.pid})")
                        proc.kill()
                        proc.wait()
                    
                    print(f"[INFO] Server shutdown complete, GPU memory freed")
                    found = True
                    
                    # Give GPU time to release memory
                    time.sleep(5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            pass
    
    if not found:
        print(f"[WARNING] No vLLM server found on port {port}")
    
    return found


def verify_gpu_memory_available(min_free_gb: float = 10.0) -> bool:
    """
    Check if sufficient GPU memory is available.
    
    Args:
        min_free_gb: Minimum free GPU memory required in GB
        
    Returns:
        bool: True if sufficient memory is available
    """
    try:
        import torch
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / (1024**3)  # Convert to GB
            total_mem = torch.cuda.mem_get_info()[1] / (1024**3)
            used_mem = total_mem - free_mem
            print(f"[INFO] GPU Memory: {used_mem:.2f}GB used, {free_mem:.2f}GB free, {total_mem:.2f}GB total")
            return free_mem >= min_free_gb
        else:
            print("[WARNING] CUDA not available")
            return False
    except Exception as e:
        print(f"[WARNING] Could not check GPU memory: {e}")
        return True  # Don't block if we can't check


def restart_vllm_server(
    model_name: str, 
    tensor_parallel_size: int = 1, 
    base_port: int = 8000,
    seed: int = 1,
) -> int:
    """
    Restart a vLLM server (useful after shutdown).
    
    Returns:
        port: The port number where the server is running
    """
    print(f"[INFO] Restarting vLLM server for {model_name}")
    shutdown_vllm_server(model_name, base_port)
    time.sleep(2)  # Give it a moment to clean up
    port = ensure_vllm_server(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        base_port=base_port,
        seed=seed,
    )
    return port
