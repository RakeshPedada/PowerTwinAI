"""
core/gpu_detector.py
--------------------
Hardware detection utility for NVIDIA GPU, CUDA runtime, VRAM capacity,
and COLMAP CUDA support.
"""

import os
import shutil
import subprocess
import json


def get_gpu_info() -> dict:
    """
    Detect system GPU, VRAM, CUDA runtime availability, and COLMAP CUDA capability.

    Returns
    -------
    dict
        {
            "name": str,
            "vram_gb": float,
            "cuda_available": bool,
            "colmap_cuda": bool,
            "driver_version": str,
            "device_type": str,
            "summary": str,
            "status_badge": str
        }
    """
    info = {
        "name": "Unknown GPU",
        "vram_gb": 0.0,
        "cuda_available": False,
        "colmap_cuda": False,
        "driver_version": "N/A",
        "device_type": "Unknown",
        "summary": "No GPU detected",
        "status_badge": "CPU Mode"
    }

    # 1. Check PyTorch CUDA if available
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["name"] = torch.cuda.get_device_name(0)
            info["device_type"] = "NVIDIA"
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
            info["status_badge"] = f"CUDA Available ({info['vram_gb']} GB)"
    except Exception:
        pass

    # 2. Check via nvidia-smi if PyTorch is not available or gave no CUDA
        nvidia_smi = shutil.which("nvidia-smi") or os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "nvidia-smi.exe")
        if os.path.exists(nvidia_smi) or shutil.which("nvidia-smi"):
            try:
                res = subprocess.run(
                    [nvidia_smi, "--query-gpu=gpu_name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if res.returncode == 0 and res.stdout.strip():
                    parts = [p.strip() for p in res.stdout.strip().split(",")]
                    if len(parts) >= 3:
                        info["name"] = parts[0]
                        info["device_type"] = "NVIDIA"
                        info["cuda_available"] = True
                        try:
                            info["vram_gb"] = round(float(parts[1]) / 1024.0, 1)
                        except ValueError:
                            pass
                        info["driver_version"] = parts[2]
                        info["status_badge"] = f"CUDA Available ({info['vram_gb']} GB)"
            except Exception:
                pass

    # 3. If still no discrete NVIDIA GPU found, inspect via Windows PowerShell / WMI
    if info["name"] == "Unknown GPU":
        try:
            ps_cmd = "Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM | ConvertTo-Json"
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=6
            )
            if res.returncode == 0 and res.stdout.strip():
                try:
                    data = json.loads(res.stdout.strip())
                    if isinstance(data, list):
                        nvidia_devs = [d for d in data if "NVIDIA" in str(d.get("Name", "")).upper()]
                        data = nvidia_devs[0] if nvidia_devs else data[0]

                    info["name"] = data.get("Name", "Integrated GPU")
                    info["driver_version"] = data.get("DriverVersion", "N/A")
                    adapter_ram = data.get("AdapterRAM", 0) or 0
                    if adapter_ram > 0:
                        info["vram_gb"] = round(float(adapter_ram) / (1024 ** 3), 1)

                    name_upper = info["name"].upper()
                    if "NVIDIA" in name_upper:
                        info["device_type"] = "NVIDIA"
                    elif "INTEL" in name_upper:
                        info["device_type"] = "Intel"
                    elif "AMD" in name_upper or "RADEON" in name_upper:
                        info["device_type"] = "AMD"
                except Exception:
                    pass
        except Exception:
            pass

    # 4. Check if installed COLMAP binary has CUDA support
    colmap_exe = r"C:\COLMAPin\colmap.exe"
    if os.path.exists(colmap_exe):
        try:
            res = subprocess.run([colmap_exe, "-h"], capture_output=True, text=True, timeout=5)
            output = (res.stdout or "") + (res.stderr or "")
            if "without CUDA" in output:
                info["colmap_cuda"] = False
            elif "with CUDA" in output or "CUDA" in output:
                info["colmap_cuda"] = True
        except Exception:
            pass

    # Build summary string
    cuda_str = "Available" if info["cuda_available"] else "Unavailable"
    info["summary"] = f"GPU: {info['name']} | VRAM: {info['vram_gb']} GB | CUDA: {cuda_str}"
    if not info["cuda_available"]:
        info["status_badge"] = f"{info['name']} (CPU Mode)"

    return info