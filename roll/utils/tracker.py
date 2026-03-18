import asyncio
import subprocess
import json
import os
import time
from datetime import datetime


import re
import subprocess

import subprocess

def get_sm_utilization_once():
    proc = subprocess.Popen(
        ['nvidia-smi', 'dmon', '-s', 'ut', '-o', 'T'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    sm_data = []
    header_lines = 0
    count = 0
    # dmon输出前两行为表头，后面每行为每块卡
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if line.startswith('#'):
            header_lines += 1
            continue
        if line:  # 数据行
            parts = line.split()
            if len(parts) >= 3:
                sm_data.append({
                    'index': parts[1],
                    'sm_utilization': int(parts[2])
                })
            count += 1
        # 通常卡数个数据行，全部读到就够了
        # 这里以系统有8卡为例：
        if count >= 8:  # 你的卡数多少就写多少，或根据实际行判断break
            break
    proc.terminate()
    return sm_data



def _parse_nvidia_smi_output(output):
    gpu_lines = []

    lines = output.strip().splitlines()
    for line in lines:
        # Remove commas and extra spaces
        parts = [field.strip() for field in line.split(',')]

        if len(parts) < 5:
            continue  # skip invalid lines

        try:
            gpu_index = int(parts[0])
            name = parts[1].strip()
            gpu_util = int(parts[2].replace('%', '').strip())
            mem_used = int(parts[3].replace('MiB', '').strip())
            mem_total = int(parts[4].replace('MiB', '').strip())

            gpu_info = {
                "index": str(gpu_index),
                "name": name,
                "utilization.gpu": gpu_util,
                "memory.used": mem_used,
                "memory.total": mem_total
            }
            gpu_lines.append(gpu_info)
        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line} | Error: {e}")
            continue

    return {"gpus": gpu_lines}


class GPUTracker:
    def __init__(self, interval=1, output_dir=".", filename="gpu_usage_log.json"):
        self.interval = interval
        self.running = False
        self.data_log = []
        self.output_dir = output_dir
        self.filename = filename

    def _get_gpu_info(self):
        # nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
        simple_cmd = ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader']
        simple_result = subprocess.run(simple_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if simple_result.returncode != 0:
            print("Error running simple nvidia-smi query:", simple_result.stderr)
            return None
        nv_smi_output = _parse_nvidia_smi_output(simple_result.stdout)
        return nv_smi_output

    async def _monitor_task(self):
        while self.running:
            gpus = self._get_gpu_info()
            if gpus:
                entry = {
                    "timestamp": time.time(),
                    "gpus": gpus
                }
                self.data_log.append(entry)
            else:
                print(f"[{datetime.now()}] Failed to retrieve GPU data.")

            await asyncio.sleep(self.interval)

    async def start(self, step_id):
        """ 启动异步监控 """
        if not self.running:
            self.running = True
            self.step_start_time = datetime.now()
            self.step_id = step_id
            self.task = asyncio.create_task(self._monitor_task())
            print(f"GPU tracker started asynchronously at {self.step_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def stop(self, other_statistic: dict = None):
        if self.running:
            self.running = False
            self.step_end_time = datetime.now()

            if hasattr(self, 'step_start_time'):
                timing_info = {
                    "step_timing": {
                        "step_id": self.step_id,
                        "start_time": self.step_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "end_time": self.step_end_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "duration_seconds": (self.step_end_time - self.step_start_time).total_seconds()
                    }
                }
            else:
                timing_info = {}
            
            if other_statistic:
                timing_info["other_statistic"] = other_statistic
            
            if len(timing_info) > 0:
                self.data_log.insert(0, timing_info)

            save_name, ext = os.path.splitext(self.filename)
            save_name = f"{save_name}_step_{self.step_id}.json"
            save_path = os.path.join(self.output_dir, save_name)
            self.save_to_file(save_path)
            print(f"GPU tracker stopped and data saved at {self.step_end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.data_log, f, indent=4)
        print(f"Data saved to: {filename}")
        self.data_log = list()

    def start_step_tracking(self, step_id):
        """Start GPU tracking for a specific step (requires running event loop)"""
        if not self.running:
            self.running = True
            self.data_log = []  # Reset data log for this step
            
            # Record start timestamp
            self.step_start_time = datetime.now()
            self.step_id = step_id
            
            try:
                self.task = asyncio.create_task(self._monitor_task())
                print(f"GPU tracker started for step {step_id} at {self.step_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            except RuntimeError:
                # No event loop running, create a background thread with its own loop
                self._start_background_tracking(step_id)

    def _start_background_tracking(self, step_id):
        """Start GPU tracking in a background thread with its own event loop"""
        import threading
        
        def run_background_loop():
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def background_task():
                self.task = asyncio.create_task(self._monitor_task())
                await self.task
            
            try:
                loop.run_until_complete(background_task())
            except asyncio.CancelledError:
                pass
            finally:
                loop.close()
        
        self.background_thread = threading.Thread(target=run_background_loop, daemon=True)
        self.background_thread.start()
        print(f"GPU tracker started in background thread for step {step_id} at {self.step_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def stop_step_tracking(self, step_id):
        """Stop GPU tracking and save with step_id suffix"""
        if self.running:
            self.running = False
            
            # Record end timestamp
            step_end_time = datetime.now()
            
            # Add timing metadata to the data log
            if hasattr(self, 'step_start_time'):
                timing_info = {
                    "step_timing": {
                        "step_id": step_id,
                        "start_time": self.step_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "end_time": step_end_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "duration_seconds": (step_end_time - self.step_start_time).total_seconds()
                    }
                }
                # Insert timing info at the beginning of the log
                self.data_log.insert(0, timing_info)
                
            # Generate filename with step_id suffix
            base_name, ext = os.path.splitext(self.filename)
            step_filename = f"{base_name}_step_{step_id}{ext}"
            step_filepath = os.path.join(self.output_dir, step_filename)
            self.save_to_file(step_filepath)
            print(f"GPU tracker stopped for step {step_id} at {step_end_time.strftime('%Y-%m-%d %H:%M:%S')}, data saved to: {step_filepath}")

    async def start_step_tracking_async(self, step_id):
        """Async version of start_step_tracking"""
        if not self.running:
            self.running = True
            self.data_log = []  # Reset data log for this step
            
            # Record start timestamp
            self.step_start_time = datetime.now()
            self.step_id = step_id
            
            self.task = asyncio.create_task(self._monitor_task())
            print(f"GPU tracker started asynchronously for step {step_id} at {self.step_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def stop_step_tracking_sync(self, step_id):
        """Synchronous version that can be called from non-async context"""
        if self.running:
            self.running = False
            
            # Record end timestamp
            step_end_time = datetime.now()
            
            # Wait for background thread if it exists
            if hasattr(self, 'background_thread') and self.background_thread.is_alive():
                # Give the background thread a moment to finish current collection
                import time
                time.sleep(0.1)
                
            # Add timing metadata to the data log
            if hasattr(self, 'step_start_time'):
                timing_info = {
                    "step_timing": {
                        "step_id": step_id,
                        "start_time": self.step_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "end_time": step_end_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "duration_seconds": (step_end_time - self.step_start_time).total_seconds()
                    }
                }
                # Insert timing info at the beginning of the log
                self.data_log.insert(0, timing_info)
                
            # Generate filename with step_id suffix
            base_name, ext = os.path.splitext(self.filename)
            step_filename = f"{base_name}_step_{step_id}{ext}"
            step_filepath = os.path.join(self.output_dir, step_filename)
            self.save_to_file(step_filepath)
            print(f"GPU tracker stopped for step {step_id} at {step_end_time.strftime('%Y-%m-%d %H:%M:%S')}, data saved to: {step_filepath}")


import threading
if __name__ == "__main__":
    
    tracker = GPUTracker(
        interval=0.2,
        output_dir="./",
        filename="gpu_usage.json"
    )
    print(tracker._get_gpu_info())

    loop = asyncio.new_event_loop()

    def run_async():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=run_async, daemon=True)
    thread.start()

    # 在事件循环中启动监控任务
    asyncio.run_coroutine_threadsafe(tracker.start(), loop)
    for i in range(5): 
        time.sleep(1)

    print("Stopping...")
    tracker.stop(filename='log.json')
    loop.call_soon_threadsafe(loop.stop)
    thread.join()