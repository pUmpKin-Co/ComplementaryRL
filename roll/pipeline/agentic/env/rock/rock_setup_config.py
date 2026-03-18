

model_service_start_script = """
#!/bin/bash

# Add /root/.local/bin to PATH in .bashrc for persistence
if ! grep -q '/root/.local/bin' ~/.bashrc; then
    echo 'export PATH="$PATH:/root/.local/bin"' >> ~/.bashrc
    echo "Added /root/.local/bin to PATH in .bashrc"
fi
# Also set for current session
export PATH="$PATH:/root/.local/bin"

# Check if rock model-service command is available
if ! /tmp/miniconda3/bin/rock model-service --help > /dev/null 2>&1; then
    echo "Installing rock model-service..."
    /tmp/miniconda3/bin/python -m ensurepip --upgrade
    /tmp/miniconda3/bin/pip install https://your-oss-bucket.com/jiaoliao-test/rock_rl-0.1.8.2a1-py3-none-any.whl alibabacloud_cr20181201 structlog swebench fastapi uvicorn --index-url http://your-pypi-server.com/pypi/simple --extra-index-url http://your-pypi-server.com/aliyun-pypi-simple/ --extra-index-url http://your-pypi-mirror.com/1/pypi/mdl --extra-index-url http://your-pypi-mirror.com/1/pypi/nebula --trusted-host your-pypi-server.com --trusted-host your-pypi-mirror.com
else
    echo "rock model-service already available, skipping installation"
fi

mkdir -p ~/.rock
touch ~/.rock/config.ini

# Stop existing service before starting
echo "Stopping existing rock model-service..."
rock model-service stop

# Start the service
echo "Starting rock model-service..."
rock model-service start
"""

model_service_stop_script = """
#!/bin/bash
rock model-service stop
"""

model_service_anti_call_llm_script = """
#!/bin/bash
rock model-service anti-call-llm --index {index} --response {response_payload}
"""

model_service_anti_call_llm_no_response_script = """
#!/bin/bash
rock model-service anti-call-llm --index {index}
"""

