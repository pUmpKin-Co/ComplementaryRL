

set_up_swebench_script= """
#!/bin/bash
chmod +x /run_tests.sh
ln -s /opt/miniconda3/envs/testbed /root/.venv
python -m pip install chardet -i https://mirrors.aliyun.com/pypi/simple/
"""


set_up_trainset_script= """
#!/bin/bash
ln -s /testbed/.venv /root/.venv
ln -s /testbed/.venv/bin/python /root/.local/bin/python
ln -s /testbed/.venv/bin/python /root/.local/bin/python3
ln -sf /testbed/.venv/bin/chardetect /usr/local/bin/chardet
mv /r2e_tests /root/r2e_tests
mv /testbed/run_tests.sh /root/run_tests.sh
ln -s /root/run_tests.sh /testbed/run_tests.sh
sed -i 's|\.venv/bin/python|/testbed/.venv/bin/python|g' /root/run_tests.sh
sed -i 's|r2e_tests|/root/r2e_tests|g' /root/run_tests.sh
chmod +x /root/run_tests.sh
"""