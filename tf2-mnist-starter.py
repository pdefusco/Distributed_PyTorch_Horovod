import os
import cdsw

workers = cdsw.launch_workers(n=2, cpu=1, memory=2,
                              code="import time; time.sleep(365*3600*10)")
print('Starting workers ...')
worker_ids = [worker["id"] for worker in workers]
running_workers = cdsw.await_workers(worker_ids,
                              wait_for_completion=False,
                              timeout_seconds=120)
worker_ips = [worker["ip_address"] for worker in \
                              running_workers["workers"]]
print('Workers:', worker_ips)
hosts_str = ",".join([worker_ip+":1" for worker_ip in worker_ips])
cmd = "horovodrun -np {} -H {} -p 2222 python3 tf2-mnist-hvd.py 2>&1".format(
                              len(worker_ips),
                              hosts_str)
print('Preparing to run: ' + cmd)
os.system(cmd)
