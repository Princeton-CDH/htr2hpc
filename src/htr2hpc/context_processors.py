from os import cpu_count, getloadavg
from socket import gethostname

import psutil


# Get host and CPU count once on load, since changing these requires a reboot

#: number of logical CPUs in the system
CPU_COUNT = cpu_count()
#: hostname for the local VM
HOSTNAME = gethostname()

# os.uname potentially useful information (release, version, hardware)


def vm_status(request):
    """Custom context processor to return information about VM
    configuration and resources."""

    # load average returns average load over last 1, 5, and 15 minutes
    load_average = getloadavg()
    return {
        "cpu_count": CPU_COUNT,
        "load_average": {
            "1": load_average[0],
            "5": load_average[1],
            "15": load_average[2],
        },
        # use to track if we're switching between VMs or if nginx+ is sticky
        "hostname": HOSTNAME,
        # system memory usage: total, available, percent, etc
        "system_memory": psutil.virtual_memory(),
    }
