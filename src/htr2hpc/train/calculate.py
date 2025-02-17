import re
import math
import datetime

def slurm_get_max_acc(slurm_output, training_mode):
    """Return a tuple of (epoch #, accuracy) for the epoch with highest accuracy"""
    if training_mode == "Segment":
        re_acc = 'stage ([\d]+).+\n[^i]*val_mean_iu:\s+\n\s+([\d.]+)'
    else:
        re_acc = 'stage ([\d]+).+\n.+(\d.\d\d\d)\s*\d/10'
    
    accuracies = re.findall(re_acc, slurm_output)
    accuracies = [(int(i[0]), float(i[1])) for i in accuracies]
    
    return max(accuracies, key=lambda x:x[1])
    
    
def slurm_get_avg_epoch(slurm_output):
    """Return average epoch duration, in seconds"""
    epoch_times = re.findall('(\d:\d\d:\d\d) •', slurm_output)
    epoch_times = [datetime.datetime.strptime(t, '%H:%M:%S') for t in epoch_times]
    epoch_times = [datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).seconds for t in epoch_times]
    
    return math.ceil(sum(epoch_times) / len(epoch_times))
    

def stats_get_max_cpu(job_stats):
    """Return max CPU usage"""
    mem_usage = re.findall('\(([\d.]+)([\w]+)\/[\d.]+[\w]+ per core', job_stats)
    if not mem_usage:
        return None
    gb_used = float(mem_usage[0][0])/1000 if mem_usage[0][1] == 'MB' else float(mem_usage[0][0])
    
    return gb_used

