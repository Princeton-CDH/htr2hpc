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
    
    if accuracies:
        return max(accuracies, key=lambda x:x[1])
        
        
def slurm_count_epoch(slurm_output):
    """Return count of epochs"""
    epoch_times = re.findall('(\d:\d\d:\d\d) •', slurm_output)
    
    if epoch_times:
        return len(epoch_times)
    
    
def slurm_get_avg_epoch(slurm_output):
    """Return average epoch duration, in seconds"""
    epoch_times = re.findall('(\d:\d\d:\d\d) •', slurm_output)
    
    if epoch_times:
        epoch_times = [datetime.datetime.strptime(t, '%H:%M:%S') for t in epoch_times]
        epoch_times = [datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).seconds for t in epoch_times]
        return math.ceil(sum(epoch_times) / len(epoch_times))
    

def stats_get_max_cpu(job_stats):
    """Return max CPU usage"""
    mem_usage = re.findall('\(([\d.]+)([\w]+)\/[\d.]+[\w]+ per core', job_stats)
    
    if mem_usage:
        gb_used = float(mem_usage[0][0])/1000 if mem_usage[0][1] == 'MB' else float(mem_usage[0][0])
        return gb_used


def calc_full_duration(slurm_output, job_stats):
    """Given a preliminary slurm job output, return duration estimate:""" 
    """Setup time, plus 50 times the average epoch plus 10% for wiggle room."""
    """Assumes the train task will take 50 epochs."""
    job_duration = re.findall('Run Time: (\d+:\d\d:\d\d)', job_stats)
    epoch_avg = slurm_get_avg_epoch(slurm_output)
    epoch_count = slurm_count_epoch(slurm_output)
    
    if epoch_avg and job_duration:
        t = datetime.datetime.strptime(job_duration[0], '%H:%M:%S')
        job_duration = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).seconds
        setup_time = job_duration - ( epoch_avg * epoch_count )
        
        return datetime.timedelta(seconds=(setup_time + ( epoch_avg * 50 * 1.1 )))
    

def calc_cpu_mem(job_stats):
    """Given a preliminary job_stats output, return recommended mem per cpu."""
    gb_used = stats_get_max_cpu(job_stats)
    
    if gb_used:
        return f"{math.ceil(gb_used + 0.3)}G"
        