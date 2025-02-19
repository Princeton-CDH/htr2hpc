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
    """Given a preliminary slurm job output, return duration estimate:
    Setup time, plus N times the average epoch plus 10% for wiggle room.
    Assumes the train task will take 50 epochs. N = 50 - count of completed
    epochs from prelim train task.
    """
    job_duration = re.findall('Run Time: (\d+:\d\d:\d\d)', job_stats)
    epoch_avg = slurm_get_avg_epoch(slurm_output)
    epoch_count = slurm_count_epoch(slurm_output)
    
    if job_duration:
        t = datetime.datetime.strptime(job_duration[0], '%H:%M:%S')
        job_duration = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).seconds
    
        if epoch_avg:
            setup_time = job_duration - ( epoch_avg * epoch_count )
            
            epoch_request = 50 - epoch_count
            # if prelim train task already came close to 50 epochs or overshot it, run second train task
            # so that --lag 10 is immediately active (epoch_request -> --min-epochs 5) and so that the 
            # estimated job time request allows room for 15 more epochs.
            epoch_time_est = 15 if epoch_request < 11 else epoch_time_est
            epoch_request = 5 if epoch_request < 11 else epoch_request
            
            return epoch_request, datetime.timedelta(minutes=math.ceil((setup_time + ( epoch_avg * epoch_time_est * 1.1 )) / 60))
    
        elif job_duration > datetime.timedelta(minutes=14):
            # if epoch_avg returns as None (no epochs completed during the first train task),
            # but job did not error out early, assume that more time is needed per epoch.
            # assume 15min per epoch and 15min setup time.
            # this means max train time should be ~14 hrs.
            # note the calc_full_duration function should run only when the first train job does not crash.
            epoch_request = 50
            
            return epoch_request, datetime.timedelta(minutes=( 15 * 51 * 1.1 ))
    

def calc_cpu_mem(job_stats):
    """Given a preliminary job_stats output, return recommended mem per cpu."""
    gb_used = stats_get_max_cpu(job_stats)
    
    if gb_used:
        return f"{math.ceil(gb_used + 0.3)}G"
        

def estimate_duration(training_data_size, training_mode):
    """Use files in input data dir to come up with estimate of prelim train duration."""
    
    if training_mode == "Segment":
        job_duration = datetime.timedelta(minutes=5) if training_data_size < 20000000 else datetime.timedelta(minutes=15)
    else:
        job_duration = datetime.timedelta(minutes=5) if training_data_size < 50000000 else datetime.timedelta(minutes=15)
        
    return job_duration
    

def estimate_cpu_mem(training_data_size, training_mode):
    """Use files in input data dir to come up with estimate of prelim mem per cpu."""
    
    if training_mode == "Segment":
        if training_data_size < 10000000:
            mem_per_cpu = "1G"
        elif training_data_size < 20000000:
            mem_per_cpu = "2G"
        elif training_data_size < 40000000:
            mem_per_cpu = "3G"
        elif training_data_size < 120000000:
            mem_per_cpu = "4G"
        elif training_data_size < 200000000:
            mem_per_cpu = "5G"
        else:
            mem_per_cpu = f"{6 + (training_data_size - 200000000) // 100000000}G"
    else:
        mem_per_cpu = "1G" if training_data_size < 50000000 else "2G"
        
    return mem_per_cpu

        