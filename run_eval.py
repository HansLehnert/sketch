"""Script for launching and logging data of different sketch implementations"""

import subprocess
import collections
import re
import csv
import argparse
import sys
import json
import os

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('runs', type=int)
parser.add_argument('--cuda-metrics', action='store_true', dest='cuda_metrics')
parser.add_argument(
    '--program-type', nargs='*', default='auto', dest='program_type')
parser.add_argument(
    '--data-tags', nargs='*', default=['default'], dest='dataset_tags')
args = parser.parse_args(sys.argv[1:])

# Detect CUDA and AVX features
try:
    subprocess.call('nvcc', stderr=subprocess.DEVNULL)
    use_cuda = True
except OSError:
    use_cuda = False

lscpu = subprocess.run(
    'lscpu', stdout=subprocess.PIPE, universal_newlines=True)
if 'avx2' in lscpu.stdout:
    use_avx = True
else:
    use_avx = False

if args.program_type == 'auto':
    program_type = ['default']
    if use_avx:
        program_type.append('avx')
    if use_cuda:
        program_type.append('cuda')
else:
    program_type = args.program_type

# Programs to check
# TODO: Move to external file
programs = {
    'bin/release/sketch': ['default'],
    'bin/release/sketch_avx_pipelined': ['avx'],
    'bin/release/sketch_cu': ['cuda']
}

# Create output directory
if not os.path.isdir('./out/'):
    os.mkdir('./out/')

# Database description
with open('datasets.json') as dataset_file:
    datasets = json.load(dataset_file)

n_runs = args.runs

for program_name in programs:
    if not any(x in programs[program_name] for x in program_type):
        print('Skipping program {}'.format(program_name))
        continue

    print('Running program {}'.format(program_name))

    runs = collections.OrderedDict()

    for dataset_name in datasets:
        dataset = datasets[dataset_name]

        if not any(x in dataset['tags'] for x in args.dataset_tags):
            continue

        print('Using dataset {}'.format(dataset_name), end='', flush=True)

        min_length = dataset['first_length']
        max_length = min_length + len(dataset['thresholds']) - 1

        base_command = [
            program_name,
            dataset['test_file'],
            dataset['control_file'],
            str(min_length),
            str(max_length),
            *[str(x) for x in dataset['thresholds']]]

        for i in range(n_runs):
            if 'cuda' in programs[program_name]:
                command = ['nvprof'] + base_command
            else:
                command = base_command

            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                universal_newlines=True)

            metrics = collections.OrderedDict()

            metrics['min-length'] = min_length
            metrics['max-length'] = max_length

            # Find transfer and hash times (for CUDA)
            if 'cuda' in programs[program_name]:
                transfer_times = re.findall(
                    r'(\S*)\s+(?:\S+\s+){4}\[CUDA memcpy (.*)\]',
                    result.stderr
                )

                for time, name in transfer_times:
                    metric_name = '{}-transfer-time'.format(name)
                    metrics[metric_name] = time

                kernel_times = re.findall(
                    r'([0-9.]+[muns.]+)\s+(?:\S+\s+){4}(?P<kernel>[^(\s]+)\(',
                    result.stderr
                )

                for time, kernel in kernel_times:
                    metrics['{}-kernel-runtime'.format(kernel)] = time

            # Find time reports in process output
            runtimes = re.findall(
                '^(.*) time: ([0-9.]*)', result.stderr, re.MULTILINE)

            for name, time in runtimes:
                metric_name = '{}-runtime [s]'.format(name.lower())
                metrics[metric_name] = time

            # Find heavy-hitters
            heavy_hitters = re.search(
                'Heavy-hitters \(total\): ([0-9]*)', result.stderr)

            if heavy_hitters is not None:
                metrics['heavy-hitters'] = heavy_hitters.group(1)

            # Add metrics into run log
            for x in metrics:
                metric_name = '{}_{}'.format(dataset_name, x)
                if metric_name not in runs:
                    runs[metric_name] = ['-'] * i + [metrics[x]]
                else:
                    runs[metric_name].append(metrics[x])

            # Fill unreported measures with invalid value
            for x in runs:
                if dataset_name in x and len(runs[x]) <= i:
                    runs[x].append('-')

            print('.', end='', flush=True)

        # Other CUDA metrics
        if 'cuda' in programs[program_name] and args.cuda_metrics:
            # Occupancy
            print('[cuda metrics]', end='', flush=True)
            command = ['nvprof', '--metrics', 'achieved_occupancy']
            command += base_command

            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                universal_newlines=True)

            occupancy = re.search(
                'Achieved Occupancy\s+([0-9.]*)\s+([0-9.]*)\s+([0-9.]*)',
                result.stderr)

            if occupancy is not None:
                runs['{}_occupancy-min'.format(dataset_name)] = (
                    [occupancy.group(1)] + ['-'] * (n_runs - 1))
                runs['{}_occupancy-max'.format(dataset_name)] = (
                    [occupancy.group(2)] + ['-'] * (n_runs - 1))
                runs['{}_occupancy-avg'.format(dataset_name)] = (
                    [occupancy.group(3)] + ['-'] * (n_runs - 1))

            # Power
            power_filename = 'out/{}_{}_power.csv'.format(
                program_name.split('/')[-1], dataset_name)
            power_log = open(power_filename, 'w')

            nvidia_smi = subprocess.Popen(
                ['nvidia-smi',
                    '-i', '0',
                    '--format=csv',
                    '--query-gpu=power.draw',
                    '-lms', '50'],
                stdout=power_log,
                stderr=subprocess.DEVNULL,
                universal_newlines=True)

            subprocess.run(
                base_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            nvidia_smi.terminate()

            power_log.close()

        print('')

    output_filename = 'out/result_{}.csv'.format(program_name.split('/')[-1])
    print('Writing results to {}'.format(output_filename))
    with open(output_filename, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['run'] + list(range(1, n_runs + 1)))

        for i in runs:
            writer.writerow([i] + runs[i])
