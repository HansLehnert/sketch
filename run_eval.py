"""Script for launching and logging data of different sketch implementations"""

import subprocess
import collections
import re
import csv
import argparse
import sys

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('runs', type=int)
parser.add_argument('type', nargs='*')
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

# Programs to check
programs = {}

programs['bin/sketch'] = {'default'}

if use_avx:
    programs['bin/sketch_avx_pipelined'] = {'avx'}

if use_cuda:
    programs['bin/sketch_cu'] = {'cuda'}

# Database description
datasets = {
    'esbrr': {
        'test_file': 'data/test.fasta',
        'control_file': 'data/control.fasta',
        'first_length': 10,
        'thresholds': [365, 308, 257, 161, 150, 145, 145, 145, 145, 145, 145],
    },

    'e2f1': {
        'test_file': 'data/E2f1-200-mm9-only.fasta',
        'control_file': 'data/control.fasta',
        'first_length': 10,
        'thresholds': [351, 294, 245, 153, 143, 138, 138, 138, 138, 138, 138]
    },

    'tcfcpl1': {
        'test_file': 'data/Tcfcpl1-200-mm9-only.fasta',
        'control_file': 'data/control.fasta',
        'first_length': 10,
        'thresholds': [457, 382, 319, 200, 186, 180, 180, 180, 180, 180, 180]
    },

    'ctcf': {
        'test_file': 'data/ctcf-200-mm9-only.fasta',
        'control_file': 'data/control.fasta',
        'first_length': 10,
        'thresholds': [673, 563, 470, 294, 274, 265, 265, 265, 265, 265, 265]
    },

    'wgEncode': {
        'test_file':
            'data/wgEncodeOpenChromChipHepg2Pol2Pk_peak-200-hg19-only.fasta',
        'control_file':
            'data/wgEncodeHepg2Pol2Pk_peak-200-hg9-control.fasta',
        'first_length': 10,
        'thresholds': [
            13400, 11700, 10950, 9600, 9100, 8500, 8000, 7500, 7400, 6000,
            4500, 3900, 3800, 3050, 2700, 2600, 2500, 2300, 2300]
    }
}

n_runs = args.runs

for program_name in programs:
    if (args.type is not None and
            not any(x in programs[program_name] for x in args.type)):
        print('Skipping program {}'.format(program_name))
        continue

    print('Running program {}'.format(program_name))

    runs = collections.OrderedDict()

    for dataset_name in datasets:
        print('Using dataset {}'.format(dataset_name), end='', flush=True)

        dataset = datasets[dataset_name]

        for i in range(n_runs):
            min_length = dataset['first_length']
            max_length = min_length + len(dataset['thresholds']) - 1

            command = [program_name,
                dataset['test_file'],
                dataset['control_file'],
                str(min_length),
                str(max_length),
                *[str(x) for x in dataset['thresholds']]]

            if 'cuda' in programs[program_name]:
                command = ['nvprof'] + command

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
                    '(\S*)\s+(?:\S+\s+){4}\[CUDA memcpy (.*)\]', result.stderr)

                for time, name in transfer_times:
                    metric_name = '{}-transfer-time'.format(name)
                    metrics[metric_name] = time

                hash_time = re.search(
                    '(\S*)\s+(?:\S+\s+){4}void hashH3', result.stderr)

                if hash_time is not None:
                    metrics['hash-runtime'] = hash_time.group(1)

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
        print('')

    output_filename = 'out/result_{}.csv'.format(program_name.split('/')[-1])
    print('Writing results to {}'.format(output_filename))
    with open(output_filename, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['run'] + list(runs.keys()))

        for i in range(n_runs):
            writer.writerow([i] + [runs[x][i] for x in runs])
