"""Script for launching and logging data of different sketch implementations"""

import subprocess
import collections
import re
import csv

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
programs = []

# programs.append('bin/sketch')

# if use_avx:
#     programs.append('bin/sketch_avx_pipelined')

if use_cuda:
    programs.append('bin/sketch_cu')

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


n_runs = 10

for program_name in programs:
    print('Running program {}'.format(program_name))

    runs = collections.OrderedDict()

    for dataset_name in datasets:
        print('Using dataset {}'.format(dataset_name), end='', flush=True)

        dataset = datasets[dataset_name]

        for i in range(n_runs):
            min_length = dataset['first_length']
            max_length = min_length + len(dataset['thresholds']) - 1

            result = subprocess.run(
                [program_name,
                    dataset['test_file'],
                    dataset['control_file'],
                    str(min_length),
                    str(max_length),
                    *[str(x) for x in dataset['thresholds']]],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                universal_newlines=True)

            measurements = collections.OrderedDict()

            measurements['min-length'] = min_length
            measurements['max-length'] = max_length

            measurements['test-runtime [s]'] = re.search(
                'Test time: ([0-9.]*)', result.stderr)
            measurements['control-runtime [s]'] = re.search(
                'Control time: ([0-9.]*)', result.stderr)
            measurements['total-runtime [s]'] = re.search(
                '(?:Total|Execution) time: ([0-9.]*)', result.stderr)
            measurements['heavy-hitters'] = re.search(
                'Heavy-hitters \(total\): ([0-9]*)', result.stderr)

            for x in measurements:
                if measurements[x] is not None:
                    try:
                        value = float(measurements[x].group(1))
                    except AttributeError:
                        value = measurements[x]

                    measurement_name = '{}_{}'.format(dataset_name, x)
                    if measurement_name not in runs:
                        runs[measurement_name] = [value]
                    else:
                        runs[measurement_name].append(value)

            print('.', end='', flush=True)
            if i == n_runs - 1:
                print('')
                print(result.stderr, end='', flush=True)
        print('')

    output_filename = 'out/result_{}.csv'.format(program_name.split('/')[-1])
    print('Writing results to {}'.format(output_filename))
    with open(output_filename, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['run'] + list(runs.keys()))

        for i in range(n_runs):
            writer.writerow([i] + [runs[x][i] for x in runs])
