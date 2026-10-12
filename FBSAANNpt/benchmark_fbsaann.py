#!/usr/bin/env python3
"""Full configured tabular benchmark through the shared ANNpt wrapper.

Dataset and algorithm switches are evaluated in memory; repository wrappers are
never edited. Train with the configured epochs/repeats/batches. Test uses every
held-out row and reports sample-weighted accuracy, not an average of batches.
Preparation downloads the UCI HIGGS and New Thyroid inputs and prepends their
CSV headers under OUTPUT/raw (or --raw-data-dir), reusing completed files.
"""
import argparse
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import re
import sys
import subprocess
import tempfile
import time
import traceback
import types
from urllib.request import urlopen

DATASETS = ['iris', 'new-thyroid', 'breast-cancer-wisconsin', 'red-wine', 'titanic',
            'tabular-benchmark', 'banking-marketing', 'adult_income_dataset',
            'diabetes-readmission', 'blog-feedback', 'covertype', 'topquark', 'higgs']
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'benchmarks/2026-10-12-seed42'
LOCAL_DATASETS = {
    #https://archive.ics.uci.edu/dataset/280/higgs
    'higgs': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz',
        'download': 'HIGGS.csv.gz', 'raw': 'HIGGS.csv', 'csv': 'HIGGSwithHeader.csv',
        'header': ('class,lepton_pT,lepton_eta,lepton_phi,missing_energy_magnitude,missing_energy_phi,'
                   'jet1pt,jet1eta,jet1phi,jet1b,jet2pt,jet2eta,jet2phi,jet2b,'
                   'jet3pt,jet3eta,jet3phi,jet3b,jet4pt,jet4eta,jet4phi,jet4b,'
                   'm_jj,m_jjj,m_lv,m_jlv,m_bb,m_wbb,m_wwbb'),
    },
    #https://archive.ics.uci.edu/dataset/102/thyroid+disease
    'new-thyroid': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data',
        'download': 'new-thyroid.data', 'raw': 'new-thyroid.data', 'csv': 'new-thyroid.csv',
        'header': 'class,a,b,c,d,e',
    },
}


def copy_stream_atomic(source, destination, prefix=b'', expected_size=None):
    """Stream large files and publish only after the entire transfer succeeds."""
    destination = Path(destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=destination.parent,
                                         prefix=destination.name + '.', suffix='.part', delete=False) as output:
            temporary = Path(output.name)
            output.write(prefix)
            copied = 0
            last_progress = time.monotonic()
            while chunk := source.read(8 * 1024 * 1024):
                output.write(chunk)
                copied += len(chunk)
                if time.monotonic() - last_progress >= 30:
                    print(f'COPY {destination.name}: {copied / (1024 * 1024):.0f} MiB', flush=True)
                    last_progress = time.monotonic()
            if not copied or (expected_size is not None and copied != expected_size):
                raise OSError(f'Incomplete transfer for {destination}: received {copied} bytes, expected {expected_size}')
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_local_csv(dataset, directory):
    """Equivalent to echo HEADER | cat - RAW > CSV, preserving every raw byte."""
    specification = LOCAL_DATASETS[dataset]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    #Separate output directories may share a raw-data directory.
    import fcntl
    with (directory / (dataset + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        converted = directory / specification['csv']
        header = (specification['header'] + '\n').encode('ascii')
        if converted.is_file() and converted.stat().st_size > len(header):
            with converted.open('rb') as source:
                if source.read(len(header)) != header:
                    raise ValueError(f'Unexpected CSV header in {converted}; move it aside before retrying')
            print('REUSING ' + str(converted), flush=True)
            return converted
        raw = directory / specification['raw']
        if not raw.is_file() or not raw.stat().st_size:
            downloaded = directory / specification['download']
            if not downloaded.is_file() or not downloaded.stat().st_size:
                print('DOWNLOAD ' + specification['url'], flush=True)
                with urlopen(specification['url'], timeout=60) as response:
                    size = response.headers.get('Content-Length')
                    copy_stream_atomic(response, downloaded, expected_size=int(size) if size else None)
            if downloaded != raw:
                print('DECOMPRESS ' + str(downloaded), flush=True)
                with gzip.open(downloaded, 'rb') as source:
                    copy_stream_atomic(source, raw)
        print('ADD HEADER ' + str(converted), flush=True)
        with raw.open('rb') as source:
            copy_stream_atomic(source, converted, prefix=header, expected_size=raw.stat().st_size)
        return converted


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def setup(args):
    cache = args.output / 'cache'
    os.environ.update(HF_HOME=str(cache / 'huggingface'),
                      HF_DATASETS_CACHE=str(cache / 'datasets'),
                      MPLCONFIGDIR=str(cache / 'matplotlib'),
                      HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source = args.output / 'source'
    sys.path.insert(0, str(source))
    config = types.ModuleType('FBSAANNpt_FBSAANN_globalDefs')
    text = (source / (config.__name__ + '.py')).read_text()
    text, count = re.subn(r'(?m)^useFBSA = True.*$', 'useFBSA = ' + str(args.method == 'fbsa'), text)
    assert count == 1
    exec(compile(text, str(source / (config.__name__ + '.py')), 'exec'), vars(config))
    sys.modules[config.__name__] = config
    definitions = types.ModuleType('ANNpt_globalDefs')
    text = (source / 'ANNpt_globalDefs.py').read_text()
    text, count = re.subn(r"(?m)^\tdatasetName = 'titanic'.*$", '\tdatasetName = ' + repr(args.dataset), text)
    assert count == 1
    exec(compile(text, str(source / 'ANNpt_globalDefs.py'), 'exec'), vars(definitions))
    definitions.dataPathName = str(args.raw_data_dir)
    definitions.modelPathNameFull = str(args.output / 'runs' / f'{args.dataset}_{args.method}.pt')
    definitions.saveModelTrainContinuous = False
    definitions.printFBSAANNmodelProperties = False
    sys.modules[definitions.__name__] = definitions
    wrapper = importlib.import_module('ANNpt_main')
    return np, torch, definitions, wrapper


def settings(definitions):
    names = ['datasetName', 'datasetNameFull', 'learningRate', 'batchSize', 'numberOfLayers',
             'hiddenLayerSize', 'trainNumberOfEpochs', 'datasetRepeat', 'datasetRepeatSize',
             'datasetHasTestSplit', 'datasetTestSplitSize', 'dataloaderMaintainBatchSize',
             'dataloaderNumWorkers', 'fbsaBackwardWidthMultiplier', 'fbsaLearningRateFactors',
             'fbsaWeightDecay', 'fbsaWeightDecayV', 'fbsaWeightDecayVBar', 'fbsaMomentum',
             'fbsaImplementation', 'fbsaReferenceCommit', 'activationFunctionTypeForward',
             'activationFunctionTypeBackward', 'trainLocal', 'useFBSA']
    return {name: getattr(definitions, name) for name in names if hasattr(definitions, name)}


def prepare(args):
    np, torch, definitions, wrapper = setup(args)
    import datasets
    datasets.disable_progress_bars()
    started = time.time()
    source_adapter = None
    if args.dataset in LOCAL_DATASETS:
        converted = ensure_local_csv(args.dataset, args.raw_data_dir)
        source_adapter = {'download_url': LOCAL_DATASETS[args.dataset]['url'],
                          'file': str(converted),
                          'conversion': 'Prepend CSV header; preserve all raw data rows and their order'}
    if args.dataset == 'topquark':
        #The original repository loader requires trust_remote_code. Its default
        #configuration is raw; HF publishes the same generated rows as Parquet.
        #Only replace file acquisition, retaining every ANNpt preprocessing step.
        from huggingface_hub import HfApi
        repo = definitions.datasetNameFull
        api = HfApi()
        revision = api.dataset_info(repo, revision='refs/convert/parquet').sha
        files = api.list_repo_files(repo, repo_type='dataset', revision=revision)
        data_files = {split: [f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}'
                             for name in sorted(files)
                             if name.startswith(f'raw/top_quark_tagging-{split}-') and name.endswith('.parquet')]
                      for split in ['train', 'validation', 'test']}
        assert all(data_files.values())
        original_load = wrapper.ANNpt_data.load_dataset
        def load_published_parquet(path, *positional, **kwargs):
            if path == repo:
                return datasets.load_dataset('parquet', data_files=data_files)
            return original_load(path, *positional, **kwargs)
        wrapper.ANNpt_data.load_dataset = load_published_parquet
        source_adapter = {'repository': repo, 'revision': revision, 'files': data_files,
                          'reason': 'Read published raw-config Parquet instead of executing the dataset script'}
    dataset = wrapper.ANNpt_data.loadDataset()
    data_path = args.output / 'data' / args.dataset
    dataset.save_to_disk(str(data_path))
    info = {'dataset': args.dataset, 'seed': args.seed,
            'sizes': {name: len(split) for name, split in dataset.items()},
            'fingerprints': {name: split._fingerprint for name, split in dataset.items()},
            'features': wrapper.ANNpt_data.countNumberFeatures(dataset['train']),
            'classes': wrapper.ANNpt_data.countNumberClasses(dataset['train'])[0],
            'field_types': wrapper.ANNpt_data.createFieldTypeList(dataset['train']),
            'settings': settings(definitions), 'elapsed_seconds': time.time() - started,
            'source_adapter': source_adapter,
            'source_hashes': {p.name: sha_file(p) for p in (args.output / 'source').glob('*.py')}}
    atomic_json(data_path / 'benchmark.json', info)
    print('PREPARED ' + json.dumps(info), flush=True)


class Progress:
    """tqdm-compatible observer; does not change data iteration or training."""
    def __init__(self, iterable, status_path, **kwargs):
        self.iterable = iterable
        self.total = len(iterable)
        self.description = ''
        self.position = 0
        self.last_write = 0
        self.status_path = status_path
    def __iter__(self):
        for batch in self.iterable:
            yield batch
            self.position += 1
    def set_description(self, description):
        self.description = description
    def set_postfix(self, **kwargs):
        now = time.time()
        if now - self.last_write > 30 or self.position + 1 == self.total:
            data = {key: float(value) for key, value in kwargs.items()}
            data.update(epoch=self.description, batches_total=self.total, time=now)
            atomic_json(self.status_path, data)
            print('PROGRESS ' + json.dumps(data), flush=True)
            self.last_write = now


def run(args):
    np, torch, definitions, wrapper = setup(args)
    from datasets import load_from_disk
    data_path = args.output / 'data' / args.dataset
    info = json.loads((data_path / 'benchmark.json').read_text())
    for name, expected in info['source_hashes'].items():
        assert sha_file(args.output / 'source' / name) == expected
    dataset = load_from_disk(str(data_path))
    data_module = wrapper.ANNpt_data
    data_module._cachedFeatureTypeList = info['field_types'][:-1]
    data_module._cachedClassFieldType = info['field_types'][-1]
    torch.manual_seed(args.seed)
    model = wrapper.ANNpt_algorithm.createModel(dataset['train'])
    initial_hash = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.layersLinearF.parameters())).hexdigest()
    #Reset AFTER model creation: auxiliary FBSA initialisation must not shift the
    #shared shuffled minibatch order relative to backprop.
    random.seed(args.seed + 1000)
    np.random.seed(args.seed + 1000)
    torch.manual_seed(args.seed + 1000)
    status_path = args.output / 'runs' / f'{args.dataset}_{args.method}.progress.json'
    wrapper.tqdm = lambda iterable, **kwargs: Progress(iterable, status_path, **kwargs)
    schedule = hashlib.sha256()
    original_train = wrapper.trainBatch
    steps = 0
    def observed_train(batch_index, batch, *rest, **kwargs):
        nonlocal steps
        for array in batch:
            schedule.update(array.numpy().tobytes())
        loss, accuracy = original_train(batch_index, batch, *rest, **kwargs)
        if not np.isfinite(loss):
            raise FloatingPointError(f'Non-finite training loss at update {steps}')
        steps += 1
        return loss, accuracy
    wrapper.trainBatch = observed_train
    started = time.time()
    wrapper.processDataset(True, dataset['train'], model)
    trained = time.time()
    #The existing loader defaults to drop_last=True, including during testing.
    #Keep its training behavior, but include the last partial test batch.
    data_module.dataloaderMaintainBatchSize = False
    data_module.dataloaderShuffle = False
    totals = dict(correct=0, rows=0, loss_sum=0.0)
    original_test = wrapper.testBatch
    def observed_test(batch_index, batch, *rest, **kwargs):
        loss, accuracy = original_test(batch_index, batch, *rest, **kwargs)
        if not np.isfinite(loss):
            raise FloatingPointError('Non-finite final test loss')
        labels = batch[1]
        predictions = model.Ztrace[-1].argmax(1).cpu()
        totals['correct'] += (predictions == labels).sum().item()
        totals['rows'] += len(labels)
        totals['loss_sum'] += loss * len(labels)
        return loss, accuracy
    wrapper.testBatch = observed_test
    wrapper.processDataset(False, dataset['test'], model)
    assert totals['rows'] == len(dataset['test'])
    repeat = definitions.datasetRepeatSize if definitions.datasetRepeat else 1
    result = {'dataset': args.dataset, 'method': args.method, 'seed': args.seed,
              'status': 'completed', 'test_accuracy': totals['correct'] / totals['rows'],
              'test_loss': totals['loss_sum'] / totals['rows'], 'test_rows_evaluated': totals['rows'],
              'test_unique_rows': len(dataset['test']) // repeat,
              'training_steps': steps, 'training_seconds': trained - started,
              'test_seconds': time.time() - trained, 'settings': settings(definitions),
              'data_fingerprints': info['fingerprints'], 'initial_forward_sha256': initial_hash,
              'minibatch_schedule_sha256': schedule.hexdigest(), 'device': str(definitions.device),
              'torch_version': torch.__version__, 'source_hashes': info['source_hashes']}
    atomic_json(args.output / 'runs' / f'{args.dataset}_{args.method}.json', result)
    print('RESULT ' + json.dumps(result), flush=True)


def report(args):
    rows = []
    verified = []
    implementations = set()
    lines = ['FBSA / BACKPROP BENCHMARK', '',
             'Configured per-dataset architecture, learning rate, repeats and epochs; paired initialization and minibatches.',
             'Backprop: existing Adam optimizer. FBSA: local SAL updates with SGD; version/settings listed below.',
             'All held-out rows evaluated, including final partial batches. No tuning on the test set.', '',
             'Dataset | Backprop test accuracy (%) | FBSA test accuracy (%) | FBSA minus backprop (pp)']
    for dataset in DATASETS:
        entry = {'dataset': dataset}
        for method in ['backprop', 'fbsa']:
            path = args.output / 'runs' / f'{dataset}_{method}.json'
            entry[method] = json.loads(path.read_text()) if path.exists() else None
        if entry['fbsa'] and entry['fbsa'].get('status') == 'completed':
            config = entry['fbsa']['settings']
            version = config.get('fbsaImplementation', 'preprint-based SAL (legacy)')
            decay = config['fbsaWeightDecay']
            decay_v = config.get('fbsaWeightDecayV')
            decay_v = decay if decay_v is None else decay_v
            decay_vbar = config.get('fbsaWeightDecayVBar')
            decay_vbar = decay_v if decay_vbar is None else decay_vbar
            implementations.add(
                f"{version}; reference={config.get('fbsaReferenceCommit', '2024 preprint')}; "
                f"momentum={config.get('fbsaMomentum', 0.0)}; "
                f"decay W/Wprime={decay}, V={decay_v}, Vbar={decay_vbar}; "
                f"gradient factors={json.dumps(config['fbsaLearningRateFactors'], sort_keys=True)}")
        if all(entry[m] and entry[m].get('status') == 'completed' for m in ['backprop', 'fbsa']):
            for key in ['data_fingerprints', 'initial_forward_sha256', 'minibatch_schedule_sha256', 'training_steps', 'source_hashes']:
                assert entry['backprop'][key] == entry['fbsa'][key], (dataset, key)
            info = json.loads((args.output / 'data' / dataset / 'benchmark.json').read_text())
            for method in ['backprop', 'fbsa']:
                result = entry[method]
                config = result['settings']
                batches, remainder = divmod(info['sizes']['train'], config['batchSize'])
                if remainder and not config['dataloaderMaintainBatchSize']:
                    batches += 1
                assert result['training_steps'] == batches * config['trainNumberOfEpochs'], dataset
                assert result['test_rows_evaluated'] == info['sizes']['test'], dataset
                assert result['source_hashes'] == info['source_hashes'], dataset
            verified.append(dataset)
            a, b = [100 * entry[m]['test_accuracy'] for m in ['backprop', 'fbsa']]
            lines.append(f'{dataset} | {a:.4f} | {b:.4f} | {b-a:+.4f}')
        else:
            def fmt(item):
                if item is None: return 'pending'
                if item['status'] != 'completed': return item['status']
                return f"{100*item['test_accuracy']:.4f}"
            lines.append(f"{dataset} | {fmt(entry['backprop'])} | {fmt(entry['fbsa'])} | —")
        rows.append(entry)
    if implementations:
        lines.extend(['', 'FBSA implementations in completed runs:'] + sorted(implementations))
    lines.extend(['', f'Verified complete paired runs: {len(verified)}/{len(DATASETS)} datasets.',
                  'Checks: identical prepared data, initial forward weights, minibatch sequence and source code;',
                  'configured training-update counts and complete test-set coverage for both methods.', '',
                  'Interpretation: one paired seed, without hyperparameter tuning or confidence intervals.',
                  'These are repository-recipe results, not a reproduction of the paper\'s tuned experiments.',
                  'Original preprocessing retains Iris Id and Breast Cancer id/Unnamed: 32 columns;',
                  'Blog Feedback is evaluated as count classification. Small datasets retain configured repetition.',
                  'Top Quark uses the publisher\'s converted raw Parquet data; all labels were checked against the original files.',
                  'Jobs share one GPU, so recorded elapsed times are not isolated speed measurements.', '',
                  'Artifacts: results.json (full metrics), protocol.json (source hashes and execution policy),',
                  'environment.json, topquark_source_audit.json, data/*/benchmark.json and logs/*.log.'])
    atomic_json(args.output / 'results.json', rows)
    atomic_json(args.output / 'verification.json', {'verified_dataset_pairs': verified,
                                                  'expected_dataset_pairs': len(DATASETS),
                                                  'complete': len(verified) == len(DATASETS)})
    (args.output / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def suite(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    manifest = {'seed': args.seed, 'datasets': DATASETS, 'methods': ['backprop', 'fbsa'],
                'raw_data_dir': str(args.raw_data_dir),
                'threads_per_job': args.threads, 'concurrent_datasets': args.jobs,
                'source_sha256': {p.name: sha_file(p) for p in (args.output / 'source').glob('*.py')},
                'protocol': 'Repository defaults; full configured epochs and repeats; paired initial forward weights and minibatch order; all test rows; no validation tuning.',
                'evaluation_override': 'Disable drop_last and shuffling only for test; aggregate correct predictions over all test rows.',
                'runner_sha256': sha_file(Path(__file__))}
    atomic_json(args.output / 'protocol.json', manifest)
    def execute(action, dataset, method='backprop'):
        log = args.output / 'logs' / f'{dataset}_{action if action == "prepare" else method}.log'
        command = [sys.executable, '-B', str(Path(__file__).resolve()), action,
                   '--dataset', dataset, '--method', method, '--output', str(args.output),
                   '--seed', str(args.seed), '--threads', str(args.threads),
                   '--raw-data-dir', str(args.raw_data_dir)]
        print('START ' + ' '.join(command[3:]), flush=True)
        with log.open('w') as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        print(f'END {dataset} {action} {method}: {result.returncode}', flush=True)
        return result.returncode
    def dataset_pair(dataset):
        metadata = args.output / 'data' / dataset / 'benchmark.json'
        if not metadata.exists():
            if execute('prepare', dataset):
                for method in ['backprop', 'fbsa']:
                    atomic_json(args.output / 'runs' / f'{dataset}_{method}.json',
                                {'dataset': dataset, 'method': method, 'status': 'data_failed',
                                 'error': f'See logs/{dataset}_prepare.log'})
                return
        for method in ['backprop', 'fbsa']:
            result = args.output / 'runs' / f'{dataset}_{method}.json'
            if result.exists() and (not args.retry_failed or json.loads(result.read_text())['status'] == 'completed'):
                continue
            execute('run', dataset, method)
    #Start the largest source while a second worker clears the smaller datasets.
    order = ['higgs'] + [name for name in DATASETS if name != 'higgs']
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(dataset_pair, name): name for name in order}
        for future in as_completed(futures):
            future.result()
            report(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'report', 'suite'])
    parser.add_argument('--dataset', choices=DATASETS, default='iris')
    parser.add_argument('--method', choices=['backprop', 'fbsa'], default='backprop')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--raw-data-dir', type=Path,
                        help='Download/reuse HIGGS and New Thyroid files here (default: OUTPUT/raw)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.raw_data_dir = (args.raw_data_dir or args.output / 'raw').resolve()
    try:
        if args.action in ('prepare', 'run'):
            #An independently started preparation/run can safely overlap the
            #suite: a second request waits, then reuses the completed artifact.
            import fcntl
            identity = args.dataset + ('_' + args.method if args.action == 'run' else '_prepare')
            with (args.output / 'runs' / (identity + '.lock')).open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                result = (args.output / 'data' / args.dataset / 'benchmark.json' if args.action == 'prepare'
                          else args.output / 'runs' / (identity + '.json'))
                if result.exists():
                    previous = json.loads(result.read_text())
                    if args.action == 'prepare' or previous.get('status') == 'completed':
                        assert previous['seed'] == args.seed, 'Use a separate output directory for a different seed'
                        print('REUSING ' + str(result), flush=True)
                        return
                globals()[args.action](args)
        else:
            globals()[args.action](args)
    except Exception as error:
        traceback.print_exc()
        if args.action == 'run':
            atomic_json(args.output / 'runs' / f'{args.dataset}_{args.method}.json',
                        {'dataset': args.dataset, 'method': args.method, 'status': 'failed',
                         'error': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
