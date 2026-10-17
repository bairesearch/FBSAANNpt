#!/usr/bin/env python3
"""Full configured tabular benchmark through the shared ANNpt wrapper.

Dataset and algorithm switches are evaluated in memory; repository wrappers are
never edited. Train with configured epochs or the production FBSA convergence
controller, for both FBSA and backprop. Test uses every
held-out row and reports sample-weighted accuracy, not an average of batches.
Training and evaluation use DATASET_BATCH_SIZES. The initial learning rate
defaults to the dataset setting and can be overridden with --learning-rate.
Preparation downloads the UCI HIGGS and New Thyroid inputs and prepends their
CSV headers under OUTPUT/raw (or --raw-data-dir), reusing completed files.
"""
import argparse
import gzip
import hashlib
import importlib
import json
import math
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
DATASET_BATCH_SIZES = {
    'iris': 128,
    'new-thyroid': 128,
    'breast-cancer-wisconsin': 128,
    'red-wine': 128,
    'titanic': 128,
    'tabular-benchmark': 128,
    'banking-marketing': 128,
    'adult_income_dataset': 128,
    'diabetes-readmission': 128,
    'blog-feedback': 128,
    'covertype': 128,	#1024
    'topquark': 128,	#1024
    'higgs': 8192,
}
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'benchmarks/2026-10-12-seed42'
CONVERGENCE_SETTINGS = [
    'trainSetLossOptimisation', 'fbsaValidationSplitSize', 'fbsaValidationSplitSeed',
    'fbsaEvaluateEveryIterations', 'fbsaMinimumTrainingIterations',
    'fbsaTrainingPatience', 'fbsaTrainingMinDelta', 'fbsaTrainingRelativeMinDelta',
    'fbsaTrainingLossGoal', 'fbsaValidationPatience', 'fbsaValidationMinDelta',
    'fbsaValidationRelativeMinDelta', 'fbsaTrainingLearningRateFactor',
    'fbsaTrainingLearningRateReductions', 'fbsaTrainingMaxNumericalRecoveries',
]
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


def load_algorithm_config(args):
    """Apply optional switches before the source's conditional settings execute."""
    path = args.output / 'source' / 'FBSAANNpt_FBSAANN_globalDefs.py'
    text = path.read_text()
    overrides = {'useFBSA': args.method == 'fbsa',
                 'fbsaOptimiseTrainingIterations': getattr(args, 'optimise_training_iterations', None),
                 'trainSetLossOptimisation': getattr(args, 'train_set_loss_optimisation', None)}
    for name, value in overrides.items():
        if value is None:
            continue
        text, count = re.subn(r'(?m)^([ \t]*)' + name + r' = .*$',
                             lambda match: match[1] + name + ' = ' + str(value), text)
        if count != 1:
            raise ValueError(f'Cannot configure {name}; create a current source snapshot')
    config = types.ModuleType('FBSAANNpt_FBSAANN_globalDefs')
    exec(compile(text, str(path), 'exec'), vars(config))
    return config


def iteration_settings(config):
    values = config if isinstance(config, dict) else vars(config)
    enabled = values.get('fbsaOptimiseTrainingIterations', False)
    result = {'fbsaOptimiseTrainingIterations': enabled}
    if enabled:
        result.update({name: values[name] for name in CONVERGENCE_SETTINGS})
    return result


def source_hashes(args):
    return {p.name: sha_file(p) for p in (args.output / 'source').glob('*.py')}


def preparation_source_hashes(hashes):
    #loadDataset() and its settings determine the saved rows, splits and transforms.
    #Model, optimiser and convergence-loop changes do not require preparing again.
    return {name: hashes[name] for name in (
        'ANNpt_data.py', 'ANNpt_globalDefs.py', 'FBSAANNpt_FBSAANN_globalDefs.py')}


def positive_learning_rate(value):
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise argparse.ArgumentTypeError('learning rate must be finite and greater than zero')
    return rate


def validate_saved_artifact(record, args, *, prepared_data=False):
    """Check preprocessing sources for data, and the full snapshot for trained runs."""
    expected = iteration_settings(load_algorithm_config(args))
    expected_batch_size = DATASET_BATCH_SIZES.get(record['dataset'])
    saved_sources, current_sources = record['source_hashes'], source_hashes(args)
    if prepared_data:
        saved_sources = preparation_source_hashes(saved_sources)
        current_sources = preparation_source_hashes(current_sources)
    differences = []
    if record['seed'] != args.seed:
        differences.append(f"seed ({record['seed']} != {args.seed})")
    changed_sources = sorted(name for name in saved_sources.keys() | current_sources.keys()
                             if saved_sources.get(name) != current_sources.get(name))
    if changed_sources:
        differences.append('source snapshot (' + ', '.join(changed_sources) + ')')
    if iteration_settings(record['settings']) != expected:
        differences.append('convergence policy')
    #Prepared rows are independent of minibatch size; completed runs are not.
    if not prepared_data and expected_batch_size is not None and record['settings']['batchSize'] != expected_batch_size:
        differences.append(f"batch size ({record['settings']['batchSize']} != {expected_batch_size})")
    #An optimiser setting does not change prepared rows. A completed run must
    #match an explicit override, and a previous override must not become a default.
    if not prepared_data:
        requested_rate = getattr(args, 'learning_rate', None)
        if requested_rate is not None and record['settings']['learningRate'] != requested_rate:
            differences.append(f"learning rate ({record['settings']['learningRate']} != {requested_rate})")
        elif requested_rate is None and record.get('learning_rate_override') is not None:
            differences.append(f"learning rate override ({record['learning_rate_override']} saved; none requested)")
    if differences:
        kind = 'prepared data' if prepared_data else 'run'
        raise ValueError(f"Saved {kind} is incompatible: " + '; '.join(differences) +
                         '; use a separate output directory')


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
    config = load_algorithm_config(args)
    sys.modules[config.__name__] = config
    definitions = types.ModuleType('ANNpt_globalDefs')
    text = (source / 'ANNpt_globalDefs.py').read_text()
    text, count = re.subn(r"(?m)^\tdatasetName = 'titanic'.*$", '\tdatasetName = ' + repr(args.dataset), text)
    assert count == 1
    exec(compile(text, str(source / 'ANNpt_globalDefs.py'), 'exec'), vars(definitions))
    #Apply before importing the wrapper/model/data modules, which copy global settings.
    definitions.batchSize = DATASET_BATCH_SIZES.get(args.dataset, definitions.batchSize)
    if getattr(args, 'learning_rate', None) is not None:
        definitions.learningRate = positive_learning_rate(args.learning_rate)
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
    return {**{name: getattr(definitions, name) for name in names if hasattr(definitions, name)},
            **iteration_settings(definitions)}


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
            'source_hashes': source_hashes(args)}
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
    validate_saved_artifact(info, args, prepared_data=True)
    run_source_hashes = source_hashes(args)
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
    schedule_checkpoints = {'0': schedule.hexdigest()}
    optimise = getattr(definitions, 'fbsaOptimiseTrainingIterations', False)
    original_propagate = wrapper.ANNpt_algorithm.trainOrTestModel
    steps = 0
    def observed_propagate(model, train_or_test, x, y, *rest, **kwargs):
        nonlocal steps
        if train_or_test:
            for array in (x, y):
                schedule.update(array.detach().cpu().numpy().tobytes())
            #Count attempted updates, including production numerical recoveries.
            steps += 1
            if optimise and steps % definitions.fbsaEvaluateEveryIterations == 0:
                schedule_checkpoints[str(steps)] = schedule.hexdigest()
                atomic_json(status_path, {'status': 'training', 'training_steps': steps,
                                         'selection_metric': 'train_loss' if definitions.trainSetLossOptimisation else 'validation_loss',
                                         'time': time.time()})
        loss, accuracy = original_propagate(model, train_or_test, x, y, *rest, **kwargs)
        if train_or_test and not optimise and not np.isfinite(float(loss.detach())):
            raise FloatingPointError(f'Non-finite training loss at update {steps}')
        return loss, accuracy
    wrapper.ANNpt_algorithm.trainOrTestModel = observed_propagate
    started = time.time()
    convergence_result = wrapper.processDataset(True, dataset['train'], model, dataset.get('validation'))
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
    test_result = wrapper.processDataset(False, dataset['test'], model)
    if optimise:
        assert convergence_result['iterations'] == steps
        assert np.isfinite(test_result['loss']) and np.isfinite(test_result['accuracy'])
    else:
        test_result = dict(rows=totals['rows'], accuracy=totals['correct'] / totals['rows'],
                           loss=totals['loss_sum'] / totals['rows'])
    assert test_result['rows'] == len(dataset['test'])
    repeat = definitions.datasetRepeatSize if definitions.datasetRepeat else 1
    result = {'dataset': args.dataset, 'method': args.method, 'seed': args.seed,
              'status': 'completed', 'test_accuracy': test_result['accuracy'],
              'test_loss': test_result['loss'], 'test_rows_evaluated': test_result['rows'],
              'test_unique_rows': len(dataset['test']) // repeat,
              'training_steps': steps, 'training_seconds': trained - started,
              'test_seconds': time.time() - trained, 'settings': settings(definitions),
              'data_fingerprints': info['fingerprints'], 'initial_forward_sha256': initial_hash,
              'minibatch_schedule_sha256': schedule.hexdigest(), 'device': str(definitions.device),
              'torch_version': torch.__version__, 'source_hashes': run_source_hashes,
              'data_source_hashes': info['source_hashes'],
              'learning_rate_override': getattr(args, 'learning_rate', None)}
    if optimise:
        result['training_convergence'] = convergence_result
        result['minibatch_schedule_checkpoints'] = schedule_checkpoints
    atomic_json(args.output / 'runs' / f'{args.dataset}_{args.method}.json', result)
    if optimise:
        atomic_json(status_path, {'status': 'completed', 'training_steps': steps,
                                 'selected_iteration': convergence_result['selected_iteration'],
                                 'stop_reason': convergence_result['stop_reason'], 'time': time.time()})
    print('RESULT ' + json.dumps(result), flush=True)


def verify_pair(dataset, entry, info):
    backprop, fbsa = entry['backprop'], entry['fbsa']
    for key in ('seed', 'data_fingerprints', 'initial_forward_sha256', 'source_hashes'):
        assert backprop[key] == fbsa[key], (dataset, key)
    policy = iteration_settings(backprop['settings'])
    assert policy == iteration_settings(fbsa['settings']) == iteration_settings(info['settings']), dataset
    assert backprop['settings']['batchSize'] == fbsa['settings']['batchSize'], dataset
    optimise = policy['fbsaOptimiseTrainingIterations']
    for result in (backprop, fbsa):
        config = result['settings']
        assert result['seed'] == info['seed'], dataset
        assert result['test_rows_evaluated'] == info['sizes']['test'], dataset
        assert preparation_source_hashes(result['source_hashes']) == preparation_source_hashes(info['source_hashes']), dataset
        assert result.get('data_source_hashes', result['source_hashes']) == info['source_hashes'], dataset
        assert result['data_fingerprints'] == info['fingerprints'], dataset
        batches, remainder = divmod(info['sizes']['train'], config['batchSize'])
        if remainder and (optimise or not config['dataloaderMaintainBatchSize']):
            batches += 1
        if optimise:
            selected = result['training_convergence']
            split = 'train' if policy['trainSetLossOptimisation'] else 'validation'
            assert result['training_steps'] == selected['iterations'], dataset
            assert selected['iterations'] >= selected['policy']['minimum_updates'] == max(
                batches, policy['fbsaMinimumTrainingIterations']), dataset
            assert 0 <= selected['selected_iteration'] <= selected['iterations'], dataset
            assert selected['selected_iteration'] % policy['fbsaEvaluateEveryIterations'] == 0, dataset
            assert selected['policy']['selection_metric'] == split + '_loss', dataset
            assert selected[split]['rows'] == info['sizes'][split], dataset
            if split == 'validation':
                #Older runs also recorded diagnostic train metrics; new runs leave them null.
                if selected['train'] is not None:
                    assert selected['train']['rows'] == info['sizes']['train'], dataset
            else:
                assert selected['validation'] is None, dataset
            if selected['stop_reason'] == 'perfect_train_fit':
                #The selected minimum-loss checkpoint can precede the perfect-fit observation.
                assert split == 'train', dataset
                assert selected['train']['loss'] <= policy['fbsaTrainingLossGoal'], dataset
            else:
                label = 'training' if split == 'train' else 'validation'
                assert selected['stop_reason'] == label + '_loss_plateau_after_lr_reductions', dataset
            assert result['minibatch_schedule_checkpoints'][str(selected['iterations'])] == result['minibatch_schedule_sha256'], dataset
        else:
            assert result['training_steps'] == batches * config['trainNumberOfEpochs'], dataset
    if optimise:
        assert backprop['training_convergence']['policy'] == fbsa['training_convergence']['policy'], dataset
        shorter, longer = sorted((backprop, fbsa), key=lambda result: result['training_steps'])
        for step, digest in shorter['minibatch_schedule_checkpoints'].items():
            assert longer['minibatch_schedule_checkpoints'].get(step) == digest, (dataset, 'minibatch prefix', step)
    else:
        for key in ('training_steps', 'minibatch_schedule_sha256'):
            assert backprop[key] == fbsa[key], (dataset, key)


def report(args):
    rows = []
    verified = []
    implementations = set()
    iteration_rows = []
    learning_rate_overrides = []
    lines = ['FBSA / BACKPROP BENCHMARK', '',
             'Configured per-dataset architecture, learning rate and repeats; paired initialization and minibatch order.',
             'Training uses configured epochs or the production convergence controller; selection details below.',
             'Backprop: existing Adam optimizer. FBSA: local SAL updates with SGD; version/settings listed below.',
             'All held-out rows evaluated, including final partial batches. No tuning on the test set.', '',
             'Dataset | Backprop test accuracy (%) | FBSA test accuracy (%) | FBSA minus backprop (pp)']
    for dataset in DATASETS:
        entry = {'dataset': dataset}
        for method in ['backprop', 'fbsa']:
            path = args.output / 'runs' / f'{dataset}_{method}.json'
            entry[method] = json.loads(path.read_text()) if path.exists() else None
            result = entry[method]
            if result and result.get('status') == 'completed':
                if result.get('learning_rate_override') is not None:
                    learning_rate_overrides.append(f"{dataset} | {method} | {result['settings']['learningRate']}")
                if result['settings'].get('fbsaOptimiseTrainingIterations', False):
                    selected = result['training_convergence']
                    mode = selected['policy']['selection_metric']
                    iteration_rows.append(f"{dataset} | {method} | {mode} | {result['training_steps']} | "
                                          f"{selected['selected_iteration']} | {selected['stop_reason']}")
                else:
                    iteration_rows.append(f"{dataset} | {method} | fixed epochs | {result['training_steps']} | — | configured epochs")
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
            info = json.loads((args.output / 'data' / dataset / 'benchmark.json').read_text())
            verify_pair(dataset, entry, info)
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
    if learning_rate_overrides:
        lines.extend(['', 'Initial learning-rate overrides:',
                      'Dataset | Method | Initial learning rate'] + learning_rate_overrides)
    if iteration_rows:
        lines.extend(['', 'Training iteration selection:',
                      'Dataset | Method | Selection | Total iterations | Selected iteration | Stop reason'] + iteration_rows)
    lines.extend(['', f'Verified complete paired runs: {len(verified)}/{len(DATASETS)} datasets.',
                  'Checks: identical prepared data, initial forward weights, source code and selection policy;',
                  'matching minibatch prefixes through the shorter run, and complete test-set coverage.',
                  'Fixed-epoch pairs also require identical update counts and complete minibatch schedules.',
                  'Convergence pairs may stop independently; selected iterations and stop reasons are checked.', '',
                  'Interpretation: one paired seed, without confidence intervals; learning-rate overrides are listed above when used.',
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
    policy = iteration_settings(load_algorithm_config(args))
    #Reject incompatible cached artifacts before writing a new protocol or launching jobs.
    for dataset in DATASETS:
        paths = [args.output / 'data' / dataset / 'benchmark.json']
        paths += [args.output / 'runs' / f'{dataset}_{method}.json' for method in ('backprop', 'fbsa')]
        for path in paths:
            if path.exists():
                record = json.loads(path.read_text())
                if path.name == 'benchmark.json' or record.get('status') == 'completed':
                    validate_saved_artifact(record, args, prepared_data=path.name == 'benchmark.json')
    training = ('production convergence, selecting minimum full-' +
                ('training' if policy['trainSetLossOptimisation'] else 'validation') + ' loss'
                if policy['fbsaOptimiseTrainingIterations'] else 'full configured epochs')
    manifest = {'seed': args.seed, 'datasets': DATASETS, 'methods': ['backprop', 'fbsa'],
                'batch_size_overrides': DATASET_BATCH_SIZES,
                'learning_rate_override': getattr(args, 'learning_rate', None),
                'raw_data_dir': str(args.raw_data_dir),
                'threads_per_job': args.threads, 'concurrent_datasets': args.jobs,
                'source_sha256': source_hashes(args), 'training_settings': policy,
                'protocol': f'Repository defaults; {training}; configured repeats; paired initial forward weights and minibatch order; all test rows; no test-set tuning.',
                'evaluation_override': 'Evaluate all test rows without shuffling or dropping partial batches; aggregate sample-weighted metrics.',
                'runner_sha256': sha_file(Path(__file__))}
    atomic_json(args.output / 'protocol.json', manifest)
    def execute(action, dataset, method='backprop'):
        log = args.output / 'logs' / f'{dataset}_{action if action == "prepare" else method}.log'
        command = [sys.executable, '-B', str(Path(__file__).resolve()), action,
                   '--dataset', dataset, '--method', method, '--output', str(args.output),
                   '--seed', str(args.seed), '--threads', str(args.threads),
                   '--raw-data-dir', str(args.raw_data_dir)]
        for option in ('optimise_training_iterations', 'train_set_loss_optimisation'):
            value = getattr(args, option, None)
            if value is not None:
                command.append('--' + ('' if value else 'no-') + option.replace('_', '-'))
        if getattr(args, 'learning_rate', None) is not None:
            command.extend(['--learning-rate', str(args.learning_rate)])
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
    order = sorted(DATASETS, key=lambda name: name != 'higgs')
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
    parser.add_argument('--learning-rate', type=positive_learning_rate,
                        help='Override the initial learning rate for this run (both methods in suite); default: dataset setting')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--optimise-training-iterations', action=argparse.BooleanOptionalAction, default=None,
                        help='Override fbsaOptimiseTrainingIterations for both methods (default: source setting)')
    parser.add_argument('--train-set-loss-optimisation', action=argparse.BooleanOptionalAction, default=None,
                        help='Select training loss; --no-train-set-loss-optimisation selects validation loss (default: source setting)')
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
                        validate_saved_artifact(previous, args, prepared_data=args.action == 'prepare')
                        print('REUSING ' + str(result), flush=True)
                        return
                globals()[args.action](args)
        else:
            globals()[args.action](args)
    except Exception as error:
        traceback.print_exc()
        if args.action == 'run':
            path = args.output / 'runs' / f'{args.dataset}_{args.method}.json'
            #A rejected reuse request must not overwrite a completed result.
            if not path.exists() or json.loads(path.read_text()).get('status') != 'completed':
                atomic_json(path, {'dataset': args.dataset, 'method': args.method, 'status': 'failed',
                                   'error': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
