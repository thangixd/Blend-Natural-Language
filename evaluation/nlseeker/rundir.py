import configparser
import errno
import hashlib
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from typing import Callable, Dict, Iterable, List, Optional, Tuple

USAGE = '''python -m evaluation.nlseeker.rundir status <results root>...
python -m evaluation.nlseeker.rundir verify <run>
python -m evaluation.nlseeker.rundir delete <run> <code>
python -m evaluation.nlseeker.rundir index-files <index manifest>...
python -m evaluation.nlseeker.rundir verify-index <index manifest>... [<code>]
python -m evaluation.nlseeker.rundir delete-index <index manifest>... <code>'''

MANIFEST = 'MANIFEST.json'


class RunDir:
    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def create(cls, results_root: Path, tag: str, config_text: str) -> 'RunDir':
        stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')
        path = Path(results_root) / f'{stamp}__{tag}.partial'
        path.mkdir(parents=True)
        run = cls(path)
        run.config_path.write_text(config_text)
        run._write_meta({'created_utc': stamp, 'datasets': {}})
        return run

    @classmethod
    def open_partial(cls, path: Path) -> 'RunDir':
        path = Path(path)
        run = cls(path)
        if not path.is_dir() or not run.config_path.is_file() or not run._meta_path.is_file():
            raise FileNotFoundError(f'not a resumable run dir: {path}')
        return run

    @property
    def config_path(self) -> Path:
        return self.path / 'config.yaml'

    @property
    def _meta_path(self) -> Path:
        return self.path / 'run_meta.json'

    def dataset_dir(self, name: str) -> Path:
        directory = self.path / name
        directory.mkdir(exist_ok=True)
        return directory

    def artifacts_dir(self, name: str) -> Path:
        directory = self.dataset_dir(name) / 'artifacts'
        directory.mkdir(exist_ok=True)
        return directory

    def meta(self) -> dict:
        return json.loads(self._meta_path.read_text())

    def update_meta(self, **fields) -> None:
        meta = self.meta()
        meta.update(fields)
        self._write_meta(meta)

    def set_dataset(self, name: str, **fields) -> None:
        meta = self.meta()
        meta['datasets'].setdefault(name, {}).update(fields)
        self._write_meta(meta)

    def dataset_status(self, name: str) -> Optional[str]:
        return self.meta()['datasets'].get(name, {}).get('status')

    def wipe_dataset(self, name: str) -> None:
        target = self.path / name
        if target.exists():
            shutil.rmtree(target)
        meta = self.meta()
        meta['datasets'].pop(name, None)
        self._write_meta(meta)

    def run_datasets(self, names: Iterable[str], run_one: Callable[[str, Path], None], min_free_bytes: int) -> bool:
        names = list(names)
        for name in names:
            if self.dataset_status(name) == 'complete':
                print(f'{name}: already complete, skipping')
                continue
            self.wipe_dataset(name)
            try:
                check_free_space(self.path, min_free_bytes)
                run_one(name, self.dataset_dir(name))
                self.seal_dataset(name)
            except Exception as error:
                traceback.print_exc()
                self.set_dataset(name, status='failed', error=f'{type(error).__name__}: {error}')
                print(f'{name}: FAILED - {error}', file=sys.stderr)
        return all(self.dataset_status(name) == 'complete' for name in names)

    def seal_dataset(self, name: str) -> None:
        manifest = _write_manifest(self.path, self.path / name / MANIFEST)
        self.set_dataset(name, status='complete', manifest_sha256=_sha256(manifest))

    def finalize(self) -> Path:
        final = self.path.with_name(self.path.name.removesuffix('.partial'))
        _write_manifest(self.path, self.path / MANIFEST)
        os.replace(self.path, final)
        self.path = final
        return final

    def _write_meta(self, meta: dict) -> None:
        _write_json(self._meta_path, meta)


def check_free_space(path: Path, min_free_bytes: int) -> None:
    free = shutil.disk_usage(path).free
    if free < min_free_bytes:
        raise OSError(errno.ENOSPC, f'{free / 2 ** 30:.1f} GiB free at {path}, '
                                    f'{min_free_bytes / 2 ** 30:.1f} GiB required')


def verify(run: Path) -> Tuple[List[str], Optional[str]]:
    manifest = Path(run) / MANIFEST
    if not manifest.is_file():
        return [f'not sealed: no {MANIFEST}'], None
    problems = _check(manifest, Path(run))
    return problems, None if problems else deletion_code(run)


def deletion_code(run: Path) -> Optional[str]:
    manifest = Path(run) / MANIFEST
    return manifest_code(manifest) if manifest.is_file() else None


def manifest_code(manifest: Path) -> str:
    return _sha256(manifest)[:12]


def delete_run(run: Path, code: str) -> bool:
    expected = deletion_code(run)
    if expected is None or code != expected or not (Path(run) / 'run_meta.json').is_file():
        return False
    shutil.rmtree(run)
    return True


def seal_index(manifest: Path, files: Iterable[Path]) -> str:
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    _write_json(Path(manifest), {Path(path).as_posix(): _entry(Path(path)) for path in files})
    return manifest_code(manifest)


def index_files(manifest: Path) -> List[Path]:
    return [Path(manifest)] + [Path(path) for path in json.loads(Path(manifest).read_text())]


def verify_index(manifest: Path, code: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
    manifest = Path(manifest)
    if not manifest.is_file():
        return [f'not sealed: no {manifest}'], None
    problems = _check(manifest, Path())
    expected = manifest_code(manifest)
    if code is not None and code != expected:
        problems.append(f'code {code!r} is not {expected!r}: a stale or different manifest')
    return problems, None if problems else expected


def delete_index(manifest: Path, code: str) -> bool:
    manifest = Path(manifest)
    if not manifest.is_file() or code != manifest_code(manifest):
        return False
    files = index_files(manifest)[1:]
    for path in files:
        path.unlink(missing_ok=True)
    manifest.unlink()
    folders = {parent for path in files for parent in path.parents if manifest.parent in parent.parents}
    for folder in sorted(folders, key=lambda path: len(path.parts), reverse=True):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    return True


def package_code(manifests: List[Path]) -> str:
    codes = [manifest_code(manifest) for manifest in manifests]
    return codes[0] if len(codes) == 1 else hashlib.sha256(' '.join(codes).encode()).hexdigest()[:12]


def verify_package(manifests: List[Path], code: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
    problems = [problem for manifest in manifests for problem in verify_index(manifest)[0]]
    if problems:
        return problems, None
    expected = package_code(manifests)
    if code is not None and code != expected:
        return [f'code {code!r} is not {expected!r}: a stale or different manifest'], None
    return [], expected


def delete_package(manifests: List[Path], code: str) -> bool:
    if not all(Path(manifest).is_file() for manifest in manifests) or code != package_code(manifests):
        return False
    for manifest in manifests:
        delete_index(manifest, manifest_code(manifest))
    return True


def status(roots: Iterable[Path]) -> List[str]:
    lines = []
    for root in roots:
        for meta_path in sorted(Path(root).glob('*/run_meta.json')):
            run = meta_path.parent
            datasets = json.loads(meta_path.read_text())['datasets']
            state = 'complete' if not run.name.endswith('.partial') else 'failed' if any(
                fields.get('status') == 'failed' for fields in datasets.values()) else 'partial'
            lakes = ', '.join(f'{name} {fields.get("status")}' for name, fields in datasets.items())
            lines.append(f'{run}: {state}; {lakes or "no lakes yet"}')
    return lines


def _check(manifest: Path, base: Path) -> List[str]:
    problems = []
    for relative, expected in json.loads(manifest.read_text()).items():
        path = base / relative
        if not path.is_file():
            problems.append(f'missing: {relative}')
        elif _entry(path) != expected:
            problems.append(f'changed: {relative}')
    return problems


def _write_manifest(run: Path, manifest: Path) -> Path:
    entries = {path.relative_to(run).as_posix(): _entry(path)
               for path in sorted(manifest.parent.rglob('*')) if path.is_file() and path != manifest}
    _write_json(manifest, entries)
    return manifest


def _entry(path: Path) -> dict:
    return {'size': path.stat().st_size, 'sha256': _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2 ** 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def write_ini(lake_ini: Path, nlseeker: Dict[str, str], alpha: float, n: int, out: Path) -> Path:
    config = configparser.ConfigParser()
    config.read(lake_ini)
    config['NLSeeker'].update({**nlseeker, 'alpha': str(alpha), 'n': str(n)})
    with open(out, 'w') as f:
        config.write(f)
    return Path(out)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < {'status': 1, 'verify': 2, 'delete': 3, 'index-files': 2, 'verify-index': 2,
                    'delete-index': 3}.get(argv[0] if argv else '', 99):
        print(USAGE, file=sys.stderr)
        return 2
    command, paths = argv[0], [Path(arg) for arg in argv[1:]]
    if command == 'status':
        print('\n'.join(status(paths)) or 'no runs')
        return 0
    if command == 'verify':
        problems, code = verify(paths[0])
        print('\n'.join(problems) if problems else f'safe to delete: {code}')
        return 1 if problems else 0
    if command == 'index-files':
        print('\n'.join(path.as_posix() for manifest in paths for path in index_files(manifest)))
        return 0
    manifests = [path for path in paths if path.suffix == '.json']
    code = next((path.name for path in paths if path.suffix != '.json'), None)
    if command == 'verify-index':
        problems, expected = verify_package(manifests, code)
        print('\n'.join(problems) if problems else f'index ok: {expected}')
        return 1 if problems else 0
    if command == 'delete-index':
        target = ' '.join(map(str, manifests))
        deleted = code is not None and delete_package(manifests, code)
    else:
        code, target = argv[2], paths[0]
        deleted = delete_run(target, code)
    if deleted:
        print(f'deleted {target}')
        return 0
    print(f'refused: {code!r} is not the code of {target}', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
