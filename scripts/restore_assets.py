"""Verify and restore downloaded Release parts using Python's standard library."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import io
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


class JoinedParts(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = path.open('rb')
            size = self.current.readinto(buffer)
            if size:
                return size
            self.current.close()
            self.current = None

    def close(self):
        if self.current:
            self.current.close()
        super().close()


def restore(group, assets, root, verify_only):
    parts = []
    for item in group['parts']:
        name = item['name']
        if Path(name).name != name:
            raise ValueError('Invalid part name')
        path = assets / name
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError(f'Part checksum differs: {name}')
        parts.append(path)
    expected = {item['path']: item for item in group['files']}
    seen = set()
    with JoinedParts(parts) as joined, io.BufferedReader(joined) as stream:
        with tarfile.open(fileobj=stream, mode='r|gz') as archive:
            for member in archive:
                name = member.name
                rel = PurePosixPath(name)
                if (not member.isfile() or rel.is_absolute() or '..' in rel.parts
                        or name not in expected or name in seen):
                    raise ValueError(f'Unexpected archive entry: {name}')
                record = expected[name]
                if member.size != record['bytes']:
                    raise ValueError(f'Entry size differs: {name}')
                target = (root / name).resolve()
                target.relative_to(root)
                skip = verify_only
                if not verify_only and target.exists():
                    if digest(target) != record['sha256']:
                        raise FileExistsError(f'Refusing to overwrite different content: {target}')
                    skip = True
                temporary = None
                sink = None
                h = hashlib.sha256()
                try:
                    if not skip:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_name(target.name + '.restoring')
                        sink = temporary.open('xb')
                    with archive.extractfile(member) as source:
                        for block in iter(lambda: source.read(8 * 1024 ** 2), b''):
                            h.update(block)
                            if sink:
                                sink.write(block)
                    if sink:
                        sink.close()
                    if h.hexdigest() != record['sha256']:
                        raise ValueError(f'Content checksum differs: {name}')
                    if not skip:
                        temporary.rename(target)
                    seen.add(name)
                finally:
                    if sink:
                        sink.close()
                        if temporary.exists():
                            temporary.unlink()
    if seen != set(expected):
        raise ValueError(f'Missing entries: {set(expected) - seen}')
    print(f'{group["name"]}: {len(seen)} files verified' + ('' if verify_only else ' and restored'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--groups', nargs='+', default=['all'])
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'release-assets.json').read_text())
    groups = {item['name']: item for item in manifest['groups']}
    names = list(groups) if args.groups == ['all'] else args.groups
    for name in names:
        if name not in groups:
            parser.error(f'Unknown group {name}; choose from {list(groups)}')
    for name in names:
        restore(groups[name], args.assets.resolve(), args.root.resolve(), args.verify_only)


if __name__ == '__main__':
    main()
