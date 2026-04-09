import json
import string
import itertools as it
import copy
import hashlib
import base64
from pathlib import Path

class JSONpath(Path):
    def read_json(self):
        with self.open('r') as file:
            return json.load(file)
            
    def write_json(self, data):
        with self.open('w') as file:
            return json.dump(data, file, separators=(',', ':'))
            
    def is_json(self):
        return self.suffix.lower() == ".json"

class AF3json:
    def __init__(self, path: JSONpath = None, data: dict = None, name: str = None, oligomer=1, seeds: set[int] = set()):
        if path:
            self.path = path
            self.name = path.stem
            self.data = path.read_json()
        elif data:
            self.path = None
            self.name = "query" if name is None else name
            self.data = data
        else:
            raise ValueError("Need json path or data")
            
        if 'sequences' not in self.data:
            raise ValueError("No 'sequences' found in json")
            
        self.orig_seeds = self.data.pop('modelSeeds', [])
        self.seeds = list(seeds) if seeds else self.orig_seeds
        self.orig_name = self.data.pop('name', None)
        
        if 'dialect' not in self.data:
            self.data['dialect'] = "alphafold3"
        elif self.data['dialect'] != "alphafold3":
            raise ValueError("Only 'alphafold3' dialect is supported")
            
        self.data['version'] = 4
        self.len = self.hash = None
        self.ids = AF3json.id_gen()
        self.len = 0
        
        for seq in self.data['sequences']:
            for item in seq:
                id_val = seq[item].get('id', [])
                seq[item]['id'] = list(it.islice(self.ids, oligomer * len(id_val)))
                if item == 'protein':
                    self.len += len(id_val) * len(seq[item]['sequence'])
                    if 'templates' not in seq[item]:
                        seq[item]['templates'] = None
                    if 'pairedMsa' not in seq[item]:
                        seq[item]['pairedMsa'] = None
                    if 'unpairedMsa' not in seq[item]:
                        seq[item]['unpairedMsa'] = None
                        
        ccd_str = self.data.pop('userCCD', None)
        self.ccd = {}
        if ccd_str:
            self.add_ccd(ccd_str)

    def add_ccd(self, ccd_str):
        ccd_name = None
        for line in ccd_str.split('\n'):
            if line.startswith('data_'):
                ccd_name = line
                self.ccd[ccd_name] = [line]
            elif ccd_name:
                self.ccd[ccd_name].append(line)
            elif line:
                raise ValueError("Found non-empty 'userCCD' but no data_* line")

    def __len__(self):
        return self.len

    def get_hash(self):
        if not self.hash:
            serialized = json.dumps(self.data, sort_keys=True, separators=(',', ':')).encode('utf-8')
            raw_hash = hashlib.sha256(serialized).digest()
            self.hash = base64.urlsafe_b64encode(raw_hash).decode().rstrip("=")
        return self.hash

    def write(self, path: JSONpath):
        data = self.data.copy()
        data['modelSeeds'] = self.seeds
        data['name'] = 'query'
        if self.ccd:
            data['userCCD'] = ''
            for ccd_name, ccd_lines in self.ccd.items():
                data['userCCD'] += '\n'.join(ccd_lines) + '\n'
        path.write_json(data)

    def iterseq(self):
        if 'sequences' not in self.data:
            raise ValueError("No 'sequences' found in json")
        for seq in self.data['sequences']:
            new_seq = seq.copy()
            new_seq['id'] = "A"
            result = copy.deepcopy(self)
            result.data["sequences"] = [new_seq]
            
            # Note: This logic existed in original code but seems to rely on an undefined `index` variable.
            sequence = self.data["sequences"][index]["sequence"] 
            result = copy.deepcopy(self)
            result.data = {"sequences": [{"protein": {"id": "A", "sequence": sequence}}]}
            return result

    @staticmethod
    def id_gen():
        letters = string.ascii_uppercase
        for let in letters:
            yield let
        for size in it.count(2):
            for p in it.product(letters, repeat=size):
                yield "".join(p)

    def __add__(self, other):
        if not isinstance(other, AF3json):
            return NotImplemented
        for seq in other.data['sequences']:
            for item in seq:
                id_val = seq[item].get('id', [])
                seq[item]['id'] = list(it.islice(self.ids, len(id_val)))
            self.data['sequences'].append(seq)
            
        for ccd_name, ccd_lines in other.ccd.items():
            if ccd_name not in self.ccd:
                self.ccd[ccd_name] = ccd_lines
        return self

    def iter_seq(self, key='protein'):
        for seq in self.data['sequences']:
            if key in seq:
                yield seq[key]
