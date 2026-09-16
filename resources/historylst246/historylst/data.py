"""Portable loader. Prediction opens inputs only; labels are opt-in."""
from pathlib import Path
import json
import numpy as np

INPUTS=('fine','coarse','support','context','emissivity','history')

class Dataset:
    def __init__(self, root, split, labels=False):
        self.root=Path(root)
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.role=self.manifest['roles'][split];self.records=self.role['scenes'];self.split=split
        names=INPUTS+(('target','formal','valid') if labels else ())
        self.arrays={k:np.load(self.root/self.role['fields'][k]['path'],mmap_mode='r',allow_pickle=False) for k in names}
    def __len__(self):return len(self.records)
    def batch(self, indices):return {k:np.array(a[indices],copy=True) for k,a in self.arrays.items()}
    def observed_history(self,index):
        h=np.asarray(self.arrays['history'][index],dtype=np.float64)
        return dict(temperature=np.where(h[:,2]>0,20*h[:,0]+300,np.nan),count30=np.rint(h[:,2]*16).astype('uint8'),
            qa_kelvin=np.where(h[:,2]>0,3*h[:,3],np.nan),emissivity=np.where(h[:,5]>0,.01*h[:,4]+.98,np.nan),
            emissivity_count30=np.rint(h[:,5]*16).astype('uint8'),sources=self.records[index]['history'])
    def sampling_probabilities(self):
        regions=sorted({r['region'] for r in self.records})
        cities={g:{r['city'] for r in self.records if r['region']==g} for g in regions}
        counts={c:sum(r['city']==c for r in self.records) for cc in cities.values() for c in cc}
        return np.array([1/(len(regions)*len(cities[r['region']])*counts[r['city']]) for r in self.records])
