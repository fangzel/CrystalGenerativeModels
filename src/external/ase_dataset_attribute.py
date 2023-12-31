# Crystal structures are saved in 'mp-####.cif' files;
# Other properties are saved in 'mp-####.pickle' files;

import functools
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import io
import torch
import pickle
import ase
import numpy as np
from torch import tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from ocpmodels.common.registry import registry
from ocpmodels.datasets.target_metadata_guesser import guess_property_metadata
from ocpmodels.preprocessing import AtomsToGraphs

def apply_one_tags(
    atoms: ase.Atoms, skip_if_nonzero: bool = True, skip_always: bool = False
):
    """
    This function will apply tags of 1 to an ASE atoms object.
    It is used as an atoms_transform in the datasets contained in this file.

    Certain models will treat atoms differently depending on their tags.
    For example, GemNet-OC by default will only compute triplet and quadruplet interactions
    for atoms with non-zero tags. This model throws an error if there are no tagged atoms.
    For this reason, the default behavior is to tag atoms in structures with no tags.

    args:
        skip_if_nonzero (bool): If at least one atom has a nonzero tag, do not tag any atoms

        skip_always (bool): Do not apply any tags. This arg exists so that this function can be disabled
                without needing to pass a callable (which is currently difficult to do with main.py)
    """
    if skip_always:
        return atoms

    if np.all(atoms.get_tags() == 0) or not skip_if_nonzero:
        atoms.set_tags(np.ones(len(atoms)))

    return atoms


class AseAtomsDatasetAttr(Dataset, ABC):
    """
    This is an abstract Dataset that includes helpful utilities for turning
    ASE atoms objects into OCP-usable data objects. This should not be instantiated directly
    as get_atoms_object and load_dataset_get_ids are not implemented in this base class.

    Derived classes must add at least two things:
        self.get_atoms_object(id): a function that takes an identifier and returns a corresponding atoms object

        self.load_dataset_get_ids(config: dict): This function is responsible for any initialization/loads
            of the dataset and importantly must return a list of all possible identifiers that can be passed into
            self.get_atoms_object(id)

    Identifiers need not be any particular type.
    """

    def __init__(
        self, config, transform=None, atoms_transform=apply_one_tags
    ) -> None:
        self.config = config

        a2g_args = config.get("a2g_args", {})

        # Make sure we always include PBC info in the resulting atoms objects
        a2g_args["r_pbc"] = True
        self.a2g = AtomsToGraphs(**a2g_args)

        self.transform = transform
        self.atoms_transform = atoms_transform

        if self.config.get("keep_in_memory", False):
            self.__getitem__ = functools.cache(self.__getitem__)

        # Derived classes should extend this functionality to also create self.ids,
        # a list of identifiers that can be passed to get_atoms_object()
        self.ids = self.load_dataset_get_ids(config)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx):
        # Handle slicing
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self.ids)))]

        # Get atoms object via derived class method
        atoms = self.get_atoms_object(self.config['src']+self.ids[idx]+self.config['pattern'])

        # Transform atoms object
        if self.atoms_transform is not None:
            atoms = self.atoms_transform(
                atoms, **self.config.get("atoms_transform_args", {})
            )

        if "sid" in atoms.info:
            sid = atoms.info["sid"]
        else:
            sid = tensor([idx])

        # Convert to data object
        data_object = self.a2g.convert(atoms, sid)

        data_object.pbc = tensor(atoms.pbc)
        # set an attribute on data_object
        attr_pkl_path = self.config['attr_pkl_dir']+self.ids[idx]+self.config['attr_pattern']
        with open(attr_pkl_path, 'rb') as file:
            attr_pkl_data = pickle.load(file)[self.attr_name]
            
        attr_func = self.config['attr_func']
        if attr_func is not None:
            attr_pkl_data = torch.tensor(attr_pkl_data, dtype=torch.float32)
            attr_pkl_data = attr_func(attr_pkl_data)
        
        setattr(data_object, self.attr_name, attr_pkl_data)

        # Transform data object
        if self.transform is not None:
            data_object = self.transform(
                data_object, **self.config.get("transform_args", {})
            )

        return data_object

    def close_db(self) -> None:
        # This method is sometimes called by a trainer
        pass

    def guess_target_metadata(self, num_samples: int = 100):
        metadata = {}

        if num_samples < len(self):
            metadata["targets"] = guess_property_metadata(
                [
                    self.get_atoms_object(self.config['src']+self.ids[idx]+self.config['pattern'])
                    for idx in np.random.choice(
                        len(self), size=(num_samples,), replace=False
                    )
                ]
            )
        else:
            metadata["targets"] = guess_property_metadata(
                [
                    self.get_atoms_object(self.config['src']+self.ids[idx]+self.config['pattern'])
                    for idx in range(len(self))
                ]
            )

        return metadata

    def get_metadata(self):
        return self.guess_target_metadata()


@registry.register_dataset("ase_read")
class AseReadDatasetAttr(AseAtomsDatasetAttr):
    """
    This Dataset uses ase.io.read to load data from a directory on disk.
    This is intended for small-scale testing and demonstrations of OCP.
    Larger datasets are better served by the efficiency of other dataset types
    such as LMDB.

    For a full list of ASE-readable filetypes, see
    https://wiki.fysik.dtu.dk/ase/ase/io/io.html

    args:
        config (dict):
            src (str): The source folder that contains your ASE-readable files

            pattern (str): Filepath matching each file you want to read
                    ex. "*/POSCAR", "*.cif", "*.xyz"
                    search recursively with two wildcards: "**/POSCAR" or "**/*.cif"

            a2g_args (dict): Keyword arguments for ocpmodels.preprocessing.AtomsToGraphs()
                    default options will work for most users

                    If you are using this for a training dataset, set
                    "r_energy":True and/or "r_forces":True as appropriate
                    In that case, energy/forces must be in the files you read (ex. OUTCAR)

            ase_read_args (dict): Keyword arguments for ase.io.read()

            keep_in_memory (bool): Store data in memory. This helps avoid random reads if you need
                    to iterate over a dataset many times (e.g. training for many epochs).
                    Not recommended for large datasets.

            atoms_transform_args (dict): Additional keyword arguments for the atoms_transform callable

            transform_args (dict): Additional keyword arguments for the transform callable

        atoms_transform (callable, optional): Additional preprocessing function applied to the Atoms
                    object. Useful for applying tags, for example.

        transform (callable, optional): Additional preprocessing function for the Data object

    """

    def load_dataset_get_ids(self, config) -> List[Path]:
        self.ase_read_args = config.get("ase_read_args", {})
        self.config = config

        if ":" in self.ase_read_args.get("index", ""):
            raise NotImplementedError(
                "To read multiple structures from a single file, please use AseReadMultiStructureDataset."
            )

        self.path = Path(config["src"])
        if self.path.is_file():
            raise Exception("The specified src is not a directory")
        
        self.attr_name = config["attr_name"]
        self.attr_pkl_dir = Path(config["attr_pkl_dir"])
        attr_files_list = [p.stem for p in self.attr_pkl_dir.glob('*'+config['attr_pattern'])]
        
        if config.get("files", None) is None:
            # if no files are specified, use the pattern to find all files
            cif_files_list = [p.stem for p in self.path.glob('*'+f'{config["pattern"]}')]
        else:
            # otherwise, use the files list
            cif_files_list = [f.stem for f in (self.path / f for f in config["files"])]
        return list(set(cif_files_list) & set(attr_files_list))


    def get_atoms_object(self, identifier):
        try:
            atoms = ase.io.read(identifier, **self.ase_read_args)
        except Exception as err:
            warnings.warn(f"{err} occured for: {identifier}")
            raise err

        return atoms
