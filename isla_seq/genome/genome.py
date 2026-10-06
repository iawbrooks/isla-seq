import copy
import subprocess
import shutil
import os
import json
from pathlib import Path
from typing import Literal
from collections import defaultdict

import send2trash
import gtfparse
import gffutils
import polars as pl
import pandas as pd
import numpy as np

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio import Entrez


def _run_command(
        cmd: list[str],
        *,
        print_cmd: bool = False,
        print_std: bool = False,
    ) -> int:
    """
    Runs the specified command and returns the returncode of that command.
    """
    if print_cmd:
        print(" > " + ' '.join(cmd))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if print_std:
        print(result.stdout)
        print(result.stderr)
    
    return result.returncode


def load_assembly_data_report(genome_dir: Path) -> dict:
    """
    
    """
    json_path = genome_dir / "ncbi_dataset/data/assembly_data_report.jsonl"
    if not json_path.is_file():
        raise FileNotFoundError(f"Could not find `assembly_data_report.jsonl`, which should have been present at path '{json_path}'")
    with json_path.open('rt') as f:
        json_dict = json.load(f)
    return json_dict


def find_genome_accession_directory(genome_dir: Path) -> Path:
    """

    """
    # Get accession name from JSON
    json_dict = load_assembly_data_report(genome_dir)
    accession = json_dict['accession']

    # Make sure accession exists
    accession_dir = genome_dir / f"ncbi_dataset/data/{accession}"
    if not accession_dir.is_dir():
        raise FileNotFoundError("Unknown error in locating accession directory. Try redownloading this genome.")

    return accession_dir


def download_ncbi_reference_genome(
        *,
        taxon_name: str,
        download_dir: Path,
        rehydrate: bool = False,
        if_exists: Literal['overwrite', 'raise', 'ignore'],
        verbose: bool = True,
    ) -> Path:
    """
    Downloads an NCBI reference genome and returns the path to that genome's root directory.
    """
    # Check inputs
    taxon_name = taxon_name.lower()
    organism_file_stem = taxon_name.replace(" ", "_")
    vprint = lambda x: print(x) if verbose else None

    # Verify download directory
    if not download_dir.is_dir():
        raise FileNotFoundError(f"Directory does not exist: '{download_dir}'")

    # Create temporary download directory (and remove it first if exists)
    temp_dir = download_dir / f".{organism_file_stem}_TEMP_NCBI_DOWNLOAD"
    try:
        temp_dir.mkdir()
    except FileExistsError:
        send2trash.send2trash(temp_dir)
        temp_dir.mkdir()

    # Download dehydrated dataset
    download_filepath = temp_dir / f"NCBI_{organism_file_stem}.zip"
    cmd = [
        "datasets", "download", "genome",
        "taxon",
        taxon_name,
        "--dehydrated", "--reference", "--assembly-source", "refseq", "--include", "cds,rna,protein,gtf,gff3,seq-report",
        "--filename", str(download_filepath),
    ]
    _run_command(cmd, print_cmd=verbose)
    
    # Unzip dataset
    cmd = ["tar", "-xf", str(download_filepath), "-C", str(temp_dir)]
    _run_command(cmd, print_cmd=verbose)

    # Find name of the downloaded assembly
    json_path = temp_dir / "ncbi_dataset/data/assembly_data_report.jsonl"
    if not json_path.is_file():
        raise FileNotFoundError(f"Could not find `assembly_data_report.jsonl`, which should have been present at path '{json_path}'")
    with json_path.open('rt') as f:
        json_dict = json.load(f)
    assembly_info = json_dict['assemblyInfo']
    assembly_name = assembly_info["assemblyName"]

    # Rename files!
    assembly_dir = download_dir / f"genome_{organism_file_stem}_{assembly_name}"
    if assembly_dir.exists():
        match if_exists:
            case 'overwrite':
                vprint(f"Assembly already exists; deleting '{assembly_dir}'")
                shutil.rmtree(assembly_dir)
                shutil.move(temp_dir, assembly_dir)
            case 'ignore':
                send2trash.send2trash(temp_dir)
            case 'raise':
                raise FileExistsError("Assembly already exists; specify `if_exists='ignore'` to overwrite")
    else:
        shutil.move(temp_dir, assembly_dir)
    vprint(f"# Dehydrated assembly '{assembly_name}' for taxon '{taxon_name}' is located at path '{assembly_dir}'")

    # Optionally rehydrate; will just do nothing if already rehydrated :)
    if rehydrate:
        rehydrate_genome(assembly_dir)

    return assembly_dir


def rehydrate_genome(assembly_dir: Path, verbose: bool = True):
    """
    Rehydrates the downloaded reference genome.
    """
    vprint = lambda x: print(x) if verbose else None
    cmd = ["datasets", "rehydrate", "--directory", str(assembly_dir)]
    _run_command(cmd, print_cmd=verbose)
    vprint("# Rehydrated!")


def parse_gtf_ncbi(
        path: Path,
        verbose: bool = False
    ) -> pl.DataFrame:
    gtf = gtfparse.parse_gtf(path)

    attributes = ['gene_id', 'transcript_id', "gene_biotype", "description", "db_xref"]
    attribute_df = pd.DataFrame(columns=attributes, index=np.arange(len(gtf)))
    
    for row_idx, attribute_split in enumerate(gtf['attribute_split']):
        if verbose and (row_idx + 1) % 100000 == 0:
            print(f"{row_idx + 1} / {len(gtf)}")
        used_keys = set()
        for attrib in attribute_split:
            key_end = attrib.index(' ')
            key = attrib[:key_end]
            val = attrib[key_end + 1:].removeprefix('"').removesuffix('"')
            if key in attributes:
                if key in used_keys:
                    attribute_df.at[row_idx, key] = f"{attribute_df.at[row_idx, key]},{val}"
                else:
                    used_keys.add(key)
                    attribute_df.at[row_idx, key] = val
    attribute_df = pl.from_pandas(attribute_df)
    assert len(attribute_df) == len(gtf)

    # Drop unnecessary columns
    gtf.drop_in_place("attribute")
    gtf.drop_in_place("attribute_split")
    
    return pl.concat([
        gtf, attribute_df
    ], how='horizontal')


class Genome():
    """
    Class providing easy access to a downloaded NCBI reference genome.
    """
    _root_dir: Path
    _cache_dir: Path
    _entrez_dir: Path
    _data_report: dict
    _assembly_name: str
    _accession_name: str

    _db_handle: gffutils.FeatureDB | None
    # _gtf: pl.DataFrame | None

    verbose: bool

    def __init__(
            self,
            root_dir: Path | str,
            cache_dir: Path | str | None = None,
            verbose: bool = False,
        ):
        """
        Parameters
        ---
        root_dir: `Path | str`
            The root directory of the downloaded reference genome. Should contain the subdirectory
            `ncbi_dataset`.
        cache_dir: `Path | str | None`, default `None`
            An optional directory at which to store cached data for this genome. Provide if the 
            genome directory will be used by multiple users. If not provided, will choose a location
            inside the genome directory.
        verbose: `bool`, default `False`
            Whether to print extra information when accessing the genome.
        """
        # Verbosity
        self.verbose = verbose

        # Validate root directory
        root_dir = Path(root_dir)
        if not root_dir.is_dir():
            raise FileNotFoundError(f"Directory does not exist: '{root_dir}'")
        if not (root_dir / "ncbi_dataset").is_dir():
            raise FileNotFoundError("Expected subfolder 'ncbi_dataset'; genome directory is improperly formatted")
        if not (root_dir / "ncbi_dataset/data").is_dir():
            raise FileNotFoundError("Expected subfolder 'ncbi_dataset/data'; genome directory is improperly formatted")
        self._root_dir = root_dir

        # Load and parse assembly data report
        self._data_report = load_assembly_data_report(root_dir)
        self._assembly_name = self._data_report['assemblyInfo']["assemblyName"]
        self._accession_name = self._data_report['accession']
        self._organism_name = self._data_report['organism']['organismName']

        # Validate accession directory
        if not self.accession_dir.exists():
            self._print("# WARNING: Dataset is not yet rehydrated! Files cannot be loaded.")

        # Validate cache directory
        if cache_dir is None:
            cache_dir = self.accession_dir / ".cache"
            self._print(f"# Chose default cache directory at '{root_dir.name}/{self.accession_dir.relative_to(root_dir).as_posix()}/.cache'")
        else:
            cache_dir = Path(cache_dir)
        
        if not cache_dir.is_dir():
            cache_dir.mkdir(exist_ok=True)
        self._cache_dir = cache_dir
        self._entrez_dir = self._cache_dir / "entrez"
        self._entrez_dir.mkdir(exist_ok=True)

        # Set handles
        self._db_handle = None


    ##################
    ### Properties ###
    ##################
    
    @property
    def root_dir(self) -> Path:
        """
        The root directory of the downloaded genome.
        """
        return self._root_dir

    @property
    def accession_dir(self) -> Path:
        """
        The directory of the specific genome accession being used.
        """
        return self.root_dir / f"ncbi_dataset/data/{self.accession_name}"

    @property
    def genomic_gtf_fp(self) -> Path:
        """
        Filepath to `genomic.gtf`.
        """
        return self.accession_dir / "genomic.gtf"

    @property
    def genomic_gff_fp(self) -> Path:
        """
        Filepath to `genomic.gff`.
        """
        return self.accession_dir / "genomic.gff"
    
    @property
    def cds_fp(self) -> Path:
        """
        Filepath to `cds_from_genomic.fna`.
        """
        return self.accession_dir / "cds_from_genomic.fna"
    
    @property
    def rna_fp(self) -> Path:
        """
        Filepath to `rna.fna`.
        """
        return self.accession_dir / "rna.fna"

    @property
    def protein_fp(self) -> Path:
        """
        Filepath to `protein.faa`.
        """
        return self.accession_dir / "protein.faa"

    @property
    def sequence_report_fp(self) -> Path:
        """
        Filepath to `sequence_report.jsonl`.
        """
        return self.accession_dir / "sequence_report.jsonl"
    
    @property
    def data_report(self) -> dict:
        """
        Dictionary containing metadata about this genome accession.
        """
        return copy.deepcopy(self._data_report)

    @property
    def assembly_name(self) -> str:
        """
        Name of the assembly, e.g. 'GRCm39'
        """
        return self._assembly_name

    @property
    def accession_name(self) -> str:
        """
        Name of the accession, e.g. 'GCF_000001635.27'
        """
        return self._accession_name

    @property
    def organism_name(self) -> str:
        """
        Name of the organism, e.g. 'mus musculus'
        """
        return self._organism_name

    @property
    def _cache_path_gff_db(self) -> Path:
        return self._cache_dir / f"gff_{self.assembly_name}.db"

    @property
    def _cache_path_gtf_parquet(self) -> Path:
        return self._cache_dir / f"gtf_{self.assembly_name}.parquet"

    @property
    def db(self) -> gffutils.FeatureDB:
        """
        Handle to the relational database generated from `genomic.gff`.

        When used for the first time, must first generate the database, which
        takes some time.
        """
        if self._db_handle is None:
            self._db_handle = self.get_db_handle()
        return self._db_handle

    def __repr__(self) -> str:
        return f"<Genome: assembly='{self.assembly_name}', accession='{self.accession_name}', organism='{self.organism_name}'>"

    #######################
    ### Private Methods ###
    #######################

    def _print(self, x):
        if self.verbose:
            print(x)

    def _load_gtf_new(self) -> pl.DataFrame:
        return parse_gtf_ncbi(self.genomic_gtf_fp, verbose=self.verbose)

    def _generate_gtf_cache(self):
        try:
            gtf = self._load_gtf_new()
            gtf.write_parquet(self._cache_path_gtf_parquet)
        except Exception as e:
            os.remove(self._cache_path_gtf_parquet)
            raise e

    def _generate_db_cache(self):
        db = gffutils.create_db(
            data = self.genomic_gff_fp,
            dbfn = self._cache_path_gff_db,
            keep_order = True,
            merge_strategy = "create_unique",
            checklines=20,
            force=True,
        )
        db.conn.close()

    def _read_entrez_cache_entry(
            self,
            id: str
        ) -> SeqRecord | None:
        fp = self._entrez_dir / f"{id}.gb"
        if fp.is_file():
            with fp.open("rt") as f:
                return SeqIO.read(f, format='genbank')
        else:
            return None

    def _write_entrez_cache_entry(
            self,
            id: str,
            record: SeqRecord,
            raise_if_exists: bool = True,
        ):
        """
        
        """
        fp = self._entrez_dir / f"{id}.gb"
        if raise_if_exists and fp.is_file():
            raise FileExistsError(f"Entrez cache entry already exists for ID '{id}'")
        with fp.open("wt") as f:
            SeqIO.write(record, f, format='genbank')

    def _fetch_entrez(
            self,
            ids: list[str],
            seq_type: Literal['protein', 'nucleotide'],
        ) -> dict[str, SeqRecord]:
        """

        """
        # Load existing IDs
        ret: dict[str, SeqRecord] = dict()
        missing_ids: list[str] = []
        for id in ids:
            record = self._read_entrez_cache_entry(id)
            ret[id] = record
            if record is None:
                missing_ids.append(id)

        # Download missing ids with Entrez
        fetch_handle = Entrez.efetch(db=seq_type, id=missing_ids, rettype="gb", retmode="text")
        try:
            records: list[SeqRecord] = [x for x in SeqIO.parse(fetch_handle, "genbank")]
        finally:
            fetch_handle.close()

        # Check ids
        if len(missing_ids) != len(records):
            raise ValueError(f"Entrez efetch result is of incorrect length ({len(records)}) compared to the query length ({len(missing_ids)})")
        for id, record in zip(missing_ids, records):
            if id != record.id:
                raise ValueError(f"ID mismatch: Queried ID ({id}) vs returned ID {record.id}")
            else:
                ret[id] = record
                self._write_entrez_cache_entry(id, record, raise_if_exists=True)

        return ret

    ######################
    ### Public Methods ###
    ######################

    def rehydrate(self):
        """
        If the genome is currently dehydrated, rehydrates using NCBI's `datasets` tool.
        """
        rehydrate_genome(self.root_dir)


    def get_db_handle(self) -> gffutils.FeatureDB:
        """
        Acquire a handle to the relational database generated from `genomic.gff`.
        """
        if not self._cache_path_gff_db.is_file():
            self._print("Generating new database; this may take a while!")
            self._generate_db_cache()

        return gffutils.FeatureDB(self._cache_path_gff_db)


    def load_gtf(self, use_cache: bool = True) -> pl.DataFrame:
        """
        Loads a slightly trimmed version of `genomic.gtf`.
        """
        if use_cache:
            if not self._cache_path_gtf_parquet.is_file():
                self._print("Generating new GTF cache; this may take a while!")
                self._generate_gtf_cache()
            return pl.read_parquet(self._cache_path_gtf_parquet)
        else:
            return self._load_gtf_new()


    def wipe_cache(
            self,
            *,
            remove_gff_db: bool = False,
            remove_gtf_parquet: bool = False,
            remove_entrez: bool = False,
        ):
        """
        Wipes specific cached data for this genome. Does not modify or delete any NCBI-downloaded genome files.
        Do not use this function lightly, as regenerating the cache can take time!
        """
        if remove_gff_db and self._cache_path_gff_db.is_file():
            os.remove(self._cache_path_gff_db)
        if remove_gtf_parquet and self._cache_path_gtf_parquet.is_file():
            os.remove(self._cache_path_gtf_parquet)
        if remove_entrez and self._entrez_dir.is_dir():
            for file in self._entrez_dir.iterdir():
                if file.is_file():
                    os.remove(file)
                elif file.is_dir():
                    shutil.rmtree(file)


    def query_features_matching_id(
            self,
            feature_id: str,
            feature_type: str,
        ) -> list[str]:
        """
        Returns a list of primary IDs of features matching the provided ID.
        """
        cmd = f"""
        SELECT id
        FROM features
        WHERE (
            id LIKE '{feature_type}-{feature_id}' OR
            id LIKE '{feature_type}-{feature_id}\\_%' ESCAPE '\\'
        )
        """
        cursor = self.db.execute(cmd)
        results = cursor.fetchall()
        return [x[0] for x in results]


    def query_features_matching_description(
            self,
            matches: str | list[str],
            feature_type: str | None = None,
            case_sensitive: bool = False,
        ) -> list[str]:
        """
        Returns a list of primary IDs of features whose descriptions contain one of the provided strings in `matches`.
        """
        # Validate params
        if isinstance(matches, str):
            matches = [matches]

        # Generate command
        if case_sensitive:
            cmd_attributes = "attributes"
        else:
            cmd_attributes = "UPPER(attributes)"
            matches = [match.upper() for match in matches]
        cmd_where = " OR ".join(f"{cmd_attributes} LIKE '%{match}%'" for match in matches)
        cmd = f"""
        SELECT id
        FROM features
        WHERE ({cmd_where})
        """
        if feature_type is not None:
            cmd += f" AND (featuretype = '{feature_type}')"

        # Execute command
        cursor = self.db.execute(cmd)
        results = cursor.fetchall()
        return [x[0] for x in results]


    def fetch_entrez(
            self,
            db_ids: list[str],
            *,
            seq_type: Literal['protein', 'nucleotide'],
        ) -> dict[str, SeqRecord]:
        """
        Fetches sequences using Entrez's efetch utility.

        Parameters
        ---
        db_ids: `list[str]`
            A list of IDs of database features. If two or more indexed features share the same `id` attribute, an error will
            be raised. To prevent this, first pass ID lists through `Genome.get_feature_accession_ids` first, and select only
            unique IDs.
        seq_type: `Literal['protein', 'nucleotide']`
            Whether the specified IDs refer to protein or nucleotide sequences. Counterintuitively, "CDS" features should be
            `"protein"`, while other features are typically `"nucleotide"`.
        """
        # Check params
        if seq_type not in ['protein', 'nucleotide']:
            raise ValueError(f"Unknown seq_type '{seq_type}'")

        # Resolve database IDs
        resolved_ids = self.get_feature_accession_ids(db_ids)
        id_mapping: dict[str, list[str]] = defaultdict(list)
        for db_id, resolved_id in zip(db_ids, resolved_ids):
            id_mapping[resolved_id].append(db_id)
        duplicate_db_ids = [k for k, v in id_mapping.items() if len(v) > 1]
        if duplicate_db_ids:
            raise ValueError(
                f"Queried database IDs point to duplicate features! The following {len(duplicate_db_ids)} keys are problematic:\n" +
                "\n".join(f"{id} : {id_mapping[id]}" for id in duplicate_db_ids) + "\n"
            )

        clean_ids = self.clean_feature_ids(db_ids, raise_if_clean=True)
        return self._fetch_entrez(clean_ids, seq_type = seq_type)


    def get_feature_accession_ids(
            self,
            db_ids: list[str],
        ) -> list[str]:
        """
        Maps database primary keys to their true feature IDs. For database entries with unique feature IDs,
        these will always be identical.
        """
        ret = []
        for db_id in db_ids:
            feature = self.db[db_id]
            true_id = feature.attributes["ID"][0]
            ret.append(true_id)
        return ret


    @staticmethod
    def clean_feature_ids(
            ids: list[str],
            raise_if_clean: bool = True
        ) -> list[str]:
        """
        Removes the featuretype-dependent prefix from the provided IDs.
        """
        known_prefixes = ["rna", "cds", "exon"]
        ret = []
        for id in ids:
            # Attempt to detect known prefix
            for pfx in known_prefixes:
                if id.startswith(f"{pfx}-"):
                    ret.append(id.removeprefix(f"{pfx}-"))
                    break
            # If none found, optionally raise an error
            else:
                if raise_if_clean:
                    raise ValueError(f"ID is already clean, or uses an unknown prefix: '{id}'")
                else:
                    ret.append(id)

        return ret
