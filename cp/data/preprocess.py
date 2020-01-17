import pandas as pd
import numpy as np
import re
import sys
import os

class BaseParser(object):

    def __init__(self, filename):
        """Base class for parsing .pdb and.cif files.

        Parameters
        ----------
        filename : path to .cif or .pdb file.
        """
        self.filename = filename
        self.file_content = self._gather_data()
        self._NMAX = 10**3

    def _gather_data(self):
        """Gather data in file as a string.

        Returns
        -------
        file_content : string containing file content..
        """
        try:
            with open(self.filename, 'r', encoding='utf8') as file:
                file_content = file.read()
                return file_content
        except FileNotFoundError:
            print("File not found, couldn't gather data.")

    def display_content(self, verbose=False):
        """Display content stored on file_content.

        Parameters
        ---------
        verbose : if True, output the entire file_content variable. Else,
        output NMAX characters from file_content.
        """
        if verbose:
            print(self.file_content)
        else:
            print(self.file_content[:self._NMAX])

class PDBParser(BaseParser):

    def __init__(self, filename):
        """Class for parsing .pdb files, gathering experiment
        and structure info.

        Parameters
        ----------
        filename : path to .pdb file
        """
        super().__init__(filename)
        self.pdbheader = {
            'SYNCHROTRON': '',
            'SOLV': 0.0,
            'WILSON': '',
            'MATTHEWS': 0.0,
        }

    def parse(self):
        """Parse experiment info from .pdb file."""
        info_regex = ["SYNCHROTRON.*\(Y/N\).*: (\w)",
                      "SOLVENT CONTENT.*\(\%\).*: (\-?\d+\.\d+)",
                      "FROM WILSON PLOT.*\(.*\).*: (\-?\d+\.\d+|NULL)",
                      "MATTHEWS COEFFICIENT.*\(.*\).*: (\-?\d+\.\d+|NULL)"]
        for index, key in enumerate(self.pdbheader.keys()):
            match = re.findall(info_regex[index], self.file_content)
            if match:
                if key == 'SOLV':
                    self.pdbheader[key] = float(match[0])
                elif(key == 'WILSON' or key == 'MATTHEWS') and match[0] != 'NULL':
                    self.pdbheader[key] = float(match[0])
                else:
                    self.pdbheader[key] = match[0]

    def header_to_df(self, columns=None):
        """Convert .pdb file header into a flatten pandas DataFrame."""
        pdbheader_df = pd.DataFrame(self.pdbheader, index=[0])
        if columns:
            return pdbheader_df.loc[:, columns]
        else:
            return pdbheader_df

    def header_to_series(self, columns=None):
        """Convert .pdb file header into a pandas Series."""
        pdbheader_series = pd.Series(self.pdbheader)
        if columns:
            return pdbheader_series.loc[columns]
        else:
            return pdbheader_series

class CIFParser(BaseParser):

    def __init__(self, filename, 
                cif_columns=[
                "index_h", "index_k", "index_l", 
                "FOBS", "SIGFOBS", "UOBS", 
                "SIGUOBS", "FC", "PHI", "FOM", 
                "RESOL", "pdbx_r_free_flag"
                ]):
        """Class for parsing .cif (Crystallography Information File) files.
        Based on regex matching on the header and reflections, separately.

        Parameters
        ----------
        filename : path to .cif file.
        cif_columns : reflection columns to be parsed.
        """
        super().__init__(filename)
        self.cifheader = {
            '_space_group':
            {
                'crystal_system':'',
                'IT_number': 0,
                'name_H-M_alt': '',
                'name_Hall': '',
            },
            '_symmetry':
            {
                'space_group_name_H-M': '',
                'space_group_name_Hall': '',
                'Int_Tables_number':0,
            },
            '_cell':
            {
                'length_a': 0.0,
                'length_b': 0.0,
                'length_c': 0.0,
                'angle_alpha': 0.0,
                'angle_beta': 0.0,
                'angle_gamma': 0.0,
                'volume': 0.0,
            }
        }

        self.reflections = {
        "_refln":
        {
            column:[] for column in cif_columns
        }
    }

    def header_parse(self):
        """Parse numeric and textual info on .cif header."""
        for super_key in self.cifheader:
            for sub_key in self.cifheader[super_key]:
                if 'name' in sub_key:
                    header_name_regex = "%s.%s .+ '(.*?)'\n"%(super_key, sub_key)
                    match = re.findall(header_name_regex, self.file_content)
                else:
                    header_num_regex = '%s.%s .+ ([0-9]+\.?[0-9]+|\w+)\n'%(super_key, sub_key)
                    match = re.findall(header_num_regex, self.file_content)
                if match:
                    # convert needed features to float
                    if super_key == "_cell":
                        self.cifheader[super_key][sub_key] = float(match[0])
                    elif (sub_key == "IT_number" or sub_key == "Int_Tables_number"):
                        self.cifheader[super_key][sub_key] = int(match[0])
                    else:
                        self.cifheader[super_key][sub_key] = match[0]



    def reflections_parse(self, integer_columns=['index_h', 'index_l', 'index_k', 'pdbx_r_free_flag']):
        """Parse reflections on .cif file.

        Parameters
        ----------
        integer_indexes : reflection columns containing 
        integer entries.
        """
        dict_size = len(self.reflections['_refln'])
        reflection_regex = "([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)[ ]+"*(dict_size-1) + "([-+]?[0-9]*\.[0-9]+|[0-9]+)\n"
        full_match = re.findall(reflection_regex, self.file_content)
        for reflection in full_match:
            for index, key in enumerate(self.reflections['_refln'].keys()):
                if key in integer_columns:
                    self.reflections['_refln'][key].append(int(reflection[index]))
                else:
                    self.reflections['_refln'][key].append(float(reflection[index]))

    def parse(self, integer_indexes=['index_h', 'index_l', 'index_k', 'pdbx_r_free_flag']):
        """Parse both header and reflections."""
        self.reflections_parse(integer_columns=integer_indexes)
        self.header_parse()

    def get_pdb_name(self):
        """Get PDB_ID of the file being parsed.

        Return
        ------
        name : PDB_ID of the file being parsed.
        """
        name = re.findall('data_(\w{4})', self.file_content)
        return name[0]

    def reflections_to_df(self, columns=None):
        """Convert reflection dictionary to a pandas DataFrame.

        Return
        ------
        refln_df : pandas DataFrame containing all information
        stored on the reflection dictionary.
        """
        refln_df = pd.DataFrame.from_dict(self.reflections['_refln'])
        if columns:
            return refln_df.loc[:, columns]
        else:
            return refln_df

    def header_to_df(self, columns=None):
        """Convert header dictionary to a pandas DataFrame.

        Return
        ------
        cifheader_df : pandas DataFrame containing all information
        stored on the header dictionary.
        """
        flat_header = {}
        for sup_key in self.cifheader:
            flat_header.update(self.cifheader[sup_key])
        cifheader_df = pd.DataFrame(flat_header, index=[0])
        if columns:
            return cifheader_df.loc[:, columns]
        else:
            return cifheader_df

    def header_to_series(self, columns=None):
        """Convert header dictionary to a pandas Series.

        Return
        ------
        cifheader_series : pandas Series containing all information
        stored on the header dictionary.
        """
        flat_header = {}
        for sup_key in self.cifheader:
            flat_header.update(self.cifheader[sup_key])
        cifheader_series = pd.Series(flat_header)
        if columns:
            return cifheader_series.loc[columns]
        else:
            return cifheader_series

    def header_refln_df(self, headercols=None, reflncols=None, phierror=False):
        """Convert header and reflection into a single DataFrame. Header
        and reflections columns can be specified via headercols and
        reflncols arguments.

        Parameters
        ----------
        headercols : header columns to be included in the final DataFrame.
        reflncols : reflection columns to be included in the final DataFrame.

        Return
        ------
        header_refln_df : DataFrame containing both header and reflection info.
        """
        refln_df = self.reflections_to_df(columns=reflncols)
        header_series = self.header_to_series(columns=headercols)
        for key in header_series.keys():
            refln_df[key] = [header_series.get(key)]*len(refln_df)
        if phierror:
            refln_df['PHI_ERROR'] = 180.0/np.pi * np.arccos(refln_df['FOM'])
        return refln_df

def mw_ext_parse(filename):
    """Gather molecular weight (MW) data stored on
    external file.

    Parameters
    ----------
    filename : path to Excel file.

    Returns
    -------
    mw_sheet : pandas sheet with parsed info.
    """
    mw_sheet = pd.read_excel(io=filename)
    return mw_sheet

def cifpdb_df(cifdf, pdbheader_series):
    """Merge CIF pandas DataFrame and PDB header
    pandas Series into a single pandas DataFrame.

    Parameters
    ---------
    cifdf : CIF pandas DataFrame.
    pdbheader_series : PDB header pandas Series.

    Returns
    -------
    master_df : pandas DataFrame containing PDB and
    CIF parsed information.
    """
    master_df = cifdf.copy()
    for key in pdbheader_series.keys():
        master_df[key] = pdbheader_series[key]
    return master_df