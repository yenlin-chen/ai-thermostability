if __name__ == '__main__':
    from __init__ import external_dir, collation_dir, processed_dir, res_to_1hot
    from retrievers import AlphaFold_Retriever
    from enm import TNM_Computer
else:
    from .__init__ import external_dir, collation_dir, processed_dir, res_to_1hot
    from .retrievers import AlphaFold_Retriever
    from .enm import TNM_Computer

import os, torch, prody, json
import numpy as np
import torch_geometric as pyg
from transformers import BertModel, BertTokenizer

from tqdm import tqdm

df_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prody.confProDy(verbosity='none')
tnm_setup_filename = '{}{}_CA{:.1f}_ALL_PHIPSIPSI'

class DeepSTABp_Dataset(pyg.data.Dataset):

    '''Dataset built upon Tm data provided by DeepSTABp

    This is an implementation of the Dataset class from PyTorch
    Geometric. The underlying proteins are taken from the GitLab of
    DeepSTABp (L-I-N-K). From the list of assessions from the DeepSTABp
    GitLab, protein structure is downloaded from AlphaFoldDB. The
    structure is then analyzed using the TNM software to compute normal
    modes based on the TNM model, which is in turn trasformed into
    dynamical coupling graphs as described in (P-A-P-E-R).

    Two node feature vectors are associated with the graph nodes: 1) 20-
    dimensional one-hot-vectors denoting the residue type, and 2) the
    1025-dimensional embedding of the AA sequence encoded using the
    ProtBert model. Additionally, the pLDDT (AlphaFold's confidence in
    the predicted position of each residue) and b-factor prediction from
    the TNM software are also used as node features.

    '''

    def __init__(self, experiment=None, organism=None, cell_line=None,
                 version='v4-higher_threshold', transform=None, device=df_device):

        self.device = device
        self.version = version

        ### ASSIGN ARGUMENTS TO CLASS ATTRIBUTES (AND CHECK FOR VALIDITY)
        if experiment not in ['lysate', 'cell', None]:
            raise ValueError('experiment must be either "lysate" or "cell"')
        self.experiment = experiment

        if organism not in ['human', None]:
            raise ValueError('organism must be either "human" or None')
        self.organism = organism

        if isinstance(cell_line, str):
            if organism != 'human':
                raise ValueError(
                    'cell line can only be specified for "human" organism'
                )
            if cell_line.lower() == 'jurkat':
                if experiment == 'lysate':
                    raise ValueError(
                        'cell line "jurkat" is not available for "lysate" experiment'
                    )
            elif cell_line.lower() != 'k562':
                raise ValueError(
                    'cell line must be "k562" or "jurkat" or None'
                )
        elif cell_line is not None:
                raise ValueError(
                    'cell line must be "k562" or "jurkat" or None'
                )
        self.cell_line = cell_line

        ### READ ACCESSIONS AND TM FROM PROSTABP
        if self.organism == 'human':
            if cell_line == 'k562':
                self.set_name = f'{self.experiment}-{self.organism}-{self.cell_line}'
            elif cell_line == 'jurkat':
                self.set_name = f'{self.experiment}-{self.organism}-{self.cell_line}'
            else:
                self.set_name = f'{self.experiment}-{self.organism}'
        else:
            self.set_name = f'{self.experiment}'
        filename = f'{self.set_name}.csv'

        self.meta_file = os.path.join(collation_dir, 'DeepSTABp', filename)
        print(f' -> Generating dataset from {self.meta_file}')
        try:
            self.meta = np.loadtxt(
                self.meta_file, dtype=np.str_, delimiter=',', skiprows=1
            )
        except FileNotFoundError as err:
            raise FileNotFoundError(
                f'{self.meta_file} does not exist. '
                f'Please run DeepSTABp.ipynb in "data_collation" directory.'
            ) from err
        print(f' -> Number of entries in meta file    : {len(self.meta)}')

        ### SOME Tm STATISTICS (BEFORE PROCESSING)
        all_Tm_dict = {e[0]: float(e[1]) for e in self.meta}
        self.all_accessions = list(all_Tm_dict.keys()) # look out for duplicates
        all_Tm = list(all_Tm_dict.values())

        # compute range of Tm (to be used for normalization)
        all_Tm_mean = np.mean(all_Tm)
        all_Tm_max  = np.amax(all_Tm)
        all_Tm_min  = np.amin(all_Tm)
        print(f'     >> mean value of Tm  : {all_Tm_mean:.4f}')
        print(f'     >> range of Tm       : {all_Tm_min:.4f}-{all_Tm_max:.4f}')

        print(' -> Number of unique accessions       :',
              len(self.all_accessions))

        assert len(self.meta) == len(self.all_accessions)

        ### EXTRACT OGT
        self.all_ogt_dict = {e[0]: float(e[2]) for e in self.meta}

        ### EXTRACT SPECIES
        self.all_species_dict = {e[0]: str(e[3]) for e in self.meta}

        ### COMMON ATTRIBUTES
        self.af_retriever = AlphaFold_Retriever()
        self.tnm_computer = TNM_Computer()

        ### CONSTRUCTOR OF PARENT CLASS
        super().__init__(self.raw_dir, transform, None, None)

        # save computation time by not re-running this function
        self.processable_accessions = self.get_processable_accessions()

        ### MORE TM STATISTICS (AFTER PROCESSING)
        print(' -> Final number of accessions        :',
              len(self.processable_accessions))

        self.Tm_dict = {a: all_Tm_dict[self.af_retriever.unmodify_accession(a)]
                        for a in self.processable_accessions}
        all_Tm = list(self.Tm_dict.values())

        self.Tm_mean = np.mean(all_Tm)
        self.Tm_max  = np.amax(all_Tm)
        self.Tm_min  = np.amin(all_Tm)
        print(f'     >> mean value of Tm: {self.Tm_mean:.4f}')
        print(f'     >> range of Tm     : {self.Tm_min:.4f}-{self.Tm_max:.4f}')
        print(' -> Number of unique accessions       :',
              len(self.Tm_dict))

        print('Dataset instantiation complete.')

    @property
    def raw_dir(self):
        return os.path.join(external_dir, 'AlphaFoldDB', 'pae')

    @property
    def downloadable_accessions(self):
        '''Accessions for which AlphaFold structures should be found.

        Reads list of accessions whose AlphaFold structures are
        unavailable and removes them from the dataset. Returned values
        are UniProt accessions. For project-internal accessions, call
        `AlphaFold_Retriever.modify_accession()` on the outputted list.
        '''
        failed_accessions = self.af_retriever.get_failed_accessions()
        return np.setdiff1d(self.all_accessions, failed_accessions)

    @property
    def raw_file_names(self):
        # project-internal accessions are used as file names
        # "<UniProt accession>-AFv<version number>"

        modified_accessions = self.af_retriever.modify_accession(
            self.downloadable_accessions
        )
        return [f'{a}.json' for a in modified_accessions]

    @property
    def processed_dir(self):
        return os.path.join(processed_dir, self.version)

    def get_processable_accessions(self):
        '''Accessions for which TNM processing should be successful.

        Reads list of accessions whose TNM results are unobtainable and
        removes them from the dataset.
        '''
        failed_accessions = self.tnm_computer.get_failed_accessions()
        modified_downloadable = self.af_retriever.modify_accession(
            self.downloadable_accessions
        )
        return np.setdiff1d(modified_downloadable, failed_accessions)

    @property
    def processed_file_names(self):
        return np.char.add(self.get_processable_accessions(), '.pt').tolist()

    def download(self):
        print('\nDownloading AlphaFold structures and PAEs...')
        successful_accessions, _ = self.af_retriever.batch_retrieve(
            self.all_accessions,
            item_list=['pdb', 'pae']
        )
        print(' -> Accessions successfully downloaded:',
              len(successful_accessions))

    def process(self):
        # pyg prints "Processing..." internally
        # no need for print statement
        # print('Processing AlphaFold structures...')

        ################################################################
        # modal analysis by TNM
        ################################################################
        modified_downloadable = self.af_retriever.modify_accession(
            self.downloadable_accessions
        )
        pdb_path_list = [self.af_retriever.path_to_file(a, item_type='pdb')
                         for a in self.downloadable_accessions]
        successful_accessions, _ = self.tnm_computer.batch_run(
            modified_downloadable, pdb_path_list,
            timeout=20, # increase if necessary
            debug=True  # set to true for WSL
        )
        print(' -> Accessions with TNM results       :',
              len(successful_accessions))

        ################################################################
        # build pyg graph objects
        ################################################################

        # ProteinBERT instances
        tokenizer = BertTokenizer.from_pretrained(
            'Rostlab/prot_bert_bfd',
            do_lower_case=False
        )
        protbert = BertModel.from_pretrained(
            'Rostlab/prot_bert_bfd'
        ).to(self.device)

        pbar = tqdm(self.get_processable_accessions(),
                    dynamic_ncols=True, ascii=True)
        for accession in pbar:
            pbar.set_description(f'Graphs {accession:<12s}')
            path_to_files = self.tnm_computer.path_to_outputs(accession)

            # get AA sequence (as resIDs)
            resnames = self.tnm_computer.get_resnames(accession)
            n_residues = resnames.size
            # convert resIDs to one-hot-encoded vectors
            resnames_1hot = np.zeros((n_residues,20), dtype=np.int_)
            for j, resname in enumerate(resnames):
                resnames_1hot[j, res_to_1hot[resname]] = 1

            # create pyg data object from TNM output
            data = pyg.data.HeteroData()
            data.accession = accession
            data['residue'].res1hot = torch.from_numpy(resnames_1hot)
            data.ogt = self.all_ogt_dict[
                self.af_retriever.unmodify_accession(accession)
            ]
            data.species = self.all_species_dict[
                self.af_retriever.unmodify_accession(accession)
            ]

            ### ADD PROTBERT ENCODING AS NODE ATTRIBUTES
            sequence = ' '.join(resnames)
            encoded_input = tokenizer(
                sequence,
                return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                bert_encoding = protbert(**encoded_input).to_tuple()[0]
            data['residue'].x = bert_encoding.detach()[0,1:-1].cpu()

            ### ADD pLDDT AS NODE ATTRIBUTES
            # AlphaFold populates the B-factor column with pLDDT scores
            # which we will include as node attributes
            atoms = prody.parsePDB(
                self.af_retriever.path_to_file(accession, item_type='pdb'),
                subset='ca'
            )
            pLDDT = atoms.getBetas() # range: [0,100]
            data['residue'].pLDDT = torch.from_numpy(pLDDT)

            ### ADD CONTACT EDGES FROM PRODY
            # contact edges saved by the TNM program is somewhat sporadic
            # here the contact map is based on the position of Ca atoms
            anm = prody.ANM(name='accession_CA')
            anm.buildHessian(atoms, cutoff=12)
            cont = -anm.getKirchhoff().astype(np.int_) # the Laplacian matrix
            np.fill_diagonal(cont, 1) # contact map completed here (with loops)
            edge_index = np.argwhere(cont==1).T # undirected graph
            data['residue', 'contact', 'residue'].edge_index = torch.from_numpy(
                edge_index
            )

            # ### ADD CONTACT EDGES FROM TNM
            # # for some accessions, TNM will generate a blank file for
            # # contact maps
            # raw_data = np.loadtxt(path_to_files['cont'], dtype=np.str_)
            # if raw_data.size == 0:
            #     tqdm.write(f' -> No contact map for {accession}')
            #     continue
            # # convert indices to 0-based
            # edge_index = raw_data[:,:2].astype(np.int_).T - 1
            # edge_index = np.hstack(
            #     (edge_index, np.flip(edge_index, axis=0))
            # )
            # # assign edge indices to HeteroData object
            # data['residue', 'contact', 'residue'].edge_index = (
            #     torch.from_numpy(edge_index)
            # )
            # # keep edge weights
            # data['residue', 'contact', 'residue'].edge_weight = (
            #     torch.from_numpy(raw_data[:,2].astype(np.float_))
            # )

            ### ADD BACKBONE CONNECTION
            res_idx = np.arange(n_residues-1)
            edge_index = np.vstack((res_idx, res_idx+1))
            edge_index = np.hstack(
                (edge_index, np.flip(edge_index, axis=0))
            )
            data['residue', 'backbone', 'residue'].edge_index = (
                torch.from_numpy(edge_index)
            )

            ### ADD DYNAMICAL COUPLING EDGES
            # 1. self-loops are not included
            # 2. indices are 0-based
            #    Yes, TNM implements two different indexing systems :(
            for k in ['coord', 'codir', 'deform']:

                # read TNM output
                raw_data = np.loadtxt(path_to_files[k], dtype=np.str_)
                raw_data = raw_data.reshape((-1,3))

                # assign edge indices to HeteroData object
                edge_index = raw_data[:,:2].astype(np.int_).T
                edge_index = np.hstack(
                    (edge_index, np.flip(edge_index, axis=0))
                )
                data['residue', k, 'residue'].edge_index = torch.from_numpy(
                    edge_index
                )

                # keep record of edge weights
                data['residue', k, 'residue'].edge_weight = (
                    torch.from_numpy(raw_data[:,2].astype(np.float_))
                )

            ### ADD PREDICTED B-FACTORS AS NODE ATTRIBUTES
            # assign B-factors as predicted by TNM to graph nodes
            raw_data = np.loadtxt(path_to_files['bfactor'], dtype=np.str_)[:,1]
            bfactor = raw_data.astype(np.float_)
            data['residue'].bfactor = torch.from_numpy(bfactor)

            # # remove duplicates from all_edges
            # all_edges = np.unique(all_edges, axis=1)
            # print(all_edges)

            ### ADD PREDICTED ALIGNED ERROR AS EDGES
            # a cutoff is applied to the PAE matrix to determine
            # which edges to keep (done for computational efficiency)

            # PAE is directed, i.e. the PAE matrix is not symmetric.
            # However the upper and lower triangle are very similar
            # so we will make all edges undirected
            pae_path = self.af_retriever.path_to_file(
                accession, item_type='pae'
            )
            with open(pae_path, 'r') as f:
                pae_dict = json.load(f)[0]
            pae = np.array(pae_dict['predicted_aligned_error'])

            edge_index = np.argwhere(pae<=4).T # directed edges
            edge_index = np.hstack( # will include duplicates
                (edge_index, np.flip(edge_index, axis=0))
            )
            edge_index = np.unique(edge_index, axis=1) # remove duplicates
            data['residue', 'pae', 'residue'].edge_index = torch.from_numpy(
                edge_index
            )

            # print(data)
            assert data.is_undirected()
            torch.save(
                data,
                os.path.join(self.processed_dir, f'{accession}.pt')
            )

        # cleaup (reclaim device memory)
        del protbert

        print(' -> Accessions successfully processed :',
              len(self.get_processable_accessions()))

    def len(self):
        return self.processable_accessions.size

    def get(self, idx):
        data = torch.load(
            os.path.join(
                self.processed_dir,
                self.processable_accessions[idx] + '.pt'
            )
        )
        return data

if __name__ == '__main__':

    dataset = DeepSTABp_Dataset(
        experiment='lysate',
        organism=None,
        cell_line=None,
        version='v5-sigma2_cutoff12_species'
    )
