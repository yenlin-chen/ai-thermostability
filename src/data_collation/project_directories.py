import os.path as osp

self_dir = osp.dirname(osp.normpath(__file__))

root_dir = osp.dirname(osp.dirname(osp.join(self_dir)))
data_dir = osp.join(root_dir, 'data')

external_dir = osp.join(data_dir, 'external')
collation_dir = osp.join(data_dir, 'collation')

deepstabp_ext = osp.join(external_dir,
                             'DeepSTABp',
                             'Melting_temperatures_of_proteins')

deepstabp_col = osp.join(collation_dir, 'DeepSTABp')
