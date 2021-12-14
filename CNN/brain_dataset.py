import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler

class BrainDataset(Dataset):
	_label_col = 'Group'
	_level_0 = ['3rd_ventricle', '4th_ventricle', 'left_accumbens',
	'right_accumbens', 'left_amygdala', 'right_amygdala', 'left_caudate',
	'right_caudate', 'left_cerebellum_white_matter',
	'left_cerebellum_cortex', 'right_cerebellum_white_matter',
	'right_cerebellum_cortex', 'left_cerebral_white_matter',
	'right_cerebral_white_matter', 'left_cingulate_cortex',
	'right_cingulate_cortex', 'left_frontal_lobe', 'right_frontal_lobe',
	'left_hippocampus', 'right_hippocampus', 'left_insula', 'right_insula',
	'left_lateral_ventricle', 'right_lateral_ventricle',
	'left_occipital_lobe', 'right_occipital_lobe', 'left_pallidum',
	'right_pallidum', 'left_parietal_lobe', 'right_parietal_lobe',
	'left_putamen', 'right_putamen','left_ventraldc', 'right_ventraldc',
	'left_temporal_lobe', 'right_temporal_lobe', 'left_thalamus_proper',
	'right_thalamus_proper']

	_level_1 = ['BrainSegNotVent', 'BrainSegNotVentSurf',
				'VentricleChoroidVol',
		        'lhCortex', 'rhCortex', 'Cortex',
				'lhCerebralWhiteMatter', 'rhCerebralWhiteMatter', 'CerebralWhiteMatter',
				'SubCortGray', 'TotalGray', 
				'BrainSegVol-to-eTIV',
		        'lhSurfaceHoles',
		        'EstimatedTotalIntraCranialVol', 
				'Left-Lateral-Ventricle', 'Right-Lateral-Ventricle',
		        'Left-Inf-Lat-Vent', 'Right-Inf-Lat-Vent',
		        'Left-Putamen', 'Right-Putamen',
				'3rd-Ventricle',
				'Left-Hippocampus', 'Right-Hippocampus',
				'Left-Amygdala', 'Right-Amygdala',
		        'Left-Accumbens-area', 'Right-Accumbens-area',
		        'WM-hypointensities', 
		        'Optic-Chiasm']
	
	_level_2 = ['VentricleChoroidVol',
				'lhCortex', 'rhCortex', 'Cortex',
				'SubCortGray', 'TotalGray',
				'BrainSegVol-to-eTIV',
				'lhSurfaceHoles', 
				'Left-Lateral-Ventricle', 'Right-Lateral-Ventricle',
				'Left-Inf-Lat-Vent', 'Right-Inf-Lat-Vent',
				'Left-Putamen', 'Right-Putamen',
				'3rd-Ventricle',
				'Left-Hippocampus', 'Right-Hippocampus',
				'Left-Amygdala', 'Right-Amygdala',
				'Left-Accumbens-area', 'Right-Accumbens-area',
				'WM-hypointensities']

	def __init__(self, file_path, expand_dim=False, level=0):
		super(BrainDataset, self).__init__()
		raw_data = pd.read_csv(file_path)

		# Label Categorization
		label = raw_data['Group']
		self.label = label.replace(['CN', 'MCI', 'AD'], [0, 1, 2]).astype(int).to_numpy()

		_level = self._level_0 if level == 0 else \
				 self._level_1 if level == 1 else \
				 self._level_2

		# Data Normalization (Z score)
		data = raw_data.loc[:, _level].to_numpy()
		scaler = StandardScaler().fit(data)
		self.data = scaler.transform(data)
		assert len(label) == len(data)

		if expand_dim:
			self.data = np.expand_dims(self.data, axis=1)

	def __getitem__(self, index):
		return self.data[index], self.label[index]
	
	def __len__(self):
		return len(self.data)

