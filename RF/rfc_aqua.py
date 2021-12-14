import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score
import statistics

def findTpTnFpFn(df, label):
  labels = [0,1,2]

  tp= df.loc[label,label]
  fp = df[label].sum() - tp
  fn = df.loc[label].sum() - tp
  
  df = df.drop(label, axis=0)
  df = df.drop(label, axis=1)
  
  tn = df.sum().sum()
  
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  specificity = tn / (tn + fp)

  return precision, recall, specificity

raw_data = pd.read_csv('../aqua_data.csv')

level_0 = ['Group', 'left_cerebral_white_matter', 'left_lateral_ventricle', 'left_inferior_lateral_ventricle', 'left_cerebellum_white_matter',

            'left_cerebellum_cortex', 'left_thalamus_proper', 'left_caudate', 'left_putamen', 'left_pallidum', '3rd_ventricle', '4th_ventricle',

           'left_hippocampus', 'left_amygdala', 'left_accumbens_area', 'left_ventraldc', 'right_cerebral_white_matter', 'right_lateral_ventricle',

           'right_inferior_lateral_ventricle', 'right_cerebellum_white_matter', 'right_cerebellum_cortex', 'right_thalamus_proper', 'right_caudate',

           'right_putamen', 'right_pallidum', 'right_hippocampus', 'right_amygdala', 'right_accumbens_area', 'right_ventraldc', 'corpus_callosum',

           'ctx_left_bankssts', 'ctx_left_caudal_anterior_cingulate', 'ctx_left_caudal_middle_frontal', 'ctx_left_cuneus', 'ctx_left_entorhinal',

           'ctx_left_fusiform', 'ctx_left_inferior_parietal', 'ctx_left_inferior_temporal', 'ctx_left_isthmus_cingulate', 'ctx_left_lateral_occipital',

           'ctx_left_lateral_orbito_frontal', 'ctx_left_lingual', 'ctx_left_medial_orbito_frontal', 'ctx_left_middle_temporal', 'ctx_left_parahippocampal',

           'ctx_left_paracentral', 'ctx_left_pars_opercularis', 'ctx_left_pars_orbitalis', 'ctx_left_pars_triangularis', 'ctx_left_pericalcarine',

           'ctx_left_postcentral', 'ctx_left_posterior_cingulate', 'ctx_left_precentral', 'ctx_left_precuneus', 'ctx_left_rostral_anterior_cingulate',

           'ctx_left_rostral_middle_frontal', 'ctx_left_superior_frontal', 'ctx_left_superior_parietal', 'ctx_left_superior_temporal',

            'ctx_left_supramarginal', 'ctx_left_frontal_pole', 'ctx_left_temporal_pole', 'ctx_left_transverse_temporal', 'ctx_left_insula',

            'ctx_right_bankssts', 'ctx_right_caudal_anterior_cingulate', 'ctx_right_caudal_middle_frontal', 'ctx_right_cuneus', 'ctx_right_entorhinal',

           'ctx_right_fusiform', 'ctx_right_inferior_parietal', 'ctx_right_inferior_temporal', 'ctx_right_isthmus_cingulate', 'ctx_right_lateral_occipital',

           'ctx_right_lateral_orbito_frontal', 'ctx_right_lingual', 'ctx_right_medial_orbito_frontal', 'ctx_right_middle_temporal',

           'ctx_right_parahippocampal', 'ctx_right_paracentral', 'ctx_right_pars_opercularis', 'ctx_right_pars_orbitalis', 'ctx_right_pars_triangularis',

           'ctx_right_pericalcarine', 'ctx_right_postcentral', 'ctx_right_posterior_cingulate', 'ctx_right_precentral', 'ctx_right_precuneus',

           'ctx_right_rostral_anterior_cingulate', 'ctx_right_rostral_middle_frontal', 'ctx_right_superior_frontal', 'ctx_right_superior_parietal',

           'ctx_right_superior_temporal', 'ctx_right_supramarginal', 'ctx_right_frontal_pole', 'ctx_right_temporal_pole', 'ctx_right_transverse_temporal',

           'ctx_right_insula']

level_1 = ['Group', 'BrainSegNotVent', 'BrainSegNotVentSurf',
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
level_2 = ['Group', 'VentricleChoroidVol',
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

target_level = level_0
roi_data = raw_data[target_level]
roi_data['Group'] = roi_data['Group'].replace('CN', 0)
roi_data['Group'] = roi_data['Group'].replace('MC', 1)
roi_data['Group'] = roi_data['Group'].replace('AD', 2)

roi_data = roi_data.assign(Group=lambda s: s['Group'].astype('int'))
roi_data = roi_data.mask(roi_data['Group'] < 0).dropna()

features = roi_data.mask(roi_data.eq("None")).dropna().astype('float')
features_x = features[list(filter(lambda x: x != "Group", list(features.columns)))].values
features_y = features['Group'].values

# ------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score

n_fold = 10
rkf = RepeatedKFold(n_splits=n_fold, n_repeats=n_fold, random_state=777)
total_acc = 0.
rfr = []
acc_list = []

labels = [0, 1, 2]
df_sum = pd.DataFrame(data = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], columns = labels, index = labels)

zero_prec = []
zero_recall = []
zero_spec = []

one_prec = []
one_recall = []
one_spec = []

two_prec = []
two_recall = []
two_spec = []

for idx, (train_idx, test_idx) in enumerate(rkf.split(features_x)):
	train_x, test_x = features_x[train_idx], features_x[test_idx]
	train_y, test_y = features_y[train_idx], features_y[test_idx]
	
	rfr.append(RandomForestClassifier(n_estimators=5000, n_jobs=25, random_state=777).fit(train_x, train_y))
	result_rf = rfr[idx].predict(test_x)
	acc_rf = accuracy_score(test_y, result_rf)
	
	df = pd.DataFrame(
			data=confusion_matrix(test_y, result_rf, labels = labels),
			columns=labels,
			index=labels
			)

	df_sum = df_sum.add(df)

	total_acc += acc_rf
	acc_list.append(acc_rf)

	if (idx + 1) % 10 == 0:
		print(idx + 1)
		
		prec_0, rec_0, spec_0 = findTpTnFpFn(df_sum, 0)
		prec_1, rec_1, spec_1 = findTpTnFpFn(df_sum, 1)
		prec_2, rec_2, spec_2 = findTpTnFpFn(df_sum, 2)

		zero_prec.append(prec_0)
		zero_recall.append(rec_0)
		zero_spec.append(spec_0)

		one_prec.append(prec_1)
		one_recall.append(rec_1)
		one_spec.append(spec_1)

		two_prec.append(prec_2)
		two_recall.append(rec_2)
		two_spec.append(spec_2)

		df_sum = pd.DataFrame(data = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], columns = labels, index = labels)

print("zero_prec = %s" %(statistics.mean(zero_prec)))
print("zero_recall = %s" %(statistics.mean(zero_recall)))
print("zero_spec = %s" %(statistics.mean(zero_spec)))
print()
print("one_prec = %s" %(statistics.mean(one_prec)))
print("one_recall = %s" %(statistics.mean(one_recall)))
print("one_spec = %s" %(statistics.mean(one_spec)))
print()
print("two_prec = %s" %(statistics.mean(two_prec)))
print("two_recall = %s" %(statistics.mean(two_recall)))
print("two_spec = %s" %(statistics.mean(two_spec)))
print()
total_acc /= (n_fold*n_fold)
print("Total Acc: %f" % total_acc)
df = pd.DataFrame(acc_list)
df.to_csv('RF_output', header=False, index=False)

#import pickle
#with open('RF_model', 'wb') as f:
#	pickle.dump(rfr, f)

# -----------------------------------------------------------
'''
n_feature = len(features_x)
index = np.arange(n_feature)

feature_importances = np.array([x.feature_importances_ for x in rfr]).mean(axis=0)
_label = target_level[1:]
f = {l:d for l, d in zip(_label, feature_importances)}
f = np.array(sorted(f.items(), key=(lambda x:x[1])))
_x = f[:, 1].astype('float')
_y = f[:, 0]

plt.figure(figsize=(8, 8))
plt.barh(index, _x, align='center')
plt.yticks(index,_y)
plt.ylim(-1, n_feature)
plt.xlim(0, 0.08)
plt.xlabel('feature importance')
plt.ylabel('feature')

ax = plt.gca()
ax.xaxis.grid(True, color='lightgrey', linestyle='--')
ax.set_axisbelow(True)
plt.plot([1/n_feature, 1/n_feature], [-10, 100], color='red')

# plt.savefig(str(n_feature) + '.png', dpi=300, bbox_inches='tight')
plt.show()
'''
