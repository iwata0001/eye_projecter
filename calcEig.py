# preData, preData2で作成したデータをモデルにするための主成分分析をする
# 実行に時間がかかるのでファイルに保存

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from ipywidgets import FloatSlider
import copy
import math
import pywt
import os

import preData2 as pre2

#主成分分析のデータ保存用 実行に時間がかかる

"""
# 多重ウェーブレットデータ固有値
covMat = np.cov(pre2.dataCentArr.T)
dataEig = np.linalg.eig(covMat)
np.save('saves/eigValLev2_'+str(pre2.LC)+'_'+str(pre2.MC)+'_'+str(pre2.HC)+'_'+str(pre2.handC), dataEig[0].real)
np.save('saves/eigVecLev2_'+str(pre2.LC)+'_'+str(pre2.MC)+'_'+str(pre2.HC)+'_'+str(pre2.handC), dataEig[1].real.T)
"""
##################################################################################################################################

"""
#　生画像データ固有値
covMat = np.cov(pre2.eyeDatasCenter.T)
eyeDatasEig = np.linalg.eig(covMat)
np.save('saves/eye_eig_val_13p_mesh_coef'+str(pre2.handCoeff), eyeDatasEig[0].real)
np.save('saves/eye_eig_vec_13p_mesh_coef'+str(pre2.handCoeff), eyeDatasEig[1].real.T)
"""


##################################################################################################################################

#　生画像データ固有値+ベクタ
covMat = np.cov(pre2.eyeDatasCenter_v1a.T)
eyeDatasEig = np.linalg.eig(covMat)
np.save('saves/100dataLab_eigval', eyeDatasEig[0].real)
np.save('saves/100dataLab_eigvec', eyeDatasEig[1].real.T)
#np.save('saves/eyedata_eigval_'+str(pre2.eyeCoeff)+'-'+str(pre2.handCoeff)+'-'+str(pre2.vecCoeff), eyeDatasEig[0].real)
#np.save('saves/eyedata_eigvec_'+str(pre2.eyeCoeff)+'-'+str(pre2.handCoeff)+'-'+str(pre2.vecCoeff), eyeDatasEig[1].real.T)
