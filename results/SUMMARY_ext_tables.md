## Main comparison with extra metrics and CatBoost single (original roster, fixed config)

### MIMIC-IV 24 h mortality

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.8337 | 0.0158 | 0.7206 | 0.1347 | 0.0392 | 0.9860 | 0.0218 |
| LR single (all features) | 0.8291 | 0.0151 | 0.7082 | 0.1385 | 0.0446 | 0.7809 | 0.0020 |
| CatBoost single (fixed config) | 0.8376 | 0.0159 | 0.7232 | 0.1337 | 0.0401 | 1.1138 | 0.0204 |
| Stack, XGB meta (library defaults) | 0.8011 | 0.0182 | 0.6739 | 0.1617 | 0.1199 | 0.4006 | 0.5248 |
| Stack, XGB meta (fixed config) | 0.8286 | 0.0161 | 0.7075 | 0.1376 | 0.0365 | 0.8962 | -0.0216 |
| Stack, LR meta | 0.8322 | 0.0188 | 0.7157 | 0.1354 | 0.0397 | 0.9648 | -0.0579 |
| Mean of Level-1 | 0.8339 | 0.0166 | 0.7084 | 0.1453 | 0.0807 | 1.9559 | -0.0117 |
| Best single Level-1 (picked in-fold) | 0.7717 | 0.0233 | 0.6210 | 0.1572 | 0.0439 | 0.8788 | 0.0117 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0620 (-0.0831, -0.0408) | -0.0996 (-0.1266, -0.0725) | yes | 0.0000 |
| CatBoost single (fixed config) | +0.0039 (-0.0033, +0.0110) | +0.0026 (-0.0051, +0.0102) | no | 0.2652 |
| LR single (all features) | -0.0046 (-0.0170, +0.0077) | -0.0124 (-0.0320, +0.0071) | no | 0.4347 |
| Mean of Level-1 | +0.0002 (-0.0060, +0.0065) | -0.0122 (-0.0255, +0.0011) | no | 0.9353 |
| Stack, LR meta | -0.0016 (-0.0122, +0.0090) | -0.0049 (-0.0199, +0.0101) | no | 0.7576 |
| Stack, XGB meta (fixed config) | -0.0051 (-0.0184, +0.0082) | -0.0131 (-0.0298, +0.0036) | no | 0.4233 |
| Stack, XGB meta (library defaults) | -0.0326 (-0.0538, -0.0114) | -0.0467 (-0.0830, -0.0104) | yes | 0.0053 |

Level-1 correlation: mean_corr_all 0.611, mean_corr_same_algo_diff_subset 0.603, mean_corr_diff_algo_same_subset 0.899

### eICU 72 h mortality

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.8529 | 0.0086 | 0.7164 | 0.1080 | 0.0190 | 1.1483 | 0.0086 |
| LR single (all features) | 0.8320 | 0.0081 | 0.6937 | 0.1130 | 0.0152 | 0.9722 | 0.0019 |
| CatBoost single (fixed config) | 0.8511 | 0.0087 | 0.7152 | 0.1089 | 0.0220 | 1.1993 | 0.0102 |
| Stack, XGB meta (library defaults) | 0.8295 | 0.0092 | 0.6878 | 0.1190 | 0.0641 | 0.5598 | 0.3554 |
| Stack, XGB meta (fixed config) | 0.8499 | 0.0085 | 0.7147 | 0.1082 | 0.0133 | 1.0039 | 0.0103 |
| Stack, LR meta | 0.8502 | 0.0085 | 0.7136 | 0.1083 | 0.0163 | 0.9927 | 0.0198 |
| Mean of Level-1 | 0.8337 | 0.0083 | 0.6790 | 0.1387 | 0.1033 | 3.1734 | -0.0051 |
| Best single Level-1 (picked in-fold) | 0.7937 | 0.0104 | 0.5644 | 0.1356 | 0.0180 | 1.0912 | 0.0019 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0592 (-0.0680, -0.0503) | -0.1520 (-0.1767, -0.1274) | yes | 0.0000 |
| CatBoost single (fixed config) | -0.0018 (-0.0047, +0.0011) | -0.0013 (-0.0052, +0.0027) | no | 0.2092 |
| LR single (all features) | -0.0209 (-0.0263, -0.0155) | -0.0227 (-0.0299, -0.0154) | yes | 0.0000 |
| Mean of Level-1 | -0.0192 (-0.0239, -0.0145) | -0.0374 (-0.0480, -0.0269) | yes | 0.0000 |
| Stack, LR meta | -0.0027 (-0.0062, +0.0009) | -0.0028 (-0.0100, +0.0044) | no | 0.1332 |
| Stack, XGB meta (fixed config) | -0.0030 (-0.0062, +0.0001) | -0.0018 (-0.0088, +0.0052) | no | 0.0599 |
| Stack, XGB meta (library defaults) | -0.0234 (-0.0305, -0.0163) | -0.0286 (-0.0380, -0.0193) | yes | 0.0000 |

Level-1 correlation: mean_corr_all 0.381, mean_corr_same_algo_diff_subset 0.358, mean_corr_diff_algo_same_subset 0.858

### ICU LOS>7 d (landmark 2 d)

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.7815 | 0.0284 | 0.5980 | 0.1724 | 0.0550 | 0.8892 | 0.0355 |
| LR single (all features) | 0.7696 | 0.0238 | 0.5789 | 0.1839 | 0.0890 | 0.6209 | 0.0076 |
| CatBoost single (fixed config) | 0.7842 | 0.0284 | 0.6031 | 0.1703 | 0.0518 | 1.0515 | 0.0245 |
| Stack, XGB meta (library defaults) | 0.7429 | 0.0276 | 0.5268 | 0.2197 | 0.1748 | 0.3048 | 0.3338 |
| Stack, XGB meta (fixed config) | 0.7658 | 0.0252 | 0.5564 | 0.1783 | 0.0645 | 0.8235 | 0.0243 |
| Stack, LR meta | 0.7762 | 0.0228 | 0.5835 | 0.1762 | 0.0707 | 0.8823 | 0.0222 |
| Mean of Level-1 | 0.7674 | 0.0211 | 0.5698 | 0.1857 | 0.0918 | 2.1464 | -0.0187 |
| Best single Level-1 (picked in-fold) | 0.7659 | 0.0196 | 0.5644 | 0.1744 | 0.0457 | 0.9926 | 0.0013 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0157 (-0.0400, +0.0087) | -0.0336 (-0.0688, +0.0016) | no | 0.1899 |
| CatBoost single (fixed config) | +0.0027 (-0.0089, +0.0142) | +0.0050 (-0.0188, +0.0289) | no | 0.6291 |
| LR single (all features) | -0.0120 (-0.0318, +0.0079) | -0.0192 (-0.0479, +0.0096) | no | 0.2174 |
| Mean of Level-1 | -0.0142 (-0.0338, +0.0054) | -0.0282 (-0.0553, -0.0011) | no | 0.1432 |
| Stack, LR meta | -0.0053 (-0.0276, +0.0169) | -0.0145 (-0.0514, +0.0224) | no | 0.6144 |
| Stack, XGB meta (fixed config) | -0.0158 (-0.0430, +0.0115) | -0.0416 (-0.0797, -0.0036) | no | 0.2347 |
| Stack, XGB meta (library defaults) | -0.0387 (-0.0727, -0.0046) | -0.0713 (-0.1192, -0.0233) | yes | 0.0288 |

Level-1 correlation: mean_corr_all 0.421, mean_corr_same_algo_diff_subset 0.431, mean_corr_diff_algo_same_subset 0.715

### ICU LOS>14 d (landmark 2 d)

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.7952 | 0.0273 | 0.3521 | 0.0899 | 0.0360 | 0.8002 | 0.1736 |
| LR single (all features) | 0.7563 | 0.0274 | 0.2809 | 0.1080 | 0.0784 | 0.4252 | -0.0570 |
| CatBoost single (fixed config) | 0.8007 | 0.0276 | 0.3500 | 0.0895 | 0.0331 | 0.9858 | 0.1372 |
| Stack, XGB meta (library defaults) | 0.7627 | 0.0353 | 0.2989 | 0.1064 | 0.0923 | 0.3462 | 1.1652 |
| Stack, XGB meta (fixed config) | 0.7932 | 0.0297 | 0.3351 | 0.0921 | 0.0410 | 0.7906 | 0.0498 |
| Stack, LR meta | 0.7997 | 0.0191 | 0.3437 | 0.0923 | 0.0411 | 0.8158 | -0.0187 |
| Mean of Level-1 | 0.7905 | 0.0276 | 0.3474 | 0.0944 | 0.0629 | 2.8076 | -0.3167 |
| Best single Level-1 (picked in-fold) | 0.7902 | 0.0318 | 0.3081 | 0.0913 | 0.0279 | 0.9586 | 0.0044 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0050 (-0.0448, +0.0349) | -0.0440 (-0.1172, +0.0291) | no | 0.7926 |
| CatBoost single (fixed config) | +0.0055 (-0.0118, +0.0228) | -0.0021 (-0.0557, +0.0515) | no | 0.5068 |
| LR single (all features) | -0.0389 (-0.0882, +0.0103) | -0.0712 (-0.1478, +0.0053) | no | 0.1118 |
| Mean of Level-1 | -0.0047 (-0.0299, +0.0205) | -0.0048 (-0.0664, +0.0568) | no | 0.6945 |
| Stack, LR meta | +0.0044 (-0.0308, +0.0397) | -0.0084 (-0.0932, +0.0764) | no | 0.7903 |
| Stack, XGB meta (fixed config) | -0.0021 (-0.0358, +0.0317) | -0.0170 (-0.0958, +0.0618) | no | 0.8970 |
| Stack, XGB meta (library defaults) | -0.0325 (-0.0736, +0.0086) | -0.0532 (-0.1336, +0.0271) | no | 0.1123 |

Level-1 correlation: mean_corr_all 0.277, mean_corr_same_algo_diff_subset 0.356, mean_corr_diff_algo_same_subset 0.466

### ICU LOS>7 d (landmark 1 d)

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.7777 | 0.0175 | 0.4739 | 0.1468 | 0.0411 | 0.9556 | 0.0522 |
| LR single (all features) | 0.7585 | 0.0157 | 0.4372 | 0.1567 | 0.0627 | 0.6762 | -0.0021 |
| CatBoost single (fixed config) | 0.7711 | 0.0204 | 0.4549 | 0.1481 | 0.0397 | 1.0748 | 0.0416 |
| Stack, XGB meta (library defaults) | 0.7193 | 0.0240 | 0.3949 | 0.1863 | 0.1454 | 0.3081 | 0.7303 |
| Stack, XGB meta (fixed config) | 0.7519 | 0.0199 | 0.4238 | 0.1532 | 0.0469 | 0.8452 | 0.0107 |
| Stack, LR meta | 0.7455 | 0.0246 | 0.4172 | 0.1539 | 0.0446 | 0.8822 | -0.0228 |
| Mean of Level-1 | 0.7497 | 0.0273 | 0.4273 | 0.1583 | 0.0722 | 2.3341 | -0.0517 |
| Best single Level-1 (picked in-fold) | 0.7405 | 0.0218 | 0.4025 | 0.1536 | 0.0412 | 0.8944 | -0.0020 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0372 (-0.0550, -0.0195) | -0.0714 (-0.1100, -0.0328) | yes | 0.0005 |
| CatBoost single (fixed config) | -0.0067 (-0.0194, +0.0061) | -0.0191 (-0.0463, +0.0082) | no | 0.2799 |
| LR single (all features) | -0.0192 (-0.0363, -0.0021) | -0.0367 (-0.0767, +0.0033) | yes | 0.0306 |
| Mean of Level-1 | -0.0280 (-0.0535, -0.0024) | -0.0466 (-0.0911, -0.0020) | yes | 0.0340 |
| Stack, LR meta | -0.0322 (-0.0521, -0.0124) | -0.0567 (-0.1022, -0.0111) | yes | 0.0037 |
| Stack, XGB meta (fixed config) | -0.0259 (-0.0460, -0.0057) | -0.0501 (-0.1075, +0.0073) | yes | 0.0155 |
| Stack, XGB meta (library defaults) | -0.0585 (-0.0852, -0.0317) | -0.0790 (-0.1417, -0.0164) | yes | 0.0003 |

Level-1 correlation: mean_corr_all 0.334, mean_corr_same_algo_diff_subset 0.391, mean_corr_diff_algo_same_subset 0.603

### ICU LOS>14 d (landmark 1 d)

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.7678 | 0.0433 | 0.2134 | 0.0714 | 0.0299 | 0.7959 | 0.1523 |
| LR single (all features) | 0.7524 | 0.0418 | 0.1918 | 0.0781 | 0.0448 | 0.5385 | -0.0325 |
| CatBoost single (fixed config) | 0.7735 | 0.0341 | 0.2218 | 0.0705 | 0.0203 | 0.9843 | 0.1254 |
| Stack, XGB meta (library defaults) | 0.7157 | 0.0312 | 0.1763 | 0.0822 | 0.0757 | 0.2926 | 1.5493 |
| Stack, XGB meta (fixed config) | 0.7559 | 0.0327 | 0.1971 | 0.0725 | 0.0305 | 0.7320 | 0.0761 |
| Stack, LR meta | 0.7620 | 0.0333 | 0.2035 | 0.0726 | 0.0277 | 0.7888 | -0.0310 |
| Mean of Level-1 | 0.7480 | 0.0438 | 0.2033 | 0.0734 | 0.0397 | 2.6814 | -0.4265 |
| Best single Level-1 (picked in-fold) | 0.7613 | 0.0369 | 0.1961 | 0.0705 | 0.0172 | 1.0068 | 0.0018 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0065 (-0.0329, +0.0199) | -0.0173 (-0.0536, +0.0191) | no | 0.6051 |
| CatBoost single (fixed config) | +0.0057 (-0.0125, +0.0238) | +0.0084 (-0.0289, +0.0457) | no | 0.5134 |
| LR single (all features) | -0.0155 (-0.0471, +0.0162) | -0.0216 (-0.0675, +0.0243) | no | 0.3127 |
| Mean of Level-1 | -0.0199 (-0.0418, +0.0021) | -0.0101 (-0.0521, +0.0319) | no | 0.0724 |
| Stack, LR meta | -0.0058 (-0.0240, +0.0125) | -0.0099 (-0.0504, +0.0306) | no | 0.5084 |
| Stack, XGB meta (fixed config) | -0.0119 (-0.0364, +0.0127) | -0.0162 (-0.0699, +0.0374) | no | 0.3166 |
| Stack, XGB meta (library defaults) | -0.0521 (-0.0873, -0.0169) | -0.0371 (-0.0871, +0.0130) | yes | 0.0068 |

Level-1 correlation: mean_corr_all 0.242, mean_corr_same_algo_diff_subset 0.358, mean_corr_diff_algo_same_subset 0.421

## Equal nested tuning budget (20 random configs, 3-fold inner CV, per outer fold)

### MIMIC-IV 24 h mortality

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.8337 | 0.0158 | 0.7206 | 0.1347 | 0.0392 | 0.9860 | 0.0218 |
| Stack, XGB meta (library defaults) | 0.8073 | 0.0182 | 0.6793 | 0.1602 | 0.1199 | 0.4088 | 0.5176 |
| XGB single (tuned) | 0.8369 | 0.0169 | 0.7220 | 0.1344 | 0.0383 | 1.0236 | 0.0387 |
| Stack, XGB meta (tuned) | 0.8367 | 0.0172 | 0.7156 | 0.1349 | 0.0386 | 1.0843 | -0.0085 |
| Stack, LR meta (tuned C) | 0.8414 | 0.0153 | 0.7254 | 0.1328 | 0.0367 | 1.0713 | -0.0170 |
| Mean of Level-1 | 0.8317 | 0.0162 | 0.7052 | 0.1476 | 0.0871 | 2.0718 | -0.0113 |
| Best single Level-1 (picked in-fold) | 0.7858 | 0.0236 | 0.6316 | 0.1543 | 0.0417 | 1.1033 | -0.0018 |

Paired difference vs **XGB single (tuned)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0511 (-0.0722, -0.0300) | -0.0904 (-0.1166, -0.0642) | yes | 0.0001 |
| Mean of Level-1 | -0.0052 (-0.0119, +0.0016) | -0.0168 (-0.0293, -0.0042) | no | 0.1223 |
| Stack, LR meta (tuned C) | +0.0045 (-0.0038, +0.0128) | +0.0034 (-0.0105, +0.0174) | no | 0.2636 |
| Stack, XGB meta (library defaults) | -0.0296 (-0.0471, -0.0120) | -0.0427 (-0.0696, -0.0157) | yes | 0.0028 |
| Stack, XGB meta (tuned) | -0.0002 (-0.0126, +0.0122) | -0.0063 (-0.0257, +0.0130) | no | 0.9750 |
| XGB single (fixed config) | -0.0032 (-0.0095, +0.0032) | -0.0014 (-0.0113, +0.0085) | no | 0.3010 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0479 (-0.0670, -0.0288) | -0.0890 (-0.1137, -0.0644) | yes | 0.0001 |
| Mean of Level-1 | -0.0020 (-0.0082, +0.0042) | -0.0154 (-0.0294, -0.0013) | no | 0.5000 |
| Stack, LR meta (tuned C) | +0.0077 (+0.0018, +0.0135) | +0.0048 (-0.0072, +0.0168) | yes | 0.0136 |
| Stack, XGB meta (library defaults) | -0.0264 (-0.0437, -0.0090) | -0.0413 (-0.0650, -0.0176) | yes | 0.0057 |
| Stack, XGB meta (tuned) | +0.0030 (-0.0071, +0.0131) | -0.0050 (-0.0222, +0.0122) | no | 0.5358 |
| XGB single (tuned) | +0.0032 (-0.0032, +0.0095) | +0.0014 (-0.0085, +0.0113) | no | 0.3010 |

Level-1 correlation: mean_corr_all 0.620, mean_corr_same_algo_diff_subset 0.612, mean_corr_diff_algo_same_subset 0.910

### eICU 72 h mortality

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.8529 | 0.0086 | 0.7164 | 0.1080 | 0.0190 | 1.1483 | 0.0086 |
| Stack, XGB meta (library defaults) | 0.8253 | 0.0094 | 0.6821 | 0.1201 | 0.0647 | 0.5583 | 0.3608 |
| XGB single (tuned) | 0.8567 | 0.0080 | 0.7220 | 0.1068 | 0.0162 | 1.0212 | 0.0330 |
| Stack, XGB meta (tuned) | 0.8474 | 0.0094 | 0.7109 | 0.1092 | 0.0165 | 1.0249 | 0.0187 |
| Stack, LR meta (tuned C) | 0.8482 | 0.0091 | 0.7105 | 0.1089 | 0.0168 | 1.0146 | 0.0166 |
| Mean of Level-1 | 0.8313 | 0.0082 | 0.6772 | 0.1421 | 0.1035 | 3.5846 | -0.0058 |
| Best single Level-1 (picked in-fold) | 0.7836 | 0.0103 | 0.5491 | 0.1406 | 0.0349 | 1.4562 | 0.0038 |

Paired difference vs **XGB single (tuned)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0731 (-0.0801, -0.0662) | -0.1729 (-0.1894, -0.1563) | yes | 0.0000 |
| Mean of Level-1 | -0.0254 (-0.0299, -0.0210) | -0.0448 (-0.0562, -0.0334) | yes | 0.0000 |
| Stack, LR meta (tuned C) | -0.0086 (-0.0125, -0.0046) | -0.0114 (-0.0201, -0.0027) | yes | 0.0004 |
| Stack, XGB meta (library defaults) | -0.0314 (-0.0398, -0.0231) | -0.0399 (-0.0507, -0.0291) | yes | 0.0000 |
| Stack, XGB meta (tuned) | -0.0093 (-0.0131, -0.0056) | -0.0111 (-0.0171, -0.0051) | yes | 0.0001 |
| XGB single (fixed config) | -0.0038 (-0.0062, -0.0015) | -0.0055 (-0.0098, -0.0013) | yes | 0.0038 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0693 (-0.0756, -0.0630) | -0.1673 (-0.1843, -0.1503) | yes | 0.0000 |
| Mean of Level-1 | -0.0216 (-0.0261, -0.0171) | -0.0393 (-0.0497, -0.0288) | yes | 0.0000 |
| Stack, LR meta (tuned C) | -0.0047 (-0.0086, -0.0009) | -0.0059 (-0.0132, +0.0015) | yes | 0.0200 |
| Stack, XGB meta (library defaults) | -0.0276 (-0.0358, -0.0194) | -0.0344 (-0.0455, -0.0232) | yes | 0.0000 |
| Stack, XGB meta (tuned) | -0.0055 (-0.0093, -0.0016) | -0.0056 (-0.0128, +0.0017) | yes | 0.0089 |
| XGB single (tuned) | +0.0038 (+0.0015, +0.0062) | +0.0055 (+0.0013, +0.0098) | yes | 0.0038 |

Level-1 correlation: mean_corr_all 0.380, mean_corr_same_algo_diff_subset 0.354, mean_corr_diff_algo_same_subset 0.872

### ICU LOS>7 d (landmark 2 d)

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.7815 | 0.0284 | 0.5980 | 0.1724 | 0.0550 | 0.8892 | 0.0355 |
| Stack, XGB meta (library defaults) | 0.7486 | 0.0284 | 0.5289 | 0.2189 | 0.1725 | 0.3147 | 0.3444 |
| XGB single (tuned) | 0.7834 | 0.0306 | 0.6085 | 0.1717 | 0.0613 | 1.0597 | 0.0371 |
| Stack, XGB meta (tuned) | 0.7752 | 0.0236 | 0.5593 | 0.1737 | 0.0543 | 1.1113 | -0.0045 |
| Stack, LR meta (tuned C) | 0.7814 | 0.0213 | 0.5840 | 0.1726 | 0.0554 | 0.9846 | -0.0077 |
| Mean of Level-1 | 0.7672 | 0.0199 | 0.5672 | 0.1878 | 0.0974 | 2.4111 | -0.0222 |
| Best single Level-1 (picked in-fold) | 0.7632 | 0.0204 | 0.5618 | 0.1749 | 0.0492 | 1.0097 | -0.0014 |

Paired difference vs **XGB single (tuned)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0201 (-0.0421, +0.0018) | -0.0467 (-0.0896, -0.0038) | no | 0.0691 |
| Mean of Level-1 | -0.0162 (-0.0392, +0.0068) | -0.0413 (-0.0780, -0.0045) | no | 0.1530 |
| Stack, LR meta (tuned C) | -0.0020 (-0.0238, +0.0199) | -0.0245 (-0.0663, +0.0173) | no | 0.8510 |
| Stack, XGB meta (library defaults) | -0.0347 (-0.0617, -0.0078) | -0.0796 (-0.1229, -0.0363) | yes | 0.0153 |
| Stack, XGB meta (tuned) | -0.0081 (-0.0292, +0.0130) | -0.0492 (-0.0779, -0.0204) | no | 0.4231 |
| XGB single (fixed config) | -0.0018 (-0.0119, +0.0083) | -0.0105 (-0.0406, +0.0197) | no | 0.7071 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0183 (-0.0376, +0.0009) | -0.0362 (-0.0735, +0.0011) | no | 0.0604 |
| Mean of Level-1 | -0.0144 (-0.0361, +0.0073) | -0.0308 (-0.0645, +0.0029) | no | 0.1762 |
| Stack, LR meta (tuned C) | -0.0001 (-0.0195, +0.0192) | -0.0140 (-0.0522, +0.0242) | no | 0.9874 |
| Stack, XGB meta (library defaults) | -0.0329 (-0.0618, -0.0041) | -0.0691 (-0.1127, -0.0256) | yes | 0.0282 |
| Stack, XGB meta (tuned) | -0.0063 (-0.0273, +0.0147) | -0.0387 (-0.0660, -0.0113) | no | 0.5294 |
| XGB single (tuned) | +0.0018 (-0.0083, +0.0119) | +0.0105 (-0.0197, +0.0406) | no | 0.7071 |

Level-1 correlation: mean_corr_all 0.449, mean_corr_same_algo_diff_subset 0.461, mean_corr_diff_algo_same_subset 0.739

### ICU LOS>14 d (landmark 2 d)

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.7952 | 0.0273 | 0.3521 | 0.0899 | 0.0360 | 0.8002 | 0.1736 |
| Stack, XGB meta (library defaults) | 0.7418 | 0.0421 | 0.2653 | 0.1100 | 0.0982 | 0.3041 | 1.2127 |
| XGB single (tuned) | 0.8073 | 0.0298 | 0.3762 | 0.0885 | 0.0295 | 1.3617 | 0.0012 |
| Stack, XGB meta (tuned) | 0.7935 | 0.0349 | 0.3174 | 0.0919 | 0.0398 | 1.1870 | -0.0249 |
| Stack, LR meta (tuned C) | 0.8101 | 0.0233 | 0.3556 | 0.0889 | 0.0314 | 1.0286 | 0.0057 |
| Mean of Level-1 | 0.7872 | 0.0287 | 0.3249 | 0.0959 | 0.0694 | 3.2051 | -0.3409 |
| Best single Level-1 (picked in-fold) | 0.7865 | 0.0280 | 0.3044 | 0.0911 | 0.0320 | 1.0787 | -0.0152 |

Paired difference vs **XGB single (tuned)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0207 (-0.0515, +0.0100) | -0.0718 (-0.1421, -0.0015) | no | 0.1704 |
| Mean of Level-1 | -0.0201 (-0.0465, +0.0063) | -0.0514 (-0.1151, +0.0124) | no | 0.1247 |
| Stack, LR meta (tuned C) | +0.0028 (-0.0160, +0.0216) | -0.0206 (-0.0779, +0.0366) | no | 0.7526 |
| Stack, XGB meta (library defaults) | -0.0654 (-0.1197, -0.0112) | -0.1109 (-0.2008, -0.0209) | yes | 0.0216 |
| Stack, XGB meta (tuned) | -0.0138 (-0.0417, +0.0141) | -0.0588 (-0.1276, +0.0101) | no | 0.3067 |
| XGB single (fixed config) | -0.0121 (-0.0290, +0.0049) | -0.0241 (-0.0574, +0.0092) | no | 0.1503 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0087 (-0.0453, +0.0279) | -0.0478 (-0.1098, +0.0143) | no | 0.6192 |
| Mean of Level-1 | -0.0081 (-0.0342, +0.0181) | -0.0273 (-0.0895, +0.0350) | no | 0.5194 |
| Stack, LR meta (tuned C) | +0.0149 (-0.0059, +0.0356) | +0.0035 (-0.0560, +0.0630) | no | 0.1472 |
| Stack, XGB meta (library defaults) | -0.0534 (-0.1030, -0.0038) | -0.0868 (-0.1612, -0.0124) | yes | 0.0368 |
| Stack, XGB meta (tuned) | -0.0017 (-0.0331, +0.0296) | -0.0347 (-0.1031, +0.0338) | no | 0.9066 |
| XGB single (tuned) | +0.0121 (-0.0049, +0.0290) | +0.0241 (-0.0092, +0.0574) | no | 0.1503 |

Level-1 correlation: mean_corr_all 0.315, mean_corr_same_algo_diff_subset 0.404, mean_corr_diff_algo_same_subset 0.498

## Diverse Level-1 roster (XGB, LR, kNN, RF, naive Bayes, MLP)

### MIMIC-IV 24 h mortality

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.8337 | 0.0158 | 0.7206 | 0.1347 | 0.0392 | 0.9860 | 0.0218 |
| LR single (all features) | 0.8291 | 0.0151 | 0.7082 | 0.1385 | 0.0446 | 0.7809 | 0.0020 |
| Stack, XGB meta (library defaults) | 0.8059 | 0.0194 | 0.6782 | 0.1620 | 0.1230 | 0.3987 | 0.5400 |
| Stack, XGB meta (fixed config) | 0.8315 | 0.0181 | 0.7113 | 0.1366 | 0.0435 | 0.9010 | -0.0067 |
| Stack, LR meta | 0.8223 | 0.0199 | 0.7054 | 0.1386 | 0.0375 | 0.8720 | -0.0482 |
| Mean of Level-1 | 0.8284 | 0.0164 | 0.6961 | 0.1457 | 0.0720 | 1.7375 | 0.0468 |
| Best single Level-1 (picked in-fold) | 0.7753 | 0.0198 | 0.6246 | 0.1561 | 0.0422 | 0.9391 | -0.0111 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0584 (-0.0737, -0.0430) | -0.0961 (-0.1261, -0.0660) | yes | 0.0000 |
| LR single (all features) | -0.0046 (-0.0170, +0.0077) | -0.0124 (-0.0320, +0.0071) | no | 0.4347 |
| Mean of Level-1 | -0.0053 (-0.0143, +0.0037) | -0.0245 (-0.0415, -0.0075) | no | 0.2289 |
| Stack, LR meta | -0.0115 (-0.0333, +0.0104) | -0.0152 (-0.0383, +0.0078) | no | 0.2792 |
| Stack, XGB meta (fixed config) | -0.0022 (-0.0146, +0.0102) | -0.0093 (-0.0254, +0.0069) | no | 0.7090 |
| Stack, XGB meta (library defaults) | -0.0278 (-0.0517, -0.0038) | -0.0424 (-0.0678, -0.0169) | yes | 0.0261 |

Level-1 correlation: mean_corr_all 0.572, mean_corr_same_algo_diff_subset 0.581, mean_corr_diff_algo_same_subset 0.830

### eICU 72 h mortality

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.8529 | 0.0086 | 0.7164 | 0.1080 | 0.0190 | 1.1483 | 0.0086 |
| LR single (all features) | 0.8320 | 0.0081 | 0.6937 | 0.1130 | 0.0152 | 0.9722 | 0.0019 |
| Stack, XGB meta (library defaults) | 0.8321 | 0.0110 | 0.6898 | 0.1198 | 0.0705 | 0.5331 | 0.4444 |
| Stack, XGB meta (fixed config) | 0.8532 | 0.0075 | 0.7178 | 0.1079 | 0.0143 | 1.0034 | 0.0116 |
| Stack, LR meta | 0.8551 | 0.0087 | 0.7176 | 0.1074 | 0.0176 | 0.9695 | 0.0250 |
| Mean of Level-1 | 0.8288 | 0.0074 | 0.6710 | 0.1374 | 0.0957 | 2.8076 | 0.0252 |
| Best single Level-1 (picked in-fold) | 0.7943 | 0.0098 | 0.5632 | 0.1357 | 0.0174 | 1.0688 | -0.0023 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0585 (-0.0652, -0.0519) | -0.1532 (-0.1774, -0.1290) | yes | 0.0000 |
| LR single (all features) | -0.0209 (-0.0263, -0.0155) | -0.0227 (-0.0299, -0.0154) | yes | 0.0000 |
| Mean of Level-1 | -0.0241 (-0.0301, -0.0181) | -0.0454 (-0.0563, -0.0344) | yes | 0.0000 |
| Stack, LR meta | +0.0022 (-0.0019, +0.0064) | +0.0012 (-0.0065, +0.0088) | no | 0.2610 |
| Stack, XGB meta (fixed config) | +0.0003 (-0.0039, +0.0045) | +0.0013 (-0.0041, +0.0067) | no | 0.8746 |
| Stack, XGB meta (library defaults) | -0.0207 (-0.0285, -0.0130) | -0.0266 (-0.0355, -0.0177) | yes | 0.0001 |

Level-1 correlation: mean_corr_all 0.340, mean_corr_same_algo_diff_subset 0.327, mean_corr_diff_algo_same_subset 0.761

### ICU LOS>7 d (landmark 2 d)

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.7815 | 0.0284 | 0.5980 | 0.1724 | 0.0550 | 0.8892 | 0.0355 |
| LR single (all features) | 0.7696 | 0.0238 | 0.5789 | 0.1839 | 0.0890 | 0.6209 | 0.0076 |
| Stack, XGB meta (library defaults) | 0.7518 | 0.0262 | 0.5428 | 0.2161 | 0.1703 | 0.3206 | 0.4057 |
| Stack, XGB meta (fixed config) | 0.7688 | 0.0205 | 0.5578 | 0.1774 | 0.0623 | 0.8210 | 0.0351 |
| Stack, LR meta | 0.7553 | 0.0230 | 0.5695 | 0.1836 | 0.0721 | 0.7231 | 0.0361 |
| Mean of Level-1 | 0.7621 | 0.0198 | 0.5568 | 0.1850 | 0.0858 | 1.8554 | 0.0010 |
| Best single Level-1 (picked in-fold) | 0.7631 | 0.0168 | 0.5593 | 0.1772 | 0.0544 | 0.9070 | -0.0084 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0185 (-0.0465, +0.0096) | -0.0387 (-0.0816, +0.0042) | no | 0.1803 |
| LR single (all features) | -0.0120 (-0.0318, +0.0079) | -0.0192 (-0.0479, +0.0096) | no | 0.2174 |
| Mean of Level-1 | -0.0194 (-0.0383, -0.0005) | -0.0412 (-0.0711, -0.0113) | yes | 0.0447 |
| Stack, LR meta | -0.0263 (-0.0512, -0.0014) | -0.0285 (-0.0570, +0.0001) | yes | 0.0398 |
| Stack, XGB meta (fixed config) | -0.0127 (-0.0405, +0.0150) | -0.0403 (-0.0803, -0.0003) | no | 0.3415 |
| Stack, XGB meta (library defaults) | -0.0297 (-0.0664, +0.0070) | -0.0552 (-0.1108, +0.0004) | no | 0.1041 |

Level-1 correlation: mean_corr_all 0.414, mean_corr_same_algo_diff_subset 0.428, mean_corr_diff_algo_same_subset 0.696

### ICU LOS>14 d (landmark 2 d)

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.7952 | 0.0273 | 0.3521 | 0.0899 | 0.0360 | 0.8002 | 0.1736 |
| LR single (all features) | 0.7563 | 0.0274 | 0.2809 | 0.1080 | 0.0784 | 0.4252 | -0.0570 |
| Stack, XGB meta (library defaults) | 0.7629 | 0.0383 | 0.3109 | 0.1043 | 0.0940 | 0.3431 | 1.2904 |
| Stack, XGB meta (fixed config) | 0.7888 | 0.0319 | 0.3226 | 0.0927 | 0.0453 | 0.7401 | 0.0923 |
| Stack, LR meta | 0.7788 | 0.0264 | 0.3335 | 0.0953 | 0.0508 | 0.6207 | 0.0216 |
| Mean of Level-1 | 0.7790 | 0.0292 | 0.3016 | 0.0940 | 0.0520 | 1.9708 | -0.2973 |
| Best single Level-1 (picked in-fold) | 0.7873 | 0.0262 | 0.3112 | 0.1172 | 0.0727 | 0.5885 | -0.8660 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0080 (-0.0435, +0.0275) | -0.0409 (-0.1204, +0.0386) | no | 0.6375 |
| LR single (all features) | -0.0389 (-0.0882, +0.0103) | -0.0712 (-0.1478, +0.0053) | no | 0.1118 |
| Mean of Level-1 | -0.0162 (-0.0456, +0.0132) | -0.0505 (-0.1199, +0.0189) | no | 0.2578 |
| Stack, LR meta | -0.0165 (-0.0582, +0.0253) | -0.0186 (-0.1127, +0.0756) | no | 0.4120 |
| Stack, XGB meta (fixed config) | -0.0064 (-0.0397, +0.0270) | -0.0295 (-0.1029, +0.0439) | no | 0.6878 |
| Stack, XGB meta (library defaults) | -0.0323 (-0.0783, +0.0138) | -0.0413 (-0.1121, +0.0296) | no | 0.1550 |

Level-1 correlation: mean_corr_all 0.305, mean_corr_same_algo_diff_subset 0.374, mean_corr_diff_algo_same_subset 0.509

## CatBoost added as a fifth Level-1 learner

### MIMIC-IV 24 h mortality

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.8337 | 0.0158 | 0.7206 | 0.1347 | 0.0392 | 0.9860 | 0.0218 |
| LR single (all features) | 0.8291 | 0.0151 | 0.7082 | 0.1385 | 0.0446 | 0.7809 | 0.0020 |
| Stack, XGB meta (library defaults) | 0.8068 | 0.0166 | 0.6756 | 0.1626 | 0.1242 | 0.3975 | 0.5420 |
| Stack, XGB meta (fixed config) | 0.8313 | 0.0140 | 0.7121 | 0.1364 | 0.0395 | 0.9018 | -0.0176 |
| Stack, LR meta | 0.8311 | 0.0177 | 0.7144 | 0.1359 | 0.0390 | 0.9427 | -0.0546 |
| Mean of Level-1 | 0.8344 | 0.0166 | 0.7085 | 0.1454 | 0.0820 | 1.9626 | -0.0082 |
| Best single Level-1 (picked in-fold) | 0.7849 | 0.0251 | 0.6336 | 0.1533 | 0.0399 | 1.0103 | 0.0052 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0488 (-0.0686, -0.0289) | -0.0870 (-0.1154, -0.0587) | yes | 0.0001 |
| LR single (all features) | -0.0046 (-0.0170, +0.0077) | -0.0124 (-0.0320, +0.0071) | no | 0.4347 |
| Mean of Level-1 | +0.0007 (-0.0052, +0.0067) | -0.0121 (-0.0247, +0.0006) | no | 0.7971 |
| Stack, LR meta | -0.0026 (-0.0122, +0.0070) | -0.0062 (-0.0177, +0.0053) | no | 0.5679 |
| Stack, XGB meta (fixed config) | -0.0024 (-0.0144, +0.0096) | -0.0085 (-0.0251, +0.0081) | no | 0.6715 |
| Stack, XGB meta (library defaults) | -0.0269 (-0.0448, -0.0089) | -0.0451 (-0.0760, -0.0141) | yes | 0.0063 |

Level-1 correlation: mean_corr_all 0.621, mean_corr_same_algo_diff_subset 0.609, mean_corr_diff_algo_same_subset 0.916

### eICU 72 h mortality

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.8529 | 0.0086 | 0.7164 | 0.1080 | 0.0190 | 1.1483 | 0.0086 |
| LR single (all features) | 0.8320 | 0.0081 | 0.6937 | 0.1130 | 0.0152 | 0.9722 | 0.0019 |
| Stack, XGB meta (library defaults) | 0.8301 | 0.0100 | 0.6890 | 0.1188 | 0.0648 | 0.5549 | 0.3810 |
| Stack, XGB meta (fixed config) | 0.8506 | 0.0091 | 0.7153 | 0.1081 | 0.0137 | 1.0042 | 0.0126 |
| Stack, LR meta | 0.8505 | 0.0088 | 0.7132 | 0.1083 | 0.0163 | 0.9870 | 0.0226 |
| Mean of Level-1 | 0.8342 | 0.0083 | 0.6802 | 0.1383 | 0.1027 | 3.1360 | -0.0038 |
| Best single Level-1 (picked in-fold) | 0.7937 | 0.0104 | 0.5644 | 0.1356 | 0.0180 | 1.0912 | 0.0019 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0592 (-0.0680, -0.0503) | -0.1520 (-0.1767, -0.1274) | yes | 0.0000 |
| LR single (all features) | -0.0209 (-0.0263, -0.0155) | -0.0227 (-0.0299, -0.0154) | yes | 0.0000 |
| Mean of Level-1 | -0.0187 (-0.0232, -0.0143) | -0.0363 (-0.0463, -0.0262) | yes | 0.0000 |
| Stack, LR meta | -0.0024 (-0.0060, +0.0011) | -0.0033 (-0.0100, +0.0035) | no | 0.1600 |
| Stack, XGB meta (fixed config) | -0.0022 (-0.0049, +0.0005) | -0.0011 (-0.0066, +0.0044) | no | 0.0979 |
| Stack, XGB meta (library defaults) | -0.0228 (-0.0319, -0.0138) | -0.0274 (-0.0395, -0.0154) | yes | 0.0001 |

Level-1 correlation: mean_corr_all 0.394, mean_corr_same_algo_diff_subset 0.366, mean_corr_diff_algo_same_subset 0.882

### ICU LOS>7 d (landmark 2 d)

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.7815 | 0.0284 | 0.5980 | 0.1724 | 0.0550 | 0.8892 | 0.0355 |
| LR single (all features) | 0.7696 | 0.0238 | 0.5789 | 0.1839 | 0.0890 | 0.6209 | 0.0076 |
| Stack, XGB meta (library defaults) | 0.7485 | 0.0264 | 0.5372 | 0.2168 | 0.1695 | 0.3084 | 0.3739 |
| Stack, XGB meta (fixed config) | 0.7662 | 0.0265 | 0.5534 | 0.1787 | 0.0688 | 0.8063 | 0.0252 |
| Stack, LR meta | 0.7753 | 0.0217 | 0.5803 | 0.1765 | 0.0698 | 0.8431 | 0.0403 |
| Mean of Level-1 | 0.7689 | 0.0210 | 0.5714 | 0.1852 | 0.0915 | 2.1215 | -0.0127 |
| Best single Level-1 (picked in-fold) | 0.7630 | 0.0211 | 0.5580 | 0.1750 | 0.0469 | 0.9800 | -0.0006 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0185 (-0.0409, +0.0038) | -0.0400 (-0.0712, -0.0088) | no | 0.0967 |
| LR single (all features) | -0.0120 (-0.0318, +0.0079) | -0.0192 (-0.0479, +0.0096) | no | 0.2174 |
| Mean of Level-1 | -0.0127 (-0.0320, +0.0066) | -0.0266 (-0.0526, -0.0005) | no | 0.1802 |
| Stack, LR meta | -0.0062 (-0.0283, +0.0158) | -0.0177 (-0.0615, +0.0261) | no | 0.5530 |
| Stack, XGB meta (fixed config) | -0.0153 (-0.0403, +0.0097) | -0.0446 (-0.0793, -0.0100) | no | 0.2102 |
| Stack, XGB meta (library defaults) | -0.0331 (-0.0658, -0.0003) | -0.0608 (-0.1015, -0.0201) | yes | 0.0481 |

Level-1 correlation: mean_corr_all 0.444, mean_corr_same_algo_diff_subset 0.446, mean_corr_diff_algo_same_subset 0.760

### ICU LOS>14 d (landmark 2 d)

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.7952 | 0.0273 | 0.3521 | 0.0899 | 0.0360 | 0.8002 | 0.1736 |
| LR single (all features) | 0.7563 | 0.0274 | 0.2809 | 0.1080 | 0.0784 | 0.4252 | -0.0570 |
| Stack, XGB meta (library defaults) | 0.7647 | 0.0315 | 0.3133 | 0.1043 | 0.0907 | 0.3493 | 1.2100 |
| Stack, XGB meta (fixed config) | 0.7953 | 0.0284 | 0.3408 | 0.0914 | 0.0393 | 0.7754 | 0.0587 |
| Stack, LR meta | 0.8010 | 0.0189 | 0.3515 | 0.0922 | 0.0486 | 0.7887 | -0.0185 |
| Mean of Level-1 | 0.7931 | 0.0274 | 0.3513 | 0.0936 | 0.0537 | 2.6535 | -0.2520 |
| Best single Level-1 (picked in-fold) | 0.7905 | 0.0330 | 0.3287 | 0.0906 | 0.0316 | 0.9913 | 0.0039 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0047 (-0.0432, +0.0337) | -0.0235 (-0.1048, +0.0579) | no | 0.7957 |
| LR single (all features) | -0.0389 (-0.0882, +0.0103) | -0.0712 (-0.1478, +0.0053) | no | 0.1118 |
| Mean of Level-1 | -0.0021 (-0.0245, +0.0203) | -0.0008 (-0.0620, +0.0604) | no | 0.8445 |
| Stack, LR meta | +0.0058 (-0.0308, +0.0424) | -0.0006 (-0.1024, +0.1012) | no | 0.7391 |
| Stack, XGB meta (fixed config) | +0.0001 (-0.0293, +0.0294) | -0.0113 (-0.0721, +0.0495) | no | 0.9957 |
| Stack, XGB meta (library defaults) | -0.0305 (-0.0639, +0.0029) | -0.0389 (-0.1178, +0.0400) | no | 0.0704 |

Level-1 correlation: mean_corr_all 0.315, mean_corr_same_algo_diff_subset 0.377, mean_corr_diff_algo_same_subset 0.554

## CatBoost replacing the MLP in Level-1

### MIMIC-IV 24 h mortality

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.8337 | 0.0158 | 0.7206 | 0.1347 | 0.0392 | 0.9860 | 0.0218 |
| LR single (all features) | 0.8291 | 0.0151 | 0.7082 | 0.1385 | 0.0446 | 0.7809 | 0.0020 |
| Stack, XGB meta (library defaults) | 0.8011 | 0.0117 | 0.6700 | 0.1641 | 0.1234 | 0.3884 | 0.5258 |
| Stack, XGB meta (fixed config) | 0.8309 | 0.0150 | 0.7107 | 0.1367 | 0.0391 | 0.9007 | -0.0074 |
| Stack, LR meta | 0.8355 | 0.0150 | 0.7202 | 0.1344 | 0.0344 | 0.9411 | -0.0127 |
| Mean of Level-1 | 0.8358 | 0.0166 | 0.7091 | 0.1444 | 0.0777 | 1.9039 | 0.0042 |
| Best single Level-1 (picked in-fold) | 0.7849 | 0.0251 | 0.6336 | 0.1533 | 0.0399 | 1.0103 | 0.0052 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0488 (-0.0686, -0.0289) | -0.0870 (-0.1154, -0.0587) | yes | 0.0001 |
| LR single (all features) | -0.0046 (-0.0170, +0.0077) | -0.0124 (-0.0320, +0.0071) | no | 0.4347 |
| Mean of Level-1 | +0.0021 (-0.0038, +0.0080) | -0.0115 (-0.0238, +0.0008) | no | 0.4558 |
| Stack, LR meta | +0.0017 (-0.0058, +0.0093) | -0.0004 (-0.0078, +0.0069) | no | 0.6268 |
| Stack, XGB meta (fixed config) | -0.0028 (-0.0112, +0.0057) | -0.0099 (-0.0212, +0.0014) | no | 0.4904 |
| Stack, XGB meta (library defaults) | -0.0326 (-0.0463, -0.0189) | -0.0506 (-0.0728, -0.0284) | yes | 0.0002 |

Level-1 correlation: mean_corr_all 0.638, mean_corr_same_algo_diff_subset 0.619, mean_corr_diff_algo_same_subset 0.948

### eICU 72 h mortality

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.8529 | 0.0086 | 0.7164 | 0.1080 | 0.0190 | 1.1483 | 0.0086 |
| LR single (all features) | 0.8320 | 0.0081 | 0.6937 | 0.1130 | 0.0152 | 0.9722 | 0.0019 |
| Stack, XGB meta (library defaults) | 0.8291 | 0.0109 | 0.6876 | 0.1191 | 0.0637 | 0.5701 | 0.3023 |
| Stack, XGB meta (fixed config) | 0.8498 | 0.0093 | 0.7141 | 0.1083 | 0.0135 | 1.0029 | 0.0050 |
| Stack, LR meta | 0.8509 | 0.0092 | 0.7136 | 0.1082 | 0.0156 | 0.9918 | -0.0017 |
| Mean of Level-1 | 0.8336 | 0.0084 | 0.6789 | 0.1380 | 0.1017 | 3.0703 | 0.0005 |
| Best single Level-1 (picked in-fold) | 0.7937 | 0.0104 | 0.5644 | 0.1356 | 0.0180 | 1.0912 | 0.0019 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0592 (-0.0680, -0.0503) | -0.1520 (-0.1767, -0.1274) | yes | 0.0000 |
| LR single (all features) | -0.0209 (-0.0263, -0.0155) | -0.0227 (-0.0299, -0.0154) | yes | 0.0000 |
| Mean of Level-1 | -0.0193 (-0.0237, -0.0148) | -0.0376 (-0.0470, -0.0281) | yes | 0.0000 |
| Stack, LR meta | -0.0020 (-0.0060, +0.0020) | -0.0029 (-0.0096, +0.0039) | no | 0.2973 |
| Stack, XGB meta (fixed config) | -0.0030 (-0.0059, -0.0002) | -0.0023 (-0.0076, +0.0029) | yes | 0.0374 |
| Stack, XGB meta (library defaults) | -0.0238 (-0.0314, -0.0162) | -0.0288 (-0.0385, -0.0191) | yes | 0.0000 |

Level-1 correlation: mean_corr_all 0.400, mean_corr_same_algo_diff_subset 0.375, mean_corr_diff_algo_same_subset 0.889

### ICU LOS>7 d (landmark 2 d)

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.7815 | 0.0284 | 0.5980 | 0.1724 | 0.0550 | 0.8892 | 0.0355 |
| LR single (all features) | 0.7696 | 0.0238 | 0.5789 | 0.1839 | 0.0890 | 0.6209 | 0.0076 |
| Stack, XGB meta (library defaults) | 0.7482 | 0.0286 | 0.5375 | 0.2165 | 0.1677 | 0.3117 | 0.3869 |
| Stack, XGB meta (fixed config) | 0.7655 | 0.0239 | 0.5491 | 0.1786 | 0.0611 | 0.8038 | 0.0227 |
| Stack, LR meta | 0.7772 | 0.0239 | 0.5778 | 0.1747 | 0.0558 | 0.8838 | 0.0213 |
| Mean of Level-1 | 0.7696 | 0.0213 | 0.5712 | 0.1835 | 0.0857 | 1.9176 | 0.0079 |
| Best single Level-1 (picked in-fold) | 0.7630 | 0.0211 | 0.5580 | 0.1750 | 0.0469 | 0.9800 | -0.0006 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0185 (-0.0409, +0.0038) | -0.0400 (-0.0712, -0.0088) | no | 0.0967 |
| LR single (all features) | -0.0120 (-0.0318, +0.0079) | -0.0192 (-0.0479, +0.0096) | no | 0.2174 |
| Mean of Level-1 | -0.0119 (-0.0304, +0.0065) | -0.0268 (-0.0518, -0.0019) | no | 0.1864 |
| Stack, LR meta | -0.0044 (-0.0256, +0.0168) | -0.0202 (-0.0599, +0.0195) | no | 0.6642 |
| Stack, XGB meta (fixed config) | -0.0161 (-0.0362, +0.0040) | -0.0489 (-0.0812, -0.0167) | no | 0.1082 |
| Stack, XGB meta (library defaults) | -0.0334 (-0.0613, -0.0054) | -0.0606 (-0.0984, -0.0227) | yes | 0.0227 |

Level-1 correlation: mean_corr_all 0.496, mean_corr_same_algo_diff_subset 0.475, mean_corr_diff_algo_same_subset 0.871

### ICU LOS>14 d (landmark 2 d)

| method | auc | auc_sd | auprc | brier | ece | slope | citl |
|---|---|---|---|---|---|---|---|
| XGB single (fixed config) | 0.7952 | 0.0273 | 0.3521 | 0.0899 | 0.0360 | 0.8002 | 0.1736 |
| LR single (all features) | 0.7563 | 0.0274 | 0.2809 | 0.1080 | 0.0784 | 0.4252 | -0.0570 |
| Stack, XGB meta (library defaults) | 0.7667 | 0.0316 | 0.3077 | 0.1053 | 0.0904 | 0.3397 | 1.1207 |
| Stack, XGB meta (fixed config) | 0.7949 | 0.0329 | 0.3524 | 0.0910 | 0.0421 | 0.7750 | 0.0560 |
| Stack, LR meta | 0.8015 | 0.0192 | 0.3581 | 0.0901 | 0.0376 | 0.8341 | -0.0193 |
| Mean of Level-1 | 0.7953 | 0.0293 | 0.3483 | 0.0918 | 0.0372 | 1.8956 | 0.0405 |
| Best single Level-1 (picked in-fold) | 0.7905 | 0.0330 | 0.3287 | 0.0906 | 0.0316 | 0.9913 | 0.0039 |

Paired difference vs **XGB single (fixed config)** (AUROC and AUPRC; "excl. 0" refers to AUROC):

| method | auc | auprc | excl. 0 | p_corrected_t |
|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0047 (-0.0432, +0.0337) | -0.0235 (-0.1048, +0.0579) | no | 0.7957 |
| LR single (all features) | -0.0389 (-0.0882, +0.0103) | -0.0712 (-0.1478, +0.0053) | no | 0.1118 |
| Mean of Level-1 | +0.0001 (-0.0195, +0.0196) | -0.0038 (-0.0660, +0.0584) | no | 0.9955 |
| Stack, LR meta | +0.0062 (-0.0263, +0.0387) | +0.0060 (-0.0762, +0.0882) | no | 0.6872 |
| Stack, XGB meta (fixed config) | -0.0003 (-0.0289, +0.0283) | +0.0003 (-0.0777, +0.0783) | no | 0.9814 |
| Stack, XGB meta (library defaults) | -0.0285 (-0.0598, +0.0027) | -0.0444 (-0.1151, +0.0263) | no | 0.0706 |

Level-1 correlation: mean_corr_all 0.434, mean_corr_same_algo_diff_subset 0.418, mean_corr_diff_algo_same_subset 0.811

## Cohort size: 10 stratified draws of n = 729 (MIMIC-IV 24 h), 3 repeats each

| method | mean_delta | min_delta | max_delta | draws_CI_below_0 | draws_CI_above_0 | draws |
|---|---|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | -0.0546 | -0.0756 | -0.0243 | 7 | 0 | 10 |
| LR single (all features) | -0.0158 | -0.0255 | 0.0136 | 0 | 0 | 10 |
| Mean of Level-1 | 0.0081 | -0.0071 | 0.0173 | 0 | 0 | 10 |
| Stack, LR meta | -0.0133 | -0.0306 | 0.0074 | 0 | 0 | 10 |
| Stack, XGB meta (fixed config) | -0.0078 | -0.0244 | 0.0069 | 0 | 0 | 10 |
| Stack, XGB meta (library defaults) | -0.0300 | -0.0561 | -0.0142 | 1 | 0 | 10 |

Mean over draws of AUROC / ECE:

| method | auc | auprc | ece |
|---|---|---|---|
| Best single Level-1 (picked in-fold) | 0.7590 | 0.6113 | 0.0796 |
| LR single (all features) | 0.7977 | 0.6772 | 0.1054 |
| Mean of Level-1 | 0.8217 | 0.6961 | 0.0959 |
| Stack, LR meta | 0.8003 | 0.6835 | 0.0819 |
| Stack, XGB meta (fixed config) | 0.8058 | 0.6776 | 0.0858 |
| Stack, XGB meta (library defaults) | 0.7836 | 0.6474 | 0.1563 |
| XGB single (fixed config) | 0.8136 | 0.6894 | 0.0792 |

## Ablation under the new protocol (fixed XGBoost, identical folds, 15 scores)

### MIMIC-IV 24 h mortality

| condition | n_features | AUROC | delta_vs_all | CI95_low | CI95_high | p_corrected_t |
|---|---|---|---|---|---|---|
| All features | 120 | 0.8337 | 0.0000 | 0.0000 | 0.0000 |  |
| Vitals only | 35 | 0.7998 | -0.0339 | -0.0503 | -0.0176 | 0.0005 |
| Labs only | 85 | 0.7785 | -0.0552 | -0.0753 | -0.0352 | 0.0000 |
| Mean aggregate only | 24 | 0.8271 | -0.0066 | -0.0130 | -0.0003 | 0.0416 |
| No clipping | 120 | 0.8344 | 0.0007 | -0.0036 | 0.0050 | 0.7333 |

### eICU 72 h mortality

| condition | n_features | AUROC | delta_vs_all | CI95_low | CI95_high | p_corrected_t |
|---|---|---|---|---|---|---|
| All features | 55 | 0.8529 | 0.0000 | 0.0000 | 0.0000 |  |
| Vitals only | 15 | 0.7923 | -0.0606 | -0.0723 | -0.0489 | 0.0000 |
| Labs only | 40 | 0.8067 | -0.0461 | -0.0539 | -0.0384 | 0.0000 |
| Mean aggregate only | 11 | 0.7930 | -0.0598 | -0.0658 | -0.0538 | 0.0000 |
| No clipping | 55 | 0.8529 | 0.0000 | -0.0011 | 0.0011 | 0.9917 |

### ICU LOS>7 d (landmark 2 d)

| condition | n_features | AUROC | delta_vs_all | CI95_low | CI95_high | p_corrected_t |
|---|---|---|---|---|---|---|
| All features | 145 | 0.7815 | 0.0000 | 0.0000 | 0.0000 |  |
| Vitals only | 35 | 0.7522 | -0.0294 | -0.0517 | -0.0070 | 0.0136 |
| Labs only | 110 | 0.7226 | -0.0590 | -0.0881 | -0.0298 | 0.0007 |
| Mean aggregate only | 29 | 0.7583 | -0.0232 | -0.0410 | -0.0054 | 0.0142 |
| No clipping | 145 | 0.7839 | 0.0024 | -0.0039 | 0.0087 | 0.4325 |

### ICU LOS>14 d (landmark 2 d)

| condition | n_features | AUROC | delta_vs_all | CI95_low | CI95_high | p_corrected_t |
|---|---|---|---|---|---|---|
| All features | 145 | 0.7952 | 0.0000 | 0.0000 | 0.0000 |  |
| Vitals only | 35 | 0.7793 | -0.0159 | -0.0505 | 0.0187 | 0.3411 |
| Labs only | 110 | 0.7233 | -0.0719 | -0.1008 | -0.0429 | 0.0001 |
| Mean aggregate only | 29 | 0.7681 | -0.0271 | -0.0537 | -0.0005 | 0.0461 |
| No clipping | 145 | 0.7954 | 0.0002 | -0.0064 | 0.0068 | 0.9557 |

## SHAP, fixed-config XGBoost on the full cohorts

Shared features: 55. Spearman rank correlation of mean |SHAP|: 0.507 (95% CI 0.280 to 0.681), p = 7.7e-05.

mimic24 top 10:

| feature | feature_std | mean_abs_shap |
|---|---|---|
| labs_lactate_last | labs_lactate_last | 0.340 |
| vitals_gcs_verbal_last | vitals_gcs_verbal_last | 0.158 |
| vitals_spo2_last | vitals_spo2_last | 0.147 |
| vitals_gcs_verbal_avg | vitals_gcs_verbal_avg | 0.133 |
| vitals_temperature_f_avg | vitals_temperature_f_avg | 0.120 |
| vitals_spo2_min | vitals_spo2_min | 0.112 |
| vitals_temperature_f_min | vitals_temperature_f_min | 0.110 |
| vitals_gcs_eye_last | vitals_gcs_eye_last | 0.106 |
| vitals_spo2_avg | vitals_spo2_avg | 0.102 |
| labs_bun_last | labs_bun_last | 0.097 |

eicu72 top 10:

| feature | feature_std | mean_abs_shap |
|---|---|---|
| vitals_Heart Rate_last | vitals_heart_rate_last | 0.306 |
| labs_WBC_last | labs_wbc_last | 0.249 |
| vitals_SpO2_min | vitals_spo2_min | 0.222 |
| labs_BUN_last | labs_bun_last | 0.211 |
| vitals_SpO2_last | vitals_spo2_last | 0.173 |
| labs_Potassium_last | labs_potassium_last | 0.158 |
| labs_Platelets_last | labs_platelets_last | 0.154 |
| vitals_Respiratory Rate_last | vitals_resp_rate_last | 0.126 |
| labs_Creatinine_last | labs_creatinine_last | 0.110 |
| vitals_Heart Rate_avg | vitals_heart_rate_avg | 0.105 |
