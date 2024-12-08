cols = ['gameNum', 'OPS', 'post_OPS', 'SB', 'post_SB', 'inducted', 'SHO', 'SV', 'WHIP', 'KPer9', 'post_SHO', 'post_SV', 'post_WHIP', 'post_KPer9', 'PitchingTripleCrowns', 'TripleCrowns', 'MVPs', 'ROYs', 'WSMVPs', 'CyYoungs', 'GoldGloves', 'ASGMVPs', 'RolaidsReliefManAwards', 'NLCSMVPs', 'ALCSMVPs', 'SilverSluggers']

Accuracy: 0.9840848806366048
[[3689   26]
 [  34   21]]
Precision: 0.44680851063829785
Recall 0.38181818181818183
F1 0.4117647058823529

Removed SV, post_SV
cols = ['gameNum', 'OPS', 'post_OPS', 'SB', 'post_SB', 'inducted', 'SHO', 'WHIP', 'KPer9', 'post_SHO', 'post_WHIP', 'post_KPer9', 'PitchingTripleCrowns', 'TripleCrowns', 'MVPs', 'ROYs', 'WSMVPs', 'CyYoungs', 'GoldGloves', 'ASGMVPs', 'RolaidsReliefManAwards', 'NLCSMVPs', 'ALCSMVPs', 'SilverSluggers']

Accuracy: 0.983554376657825
[[3690   25]
 [  37   18]]
Precision: 0.4186046511627907
Recall 0.32727272727272727
F1 0.3673469387755102

If you use just awards you can get precision up to 0.62