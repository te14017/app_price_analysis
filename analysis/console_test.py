# -*- coding: utf-8 -*-

### This section is used for copy and paste to Ipython console for data preview and test

import pandas as pd
import analysis.analysisUtil as util
import cleanup.cleanup as cleanup
import cleanup.cleanupUtil as cleanUtil
import analysis.analysis as anal
X = anal.prepare_data()
X.drop('text', axis=1, inplace=True)

# pipeline test
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import GaussianNB
pipe = Pipeline([('selector', SelectKBest(score_func=chi2, k=10)),
                 ('clf', GaussianNB())])

# text feature test
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
count_vector = CountVectorizer(stop_words='english', ngram_range=(1,1))
tfidf = TfidfTransformer()
text_words = count_vector.fit_transform(X_train['text'])
text_freq = tfidf.fit_transform(text_words)
df_dense = pd.DataFrame(data = text_freq.toarray())
X_concat = pd.concat([X_train, df_dense], axis=1)

# merge priority: Tools, Arcade + Action, Sports, Productivity,
#                 Photography + Puzzle, Finance + Board, Medical+ Adventure,
#                   

# Video Players & Editors: 0.35 (NB,Tree,GB), 244
# Tools: 0.29 (KN), 817
# Sports: 0.41 (Tree,GB), 701
# Role Playing: 0.48 (Tree,GB,MLP), 314
# Racing: 0.47 (SGD,GB), 0.39(KN,Tree), 108
# Productivity: 0.41 (NB,Tree,GB,MLP), 812
# Photography: 0.48 (KN), 0.42(Tree,GB,MLP), 413
# Medical: 0.46 (NB,Tree,GB,MLP), 456
# Maps & Navigation: 0.47 (KN,NB,Tree), 0.41(GB,MLP), 238
# Health & Fitness: 0.45 (NB,Tree,GB,MLP), 761
# Finance: 0.37 (SGD), 0.25(KN,NB,Tree,GB), 236
# Entertainment: 0.5(NB,Tree,MLP), 0.48(KN,GB), 904
# Educational: 0.5(SGD), 0.48(NB,Tree,GB), 550
# Communication: 0.42 (NB,Tree,GB,MLP), 375
# Card: 0.38 (KN), 0.35(NB,Tree,GB), 206
# Business: 0.53 (SGD), 0.5(KN,Tree,GB), 367
# Board: 0.51 (KN), 0.42(NB,Tree,GB,MLP), 166
# Arcade: 0.44 , 447
# Adventure: 0.47, 500
# Action: 0.40 (SGD), 0.31(NB,Tree,GB,MLP), 296
# Strategy: 0.43 (Tree) 0.37 (KN,GB), 272
# Puzzle: 0.48 (NB,Tree,GB,MLP), 865

# Personalization: 0.70 (NB,Tree), 0.68(GB,MLP), 935
# Music & Audio: 0.60 (NB,Tree,GB,MLP), 933
# Lifestyle: 0.54 (Tree, GB), 0.52(NB,MLP), 952
# Education: 0.52 (Tree,GB,MLP), 1156
# Casual: 0.55 (NB), 0.53(Tree,GB,MLP), 524
# Books & Reference: 0.56 (NB,Tree,GB,MLP), 931
# Weather: 0.51 (KN,Tree,GB), 167
# Travel & Local: 0.54 (KN), 425
# Social: 0.58 (KN), 0.48(SGD,Tree,GB), 145
# Simulation: 0.56 (KN), 0.52(SGD,Tree,GB), 343

# Video Players & Editors_Tools: 0.34,                  UP1
# Video Players & Editors_Role Playing: 0.44            UP1
# Video Players & Editors_Finance: 0.38                 UP2
# Tools_Finance: 0.34                                   UP2
# Tools_Card: 0.35                                      UP1 -
# Sports_Maps & Navigation： 0.48                        UP1 -
# Role Playing_Productivity: 0.47                       UP1 -
# Role Playing_Maps & Navigation: 0.53                  UP2
# Role Playing_Finance: 0.48                            UP1
# Racing_Arcade: 0.50                                   UP2 *
# Productivity_Maps & Navigation: 0.48                  UP1
# Photography_Communication: 0.48                       UP2
# Photography_Board: 0.47                               UP2
# Photography_Puzzle: 0.50                              UP2 -
# Medical_Adventure: 0.48                               UP1 -
# Medical_Puzzle: 0.50                                  UP2
# Maps & Navigation_Puzzle: 0.49                        UP2
# Finance_Board: 0.46                                   UP2 -
# Finance_Arcade: 0.47                                  UP2 *
# Finance_Puzzle: 0.48                                  UP1
# Entertainment_Card: 0.50                              UP2 *
# Educational_Card: 0.49                                UP1
# Educational_Board: 0.52                               UP2 *
# Communication_Puzzle: 0.48                            UP1
# Card_Arcade: 0.50                                     UP2 *
# Board_Arcade: 0.52                                    UP2 *
# Arcade_Action: 0.52                                   UP2 * -
# Action_Puzzle: 0.48                                   UP1 

# Video Players & Editors_Racing: 0.27
# Racing_Finance: 0.28
# Productivity_Board: 0.3