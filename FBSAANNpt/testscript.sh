source activate pytorchsenv
replacesubstring "#datasetName = 'tabular-benchmark'" "datasetName = 'tabular-benchmark'" ANNpt_globalDefs.py
python ANNpt_main.py
replacesubstring "#datasetName = 'blog-feedback'" "datasetName = 'blog-feedback'" ANNpt_globalDefs.py
python ANNpt_main.py
replacesubstring "#datasetName = 'titanic'" "datasetName = 'titanic'" ANNpt_globalDefs.py
python ANNpt_main.py
replacesubstring "#datasetName = 'red-wine'" "datasetName = 'red-wine'" ANNpt_globalDefs.py
python ANNpt_main.py
replacesubstring "#datasetName = 'breast-cancer-wisconsin'" "datasetName = 'breast-cancer-wisconsin'" ANNpt_globalDefs.py
python ANNpt_main.py
replacesubstring "#datasetName = 'diabetes-readmission'" "datasetName = 'diabetes-readmission'" ANNpt_globalDefs.py
python ANNpt_main.py
replacesubstring "#datasetName = 'banking-marketing'" "datasetName = 'banking-marketing'" ANNpt_globalDefs.py
python ANNpt_main.py
replacesubstring "#datasetName = 'adult_income_dataset'" "datasetName = 'adult_income_dataset'" ANNpt_globalDefs.py
python ANNpt_main.py
replacesubstring "#datasetName = 'covertype'" "datasetName = 'covertype'" ANNpt_globalDefs.py
python ANNpt_main.py
replacesubstring "#datasetName = 'higgs'" "datasetName = 'higgs'" ANNpt_globalDefs.py
python ANNpt_main.py
replacesubstring "#datasetName = 'topquark'" "datasetName = 'topquark'" ANNpt_globalDefs.py
python ANNpt_main.py
replacesubstring "#datasetName = 'iris'" "datasetName = 'iris'" ANNpt_globalDefs.py
python ANNpt_main.py
replacesubstring "#datasetName = 'new-thyroid'" "datasetName = 'new-thyroid'" ANNpt_globalDefs.py
python ANNpt_main.py
