Extension:
High priority:
[] ContextAR (600M)
[] ContextAR (600M)+ q-former
[] ContextFlow 2B
Low priority:
[] Sequence training
[] Light version, light-version inherit from the base version

refactor the data pipeline
[] refactor the custom dataset and custom dataset v2
[] fix the bugging in the padding of custom dataset
optimize the code
[] can we simplify the reindex_filtered_dict function (refer to customdataset.py)?
[] clean the configs
[] organize the unit testing files
[] remove third image for libero dataset
[] merge pi0-incontextv12 and pi0-incontextv18
[] merge the light versions
[] simplify the data preparation process

[] remove old incontext data loader
[] update aloha with custom incontext data loader
[] modify the inference code to use customdataconfig