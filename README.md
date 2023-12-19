
It is important to analyze MS2 spectrum from mass spectrometry in the field of proteomics. Because there was rapid development of mass spectrometry, we can obtain massive amount of MS2 spectrum in a short time. For that reason, there is huge advance if we can describe MS2 spectrum effectively and exactly. There were several spectrum embedding method such as GLEAMS, CLERMS and MS2deepScore using deep learning to get spectrum embedding that describes MS2 spectrum well. however these spectrum embedder do not have any sequential information to the MS2 embedding. So we introduce our model SEMS(Spectrum Embedding with multimodality using Sequence). Our model basically has 2 embedders which are spectrum embedder and sequence embedder. We found this idea from CLIP. With our model there are chances to make spectral library search faster and detect unknown modified peptides. 

........................................

implement
before start variable mgf must be needed
please download mgf file at https://massive.ucsd.edu/ProteoSAFe/status.jsp?task=adfc82524d394cb9be709fa3b2379bad
![image](https://github.com/tjdudgml3/Sequence-Spectrum-embedding/assets/67582418/6e6c1806-79ca-4f96-82dd-875b2baa3afc)

test dataset are in dataset folder. 

to train SEMS you have to change data part to dataset.train_pos as explained in train.ipynb
to validate SEMS you can validate in validataion.ipynb

dataset are from MassiveKB mgf file. and i made dataframe from mgf.
positive pair and negative pairs are made by make_datast.ipynb

spectrum proprecess are performed by data_process.py

dataloader is made by datagenerator.py  
..............................
