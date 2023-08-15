# Medium-range Trajectory Prediction Network Compliant to Physical Constraint for Oceanic Eddy

This paper proposed a novel neural network structure to achieve medium-range eddy trajectory prediction, named **ETPNet**, which is compliant with the physical constraint. This structure contains a variant of the long short-term memory (LSTM) cell, enhancing the dynamic interaction and representation ability of the features, constraints, and knowledge. The pipeline of the whole prediction framework is shown as follows:

<p align="center">
  <img src="img/model.png" alt="Centered Image" width="600" />
</p>

And the structure of the variant LSTM is illustrated as 

<p align="center">
  <img src="img/TraceLSTM.png" alt="Centered Image" width="600" />
</p>


## Experimental Results
| Method (MAGE Loss) |     MGD     |     SGD     |    MSE     |    MAE     |
| :----------------- | :---------: | :---------: | :--------: | :--------: |
| EGRU               |   45.9186   |  321.4303   |   0.1964   |   0.2766   |
| ELSTM              |   32.1280   |  224.8962   |   0.5132   |   0.1818   |
| Seq2Seq            |   40.5100   |  283.5702   |   0.1219   |   0.2452   |
| MSeq2Seq           |   12.5271   |   87.6897   |   0.0124   |   0.0757   |
| ETPNet-Ego         |   10.4589   |   73.2123   |   0.0088   |   0.0634   |
| ETPNet-Physical    |   10.6959   |   74.8712   |   0.0090   |   0.0646   |
| ETPNet             | **10.1715** | **71.2008** | **0.0085** | **0.0611** |

| Method (L1Loss) |   MGD   |   SGD    |  MSE   |  MAE   |
| :-------------- | :-----: | :------: | :----: | :----: |
| EGRU            | 58.4550 | 409.1852 | 0.3792 | 0.3311 |
| ELSTM           | 36.4317 | 255.0218 | 0.6370 | 0.2060 |
| Seq2Seq         | 37.8517 | 264.9622 | 0.1120 | 0.2155 |
| MSeq2Seq        | 14.2436 | 99.7053  | 0.0145 | 0.0848 |
| ETPNet-Ego      | 14.1674 | 99.1718  | 0.0169 | 0.0859 |
| ETPNet-Physical | 18.1555 | 127.0886 | 0.0264 | 0.1094 |
| ETPNet          | 10.2154 | 71.5075  | 0.0086 | 0.0616 |
## Citing

if you find this work is helpful to your research, please consider citing our paper.

```latex
@article{10190732,
  author={Ge, Linyao and Huang, Baoxiang and Chen, Xiaoyan and Chen, Ge},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Medium-Range Trajectory Prediction Network Compliant to Physical Constraint for Oceanic Eddy}, 
  year={2023},
  volume={61},
  number={},
  pages={1-14},
  doi={10.1109/TGRS.2023.3298020}
```
