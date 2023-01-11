# Interpretable spatial cell learning enhances the characterization of patient tissue microenvironments with highly multiplexed imaging data

<img src="image/github.png" alt="Logo" width="600" align = "right">

In highly multiplexed imaging data analysis, there is a gap between tissue phenotype identification and its relevant microenvironment characterization. Here we present **InterSTELLAR**, a geometric deep learning framework for multiplexed imaging data, to directly link tissue-scale outcomes with the corresponding cell communities that are clinically relevant. Using a publicly available breast cancer imaging mass cytometry dataset, InterSTELLAR allows simultaneous tissue phenotype prediction and interested community detection, with better performance than reference methods. Downstream analyses demonstrate InterSTELLAR is able to capture specific microenvironment features from different clinical cancer subtypes, as well as discover potential relationships to patient prognosis. InterSTELLAR represents an application of geometric deep learning with direct benefits for extracting enhanced microenvironment characterization for multiplexed imaging of patient samples.

## Contents

- [Directory structure of InterSTELLAR](#directory-structure-of-interstellar)
- [Customize environment for InterSTELLAR](#customize-environment-for-interstellar)
  - [Our InterSTELLAR environment](#our-interstellar-environment)
  - [Installation](#installation)
- [Implement InterSTELLAR](#implement-interstellar)
  - [Download example data](#download-example-data)
  - [Implement InterSTELLAR with scripts](#implement-interstellar-with-scripts)
- [Contact](#contact)
- [References](#references)

## Directory structure of InterSTELLAR
```
InterSTELLAR
|---InterSTELLAR
|---|---Graph_Construction_Functions.py
|---|---InterSTELLAR_Network.py
|---|---InterSTELLAR_Train.py
|---|---InterSTELLAR_Utility.py
|---Demo
|---|---Graph_Construction_Demo.py
|---|---Training_DeepSNiF_script.py
|---|---InterSTELLAR_Train_Demo.py
|---|---InterSTELLAR_Predict_Demo.py
|---trained_weights
|---|---trained_weights_10fold
|---|---trained_weights_ETA
```

## Customize environment for InterSTELLAR
### Our InterSTELLAR environment
- Windows 10 64bit
- Python 3.6
- PyTorch 1.10.2
- PyTorch-Geometric 2.0.4
- NVIDIA GPU + CUDA
- Note that Linux also works with such configurations

### Installation
- Create a virtual environment and install PyTorch and PyTorch_Geometric (we run all the codes in a Anaconda Powershell Prompt).
```
$ conda create -n 'InterSTELLAR' python=3.6
$ conda activate InterSTELLAR (some systems recommend: source activate IMC_Denoise)
$ conda install -c anaconda brotlipy
$ pip install torch==1.10.2 torch-geometric==2.0.4
$ conda install -c anaconda cudnn=7.6.5 cudatoolkit=10.1.243
```
- Download the source code and install the package in your folder.
```
$ git clone https://github.com/PENGLU-WashU/InterSTELLAR.git
$ cd InterSTELLAR
$ pip install -e .
```

## Implement InterSTELLAR
### Download example data
- Please go to https://doi.org/10.5281/zenodo.7516958 and download **graph_data.npy**. 
- Previously trained weights can be accessed in this repository.
### Implement InterSTELLAR with scripts
- Activate the InterSTELLAR environment.
```
$ conda activate InterSTELLAR
```
- Graph construction from raw data.
  ```
  python Demo/Graph_Construction_Demo.py --dataset 'graph_data.npy' --neighbour_thresh '40' --myeps '1e-4'  --built_graph_data 'constructed_graph_data.npy' 
  ```
- InterSTELLAR training with the demo data.
  ```
  python Demo/InterSTELLAR_Train_Demo.py --dataset 'constructed_graph_data.npy' --fold_order '5' --k_sample_val '8' --epoch_num '30' --lr '3e-4' --eta '0.85' --n_classes '3' --out_channels '10' --batch_size '8' --lambda_reg '3e-5' --results_dir 'saved_model_interstellar5.pt' --GPU 'True'
  ```
- InterSTELLAR prediction with the test dataset from the demo data.
  ```
  python Demo/InterSTELLAR_Predict_Demo.py --trained_weights 'saved_model_interstellar5.pt' --dataset 'constructed_graph_data.npy' --n_classes '3' --out_channels '10' --GPU 'True' --save_results 'True'
  ```   

## Contact

Peng Lu - [@penglu10](https://twitter.com/penglu10) - penglu@wustl.edu
<br/>Project Link: [https://github.com/PENGLU-WashU/IMC_Denoise](https://github.com/PENGLU-WashU/IMC_Denoise)
<br/>Lab Website: [Thorek Lab WashU](https://sites.wustl.edu/thoreklab/)

## References
