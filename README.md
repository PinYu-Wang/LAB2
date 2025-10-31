# LAB2    
EEG Classification with BCI competition dataset.    

## Requirements
Python 3.8+ and the following packages: torch, numpy, pandas, matplotlib      

## File
data/:     
  EEG dataset files    
models/:    
  EEGNet.py: EEGNet model definition    
  DeepConvNet.py: DeepConvNet model definition    
dataloader.py: Dataset reading and preparation    
main.py: Main script for training and testing    


## How to Run
param:  
--model: "EEGNet" or "DeepConvNet" (default: EEGNet)    
--batch_size: batch size (default: 64)    
--lr: learning rate (default: 0.01)    
--num_epochs: number of training epochs (default: 150)    
--dropout: dropout rate (default: 0.25)    
--elu_alpha: ELU activation alpha (default: 1.0)    
--F1: number of temporal filters in EEGNet (default: 16)    
--D: number of spatial filters in EEGNet (default: 2)    
--kernel_len: kernel length in EEGNet (default: 64)    
--weight_decay: L2 regularization (default: 0.001)    
--save_dir: directory to save results (default: ./result)       

example: python main.py --model EEGNet --batch_size 64 --num_epochs 150





