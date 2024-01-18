
# Setup
  
1. Download .zip of project  
2. Extract to folder  
3. Create folder weights/ to store weights  

### Training a model:

1. Either:  
	* Unzip img_align_celeba into data/ 
	* Create custom dataset (see below)  
2. Change num_epochs in config.py to desired amount  
3. Run train.py  

### To run pretrained model:

1. Put .pth file in weights/  
2. Change generator_load in config.py to file location of weights  
3. Run main.py

# Dataset Structure
  
├─── data/  
│└─── folder-name-doesnt-matter/  
│└───── images  
  
* Put the path to data/ into config.py's dataroot variable  
* This should work with any collection of images  

## Links
  
[celebrity faces dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)  
  
[tutorial link](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)  
