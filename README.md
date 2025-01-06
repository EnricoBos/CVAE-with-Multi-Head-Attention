# Conditional Variational Autoencoder with Multi Head Attention

* This repository provides the code for training and evaluating a Conditional Variational Autoencoder (CVAE) with multi-head attention on an image dataset. The project demonstrates the process of training the CVAE model with multi-head attention and evaluating its performance. The dataset required for this project can be downloaded from [Link](https://vis-www.cs.umass.edu/lfw/). Ensure you download both the image data and the accompanying text file containing attributes.



## Overview
The project involves:
- Image processing: Preprocessing images by cropping and resizing
- Model Training: Training a Conditional Variational Autoencoder (CVAE) with multi-head attention on the prepared dataset.
- Evaluation:  Assessing the performance of the trained model by generating samples using random sampling from a normal distribution and analyzing the generated images.

## Image Preprocessing and Dataset Details
* Image Preprocessing:
	- Images are cropped and resized to a consistent format for input to the model. 
The pixel values of the images are rescaled by dividing them by 255, ensuring they fall within the range [0, 1].

* Attribute Selection:
	- From the attributes matrix provided with the dataset, the columns "sunglasses" and "smiling" are selected.
	These attributes are used as conditional inputs to the CVAE during training.

* Multi-Head Attention:
	- The multi-head attention mechanism is structured with 2 heads, enabling the model to capture complex relationships in the data.



## Project Structure
	-  mh_cvae_face_main.py: The main script for training and evaluating the Transformer model.
	-  t_cvae_class.py: A class implementation for the Conditional Variational Autoencoder (CVAE) with multi head attention.
	-  checkpoints/: Directory for saving model checkpoints.



## Dependencies
Make sure you have the following packages installed:
	tensorflow numpy pandas scikit-learn 


## Usage
* Training
	-  To train the model, set the choice variable 'enable_trainig'  to True. If you want to start training without a checkpoint, ensure that any existing checkpoints are deleted.

* Evaluation
	- To evaluate the model, set the choice variable 'enable_trainig' to False. Make sure the model weights are saved in checkpoints folder
		

## Example Output
- **Some Original Images**  
![processed_original_2](https://github.com/user-attachments/assets/f5728146-a750-410a-bb33-76fa84c110bf) ![processed_original_1](https://github.com/user-attachments/assets/c9029bdf-6153-4ce6-bc60-94e40bd8e17d)



- **Generated Images with Sunglasses and Smiling**  
	- Images with sunglass and no-smiling , smiling
   
   	![generated_0](https://github.com/user-attachments/assets/7fbdcf72-010a-4ef8-949f-6237c2749e92) ![generated_2](https://github.com/user-attachments/assets/9124c4ba-07d8-468a-9633-c5e3cb189b37)

	- Images with no sunglass and no-smiling , smiling

	![generated_1](https://github.com/user-attachments/assets/1f9dad68-0f01-4ecb-aeed-756ccbc0928c)  ![generated_3](https://github.com/user-attachments/assets/767c6f0a-b42b-4ea6-8616-b33b1fee8b55)


## Authors

* Enrico Boscolo
