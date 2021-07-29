# Semantic Segmentation with LandCover.ai Dataset

In this project I used LandCover.ai Dataset Version 2. It has 4 classes: building (1), woodland (2), water(3) and road(4). Goal of this project is deteck those areas in images learning from pixels. The project is an example of semantic segmentation. I tried 3 different versions. All in the U-net model.

## Version 1

In this version I used some of the dataset because I thought that it may has a faster computation. Results was promising so I added more pictures for second version. 
In the first version I used 120 pictures for training. For a deep learning project it is a pretty small amount. I used OpenCV library to read images but I found this method is slowing me and using lots of RAM. I worked on GPU in Colab. You can directly open the notebook I worked from the link below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BSKVKPn76X-PXFIAFIbgUOh57PB98KFx)

### Results

![image](https://user-images.githubusercontent.com/69505652/127444919-f1431721-8bb3-433d-b50d-b8f75a0b66aa.png)

The first image in the below represents the model's result and the picture below it represents the actual labels.

![image](https://user-images.githubusercontent.com/69505652/127445043-b218c298-f071-4c31-81b8-7894afe49ce6.png)


## Version 2

In the second version, I knew that I should use more images but I was still dealing with handling image reading with tensorflow so I was still using OpenCV. So that I could not increased the image amount too much. I trained the model with 300 images. Results was again promising. I worked on GPU in Colab. You can directly open the notebook I worked from the link below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VRcu9FD2ZVbFtanhBDP9QtUEUMgbDPmP#scrollTo=qlOdLXKzAHuL)

### Results

![image](https://user-images.githubusercontent.com/69505652/127445184-87271477-a843-4477-b191-0597a19e8021.png)

The first image in the below represents the model's result and the picture below it represents the actual labels.

![image](https://user-images.githubusercontent.com/69505652/127445240-f507e632-f978-43a4-a2e7-7ee883e6cc66.png)
![image](https://user-images.githubusercontent.com/69505652/127445276-ea302f7e-b91c-4db2-a0dc-9f7b7af3e919.png)

## Version 3
In version 3, I decided to resize the images. Resizing the images helped me a lot about RAM usage but I believe it decreased the success of the model. I worked on GPU in Colab. You can directly open the notebook I worked from the link below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aUWLxLwY5vYeFKDXqeF7Rmoqx_OoG_PZ)

### Results

![image](https://user-images.githubusercontent.com/69505652/127445329-65cca848-22bf-410f-8396-bc335f5c7d39.png)

The first image in the below represents the model's result and the picture below it represents the actual labels.

![image](https://user-images.githubusercontent.com/69505652/127445472-5f51d8a4-cdf8-4c40-8567-9c9d1dcbe74f.png)
![image](https://user-images.githubusercontent.com/69505652/127445417-80c3df8d-e41d-4e88-9bc7-087db61a0092.png)

## Testing the Model With Images from Istanbul Technical University 
<b><i>Version 3</i></b>
![image](https://user-images.githubusercontent.com/69505652/127445607-0c0987b2-ebb1-4403-a21a-38fa5a24dd22.png)

<b><i>Version 2</i></b>
![image](https://user-images.githubusercontent.com/69505652/127445818-564fb77f-7d2f-4025-8398-dd40829ba569.png)
![image](https://user-images.githubusercontent.com/69505652/127445855-4d871138-2d59-4b9c-90e9-d67d197f4cec.png)

# Final Comments

I developed this project to learn more about deep learning, tensorflow and keras. I believe I accomplished my main goal. The project is open to any kind of improvement. I will be happy to hear thoughts on this project. Thank you for checking!

# Citation

@InProceedings{Boguszewski_2021_CVPR,
      author = {Boguszewski, Adrian and Batorski, Dominik and Ziemba-Jankowska, Natalia and Dziedzic, Tomasz and Zambrzycka, Anna},
      title = {LandCover.ai: Dataset for Automatic Mapping of Buildings, Woodlands, Water and Roads from Aerial Imagery},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
      month = {June},
      year = {2021},
      pages = {1102-1110}
}
