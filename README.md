# Semantic Segmentation with LandCover.ai Dataset

In this project I used LandCover.ai Dataset Version 2. It has 4 classes: building (1), woodland (2), water(3) and road(4). Goal of this project is deteck those areas in images learning from pixels. The project is an example of semantic segmentation. I tried 3 different versions. All in the U-net model.

### Version 1

In this version I used some of the dataset because I thought that it may has a faster computation. Results was promising so I added more pictures for second version. 
In the first version I used 120 pictures for training. For a deep learning project it is a pretty small amount. I used OpenCV library to read images but I found this method is slowing me and using lots of RAM. I worked on GPU in Colab. You can directly open the notebook I worked from the link below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BSKVKPn76X-PXFIAFIbgUOh57PB98KFx)

### Version 2

In the second version, I knew that I should use more images but I was still dealing with handling image reading with tensorflow so I was still using OpenCV. So that I could not increased the image amount too much. I trained the model with 300 images. Results was again promising. I worked on GPU in Colab. You can directly open the notebook I worked from the link below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VRcu9FD2ZVbFtanhBDP9QtUEUMgbDPmP#scrollTo=qlOdLXKzAHuL)

### Version 3
In version 3, I decided to resize the images. Resizing the images helped me a lot about RAM usage but I believe it decreased the success of the model. I worked on GPU in Colab. You can directly open the notebook I worked from the link below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aUWLxLwY5vYeFKDXqeF7Rmoqx_OoG_PZ)

## Citation

@InProceedings{Boguszewski_2021_CVPR,
      author = {Boguszewski, Adrian and Batorski, Dominik and Ziemba-Jankowska, Natalia and Dziedzic, Tomasz and Zambrzycka, Anna},
      title = {LandCover.ai: Dataset for Automatic Mapping of Buildings, Woodlands, Water and Roads from Aerial Imagery},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
      month = {June},
      year = {2021},
      pages = {1102-1110}
}
