# ENet - Real Time Semantic Segmentation

A Neural Net Architecture for real time Semantic Segmentation. <br/>
In this repository we have reproduced the ENet Paper - Which can be used on
mobile devices for real time semantic segmentattion. The link to the paper can be found here: [ENet](https://arxiv.org/pdf/1606.02147.pdf)

## How to use?

0. This repository comes in with a handy notebook which you can use with Colab. <br/>
You can find a link to the notebook here: [
ENet - Real Time Semantic Segmentation](https://github.com/iArunava/ENet-Real-Time-Semantic-Segmentation/blob/master/ENet-Real%20Time%20Semantic%20Segmentation.ipynb) <br/>
Open it in colab: [Open in Colab](https://colab.research.google.com/github/iArunava/ENet-Real-Time-Semantic-Segmentation/blob/master/ENet-Real%20Time%20Semantic%20Segmentation.ipynb)

---


0. Clone the repository and cd into it
```
git clone https://github.com/iArunava/ENet-Real-Time-Semantic-Segmentation.git
cd ENet-Real-Time-Semantic-Segmentation/
```

1. Use this command to train the model
```
python3 init.py --mode train -iptr path/to/train/input/set/ -lptr /path/to/label/set/
```

2. Use this command to test the model
```
python3 init.py --mode test -m /path/to/the/pretrained/model.pth -i /path/to/image/to/infer.png
```

3. Use `--help` to get more commands
```
python3 init.py --help
```

## Some results

![enet infer 1](https://user-images.githubusercontent.com/26242097/51782315-4b88d300-214c-11e9-9c92-3444c6582a80.png)
![enet infer 4](https://user-images.githubusercontent.com/26242097/51782341-a02c4e00-214c-11e9-8566-f2092ddad086.png)
![enet infer 6](https://user-images.githubusercontent.com/26242097/51782371-01542180-214d-11e9-80b8-55807f83f776.png)
![enet infer 5](https://user-images.githubusercontent.com/26242097/51782353-c3ef9400-214c-11e9-8c66-276795c83f08.png)
![enet infer 2](https://user-images.githubusercontent.com/26242097/51782324-6b1ffb80-214c-11e9-9f92-741954699f4d.png)

## References
1. A. Paszke, A. Chaurasia, S. Kim, and E. Culurciello.
Enet: A deep neural network architecture
for real-time semantic segmentation. arXiv preprint
arXiv:1606.02147, 2016.

## Citations

```
@inproceedings{ BrostowSFC:ECCV08,
  author    = {Gabriel J. Brostow and Jamie Shotton and Julien Fauqueur and Roberto Cipolla},
  title     = {Segmentation and Recognition Using Structure from Motion Point Clouds},
  booktitle = {ECCV (1)},
  year      = {2008},
  pages     = {44-57}
}

@article{ BrostowFC:PRL2008,
    author = "Gabriel J. Brostow and Julien Fauqueur and Roberto Cipolla",
    title = "Semantic Object Classes in Video: A High-Definition Ground Truth Database",
    journal = "Pattern Recognition Letters",
    volume = "xx",
    number = "x",   
    pages = "xx-xx",
    year = "2008"
}
```

## License

The code in this repository is distributed under the BSD v3 Licemse.<br/>
Feel free to fork and enjoy :)
