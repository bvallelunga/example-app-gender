# Gender

## App Design
This app makes use of a pretrained keras implementation of the mini-Xception model described in the Bonn-Rhein-Sieg 
University of Applied Sciences paper, ["Real-time Convolutional Neural Networks for Emotion and Gender Classification"][1]. 
The pretrained model can be found at the paper's accompanying github repo, [oarriaga/face_classication][2]. 

The model is trained on the IMDB gender dataset, a labeled dataset of ~460K RGB face images each of which belongs in either 
the "man" or "woman" class. The authors of the mini-Xception model report an accuracy of 95% on the IMDB dataset.

## Contributing
Code should be written for Python 3, include documentation (docstrings & comments), follow PEP 8 and pass all unittests.
To run the unittests, simply run `python -m unittest` from the repo directory.   

 
[1]: https://arxiv.org/abs/1710.07557
[2]: https://github.com/oarriaga/face_classification

