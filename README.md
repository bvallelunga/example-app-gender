# Gender
Quickly determine the gender of a face. Trained on hundreds of thousands of IMDB images."

**Possible Use Cases**
  * Adding gender classification to a face detection system
  * Targeted messaging to your users based on their gender/interests
    * Especially useful if you do not ask what a users gender is during registration
  * Categorizing profiles from social networks by gender


## Input Scheme
The input should contain a base64 encoded image of a face. The image must be at least 
64 x 64 pixels. In order to get the best results, make sure your input image is a tight face shot.
``` json
{
  "image": "BASE_64_ENCODED_IMAGE"
}
```

## Output Scheme
The output will map each gender to a percentage. The percentages measure how confident 
the model is that the person is of that gender.
``` json
{
  "woman": 0.91,
  "man": 0.09
}
```


## Training
The model was trained by the [B-IT-BOTS robotics team][1] on over 500,000 IMDB images. 


[1]: https://mas-group.inf.h-brs.de/?page_id=622
