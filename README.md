# Potato_Disease_Classification_using_CNN

This prpject help farmers to decide that whether there is an early signs of blight in a particular potato leaf or not.

### Deep Learning Model Used: CNN
> 80% of the data is split into training.
> Out of rest 20%-- 10% on Test and 10% on Validation Data
> Model Pipeline:
  1.) The input image will be reshaped to (256,256,3)
  2.) The pixels will then be converted to 0 and 1
  3.) First Convolution layer with 32 filters with relu activation function and padding='valid' is used followed by maxpooling of size (2,2)
  4.) Similarly 5 more convolutional layer with 64 filters each of size (3,3) has been made.
  5.) Flatten is then used to flaten it to vector.
  6.) A dense layer is then passed of 64 neurons and softmax activation function is then applied to get desired output

### Deployement Used: Streamlit web app
> Code can be found in main.py where the web app after getting image is first converting to numpy array and then doing prediction and displaying the class with highest probability.

The project can be helpful in Agricultral usage where farmers suffer for not being able to capture early blight.
