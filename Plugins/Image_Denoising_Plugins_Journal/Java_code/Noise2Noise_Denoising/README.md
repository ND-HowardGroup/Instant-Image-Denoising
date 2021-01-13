

# [ImageJ](https://imagej.net/) + [TensorFlow](https://www.tensorflow.org) integration layer

This component is a library which can translate between ImageJ images and
TensorFlow tensors.

It also contains a demo ImageJ command for denoising images using a
TensorFlow image model, adapted from the trained U-Net model[TensorFlow image denoising
tutorial](https://www.tensorflow.org/tutorials/image_recognition).

## Quickstart

```sh
git clone https://github.com/varunmannam/Image_denoising
cd Image_denoising
mvn -Pexec
```

This requires [Maven](https://maven.apache.org/install.html).  Typically `brew
install maven` on OS X, `apt-get install maven` on Ubuntu, or [detailed
instructions](https://maven.apache.org/install.html) otherwise.
