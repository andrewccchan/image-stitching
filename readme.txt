## Compile from source

The main program is written in C++ with OpenCV. Build from source with the
following command.
```bash
# cmake .
# make
```

## Compose an panorama

First, you need a file that provides the paths (absolute path is recommended)
to each image and its estimated focal length (in pixel). An example is given
below:

```
/path/to/images/IMG001.JPG 750
/path/to/images/IMG002.JPG 750
/path/to/images/IMG003.JPG 750
/path/to/images/IMG004.JPG 750
```

With the setting file ready, compose your images into an HDR image with

```bash
./stitch path_to_setting_file
```

## Descriptions:
-   src: all source code are put under this folder.
-   img: image seuqences along with the corresponding setting files.
