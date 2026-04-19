# usage

to use, `cargo run` or `./mosaics` with `./palette/` filled with your chosen palette images, and your target image placed in `mosaic.jpg`. the paths are relative to the current working directory and there are no command line arguments. (it was a really quick project so I didn't have time to make a command line interface)

# description

given an input image, and a collection of palette images, this reconstructs the input image using tiles made out of the palette images. it's implemented with a kd tree and compares images/tiles using average color, in the CIELAB color space for greater accuracy. `rayon` is used wherever possible to speed up computation by using multiple cores. 

it's inspired by, but not based on, the UIUC CS 225 "mosaics" MP; I made this project because I wanted to create some mosaics of my own but the CS 225 version was crashing when given a lot of palette images. the code in this project is all original and not copied from the MP, and the implementation is fairly different. 

