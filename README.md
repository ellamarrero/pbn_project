# Paint-By-Numbers (Crayon-By-Numbers)

### Objective: 
Create a paint-by-numbers (using Crayola crayons) from any image! The code currently relies on the color values listed for crayola crayon boxes (RGB codes' source is [Wikipedia](https://en.wikipedia.org/wiki/List_of_Crayola_crayon_colors)).


### Background:
I completed this project as a a fun way to learn about image processing techniques, particularly those used in remote image analysis, e.g. satellite imagery processing. xThroughout the project, I built an understanding of image processing through educational youtube videos (e.g. [What is a convolution? - 3Blue1Brown](https://www.youtube.com/watch?v=KuXjwB4LzSA&pp=ygUQaW1hZ2UgcHJvY2Vzc2luZw%3D%3D)), skimage documentation, and online blogs/forums. 


### Example:
As an example, it takes an image (like that of my kitten): 
<p align="left">
  <img src="https://github.com/ellamarrero/pbn_project/blob/main/data/tig.jpeg" alt="Tig (Original Image)"width="200"/>
</p>

Segments it using simple linear iterative clustering (SLIC), and averages regions using region adjacency graph thresholding (RAG):
<p align="left">
  <img src="https://github.com/ellamarrero/pbn_project/blob/main/output/Tig_pre_crayon.jpeg" alt="Tig (Processed Image)" width="200"/>
</p>


Then replaces the colors with the closest match from a given Crayola crayon box number:
  *Note that this may change the overall color scheme from warm to cool or match two different colors to one similar crayon color, reducing the overall number of colors in the final image*
<p align="left">
  <img src="https://github.com/ellamarrero/pbn_project/blob/main/output/tig_result.jpeg" alt="Tig (Crayon PBN Image)" width="200"/>
</p>


And finally, it creates an outline of the image with the segments matched to their closest Crayola crayon to be colored in!: 
<p align="left">
  <img src="https://github.com/ellamarrero/pbn_project/blob/main/output/tig.jpeg" alt="Tig (Crayon PBN Image)" width="200"/>
</p>

Another kitten example: 
<p align="left">
    <img src="https://github.com/ellamarrero/pbn_project/blob/main/data/bea.jpeg" alt="Bea (Original Image)" width="250"/>
    <img src="https://github.com/ellamarrero/pbn_project/blob/main/output/Bea_pre_crayon.jpeg" alt="Bea (Segmented Image)" width="250"/>
    <img src="https://github.com/ellamarrero/pbn_project/blob/main/output/Bea_result.jpeg" alt="Bea (Crayon PBN Image)" width="250"/>
</p>

<p align="left">
    <img src="https://github.com/ellamarrero/pbn_project/blob/main/output/bea.jpeg" alt="Bea ( PBN Image)" width="200"/>
</p>

### Next steps
This project is far from perfectly executed. Upon printing and coloring these images myself, I found that not all colors translated well on paper, partially due to color perception (real colors in images do not line up with our perceived colors), and partially due to the use of crayons to color (coloring is not entirely uniform, and often lighter than the color that corresponds to its RGB value). One temporary workaround to this was to use the functionality of Apple's Photos app to max out the saturation and luminance of images, and up the brightness, so more colors would stand out (otherwise many muddle into gray/brown). 

To improve on the project, I'd like to change it so any user can upload their own paints/colors and their RGB values (or CIELAB values, as some paint palettes provide that) to fill in the painting with. This would allow the end result to potentially be more accurate, and make it so users aren't required to own a pack of crayola crayons. Other modifications to the code would include:

  * creating a more efficient way to list unique color values in image (currently counting unique sets in numpy array)

  * throwing error to user input for crayon box if not in actual crayon box set

  * allow users to just print outline image if they want to just use segmentation part of code