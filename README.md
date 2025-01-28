# Paint-By-Numbers (Crayon-By-Numbers)
The goal of this project was the convert any given image to a paint-by-numbers project that can be easily executed by the user. It relies on crayola crayon boxes (the source of which crayon is in which box, and the crayons' RGB codes is [Wikipedia](https://en.wikipedia.org/wiki/List_of_Crayola_crayon_colors)). 

This project was chosen as a fun way to learn image processing techniques used in processing satellite imagery and other practical contexts. It was completed with the generous help of skimage documentation, online blogs, and stackexchange to determine best practices for image processing.

As an example, it takes an image (like that of my family's kitten): 
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

Another kitten example (you'll notice that the final product is now my profile photo): 
<p align="left">
    <img src="https://github.com/ellamarrero/pbn_project/blob/main/data/bea.jpeg" alt="Bea (Original Image)" width="250"/>
    <img src="https://github.com/ellamarrero/pbn_project/blob/main/output/Bea_pre_crayon.jpeg" alt="Bea (Segmented Image)" width="250"/>
    <img src="https://github.com/ellamarrero/pbn_project/blob/main/output/Bea_result.jpeg" alt="Bea (Crayon PBN Image)" width="250"/>
</p>

<p align="left">
    <img src="https://github.com/ellamarrero/pbn_project/blob/main/output/bea.jpeg" alt="Bea ( PBN Image)" width="200"/>
</p>
