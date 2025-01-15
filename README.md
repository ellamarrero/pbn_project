# Paint-By-Numbers (Crayon-By-Numbers)
The goal of this project was the convert any given image to a paint-by-numbers project that can be easily executed by the user. It relies on crayola crayon boxes (the source of which crayon is in which box, and the crayons' RGB codes is [Wikipedia](https://en.wikipedia.org/wiki/List_of_Crayola_crayon_colors)). 

This project was completed by Ella Marrero, with the generous help of skimage documentation, online blogs, and stackexchange to determine best practices for image processing. 

As an example, it takes an image (like that of my family's kittens): 

Segements it using simple linear iterative clustering (SLIC), and averages regions using region adjacency graph thresholding (RAG):
![Original image of Tig](https://github.com/ellamarrero/pbn_project/data/tig.jpeg "Tig (Original Image)")

Then replaces the colors with the closest match from a given Crayola crayon box number:
**Note that this may change the overall color scheme from warm to cool or match two different colors to one similar crayon color, reducing the overall number of colors in the final image**
![Segmented image of Tig](https://github.com/ellamarrero/pbn_project/output/Tig_pre_crayon.jpeg "Tig (Processed Image)")


And finally, it creates an outline of the image with the segments matched to their closest Crayola crayon: 
![Crayon image of Tig](https://github.com/ellamarrero/pbn_project/output/tig_result.jpeg "Tig (Crayon PBN Image)")

