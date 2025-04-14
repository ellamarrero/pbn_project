# import necessary libraries
import os # to get working directory
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
import imageio.v3 as iio 
import scipy
import math
import pandas as pd
from itertools import combinations, product, chain


def load_crayola_colors(cbox = 96): 
    """
    Loads in CSV of crayola box colors/RGB values, filters for specific num crayons
    Note: source of CSV is wikipedia page for crayon colors 
    Inputs:
    - cbox (int) - number of crayons in box (16, 24, 48, 64, 96, 120)
    Returns:
    - tuple (pd.DataFrame, list) - dataframe with color names and RGB vlaues, list of colors cielab space colors
    """
    crayola_box = pd.read_csv('crayola_colors.csv')
    box = crayola_box[['name','r','g','b']][crayola_box[f'box{cbox}'] =='Yes'] 
    box['srgb'] = list(zip(box['r']/255, box['g']/255, box['b']/255)) # create one rgb value for conversion
    return box


def segment_image(img): 
    """
    Given an image, segments using simple linear iterative clustering and joins segments using region 
    adjacent graph thresholding. 
    Inputs:
    - img (image) - image to be segmented 
    Returns:
    - tuple (segmented image, image segment labels) 
    """
    segments_slic = ski.segmentation.slic(img, n_segments=300, compactness=20, sigma=1, start_label=1)
    img_slic = ski.color.label2rgb(segments_slic, img, kind='avg', bg_label=None) # specify background color or will average to 0

    img_rag = ski.graph.rag_mean_color(img, segments_slic)
    reg_labels = ski.graph.cut_threshold(segments_slic, img_rag,  thresh = 20)
    img_seg = ski.color.label2rgb(reg_labels, img, kind='avg', bg_label=None) # rag might need padding to deal with edge issue.

    return img_seg, reg_labels


def get_uniq_colors(img_seg): 
    """
    Given a segmented image (i.e. simplified), identify all unique colors in image
    Inputs:
    - img_seg (segmented image)
    Returns:
    - color_lst (list of unique cielab colors)
    """
    colors_only = np.vstack(img_seg)
    colors_only_tup = list(map(tuple, colors_only)) 
    uniq_clrs = {x: colors_only_tup.count(x) for x in set(colors_only_tup)}
    uniq_clrs_srgb = [np.array(x)/255 for x in uniq_clrs.keys()] 

    return uniq_clrs_srgb

def calc_delta_E(clr1, clr2):
    '''
    Helper function, given 2 colors calculates CIELab Delta E diff
    '''
    return ski.color.deltaE_ciede2000(clr1, clr2)


def match_colors(clr_lst, cbox):
    """
    Get all possible matches of color and crayon color to find closest match using CIE2000 
    color distance calculation
    Inputs:
    - clr_lst (lst) - list of colors in image
    - cbox (int) - number of crayons in box (16, 24, 48, 64, 96, 120), default is 96
    Returns:
    - color_matches_rgb (dict), crayons_needed (np.array) - dictionary matching colors to crayon colors, 
    dictionary of crayon crayon colors neeed 
"""
    # load color list with standardized RGB values 
    box_df = load_crayola_colors(cbox) 
    box_srgb = [r for r in box_df['srgb']] # convert to list 

    # create df with combo of all colors in image and all crayon colors
    color_combos = pd.DataFrame(list(product(clr_lst,box_srgb)), columns=["img_clr_srgb", "avail_clr_srgb"])

    # convert colors to LAB space
    color_combos['img_clr_lab'] = color_combos.apply(lambda x: ski.color.rgb2lab(x['img_clr_srgb']), axis=1)
    color_combos['avail_clr_lab'] = color_combos.apply(lambda x: ski.color.rgb2lab(x['avail_clr_srgb']), axis=1)

    # calculate cie2000 for all available colors to fill in image
    color_combos['delta_e'] = color_combos.apply(lambda x: calc_delta_E(x['img_clr_lab'], x['avail_clr_lab']), axis=1)
    color_combos['img_clr_id'] = color_combos.apply(lambda x: str(x['img_clr_lab']), axis=1) # necessary, can't hash a list

    # get smallest delta E value for each color in image
    color_combos = color_combos.iloc[color_combos.groupby('img_clr_id').delta_e.idxmin()].reset_index(drop=True)

    crayons_needed = box_df[box_df['srgb'].isin(color_combos['avail_clr_srgb'])]
    crayons_needed = crayons_needed[['name']].reset_index(drop=True)
    crayons_needed['id'] = np.arange(crayons_needed.shape[0])

    return color_combos, crayons_needed


# replace image colors with closest crayon colors 
def replace_img_colors(img_seg, color_crayon_match):
    """
    Given segmented image and list of dicts matching colors in image to new color, replace colors in 
    image with new colors. 
    Inputs: 
    - img_seg (segmented image)
    - color_crayon_match: dictionary matching colors to crayon colors
    Returns:
    - new_img (segmented image): image with colors replaced
    """
    new_img = img_seg.copy()
    # replace the colors! 
    # mask idea https://stackoverflow.com/questions/61808326/how-to-replace-all-rgb-values-in-an-numpy-image-arrray-based-on-an-target-pixel
    for nc in color_crayon_match.items():   
        old_color = nc[0]
        new_color = nc[1]
        mask = np.all(new_img == old_color, axis=2)
        new_img[mask, :] = new_color

    return new_img


# helper function, identifies centroid coordinates of each region
def identify_region_centers(img_labels):
    """
    Given image split into regions, determine where center of each region would be.
    Inputs:
    - img_labels (labelled image)
    Returns: 
    - region-centroids (pd.DataFrame): dataframe with label and coordinates of centroid 
    """
    region_centroids = pd.DataFrame(ski.measure.regionprops_table(img_labels,properties=['centroid']))
    # round down centroids
    region_centroids['x'] = region_centroids['centroid-0'].apply(math.floor)
    region_centroids['y'] = region_centroids['centroid-1'].apply(math.floor)

    return region_centroids


def label_regions(img_crayon, img_labels, crayon_info, outpath):
    """
    Given a segmented image, write numbers on each region corresponding to its crayon color.
    Inputs:
    - img_crayon: image with crayon colors assinged to segments
    - img_labels: image labels
    - crayon_info: crayon colors in the image (names of crayons) 
    - outpath: str, desired outpath for temp image
    Returns:
    - nothing, writes modified image to temporary file in outpath. 
    """
    only_labels = ski.segmentation.find_boundaries(img_labels,connectivity = 1)
    only_labels = only_labels*-1

    # get centroids of regions
    region_centers = identify_region_centers(img_labels)

    temp_img = only_labels.copy() 
    crayon_key = {}
    # Display the image on the axes
    plt.figure(figsize=(8, 6), dpi=200)
    num = region_centers.shape[0]
    for i in range(0,num):
        tx = region_centers['x'][i]
        ty = region_centers['y'][i]
        tcolor = img_crayon[tx, ty]
        tlabel = crayon_info.loc[(crayon_info['r'] == tcolor[0]) & (crayon_info['g'] == tcolor[1]) & (crayon_info['b'] == tcolor[2])].color_id.item()
        plt.text(ty, tx, tlabel, color='black', va='center', ha='center', size=7)

    plt.imshow(temp_img,cmap='gray')
    plt.axis('off')
    plt.savefig(f'{outpath}/temp.png', dpi=200) # save as temp image to set DPI and labels 
    plt.close()


def save_pbn_image(outpath, name, final_img, key, nbox):
    """
    Given a segmented image, write numbers on each region corresponding to its crayon color.
    Inputs:
    - outpath: str, desired outpath for image
    - name: str, name of image
    - final_img: image labeled with crayon colors
    - key:  dataframe matching label numbers to crayon names
    - nbox: number of crayons in box (user-supplied)
    Returns:
    - nothing, writes modified image to file in outpath. 
    """
    final_img = iio.imread(f'{outpath}/temp.png')
    key = key[['color_id','name']]

    # plot image
    plt.figure(figsize=(8,11), dpi=200) # to fit on a 8.5x11 piece of paper
    plt.imshow(final_img,cmap='gray')
    plt.title(name, fontsize=14, fontweight="bold")
    plt.axis('off')

    # plot table with key
    key_table = plt.table(cellText=key.values,
                rowLabels = None, 
                colLabels=["Color Number", f"Crayon Name ({nbox}-pack)"],
                loc='bottom', fontsize=12,
                cellLoc='center')

    key_table.scale(xscale=1, yscale=2)

    plt.tight_layout()
    plt.savefig(f'{outpath}/{name}.jpeg', bbox_inches='tight')
    plt.close() 


def create_pbn(img_path, name, outpath, crayon_box = 96, no_crayon = False):
    """
    Given the filepath to an image, and the number of crayons in crayon box, converts image to a paint-by-numbers
    style crayon drawing.
    Inputs:
    - img_path: str, filepath to image
    - name: str, name of image
    - outpath: str, desired outpath for image
    - crayon_box: int, number of crayons in crayola box
    - no_crayon: bool, whether or not to save a version of segmeneted image before crayon replacement colors
    Returns:
    - nothing, saves final PBN project and result image for reference.
    """
    # read in image
    img = iio.imread(img_path)

    # given image, segment using SLIC and RAG
    print("segmenting image...\n")
    img_seg, img_labels = segment_image(img)
    # save version of image with no crayon color correction 

    if no_crayon: 
        plt.axis('off')
        plt.imshow(img_seg) # show potential results, save for ref
        plt.savefig(f'{outpath}/{name}_pre_crayon.jpeg', bbox_inches='tight')
        plt.close() 

    # color identification (in segmented image), match to closest crayon color
    print("matching image colors...\n")
    img_colors = get_uniq_colors(img_seg) 
    img_match_colors, crayons = match_colors(img_colors, cbox = crayon_box)

    # replace colors in image with closest crayon color
    print("replacing image colors with matches...\n")
    img_crayon = replace_img_colors(img_seg, img_match_colors)
    plt.axis('off')
    plt.imshow(img_crayon) # show potential results, save for ref
    plt.savefig(f'{outpath}/{name}_result.jpeg', bbox_inches='tight')
    plt.close() 

    # label regions in image with new color label
    print("creating labeled version...\n")
    img_pbn = label_regions(img_crayon, img_labels, crayons, outpath)

    # save image with key to perform
    print("saving output...\n")
    save_pbn_image(outpath, name, img_pbn, crayons, crayon_box)
