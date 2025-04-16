# import libraries
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
import imageio.v3 as iio 
import scipy
import math
import pandas as pd
from itertools import combinations, product, chain


def load_crayola_colors(cbox): 
    """
    Loads in CSV of crayola box colors/RGB values, filters for specific num crayons
    Note: source of CSV is wikipedia page for Crayola crayon colors 
    Inputs:
    - cbox (int): number of crayons in box (16, 24, 48, 64, 96, 120)
    Returns:
    - box (pd.DataFrame): df with crayon color names and RGB values
    """
    crayola_box = pd.read_csv('crayola_colors.csv') # NOTE: if project downloaded, will need to change this 
    box = crayola_box[['name','r','g','b']][crayola_box[f'box{cbox}'] =='Yes']  # filter to given box num
    box['rgb'] = list(zip(box['r'], box['g'], box['b'])) # create singular rgb value
    return box


def segment_image(img): 
    """
    Given an image, segments using simple linear iterative clustering (SLIC) and joins segments using region 
    adjacent graph thresholding. 
    Inputs:
    - img (image): image to be segmented 
    Returns:
    - tuple (segmented image, image segment labels)
    """
    # segment image using SLIC to get superpixels in image
    segments_slic = ski.segmentation.slic(img, n_segments=300, compactness=20, sigma=1, start_label=1) 
    img_slic = ski.color.label2rgb(segments_slic, img, kind='avg', bg_label=None) # specify background color or will average to 0

    # average colors within superpixels to get simplified image
    img_rag = ski.graph.rag_mean_color(img, segments_slic)
    reg_labels = ski.graph.cut_threshold(segments_slic, img_rag,  thresh = 20)
    img_seg = ski.color.label2rgb(reg_labels, img, kind='avg', bg_label=None) 

    return img_seg, reg_labels


def get_uniq_colors(img_seg): 
    """
    Identify all unique colors in an image (note: works best with processed images, e.g. superpixels
    to minimize number of colors in image)
    Inputs:
    - img_seg (image): note that in this code, image is likely segmented and averaged
    Returns:
    - uniq_clrs_df (pd.DataFrame): df where rows are RGB colors in image
    """
    # get long array of all colors present in image
    colors_only = np.vstack(img_seg) 
    colors_only_tup = list(map(tuple, colors_only)) # convert RGB values to tuples to make them unique
    # get unique colors in image via counting in dict
    uniq_clrs = {x: colors_only_tup.count(x) for x in set(colors_only_tup)}
    uniq_clrs_df = pd.DataFrame(columns = ["rgb", "srgb"]) # convert to df
    uniq_clrs_df['rgb'] = pd.Series([np.array(x) for x in uniq_clrs.keys()])

    return uniq_clrs_df


def calc_delta_E(clr1, clr2):
    '''
    Helper function, given 2 colors calculates CIELab Delta E
    Inputs:
    - clr1 (np.array or set/tuple): RGB color 1 value [R, G, B]
    - clr2 (np.array or set/tuple): RGB color 2 value [R, G, B]
    Returns:
    - (int) - Delta E value
    '''
    return ski.color.deltaE_ciede2000(clr1, clr2)


def match_colors(clr_lst, cbox):
    """
    Get all possible matches of color and crayon color to find closest match using CIE2000 
    color distance calculation
    Inputs:
    - clr_lst (lst): list of colors in image
    - cbox (int): number of crayons in box (16, 24, 48, 64, 96, 120)
    Returns:
    - color_crayons (pd.DataFrame) - df where each row is a color in simplified image, with crayon
    name, generated ID, RGB value, standardized RGB value, and CIELAB value. note: will be duplicate
    colors if differnet image colors are matched with same crayon color
"""
    # load color list with standardized RGB values 
    box_df = load_crayola_colors(cbox) 

    # create df with combo of all colors in image and all crayon colors
    color_combos = pd.DataFrame(product(list(clr_lst['rgb']),list(box_df['rgb'])), columns=["img_clr_rgb", "avail_clr_rgb"])

    # convert to standardized RGB, convert to CIELAB
    color_combos['img_clr_lab'] = color_combos.apply(lambda x: ski.color.rgb2lab(x['img_clr_rgb']/255), axis=1)
    color_combos['avail_clr_lab'] = color_combos.apply(lambda x: ski.color.rgb2lab(np.array(x['avail_clr_rgb'])/255), axis=1)

    # calculate cie2000 for all available colors for each color in image
    color_combos['delta_e'] = color_combos.apply(lambda x: calc_delta_E(x['img_clr_lab'], x['avail_clr_lab']), axis=1)
    color_combos['img_clr_id'] = color_combos.apply(lambda x: str(x['img_clr_lab']), axis=1) # necessary, can't hash a list

    # get smallest delta E value for each color in image
    color_combos = color_combos.iloc[color_combos.groupby('img_clr_id').delta_e.idxmin()].reset_index(drop=True)
    
    # join crayon name information to df
    color_crayons = color_combos.merge(box_df, left_on="avail_clr_rgb", right_on="rgb")
    color_crayons = color_crayons[['img_clr_rgb','avail_clr_rgb','name']]
    color_crayons = color_crayons.reset_index(drop=True)
    color_crayons['id'] = color_crayons.name.factorize()[0] # generate ID for labelling in image

    return color_crayons


def replace_img_colors(img_seg, color_crayon_match):
    """
    Given segmented image and df with old and new colors in image, replace colors in image. 
    Inputs: 
    - img_seg (segmented image): image segmented into superpixels
    - color_crayon_match (pd.DataFrame): df with image colors and corresponding crayon colors
    Returns:
    - new_img (segmented image): image with colors replaced
    """
    new_img = img_seg.copy()
    # replace the colors in the image! 
    # mask idea https://stackoverflow.com/questions/61808326/how-to-replace-all-rgb-values-in-an-numpy-image-arrray-based-on-an-target-pixel
    for i, row in color_crayon_match.iterrows():   
        old_color = row['img_clr_rgb']
        new_color = row['avail_clr_rgb']
        mask = np.all(new_img == old_color, axis=2)
        new_img[mask, :] = new_color

    return new_img


def identify_region_centers(img_labels):
    """
    Given image split into regions, calculate center of each region
    Inputs:
    - img_labels (labelled image)
    Returns: 
    - region_centroids (pd.DataFrame): dataframe with label value and coordinates of centroid 
    """
    region_centroids = pd.DataFrame(ski.measure.regionprops_table(img_labels,properties=['centroid']))
    # round down centroids
    region_centroids['x'] = region_centroids['centroid-0'].apply(math.floor)
    region_centroids['y'] = region_centroids['centroid-1'].apply(math.floor)

    return region_centroids


def label_regions(img_crayon, img_labels, crayon_df, outpath):
    """
    Given an image, write crayon IDs  on each image segment, save to outpath
    Inputs:
    - img_crayon (image): image with crayon colors replacing original colors
    - img_labels (labelled image): image labels
    - crayon_df (pd.DataFrame): df with available colors in the image & their names
    - outpath (str): desired outpath for temp image
    Returns:
    - nothing, writes modified image to temporary file in outpath. 
"""
    only_labels = ski.segmentation.find_boundaries(img_labels,connectivity = 1)
    only_labels = only_labels*-1 # invert to create outline of image instead of pixels themselves

    # get centroids of regions
    region_centers = identify_region_centers(img_labels)

    temp_img = only_labels.copy() 
    crayon_key = {}
    # Display the image on the axes
    plt.figure(figsize=(8, 6), dpi=200)
    num_regions = region_centers.shape[0]
    for i in range(0,num_regions):
        # for each region center, mark with crayon ID number
        tx = region_centers['x'][i]
        ty = region_centers['y'][i]
        tcolor = img_crayon[tx, ty]
        tlabel = crayon_df[crayon_df.apply(lambda x: sum(x['avail_clr_rgb']==tcolor), axis=1)>=3].id.unique().item()
        plt.text(ty, tx, tlabel, color='black', va='center', ha='center', size=7)

    plt.imshow(temp_img,cmap='gray')
    plt.axis('off')
    plt.savefig(f'{outpath}/temp.png', dpi=200) # save as temp image to set DPI and labels 
    plt.close()


def save_pbn_image(outpath, name, key, nbox):
    """
    Given outpath, load temp image created in label_regions, (blank regions labelled with crayon IDs),
    attach key, and save to outpath.
    Inputs:
    - outpath (str): desired outpath for image
    - name (str): name of image (used in image itself and final filepath)
    - key (pd.DataFrame): dataframe matching label numbers to crayon names
    - nbox (int): number of crayons in box (16, 24, 48, 64, 96, 120), used in key label
    Returns:
    - nothing, saves fill-in image with key in outpath. 
    """
    final_img = iio.imread(f'{outpath}/temp.png') # read in labelled image
    key = key[['id','name']].drop_duplicates('name')

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


def create_pbn(img_path, name, outpath, crayon_box = 96, pre_crayon_save = False):
    """
    Given the filepath to an image, and the number of crayons in crayon box, converts image to a paint-by-numbers
    style crayon drawing.
    Inputs:
    - img_path: str, filepath to image
    - name: str, name of image
    - outpath: str, desired outpath for image
    - crayon_box: int, number of crayons in crayola box (16, 24, 48, 64, 96, 120)
    - pre_crayon_save: bool, whether or not to save a version of image before crayon replacement colors
    Returns:
    - nothing, saves final PBN project and result image for reference at outpath
    """
    # read in image
    img = iio.imread(img_path)

    # given image, segment using SLIC and RAG
    print("segmenting image...\n")
    img_seg, img_labels = segment_image(img)
    # save version of image with no crayon color correction 

    if pre_crayon_save: 
        plt.axis('off')
        plt.imshow(img_seg) # show potential results, save for ref
        plt.savefig(f'{outpath}/{name}_pre_crayon.jpeg', bbox_inches='tight')
        plt.close() 

    # color identification (in segmented image), match to closest crayon color
    print("matching image colors...\n")
    img_colors = get_uniq_colors(img_seg) 
    img_crayon_colors_df = match_colors(img_colors, cbox = crayon_box)

    # replace colors in image with closest crayon color
    print("replacing image colors with matches...\n")
    img_crayon = replace_img_colors(img_seg, img_crayon_colors_df)
    plt.axis('off')
    plt.imshow(img_crayon) # show potential results, save for ref
    plt.savefig(f'{outpath}/{name}_result.jpeg', bbox_inches='tight')
    plt.close() 

    # label regions in image with new color label
    print("creating labeled version...\n")
    img_pbn = label_regions(img_crayon, img_labels, img_crayon_colors_df, outpath)

    # save image with key to perform
    print("saving output...\n")
    save_pbn_image(outpath, name, img_pbn, img_crayon_colors_df, crayon_box)
