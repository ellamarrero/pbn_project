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
    box ['srgb'] = list(zip(box['r']/255, box['g']/255, box['b']/255)) # create one rgb value for conversion
    rgb = box['srgb'] # create np array of colors
    lab = [ski.color.rgb2lab(r) for r in rgb]

    return box, lab # return crayola colors in lab color space


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


def count_uniq_colors(img_seg): 
    """
    Given a segmented image (i.e. simplified), identify all unique colors in image and
    covnert to cielab color space
    Inputs:
    - img (segmented image)
    Returns:
    - color_lst (list of unique cielab colors)
    """
    colors_only = np.vstack(img_seg)
    colors_only_tup = list(map(tuple, colors_only)) 
    uniq_clrs = {x: colors_only_tup.count(x) for x in set(colors_only_tup)}

    # convert to lab space
    uniq_colors_lab = {c: list(ski.color.rgb2lab(c, channel_axis = -1)) for c in uniq_clrs}
    color_lst = list(uniq_colors_lab.values()) # get just lab colors

    return color_lst


def get_crayon_names(crayon_matches, cbox):
    """
    Helper function ,gets relevant crayon names from box df 
    (wouldn't be necessary in a better version of this code)
    Inputs:
    - crayon_matches (dict) - dictionary of color in image and matched cielab color
    - cbox (int) - number of crayons in box (16, 24, 48, 64, 96, 120), default is 96
    Returns:
    - crayon_info (np.array) - array of colors in image and their corresponding names
    """
    # find RGB values of color sin image
    crayon_needs = [cbox.loc[(cbox['r'] == t[0][0]) & (cbox['g'] == t[0][1]) & (cbox['b'] == t[0][2])]['name'] for t in crayon_matches.values()]
    crayon_needs = np.unique(crayon_needs) # get unique RGB list as array
    crayon_info = cbox[cbox['name'].isin(crayon_needs)] # filter crayon info for relevant colors
    crayon_info['color_id'] = np.arange(crayon_info.shape[0]) # Create ID numbers of crayola crayon 

    return crayon_info


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
    box_df, lab = load_crayola_colors(cbox)
    # create match of all colors in image and all crayon colors in LAB space
    color_combos_lst = list(product(clr_lst,lab)) 
    # calculate cie2000 for colors to find closest crayon color match
    color_match_lst_dict = [{"c":c[0], "m":c[1], 'cie76':ski.color.deltaE_ciede2000(c[0],c[1])} for c in color_combos_lst] # note: list of dictionaries

    # for each color, identify color match that minimizes CIE2000, convert to RGB colorspace
    color_matches_rgb = {}
    for o in clr_lst:
        de_values = [c['cie76'] for c in color_match_lst_dict if c['c'] == o]
        min_diff = min(de_values)
        best = [c['m'] for c in color_match_lst_dict if c['cie76'] == min_diff]
        color_matches_rgb[tuple(ski.util.img_as_ubyte(ski.color.lab2rgb(o)))] = tuple(ski.util.img_as_ubyte(ski.color.lab2rgb(best)))

    crayons_needed = get_crayon_names(color_matches_rgb, box_df)

    return color_matches_rgb, crayons_needed


# replace image colors with closest crayon colors 
def replace_img_colors(img_seg, color_crayon_match):
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
    region_centroids = pd.DataFrame(ski.measure.regionprops_table(img_labels,properties=['centroid']))
    # round down centroids
    region_centroids['x'] = region_centroids['centroid-0'].apply(math.floor)
    region_centroids['y'] = region_centroids['centroid-1'].apply(math.floor)

    return region_centroids


def label_regions(img_crayon, img_labels, crayon_info, outpath):
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
    img_colors = count_uniq_colors(img_seg) 
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
