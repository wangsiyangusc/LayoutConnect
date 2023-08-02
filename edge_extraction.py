from PIL import Image
import cv2 as cv
import numpy as np
import os
import glob


def image_bin_conversion(image_dir):
    img_arr = cv.imread(image_dir, cv.IMREAD_GRAYSCALE)
    _, thresh_img = cv.threshold(img_arr, 127, 255, cv.THRESH_BINARY_INV)
    return thresh_img


def batch_image_to_grayscale(raw_data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_files = glob.glob(os.path.join(raw_data_dir, '*.png'))

    for image_file in image_files:
        # Load the image
        bin_img = image_bin_conversion(image_file)

        # Extract the filename without extension
        image_filename = os.path.splitext(os.path.basename(image_file))[0]

        # Split the image into individual clips
        clip_filename = f"bw_{image_filename}.png"

        # Save the clip as a PyTorch tensor file
        clip_path = os.path.join(output_dir, clip_filename)
        # torch.save(clip_tensor, clip_path)
        params = [cv.IMWRITE_PNG_COMPRESSION, 1, cv.IMWRITE_PNG_BILEVEL, 1]
        cv.imwrite(clip_path, bin_img, params)


def edge_pixel_extraction(image_dir, size, extraction_width):
    image = cv.imread(image_dir, cv.IMREAD_UNCHANGED)
    image = Image.open(image_dir)
    image = np.asarray(image)
    img_arr = image
    edge_arr = np.zeros([size * 4 - 4, extraction_width])
    for j in range(size - 1):
        edge_arr[j, :] = img_arr[0:extraction_width, j]
    for j in range(size - 1):
        edge_arr[j + size - 1, :] = np.flip(np.transpose(img_arr[j+1, 0:extraction_width]), 0)
    for j in range(size - 1):
        edge_arr[j + size * 2 - 2, :] = img_arr[size - extraction_width:size, j+1]
    for j in range(size - 1):
        edge_arr[j + size * 3 - 3, :] = np.flip(np.transpose(img_arr[j, (size - extraction_width):size]), 0)
    edge_arr_int = edge_arr.astype(np.int8)
    return edge_arr_int


def edge_pin_extraction(edge_arr):
    edge_tot_len = np.size(edge_arr, 0)
    extraction_width = np.size(edge_arr, 1)
    edge_side_len = int(edge_tot_len / 4)
    edge_metal_raw = np.zeros(edge_tot_len)  # stores metal existence at the edge and pin orientation
    edge_pin_info = np.zeros(edge_tot_len)  # first col pin num
    pin_num = 0
    for j in range(edge_tot_len):
        if sum(edge_arr[j, :]) > extraction_width - 1:
            # if all pixels are filled, out metal pin exists
            edge_metal_raw[j] = 1
    # operate on each side
    for side in range(4):
        for j in range(edge_side_len):
            idx = j + edge_side_len * side
            if edge_metal_raw[idx] == 1:
                edge_pin_info[idx] = pin_num
                if j < edge_side_len:
                    if edge_metal_raw[idx + 1] == 0:
                        pin_num += 1
            elif edge_metal_raw[idx] == 0:
                edge_pin_info[idx] = -1
    edge_pin_info = edge_pin_info.astype(np.int8)
    # now store pins by pin top location and corresponding width
    side_pins = [[], [], [], []]

    for side in range(4):
        pin_width_cnt = 0
        first_metal_pix_flag = 1
        # keeps track of total num of pins
        pin_num_cnt = 0
        for pin_loc in range(edge_side_len):
            # index for the overall arr
            idx = pin_loc + edge_side_len * side
            # pin nums
            curr_pin_num = edge_pin_info[idx]

            # end of arr exception
            if idx < edge_tot_len-1:
                next_pin_num = edge_pin_info[idx+1]
            else:
                next_pin_num = -1

            # if current location has a pin
            if curr_pin_num >= 0:
                # add pin width
                pin_width_cnt += 1
                # if starting new pin
                if first_metal_pix_flag:
                    # append new pin to list
                    side_pins[side].append([pin_loc, 0])
                    # disable new pin
                    first_metal_pix_flag = 0
                    # pin num count + 1
                    pin_num_cnt += 1
                # not starting new pin
                else:
                    # if pin ends
                    if next_pin_num < 0:
                        # enable new pin
                        first_metal_pix_flag = 1
                        # set pin width for this pin
                        side_pins[side][pin_num_cnt-1][1] = pin_width_cnt
                        # reset pin width for next pin
                        pin_width_cnt = 0
    return side_pins


if __name__ == '__main__':
    sp_edge = edge_pixel_extraction("C:\\Users\\Siyang_Wang_work\\USC\\SPORT_LAB\\DRC_ML\\DataFiles\\all_data\\train_dataset_processed\\images1024_bin\\bw_patid_MX_Benchmark2_clip_hotspot1_7_orig_0.png", 1024, 4)
    sp_edge = np.array(sp_edge)
    for i in range(1024):
        print(type(sp_edge[i, 0]))
        print(sp_edge[i, :])

    sp_pins = edge_pin_extraction(sp_edge)
    for edge_side in range(4):
        print(sp_pins[edge_side])

    """
    batch_image_to_grayscale("C:\\Users\\Siyang_Wang_work\\USC\\SPORT_LAB\\DRC_ML\\DataFiles\\all_data\\train_dataset",
                       "C:\\Users\\Siyang_Wang_work\\USC\\SPORT_LAB\\DRC_ML\\DataFiles\\all_data\\train_dataset_processed\\images1024_bin")
    """
