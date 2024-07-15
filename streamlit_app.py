import cv2 as cv
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import pandas as pd
import copy
import imutils
import streamlit as st

def letter_division(inp_img):
    def read_images_in_grayscale(inp_img):
        img = cv.imread(inp_img)
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return gray_image

    image_loc = inp_img
    image = read_images_in_grayscale(image_loc)
    gray = copy.deepcopy(image)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blurred, 50, 200, 255)
    thresh_inv = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    ctrs = cv.findContours(thresh_inv.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    ctrs = imutils.grab_contours(ctrs)

    digitCnts = []
    for c in ctrs:
        cv.drawContours(thresh_inv, c, -1, (0, 255, 0), 1)

    from collections import Counter
    boundingBoxes = [list(cv.boundingRect(c)) for c in ctrs]
    c = Counter([i[2] for i in boundingBoxes])
    mode = c.most_common(1)[0][0]
    if mode > 1:
        diam = mode
    else:
        diam = c.most_common(2)[1][0]

    hist = cv.reduce(thresh_inv.T, 1, cv.REDUCE_AVG).reshape(-1)
    black_pixels = np.where(hist == 0)[0]
    diameter = diam

    segment_indices = [0]
    for i in range(1, len(black_pixels)):
        if black_pixels[i] - black_pixels[i - 1] > diameter - 3:
            segment_indices.append(i)
    segment_indices.append(len(black_pixels))

    space_start_positions = []
    space_end_positions = []
    for i in range(len(segment_indices) - 1):
        space_start_positions.append(black_pixels[segment_indices[i]])
        space_end_positions.append(black_pixels[segment_indices[i + 1] - 1])

    space_df = pd.DataFrame({
        'space_start': space_start_positions,
        'space_end': space_end_positions,
    })

    df = copy.deepcopy(space_df.loc[space_df['space_end'] - space_df['space_start'] > diam])
    df.reset_index(drop=True, inplace=True)
    df['mid_points'] = (df['space_end'] - df['space_start']) / 2 + df['space_start']
    df.reset_index(drop=True, inplace=True)
    df1 = pd.DataFrame()
    df1['mid_points'] = df['mid_points'][1:]
    df1.reset_index(drop=True, inplace=True)
    df_char = pd.DataFrame()
    df_char['char_start'] = df['mid_points'][:-1]
    df_char['char_end'] = df1['mid_points']

    cropped_letters = []
    for index, row in df_char.iterrows():
        start = int(row['char_start'])
        end = int(row['char_end'])
        cropped_letter = image[:, start:end]
        cropped_letters.append(cropped_letter)

    output_directory = 'cropped_img'
    os.makedirs(output_directory, exist_ok=True)
    for i, cropped_letter in enumerate(cropped_letters):
        image_path = os.path.join(output_directory, f"letter{i+1}.jpg")
        cv.imwrite(image_path, cropped_letter)

def letter_prediction(inp_img):
    def read_images_in_grayscale(inp_img):
        img = cv.imread(inp_img)
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return gray_image

    image_loc = inp_img
    image = read_images_in_grayscale(image_loc)
    image = cv.resize(image, (200, 200))
    gray = copy.deepcopy(image)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blurred, 50, 200, 255)
    thresh_inv = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    ctrs = cv.findContours(thresh_inv.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    ctrs = imutils.grab_contours(ctrs)

    digitCnts = []
    for c in ctrs:
        cv.drawContours(thresh_inv, c, -1, (0, 255, 0), 1)

    from collections import Counter
    boundingBoxes = [list(cv.boundingRect(c)) for c in ctrs]
    c = Counter([i[2] for i in boundingBoxes])
    mode = c.most_common(1)[0][0]
    if mode > 1:
        hor_diam = mode
    else:
        hor_diam = c.most_common(2)[1][0]

    boundingBoxes = [list(cv.boundingRect(c)) for c in ctrs]
    c = Counter([i[3] for i in boundingBoxes])
    mode = c.most_common(1)[0][0]
    if mode > 1:
        ver_diam = mode
    else:
        ver_diam = c.most_common(2)[1][0]

    horizontal_projection = np.sum(thresh_inv, axis=0)
    start_points_horizontal = np.where(horizontal_projection > 0)[0][0]
    vertical_projection = np.sum(thresh_inv, axis=1)
    start_points_vertical = np.where(vertical_projection > 0)[0][0]
    right = 3 * hor_diam
    bottom = 5 * ver_diam
    left = start_points_horizontal
    top = start_points_vertical
    right = left + right
    bottom = top + bottom
    cropped_image = thresh_inv[top:bottom, left:right]

    horizontal_divisions = 3
    vertical_divisions = 2
    height, width = cropped_image.shape[:2]
    cell_height = height // horizontal_divisions
    cell_width = width // vertical_divisions

    for i in range(1, horizontal_divisions):
        y = i * cell_height
        cv.line(cropped_image, (0, y), (width, y), (0, 255, 0), 1)

    for i in range(1, vertical_divisions):
        x = i * cell_width
        cv.line(cropped_image, (x, 0), (x, height), (0, 255, 0), 1)

    white_threshold = 40
    binary_representation = []
    for i in range(horizontal_divisions):
        for j in range(vertical_divisions):
            cell = cropped_image[i * cell_height: (i + 1) * cell_height, j * cell_width: (j + 1) * cell_width]
            avg_intensity = np.mean(cell)
            dots_present = 1 if avg_intensity > white_threshold else 0
            binary_representation.append(dots_present)

    binary_string = ''.join(map(str, binary_representation))
    binary_to_english = {
        "100000": "a", "101000": "b", "110000": "c", "110100": "d", "100100": "e", "111000": "f",
        "111100": "g", "101100": "h", "011000": "i", "011100": "j", "100010": "k", "101010": "l",
        "110010": "m", "110110": "n", "100110": "o", "111010": "p", "111110": "q", "101110": "r",
        "011010": "s", "011110": "t", "100011": "u", "101011": "v", "011101": "w", "110011": "x",
        "110111": "y", "100111": "z"
    }
    predicted_english = binary_to_english.get(binary_string, "?")
    return predicted_english

def word_prediction(directorypath):
    directory_path = directorypath
    predicted_word = ""

    def extract_number(filename):
        return int(''.join(filter(str.isdigit, filename)))

    sorted_files = sorted(os.listdir(directory_path), key=extract_number)

    for filename in sorted_files:
        file_path = os.path.join(directory_path, filename)
        predicted_char = letter_prediction(file_path)
        predicted_word += predicted_char

    return predicted_word

def main():
    st.title('Braille to English Translator')
    if not os.path.exists('uploaded_images'):
        os.makedirs('uploaded_images')

    uploaded_file = st.file_uploader("Upload a Braille Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        uploaded_image_path = os.path.join('uploaded_images', uploaded_file.name)
        image.save(uploaded_image_path)

        if st.button('Translate'):
            letter_division(uploaded_image_path)
            predicted_word = word_prediction('cropped_img')
            st.success('Translation:')
            st.write(predicted_word)

if __name__ == '__main__':
    main()
