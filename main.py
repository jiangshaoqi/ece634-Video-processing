import cv2
import sys
import numpy as np
import math
import time


def blockMatching(block, area, block_to_row_start, block_to_col_start):
    diff_matrix = np.zeros((area.shape[0] - block.shape[0], area.shape[1] - block.shape[1]))
    for row in range(area.shape[0] - block.shape[0]):
        for col in range(area.shape[1] - block.shape[1]):
            temp_block = area[row:(row+block.shape[0]), col:(col+block.shape[0])]
            temp_diff = sum(sum(abs(temp_block.astype(int) - block.astype(int))))
            diff_matrix[row][col] = temp_diff
    best_match = np.unravel_index(diff_matrix.argmin(), diff_matrix.shape)
    return best_match[0]+block_to_row_start, best_match[1]+block_to_col_start


def getVideoInfo(video_name):
    content, frame, fps = video_name.split('_')
    width, height = frame.split('x')
    return int(width), int(height), int(fps)


def analyzeVideo(video_file, width, height, block_size, search_range):
    file_p = open(video_file, "rb")
    frame_size = width * height * 2 // 3
    small_width = width // 2
    small_height = height // 2
    file_p.seek(0, 2)
    end_p = file_p.tell()
    num_frame = end_p // frame_size
    file_p.seek(0, 0)
    for i in range(num_frame):
        print(i)
        s_time = time.time()
        # current image display
        y_1 = np.zeros(shape=(height, width), dtype=np.uint8, order="c")
        u_1 = np.zeros(shape=(small_height, small_width), dtype=np.uint8, order="c")
        v_1 = np.zeros(shape=(small_height, small_width), dtype=np.uint8, order="c")
        for row in range(height):
            for col in range(width):
                y_1[row][col] = ord(file_p.read(1))
        for row in range(small_height):
            for col in range(small_width):
                u_1[row][col] = ord(file_p.read(1))
        for row in range(small_height):
            for col in range(small_width):
                v_1[row][col] = ord(file_p.read(1))
        u_1 = cv2.resize(u_1, (u_1.shape[1]*2, u_1.shape[0]*2), interpolation=cv2.INTER_NEAREST)
        v_1 = cv2.resize(v_1, (v_1.shape[1]*2, v_1.shape[0]*2), interpolation=cv2.INTER_NEAREST)
        yuv_1 = cv2.merge([y_1, u_1, v_1])
        bgr_1 = cv2.cvtColor(yuv_1, cv2.COLOR_YUV2BGR)
        # next image display
        y_2 = np.zeros(shape=(height, width), dtype=np.uint8, order="c")
        u_2 = np.zeros(shape=(small_height, small_width), dtype=np.uint8, order="c")
        v_2 = np.zeros(shape=(small_height, small_width), dtype=np.uint8, order="c")
        jump_frame = 2
        jump_byte_num = jump_frame * (height * width + 2 * small_height * small_width)
        for temp in range(jump_byte_num):
            file_p.read(1)
        for row in range(height):
            for col in range(width):
                y_2[row][col] = ord(file_p.read(1))
        for row in range(small_height):
            for col in range(small_width):
                u_2[row][col] = ord(file_p.read(1))
        for row in range(small_height):
            for col in range(small_width):
                v_2[row][col] = ord(file_p.read(1))
        u_2 = cv2.resize(u_2, (u_2.shape[1]*2, u_2.shape[0]*2), interpolation=cv2.INTER_NEAREST)
        v_2 = cv2.resize(v_2, (v_2.shape[1]*2, v_2.shape[0]*2), interpolation=cv2.INTER_NEAREST)
        yuv_2 = cv2.merge([y_2, u_2, v_2])
        bgr_2 = cv2.cvtColor(yuv_2, cv2.COLOR_YUV2BGR)
        # calculate motion field, block_size is length, search range is the difference to the block
        row_block_num = height // block_size
        col_block_num = width // block_size
        row_motion_mat = np.zeros((row_block_num, col_block_num))
        col_motion_mat = np.zeros((row_block_num, col_block_num))
        for row in range(row_block_num):
            for col in range(col_block_num):
                block_row_start = row * block_size
                block_row_end = (row + 1) * block_size - 1
                block_col_start = col * block_size
                block_col_end = (col + 1) * block_size - 1
                search_row_start = (block_row_start - search_range) if (block_row_start > search_range) else 0
                search_row_end = (block_row_end + search_range) if (block_row_end + search_range < height) else (height - 1)
                search_col_start = (block_col_start - search_range) if (block_col_start > search_range) else 0
                search_col_end = (block_col_end + search_range) if (block_col_end + search_range < width) else (width - 1)
                temp_block = y_1[block_row_start:block_row_end+1, block_col_start:block_col_end+1]
                temp_area = y_2[search_row_start:search_row_end+1, search_col_start:search_col_end+1]
                temp_to_row_start = search_row_start - block_row_start
                temp_to_col_start = search_col_start - block_col_start
                row_move, col_move = blockMatching(temp_block, temp_area, temp_to_row_start, temp_to_col_start)
                row_motion_mat[row][col] = row_move
                col_motion_mat[row][col] = col_move
        field_img = np.zeros([height, width, 3], dtype=np.uint8)
        field_img.fill(255)
        for row in range(row_block_num):
            for col in range(col_block_num):
                arrow_start_r = row * block_size
                arrow_start_c = col * block_size
                arrow_start = (arrow_start_c, arrow_start_r)
                arrow_end_r = arrow_start_r + row_motion_mat[row][col]
                arrow_end_c = arrow_start_c + col_motion_mat[row][col]
                arrow_end = (int(arrow_end_c), int(arrow_end_r))
                field_img = cv2.arrowedLine(field_img, arrow_start, arrow_end, (0, 0, 255), 1)
        # rebuild the image
        re_y_1 = np.copy(y_1)
        re_u_1 = np.copy(u_1)
        re_v_1 = np.copy(v_1)
        for row in range(row_block_num):
            for col in range(col_block_num):
                current_y = y_1[row * block_size:(row + 1) * block_size, col * block_size:(col + 1) * block_size]
                current_u = u_1[row * block_size:(row + 1) * block_size, col * block_size:(col + 1) * block_size]
                current_v = v_1[row * block_size:(row + 1) * block_size, col * block_size:(col + 1) * block_size]
                des_block_row_start = row * block_size + int(row_motion_mat[row][col])
                des_block_col_start = col * block_size + int(col_motion_mat[row][col])
                des_block_row_end = des_block_row_start + block_size
                des_block_col_end = des_block_col_start + block_size
                re_y_1[des_block_row_start:des_block_row_end, des_block_col_start:des_block_col_end] = current_y
                re_u_1[des_block_row_start:des_block_row_end, des_block_col_start:des_block_col_end] = current_u
                re_v_1[des_block_row_start:des_block_row_end, des_block_col_start:des_block_col_end] = current_v
        re_yuv = cv2.merge([re_y_1, re_u_1, re_v_1])
        re_bgr = cv2.cvtColor(re_yuv, cv2.COLOR_YUV2BGR)
        # error image
        error_y = abs(np.array(re_y_1, dtype=float) - np.array(y_2, dtype=float))
        error_y = np.array(error_y, dtype=np.uint8)
        mse = (error_y ** 2).mean()
        psnr = 10 * math.log10((255 ** 2) / mse)
        print("psnr: " + str(psnr))
        # stop time
        e_time = time.time()
        print("time spend: " + str(e_time - s_time))
        # plot
        img_hori_1 = np.concatenate((re_bgr, bgr_2), axis=1)
        img_hori_2 = np.concatenate((field_img, bgr_1), axis=1)
        img_verti = np.concatenate((img_hori_1, img_hori_2), axis=0)
        cv2.imshow("error", error_y)
        cv2.imshow("window", img_verti)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    file_p.close()
    return


if __name__ == '__main__':
    # this program works for Part 1: integer-pel accuracy
    video_file = sys.argv[1]
    block_size = int(sys.argv[2])
    search_range = int(sys.argv[3])
    video_name = video_file.split(".", 1)[0]
    width, height, fps = getVideoInfo(video_name)
    analyzeVideo(video_file, width, height, block_size, search_range)
