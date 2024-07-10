import sys
import cv2
import os
import numpy as np
import onnxruntime as ort
import time

# 差异算法函数
def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

def dHash(img):
    hash_str = ''
    img = cv2.resize(img, (9, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# 提取视频差异帧序号列表
def extract_diff_frame_indices(file_path, img_dif, img_fream):
    cap = cv2.VideoCapture(file_path)
    dict_img = {"1": []}
    diff_frame_indices = []

    if cap.isOpened():
        rate = int(cap.get(5))  # 帧速率
        FrameNumber = int(cap.get(7))  # 视频文件的帧数

        # 生成间隔 img_fream 帧的索引序号列表
        frame_indices = [i for i in range(0, FrameNumber, int(img_fream))]

        for c in frame_indices:
            sm_img = 0

            # 设置视频的当前帧位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, c)
            success, frame = cap.read()

            if not success:
                break

            if c % rate == 0:
                if "2" in dict_img:
                    dict_img["1"] = dict_img["2"]
                else:
                    dict_img["1"] = dHash(frame)

                dict_img["2"] = dHash(frame)
                sm_img = cmpHash(dict_img["1"], dict_img["2"])

            if sm_img >= img_dif:
                diff_frame_indices.append(c)

        cap.release()
        return diff_frame_indices
    else:
        return []

def plot_one_box(x, img, color=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or (0, 0, 255)  # 默认红色
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

def _make_grid(nx, ny):
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

def cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride):
    row_ind = 0
    grid = [np.zeros(1)] * nl
    for i in range(nl):
        h, w = int(model_w / stride[i]), int(model_h / stride[i])
        length = int(na * h * w)
        if grid[i].shape[2:4] != (h, w):
            grid[i] = _make_grid(w, h)

        outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
            grid[i], (na, 1))) * int(stride[i])
        outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
            anchor_grid[i], h * w, axis=0)
        row_ind += length
    return outs

def post_process_opencv(outputs, model_h, model_w, img_h, img_w, thred_nms, thred_cond):
    conf = outputs[:, 4].tolist()
    c_x = outputs[:, 0] / model_w * img_w
    c_y = outputs[:, 1] / model_h * img_h
    w = outputs[:, 2] / model_w * img_w
    h = outputs[:, 3] / model_h * img_h
    p_cls = outputs[:, 5:]
    if len(p_cls.shape) == 1:
        p_cls = np.expand_dims(p_cls, 1)
    cls_id = np.argmax(p_cls, axis=1)

    p_x1 = np.expand_dims(c_x - w / 2, -1)
    p_y1 = np.expand_dims(c_y - h / 2, -1)
    p_x2 = np.expand_dims(c_x + w / 2, -1)
    p_y2 = np.expand_dims(c_y + h / 2, -1)
    areas = np.concatenate((p_x1, p_y1, p_x2, p_y2), axis=-1)

    areas = areas.tolist()
    ids = cv2.dnn.NMSBoxes(areas, conf, thred_cond, thred_nms)
    if len(ids) > 0:
        ids = [i[0] for i in ids]  # Convert ndarray to list of integers
        return np.array(areas)[ids], np.array(conf)[ids], cls_id[ids]
    else:
        return [], [], []

def infer_img(img0, net, model_h, model_w, nl, na, stride, anchor_grid, thred_cond, thred_nms=0.4):
    img = cv2.resize(img0, (model_w, model_h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

    outs = cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride)

    img_h, img_w, _ = np.shape(img0)
    boxes, confs, ids = post_process_opencv(outs, model_h, model_w, img_h, img_w, thred_nms, thred_cond)

    return boxes, confs, ids

# 推理特定序号的帧并保存
def infer_and_save_frames(file_path, output_folder, frame_indices, net, model_h, model_w, nl, na, stride, anchor_grid, thred_cond):
    thred_cond = float(thred_cond)
    cap = cv2.VideoCapture(file_path)
    os.makedirs(output_folder, exist_ok=True)

    pred_box_counts = 0  # 初始化预测框数量
    saved_frames = 0  # 初始化保存帧数量

    if cap.isOpened():
        success, frame = cap.read()
        frame_index = 0
        frameindex = 0
        while success:
            if frame_index + 1 in frame_indices:
                det_boxes, scores, ids = infer_img(frame, net, model_h, model_w, nl, na, stride, anchor_grid, thred_cond)
                pred_box_counts += len(det_boxes)  # 累加预测框数量

                if len(det_boxes) > 0:
                    for box, score, id in zip(det_boxes, scores, ids):
                        plot_one_box(box.astype(np.int16), frame, color=None, line_thickness=None)

                    cv2.imwrite(f'{output_folder}/frame_{frameindex:05d}.jpg', frame)
                    frameindex += 1
                    saved_frames += 1

            success, frame = cap.read()
            frame_index += 1

        cap.release()

    return pred_box_counts, saved_frames



if len(sys.argv) < 8:
    print("Usage: python script.py 输入文件名 临时文件夹名 差距帧阈值 模型预测阈值 模型输入大小 模型名称 差距帧数")
    print("Usage: python script.py 2min.mp4 test 6 0.50 320 best-p.onnx 30")
    sys.exit(1)


if __name__ == '__main__':
    model_pb_path = sys.argv[6]
    so = ort.SessionOptions()
    net = ort.InferenceSession(model_pb_path, so)



    model_h = sys.argv[5]
    model_w = sys.argv[5]
    model_h = int(model_h)
    model_w = int(model_w)
    nl = 3
    na = 3
    stride = [8., 16., 32.]
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, -1, 2)

    file_path = sys.argv[1]
    output_folder = sys.argv[2]
    img_dif = sys.argv[3]
    img_dif = int(img_dif)
    img_fream = sys.argv[7]
    img_fream = int(img_fream)  # 将 img_fream 转换为整数
    thred_cond = sys.argv[4]

    # 提取视频差异帧序号列表
    start_time = time.time()
    diff_frame_indices = extract_diff_frame_indices(file_path, img_dif, img_fream)
    # 推理特定序号的帧并保存到临时文件夹
    total_pred_boxes, saved_frames = infer_and_save_frames(file_path, output_folder, diff_frame_indices, net, model_h, model_w, nl, na, stride, anchor_grid, thred_cond)
    output_txt_path = os.path.splitext(sys.argv[2])[0] + '.txt'
    with open(output_txt_path, 'w') as f:
        f.write(f'预测数: {total_pred_boxes}\n')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Use time: {elapsed_time:.2f}s")
