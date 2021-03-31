import dlib
import numpy
import cv2
import math
import sys
from cv2 import cv2
import numpy as np


class DlibOpencvEular():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "./model/shape_predictor_68_face_landmarks.dat")
        self.POINTS_NUM_LANDMARK = 68

    # 获取最大的人脸
    def _largest_face(self, dets):
        if len(dets) == 1:
            return 0

        face_areas = [(det.right() - det.left()) * (det.bottom() - det.top()) for det in dets]

        largest_area = face_areas[0]
        largest_index = 0
        for index in range(1, len(dets)):
            if face_areas[index] > largest_area:
                largest_index = index
                largest_area = face_areas[index]

        # print("largest_face index is {} in {} faces".format(largest_index, len(dets)))

        return largest_index

    # 从dlib的检测结果抽取姿态估计需要的点坐标
    def get_image_points_from_landmark_shape(self, landmark_shape):
        if landmark_shape.num_parts != self.POINTS_NUM_LANDMARK:
            print("ERROR:landmark_shape.num_parts-{}".format(landmark_shape.num_parts))
            return False, None

        # 2D image points. If you change the image, you need to change vector
        image_points = numpy.array([
            (landmark_shape.part(30).x, landmark_shape.part(30).y),  # Nose tip
            (landmark_shape.part(8).x, landmark_shape.part(8).y),  # Chin
            (landmark_shape.part(36).x, landmark_shape.part(36).y),  # Left eye left corner
            (landmark_shape.part(45).x, landmark_shape.part(45).y),  # Right eye right corne
            (landmark_shape.part(48).x, landmark_shape.part(48).y),  # Left Mouth corner
            (landmark_shape.part(54).x, landmark_shape.part(54).y)  # Right mouth corner
        ], dtype="double")

        return True, image_points

    # 用dlib检测关键点，返回姿态估计需要的几个点坐标
    def get_image_points(self, img):
        # gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )  # 图片调整为灰色
        dets = self.detector(img, 0)

        if 0 == len(dets):
            # print("ERROR: found no face")
            return False, None
        largest_index = self._largest_face(dets)
        face_rectangle = dets[largest_index]
        landmark_shape = self.predictor(img, face_rectangle)

        return self.get_image_points_from_landmark_shape(landmark_shape)

    # 获取旋转向量和平移向量
    def get_pose_estimation(self, img_size, image_points):
        # 3D model points.======== dlib
        model_points = numpy.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner

        ])
        # print(f"model_points.shape = {model_points.shape}")
        # Camera internals

        focal_length = img_size[1]
        center = (img_size[1] / 2, img_size[0] / 2)
        camera_matrix = numpy.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        # print("Camera Matrix :{}".format(camera_matrix))

        dist_coeffs = numpy.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # print("Rotation Vector:\n {}".format(rotation_vector))
        # print("Translation Vector:\n {}".format(translation_vector))
        return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs

    # 从旋转向量转换为欧拉角
    def get_euler_angle(self, rotation_vector):
        # calculate rotation angles
        theta = cv2.norm(rotation_vector, cv2.NORM_L2)

        # transformed to quaterniond
        w = math.cos(theta / 2)
        x = math.sin(theta / 2) * rotation_vector[0][0] / theta
        y = math.sin(theta / 2) * rotation_vector[1][0] / theta
        z = math.sin(theta / 2) * rotation_vector[2][0] / theta

        ysqr = y * y
        # pitch (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        # print('t0:{}, t1:{}'.format(t0, t1))
        pitch = math.atan2(t0, t1)

        # yaw (y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        if t2 > 1.0:
            t2 = 1.0
        if t2 < -1.0:
            t2 = -1.0
        yaw = math.asin(t2)

        # roll (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        roll = math.atan2(t3, t4)

        # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

        # 单位转换：将弧度转换为度
        Y = int((pitch / math.pi) * 180)
        X = int((yaw / math.pi) * 180)
        Z = int((roll / math.pi) * 180)

        return Y, X, Z

    def predict(self, img_input):
        ''':type
        输入一张图像，输出人脸欧拉角
        '''
        size = img_input.shape
        ok_face_kp, image_points = self.get_image_points(img_input)
        if not ok_face_kp:
            return None
        ok_3d_face, rotation_vector, translation_vector, camera_matrix, dist_coeffs = self.get_pose_estimation(size,
                                                                                                               image_points)
        if not ok_3d_face:
            return None
        pitch, yaw, roll = self.get_euler_angle(rotation_vector)
        euler = [pitch, yaw, roll]
        return euler

def contrast(img0):
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)  # 彩色转为灰度图片
    m, n = img1.shape
    # 图片矩阵向外扩展一个像素
    img1_ext = cv2.copyMakeBorder(img1, 1, 1, 1, 1, cv2.BORDER_REPLICATE) / 1.0  # 除以1.0的目的是uint8转为float型，便于后续计算
    rows_ext, cols_ext = img1_ext.shape
    b = 0.0
    for i in range(1, rows_ext - 1):
        for j in range(1, cols_ext - 1):
            b += ((img1_ext[i, j] - img1_ext[i, j + 1]) ** 2 + (img1_ext[i, j] - img1_ext[i, j - 1]) ** 2 +
                  (img1_ext[i, j] - img1_ext[i + 1, j]) ** 2 + (img1_ext[i, j] - img1_ext[i - 1, j]) ** 2)

    cg = b / (4 * (m - 2) * (n - 2) + 3 * (2 * (m - 2) + 2 * (n - 2)) + 2 * 4)  # 对应上面48的计算公式
    print("图片对比度为：",cg)

if __name__ == '__main__':
    fileaddress = sys.argv[1]
    # f = open("output.txt", "w")
    model = DlibOpencvEular()
    # path_image = "./zhengdui.png"

    # while True:
    image = cv2.imread(fileaddress)
    result = model.predict(image)
    print(f"人脸部分姿态评估result (正对欧拉角为[180 , 0 , 0]): {result}")




    rgb_image = cv2.imread(fileaddress)
    # rgb_image = cv2.imread("./images/d120.jpeg")
    image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    print("图片亮度值为（越高越亮）：", image.mean())

    # # 评估图像是否存在过曝光或曝光不足
    # img = cv2.imread(fileaddress)
    # # 把图片转换为单通道的灰度图
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # 获取形状以及长宽
    # img_shape = gray_img.shape
    # height, width = img_shape[0], img_shape[1]
    # size = gray_img.size
    # # 灰度图的直方图
    # hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    # # 计算灰度图像素点偏离均值(128)程序
    # a = 0
    # ma = 0
    # # np.full 构造一个数组，用指定值填充其元素
    # reduce_matrix = np.full((height, width), 128)
    # shift_value = gray_img - reduce_matrix
    # shift_sum = np.sum(shift_value)
    # da = shift_sum / size
    # # 计算偏离128的平均偏差
    # for i in range(256):
    #     ma += (abs(i - 128 - da) * hist[i])
    # m = abs(ma / size)
    # # 亮度系数
    # k = abs(da) / m
    # print("亮度系数",k)
    # if k[0] > 1:
    #     # 过亮
    #     if da > 0:
    #         print("过亮")
    #     else:
    #         print("过暗")
    # else:
    #     print("亮度正常")

    # 评估图片对比度
    img0 = cv2.imread(fileaddress)
    contrast(img0)

    # 评估图片模糊程度
    image = cv2.imread(fileaddress)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print('图片清晰程度(越高越清晰）:', cv2.Laplacian(gray, cv2.CV_64F).var())

