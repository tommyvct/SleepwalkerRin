import cv2 as cv
import numpy as np
from sklearn import cluster
import time

OPENCL: bool = True
SCALE: float = 0.65




def img_read(filename, flags=cv.IMREAD_COLOR):
    """
    Wrapper of `cv.imread()`

    :param filename: Name of file to be loaded.
    :param flags: Flag that can take values of cv::ImreadModes
    :return:
    """

    if OPENCL:
        img = cv.imread(filename, flags)
        ret = cv.UMat(img)
        # ret = cv.UMat(ret)
    else:
        img = cv.imread(filename, flags)
        ret = img

    if SCALE == 1.0:
        return ret, img.shape[1], img.shape[0]
    else:
        return cv.resize(ret, dsize=(0, 0), fx=SCALE, fy=SCALE), img.shape[1], img.shape[0]


def video_readframe(video_source: cv.VideoCapture):
    """
    Wrapper of cv2.VideoCapture.read()

    :param video_source: cv2.VideoCapture object
    :return: frame
    """
    ret, frame = video_source.read()
    if not ret:
        print("Can't read frame. Panik.")
        return None
    if OPENCL:
        frame = cv.UMat(frame)

    if SCALE == 1.0:
        return frame
    else:
        return cv.resize(frame, dsize=(0, 0), fx=SCALE, fy=SCALE)


def benchmark(func_benchmark, laps=0):
    """
    :param func_benchmark: function to benchmark
    :param laps: laps to run, 0 means forever
    """

    average_time: float = 0.0
    laps: int = 0
    total_time = 0.0
    lap_count = 0

    while True:
        start_time = time.perf_counter()

        func_benchmark()

        lap_time = time.perf_counter() - start_time
        total_time += lap_time
        lap_count += 1
        print(f"Frame: {lap_count}, frame time: {lap_time}, FPS: {(1 / lap_time) if lap_time > 0 else 999}, "
              f"average frame time: {total_time / lap_count}, average FPS: {lap_count / total_time}")

        if laps > 0 and lap_count > laps:
            return


def frame_template_match(frame, templates: list) -> list:
    """
    Match multiple template for a given frame.

    :param frame: The frame to match with format of cv2.UMat or NumPy array
    :param templates: list of templates to match. Each template have a format of
                        (template_image, template_width, template_height)
    :return: list of matching result coordinates. Each coordinate hava a format of
                        (coordinate, template_width, template_height)
    """

    result = []
    for template in templates:
        match_result = cv.matchTemplate(frame, template[0], cv.TM_CCOEFF_NORMED)

        if OPENCL:
            match_result = match_result.get()

        threshold = 0.2
        loc = np.where(match_result >= threshold)
        loc = np.dstack((loc[1], loc[0]))
        loc = loc.squeeze()

        try:
            labels = cluster.DBSCAN(eps=5, min_samples=1).fit(loc).labels_
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            for i in range(0, num_clusters):
                try:
                    t = loc[labels.tolist().index(i)]
                    # print(t)
                    result.append((t, template[1], template[2]))
                except ValueError:
                    pass
        except ValueError:
            # print("Not Found")
            pass
    return result


if __name__ == '__main__':
    # Make template list
    template, template_width, template_height = img_read("target.png")
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    templates: list = []
    templates.append((template, template_width, template_height))

    # Read video
    video = cv.VideoCapture("C:\\Users\\ws103\\Desktop\\TommyDash\\source\\2021.04.25-21.25.mp4")
    while video.isOpened():
        # for each frame, do Canny processing first
        frame_rgb = video_readframe(video)
        frame = cv.Canny(frame_rgb, 300, 550)

        # do template matching
        results = frame_template_match(frame, templates)

        # plot result
        for pt in results:
            frame_rgb = cv.rectangle(frame_rgb,
                         pt, (pt[0][0] + pt[1] * SCALE, pt[0][1] + pt[2] * SCALE),
                         (0, 0, 255), 2)
            print(pt[0])
        cv.imshow("sleepwalker", frame_rgb)



    # benchmark(testament, 600)

    # cv.imshow("base", base_rgb)
    # cv.waitKey(0)
    video.release()
    cv.destroyAllWindows()
