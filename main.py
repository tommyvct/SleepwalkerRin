import cv2
import cv2 as cv
import numpy as np
from sklearn import cluster
import time
from threading import Thread
from queue import Queue

OPENCL: bool = True
SCALE: float = 0.5
CANNY_VIEW: bool = False
MULTI_THREAD: bool = True

BENCHMARK_sigma: float = 0.0
BENCHMARK_count: int = 0

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


def frame_single_template_match(frame, template: tuple) -> list:
    """
    Match given template for a given frame
    :param frame: The frame to match with format of cv2.UMat or NumPy array
    :param template: The template to match. Each template have a format of
            (template_image, template_width, template_height, threshold)
    :return: list of matching result coordinates. Each coordinate hava a format of
            (coordinate, template_width, template_height)
    """

    result = []
    match_result = cv.matchTemplate(frame, template[0], cv.TM_CCOEFF_NORMED)

    if OPENCL:
        match_result = match_result.get()

    threshold = template[3]
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


def frame_multi_template_match_MT(frame, templates: list) -> list:
    """
    Match multiple template for a given frame.

    :param frame: The frame to match with format of cv2.UMat or NumPy array
    :param templates: list of templates to match. Each template have a format of
            (template_image, template_width, template_height, threshold)
    :return: list of matching result coordinates. Each coordinate hava a format of
            (coordinate, template_width, template_height)
    """
    # global BENCHMARK_sigma
    # global BENCHMARK_count
    # start_time = time.perf_counter()


    queue = Queue()
    # thread_list = []

    for template in templates:
        t = Thread(target=(lambda q, f, t: q.put(frame_single_template_match(f, t)))(queue, frame, template))
        t.start()
        t.join()

    # for thread in thread_list:
    #     thread.join()

    result = []
    while not queue.empty():
        result += queue.get()

    # lap_time = time.perf_counter() - start_time
    # BENCHMARK_sigma += lap_time
    # BENCHMARK_count += 1
    # print(f"Frame time: {lap_time}, FPS: {(1 / lap_time) if lap_time > 0 else 999}, "
    #         f"average frame time: {BENCHMARK_sigma / BENCHMARK_count}, average FPS: {BENCHMARK_count / BENCHMARK_sigma}")

    return result


def frame_multi_template_match(frame, templates: list) -> list:
    """
    Match multiple template for a given frame.

    :param frame: The frame to match with format of cv2.UMat or NumPy array
    :param templates: list of templates to match. Each template have a format of
            (template_image, template_width, template_height, threshold)
    :return: list of matching result coordinates. Each coordinate hava a format of
            (coordinate, template_width, template_height)
    """

    # global BENCHMARK_sigma
    # global BENCHMARK_count
    # start_time = time.perf_counter()

    result = []
    for template in templates:
        match = frame_single_template_match(frame, template)
        result += match

    # lap_time = time.perf_counter() - start_time
    # BENCHMARK_sigma += lap_time
    # BENCHMARK_count += 1
    # print(f"Frame time: {lap_time}, FPS: {(1 / lap_time) if lap_time > 0 else 999}, "
    #         f"average frame time: {BENCHMARK_sigma / BENCHMARK_count}, average FPS: {BENCHMARK_count / BENCHMARK_sigma}")

    return result

def null(x): pass


if __name__ == '__main__':
    # Make template list
    templates: list = []

    template, template_width, template_height = img_read("targets/cone.png")
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    templates.append((template, template_width, template_height, 0.32))

    template, template_width, template_height = img_read("targets/cloud.png")
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    templates.append((template, template_width, template_height, 0.2))

    template, template_width, template_height = img_read("targets/eighth.png")
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    templates.append((template, template_width, template_height, 0.3))


    pause: bool = False
    save_index = 0

    frame_rgb = None
    frame = None
    results = None

    cv.namedWindow("sleepwalker")
    cv.createTrackbar("Threshold 1", "sleepwalker", 863, 1500, null)
    cv.createTrackbar("Threshold 2", "sleepwalker", 452, 1500, null)

    # Read video
    video = cv.VideoCapture("musedash.mp4")
    while video.isOpened():
        keycode = cv.waitKey(1)
        if keycode == ord(' '):
            print("[PAUSE]"if not pause else "[RESUME]")
            pause = not pause
        elif keycode == ord('s'):
            cv.imwrite(str(save_index) + ".png", frame_rgb)
            cv.imwrite(str(save_index) + "_Canny.png", frame)
            print(f"Image {save_index} saved")
            save_index += 1
            continue
        elif keycode == ord('c'):
            CANNY_VIEW = not CANNY_VIEW
            continue
        else:
            pass



        # for each frame, do Canny processing first
        if not pause:
            frame_rgb = video_readframe(video)
        frame = cv.Canny(frame_rgb,
                         cv2.getTrackbarPos("Threshold 1", "sleepwalker"),
                         cv2.getTrackbarPos("Threshold 2", "sleepwalker"))

        # do template matching
        results = frame_multi_template_match_MT(frame, templates) if MULTI_THREAD \
            else frame_multi_template_match(frame, templates)

        # plot result
        show = cv.cvtColor(frame, cv.COLOR_GRAY2BGR) if CANNY_VIEW else frame_rgb
        for pt in results:
            show = cv.rectangle(show,
                                pt[0], (int(pt[0][0] + pt[1] * SCALE), int(pt[0][1] + pt[2] * SCALE)),
                                (0, 0, 255), 2)
            print(pt[0])
        cv.imshow("sleepwalker", show)






    video.release()
    cv.destroyAllWindows()
    exit(0)



