import cv2
import cv2 as cv
import numpy as np
from sklearn import cluster
import time

OPENCL: bool = True
SCALE: float = 0.65

average_time: float = 0.0
laps: int = 0

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
        ret = cv.UMat(ret)
    else:
        img = cv.imread(filename, flags)
        ret = img

    if SCALE == 1.0:
        return ret, img.shape
    else:
        return cv2.resize(ret, dsize=(0, 0), fx=SCALE, fy=SCALE), img.shape

def benchmark(func_benchmark, laps=0):
    """
    :param func_benchmark: function to benchmark
    :param laps: laps to run, 0 means forever
    """

    total_time = 0.0
    lap_count = 0

    while True:
        start_time = time.perf_counter()

        func_benchmark()

        lap_time = time.perf_counter() - start_time
        total_time += lap_time
        lap_count += 1
        print(f"frame time: {lap_time}, FPS: {(1 / lap_time) if lap_time > 0 else 999}, "
              f"average frame time: {total_time / lap_count}, average FPS: {lap_count / total_time}")

        if laps > 0 and lap_count > laps:
            return


if __name__ == '__main__':
    base_rgb, _ = img_read("base.png", cv.IMREAD_COLOR)
    template, template_shape = img_read("target.png")

    base = cv.Canny(base_rgb, 300, 550)

    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    template_width, template_height = template_shape[1], template_shape[0]

    def testament():
        match_result = cv.matchTemplate(base, template, cv.TM_CCOEFF_NORMED)

        if OPENCL:
            match_result = match_result.get()

        threshold = 0.2
        loc = np.where(match_result >= threshold)

        loc = np.dstack((loc[1], loc[0]))
        loc = loc.squeeze()
        final_result = []
        try:
            labels = cluster.DBSCAN(eps=5, min_samples=1).fit(loc).labels_
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            for i in range(0, num_clusters):
                try:
                    t = loc[labels.tolist().index(i)]
                    # print(t)
                    final_result.append(t)
                except ValueError:
                    pass
        except ValueError:
            print("Not Found")

        for pt in final_result:
            cv.rectangle(base_rgb, pt, (pt[0] + template_width, pt[1] + template_height), (0, 0, 255), 2)
            # print(pt)

    benchmark(testament, 600)

    # cv.imshow("base", base_rgb)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
