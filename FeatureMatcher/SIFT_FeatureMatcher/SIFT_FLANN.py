import numpy as np
import cv2 as cv

class SIFTFLANNMatcher:
    """
    This class performs feature matching using SIFT and FLANN algorithms.

    Methods:
        __init__(self, images : list) -> None
            Initializes the SIFTBFMMatcher class.

        feature_matching(self) -> numpy.ndarray
            Perform feature matching using SIFT and FLANN algorithms.
            Apply SIFT algorithm to extract key points and descriptors from the input images,
            and then uses FLANN algorithm to find matches between the descriptors.
            Finally, it draws the good matches on the images and returns the result.
    """

    def __init__(self, images : list):
        """
        Initializes the SIFTFLANNMatcher class.

        Args:
            img1 (numpy.ndarray): First image.
            img2 (numpy.ndarray): Second image.
        """
        self.img1 = images[0]
        self.img2 = images[1]

    def feature_matching(self) -> np.ndarray:
        """
        Perform feature matching using SIFT and FLANN algorithms.

        This method applies SIFT algorithm to extract key points and descriptors from the input images, and then uses FLANN algorithm to find matches between the descriptors. Finally, it draws the good matches on the images and returns the result.

        Args:
            None

        Returns:
            numpy.ndarray: Image with matched features displayed.
        """

        # convert images to grayscale
        img1 = cv.cvtColor(self.img1, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(self.img2, cv.COLOR_BGR2GRAY)

        # apply SIFT algorithm and get key points
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # apply FLANN algorithm for matches
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=7) # you can replace flann tree algorithm with one "1"
        search_params = dict()
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # draw only good matches by ratio test as per Lowe's paper
        matchesMask = [[0, 0] for i in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]

        img = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=0, matchesMask=matchesMask)
        return img



