import numpy as np
import cv2 as cv

class SIFTBFMMatcher:
    """
    This class performs feature matching using SIFT and Brute-Force Matcher algorithms.

    Methods:
        __init__(self, images : list) -> None
            Initializes the SIFTBFMMatcher class.

        feature_matching(self) -> numpy.ndarray
            Perform feature matching using SIFT and Brute-Force Matcher algorithms.
            Apply SIFT algorithm to extract key points and descriptors from the input images,
            and then uses Brute-Force Matcher algorithm to find matches between the descriptors.
            Finally, it draws the good matches on the images and returns the result.
    """

    def __init__(self, images : list):
        """
        Constructor for the SIFTBFMMatcher class.

        Args:
            images (list): A list containing the two images to be matched.
        """
        # Assign the first image to self.img1
        self.img1 = images[0]
        # Assign the second image to self.img2
        self.img2 = images[1]

    def feature_matching(self) -> np.ndarray:
        """
        Perform feature matching using SIFT and Brute-Force Matcher algorithms.

        This method applies SIFT algorithm to extract key points and descriptors from the input images,
        and then uses Brute-Force Matcher algorithm to find matches between the descriptors.
        Finally, it draws the good matches on the images and returns the result.

        Returns:
            numpy.ndarray: Image with matched features displayed.
        """

        # Read images as grayscale
        img1 = cv.cvtColor(self.img1, cv.COLOR_BGR2GRAY)  # Load first image as grayscale
        img2 = cv.cvtColor(self.img2, cv.COLOR_BGR2GRAY)  # Load second image as grayscale

        # Apply SIFT algorithm and get key points
        sift = cv.SIFT_create()  # Create SIFT feature detector
        kp1, des1 = sift.detectAndCompute(img1, None)  # Detect and compute SIFT features in first image
        kp2, des2 = sift.detectAndCompute(img2, None)  # Detect and compute SIFT features in second image

        # Apply Brute-Force Matcher for matches
        bf_matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=True)  # Brute-Force Matcher
        matches = bf_matcher.match(des1, des2)  # Find matches between descriptors

        # Draw matches on images
        img = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img

