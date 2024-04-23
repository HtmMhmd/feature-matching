import numpy as np
import cv2

class ORBBFMMatcher:
    """
    Class for performing feature matching using ORB and Brute-Force Matcher.

    Attributes:
        images (list): List of images to perform feature matching on.

    Methods:
        feature_matching(self)
            Performs feature matching using ORB and Brute-Force Matcher.
    """
    def __init__(self, images):
        """
        Initializes the FeatureMatcher class.

        Args:
            images (list): A list of images to perform feature matching on.
        """
        self.images = images

    def feature_matching(self) -> np.ndarray:
        """
        Performs feature matching using ORB and Brute-Force Matcher.

        This method converts the images to grayscale, detects keypoints and descriptors using ORB, matches the descriptors,
        sorts the matches based on distance, and displays the top 5 matches on the images.

        Returns:
            numpy.ndarray: Image with matched features displayed.
        """
        # Read query and train images
        query_img = self.images[0]
        train_img = self.images[1]

        # Convert images to grayscale
        query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

        # Initialize ORB feature detector
        orb = cv2.ORB_create()

        # Detect keypoints and compute descriptors in the query and train images
        queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
        trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

        # Initialize Brute-Force Matcher
        matcher = cv2.BFMatcher()

        # Match descriptors between query and train images
        matches = matcher.match(queryDescriptors, trainDescriptors)

        # Sort matches based on distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw top 5 matches on the query and train images and return the result
        final_img = cv2.drawMatches(query_img, queryKeypoints, train_img, trainKeypoints, matches[:5], None, flags=0)
        return final_img
