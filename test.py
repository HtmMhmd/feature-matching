from FeatureMatcher.ORB_FeatureMatcher import ORBBFMMatcher
from FeatureMatcher.SIFT_FeatureMatcher import SIFTFLANNMatcher, SIFTBFMMatcher
import cv2
import numpy as np

def main():
    """
    This script demonstrates feature matching using the ORB algorithm.
    It reads two images, performs feature matching, and displays the matches.

    Usage:
    - Make sure the images 'hello.JPG' and 'hello.png' are present in the same directory as this script.
    - Run the script and a window will open showing the matches.

    Note: This script requires the ORB_FeatureMatcher module.

    """

    # Read images
    images = [cv2.imread('hello.JPG'), cv2.imread('hello.png')]

    # Create a FeatureMatcher object
    feature_matcher = SIFTBFMMatcher(images)

    # Perform feature matching
    final_img = feature_matcher.feature_matching()

    # Display the matches
    cv2.imshow("Matches", final_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
