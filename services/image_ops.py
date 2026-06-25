import cv2
import numpy as np
from PIL import Image, ImageOps


def _order_points(pts):
    """Orders 4 points as: [top-left, top-right, bottom-right, bottom-left]"""
    pts = pts.reshape((4, 2))
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def _load_as_bgr(image_input):
    """
    Accepts a file path, a PIL Image, or a numpy (OpenCV) array, and
    returns a BGR numpy array ready for OpenCV.

    File paths and PIL Images go through Pillow first so EXIF rotation
    gets baked into the actual pixels — cv2.imread/imdecode ignore that
    metadata entirely, so a photo that looks upright everywhere else
    can load sideways or upside-down here without any error.
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, np.ndarray):
        return image_input
    else:
        raise ValueError("Input must be a file path, a PIL Image, or a numpy image array.")

    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _find_quadrilateral(hull):
    """
    Tries to simplify a contour's convex hull down to exactly 4 points,
    trying progressively looser tolerances until one works.

    Returns the 4-point approximation, or None if no tolerance in the
    range produces a clean quadrilateral. We deliberately never fall
    back to picking raw hull extremes (e.g. via sum/diff of all hull
    points) — noise such as a shadow or glare merging into the grid's
    silhouette can push a hull point past the real edge of the page,
    and warpPerspective will then sample outside the source image,
    filling the gap with solid black (visible as a sharp diagonal wedge
    in the warped output).
    """
    peri = cv2.arcLength(hull, True)
    for eps_factor in (0.02, 0.03, 0.05, 0.08, 0.1):
        approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
        if len(approx) == 4:
            return approx
    return None


def process_image(image_input, side_length=450):
    """
    Reads an image from a path, a PIL Image, or a numpy array.
    Isolates the Sudoku grid robustly and returns a warped perspective.
    """
    img = _load_as_bgr(image_input)
    if img is None:
        raise FileNotFoundError(f"Could not read image from: {image_input}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dynamic thresholding based on average brightness
    avg_brightness = np.mean(gray)
    if avg_brightness < 127:
        thresh_adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 5
        )
    else:
        thresh_adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 5
        )

    # Find external contours
    contours, _ = cv2.findContours(thresh_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sudoku_contour = None

    for c in sorted_contours:
        if cv2.contourArea(c) < (img.shape[0] * img.shape[1] * 0.1):
            continue

        hull = cv2.convexHull(c)
        approx = _find_quadrilateral(hull)

        if approx is not None:
            sudoku_contour = approx
            break 

    if sudoku_contour is None:
        raise ValueError("Error: Could not identify a 4-sided grid shape in the image.")

    src_pts = _order_points(sudoku_contour)
    dst_pts = np.array([
        [0, 0], [side_length - 1, 0], [side_length - 1, side_length - 1], [0, side_length - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(gray, matrix, (side_length, side_length))

    return warped