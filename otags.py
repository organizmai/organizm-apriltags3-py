import numpy as np
from scipy.linalg import hadamard
import cv2
import random
import cProfile

Tag_d22 = np.array([	list('wwwwwwwwwwwwwwwwwwwwwwwwww'),
						list('wbbbbbbbbbbbbbbbbbbbbbbbbw'),
						list('wbddddddddddddddddddddddbw'),
						list('wbbbbbbbbbbbbbbbbbbbbbbbbw'),
						list('wwwwwwwwwwwwwwwwwwwwwwwwww')])

def angle(v1, v2):
  return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# # Assume a convex quadrilateral
# def order_points(pts):
#     # sort the points based on their x-coordinates
#     xSorted = pts[np.argsort(pts[:, 0]), :]

#     # grab the left-most and right-most points from the sorted
#     # x-roodinate points
#     leftMost = xSorted[:2, :]
#     rightMost = xSorted[2:, :]

#     # now, sort the left-most coordinates according to their
#     # y-coordinates so we can grab the top-left point
#     leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
#     tl = leftMost[0]
#     # Get the remaining points in a list
#     rem = np.asarray([leftMost[1], *rightMost])
#     # Calcualte the angle of the vector formed by the top left
#     # point and each of the remaining points with the vector
#     # (1,0), a horizontal vector
#     angles = [angle(x - tl, [1,0]) for x in rem]
    
#     # Sort the remaining points by their angle to the origin point, 
#     # and then create the output array. 
#     ordered =  np.array([tl, *[x for _,x in sorted(zip(angles,rem), key=lambda x: x[0])]], dtype="float32")

#     edge_lengths = np.array([   dist(ordered[0], ordered[1]),
#                                 dist(ordered[1], ordered[2]),
#                                 dist(ordered[2], ordered[3]),
#                                 dist(ordered[3], ordered[0])])

#     start_pt = np.argmax(edge_lengths)
#     new_ordered = np.array([ordered[(start_pt)%4],
#                 ordered[(start_pt+1)%4],
#                 ordered[(start_pt+2)%4],
#                 ordered[(start_pt+3)%4]])

#     return new_ordered

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    # M = cv2.getPerspectiveTransform(rect, dst)
    M = cv2.findHomography(rect, dst)[0]
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


class OTag(object):
    """An Organizm Tag, which is defined a number of data bits. Each tag is 
    a border of white surrounding a border of black which surounds a linear
    data string."""
    def __init__(self, d_bits):
        super(OTag, self).__init__()
        # User defined parameters
        self.d_bits = d_bits
        
        # Constant parameters
        self.d_height = 1
        self.d_width = d_bits

        # TODO: Maybe generalize this to contain different 
        # data shapes. 
        self.full_height = self.d_height + 4
        self.black_height = self.d_height + 2
        self.full_width = d_bits + 4
        self.black_width = d_bits + 2


    """Assumes that the image border is the black border, and that
    the image is binary.
    """
    def raw_decode_tag(self, quad, img):

        # Step one, extract a rectified version of the quad
        quad_ordered = order_points(quad)

        code = four_point_transform(img, quad)

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(code,(5,5),0)
        ret, bin_quad = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        # Step two, extract the data from the binarized version of the image
        im_height, im_width = bin_quad.shape

        x_pix_size = im_width / self.black_width
        y_pix_size = im_height / self.black_height

        x0_corners = np.linspace(0, im_width , self.black_width+1, dtype=int)
        y0_corners = np.linspace(0, im_height, self.black_height+1, dtype=int)

        xv, yv = np.meshgrid(x0_corners, y0_corners, sparse=False, indexing='ij')

        res = np.zeros(self.d_bits)
        for i in range(self.d_bits):
            d_slice = bin_quad[yv[1][1]:yv[2][2], xv[1+i][1]:xv[2+i][2]]

            res[i] = int(np.mean(d_slice))
            code = cv2.rectangle(code, (xv[1+i][1], yv[1][1]), (xv[2+i][2], yv[2][2]), (255,255,255), 1) 



        res = res > 255/2
        res = res.astype(int)
        print(res, flush=True)
        cv2.imshow("Display Code", code)


class CodeFamily(object):
	"""A code family generator class"""
	def __init__(self, bits, min_hamming, complexity):
		super(CodeFamily, self).__init__()
		self.bits = bits
		self.min_hamming = min_hamming
		self.complexity = complexity

		self.result_list = np.array([], dtype=int)

		self.start = random.randint(0, 2**32)
		self.large_prime = 982451653 # This is a large prime


	# Checks if the hamming distance of the given value and it's mirror
	# are less than the minmum hamming threshold.
	def is_hamming_from_existing(self, val):
		if self.result_list.shape[0] == 0:
			return True

		# Tiled to make unambiguous for square matrix broadcasting.
		val_tile = np.tile(val,(self.result_list.shape[0],1)) 
		val_mirror_tile = np.flip(val_tile, axis=1)

		min1 = np.amin(np.sum(np.logical_xor(self.result_list, val_tile), axis=1))
		if min1 > self.min_hamming:
			return False
		min2 = np.amin(np.sum(np.logical_xor(self.result_list, val_mirror_tile), axis=1))
		if min2 > self.min_hamming:
			return False
		return True

		# for code in self.result_list:
		# 	h_dist_orig = bin(int(val,2) ^ int(code,2)).count("1")
		# 	h_dist_mirror = bin(int(val_mirror,2) ^ int(code,2)).count("1")
		# 	if (h_dist_orig < self.min_hamming) or (h_dist_mirror < self.min_hamming):
		# 		return False

		# return True





	# Complexity is the number of groups of bits. 
	def check_complexity(self, val):
		res = np.sum(np.absolute(np.diff(val)))
		return res > self.complexity

	def run(self):
		for i in range(2**(self.bits)):
			val = self.start + self.large_prime * i
			
			# Convert to a numpy array with binary values
			bin_val = np.array(list(bin(val)[-self.bits-2:-1]), dtype=int)

			# print(bin_val, bin(val), bin_val) 
			if self.check_complexity(bin_val):
				mirror = np.flip(bin_val)
				h_dist_self = np.sum(np.logical_xor(bin_val, mirror))
				if h_dist_self > self.min_hamming:
					if self.is_hamming_from_existing(bin_val):
						if self.result_list.shape[0] == 0:
							self.result_list = np.array([bin_val])
						else:
							np.append(self.result_list, np.array([bin_val]), axis=0)
						print(bin_val, flush=True)



cf = CodeFamily(22, 6, 3)

cProfile.run("cf.run()")

# print(len(cf.result_list))
