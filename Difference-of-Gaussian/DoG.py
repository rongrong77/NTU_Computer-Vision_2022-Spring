import numpy as np
import cv2


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        first_octave = [image] + [cv2.GaussianBlur(image, (0, 0), self.sigma**i) for i in range(1, self.num_guassian_images_per_octave)]
        # down-sample the image
        DSImage = cv2.resize(first_octave[-1], 
                             (image.shape[1]//2, image.shape[0]//2), 
                             interpolation = cv2.INTER_NEAREST)
        # filter down-sampled images in the next octave
        second_octave = [DSImage] + [cv2.GaussianBlur(DSImage, (0, 0), self.sigma**i) for i in range(1, self.num_guassian_images_per_octave)]
        # combine two octaves
        gaussian_images = [first_octave, second_octave]
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        #iterate through each octave for subtraction
        for i in range(self.num_octaves):
            GI = gaussian_images[i]
            dog_image = []
            for j in range(self.num_DoG_images_per_octave):
                dog = cv2.subtract(GI[j], GI[j+1])
                dog_image.append(dog)
                #save DoG images to disk
                M, m = max(dog.flatten()), min(dog.flatten())
                norm = (dog-m)*255/(M-m)
                cv2.imwrite(f'testdata/DoG{i+1}-{j+1}.png', norm)
            dog_images.append(dog_image)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        
        # method_1 (slower)
#         keypoints = []
#         # iterate throught each ocvate (n=2 here)
#         for i in range(self.num_octaves):
#             # transform an octave into a 3-dimensional array
#             dogs = np.array(dog_images[i])
#             height, width = dogs[i].shape
#             # examine every 3*3 cube for local extremum
#             #iterate through Dog image number 1 and 2 (other than 0 and 3)
#             for dog in range(1, self.num_DoG_images_per_octave-1):
#                 #iterate through every pixel
#                 for x in range(1, width-2):
#                     for y in range(1, height-2):
#                         pixel = dogs[dog,y,x]
#                         cube = dogs[dog-1:dog+2, y-1:y+2, x-1:x+2]
#                         # to check if it's local extremum
#                         if (np.absolute(pixel) > self.threshold) and ((pixel >= cube).all() or (pixel <= cube).all()):
#                             keypoints.append([y*2, x*2] if i else [y, x])

        # method_2 (faster)
        keypoints = np.array([],dtype='int64').reshape((0,2))
        for i in range(self.num_octaves):
            dogs = np.array(dog_images[i])
            cube = np.array([np.roll(dogs,(x,y,z),axis=(2,1,0)) for z in range(-1,2) for y in range(-1,2) for x in range(-1,2)])
            mask = (np.absolute(dogs)>=self.threshold)&((np.min(cube,axis=0)==dogs)|(np.max(cube,axis=0)==dogs))
            for j in range(1, self.num_DoG_images_per_octave-1):
                m = mask[j]
                x, y = np.meshgrid(np.arange(m.shape[1]),np.arange(m.shape[0]))
                kp = np.stack([y[m],x[m]]).T*2 if i else np.stack([y[m],x[m]]).T
                keypoints = np.concatenate([keypoints,kp])
                
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(np.array(keypoints), axis = 0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints