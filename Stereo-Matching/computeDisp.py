import numpy as np
import cv2
import cv2.ximgproc as xip


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    
    
    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency


    
    def census_transform(img):
        c_img = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                binary_code = 0
                center = img[i, j]
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if 0 <= i + dx < img.shape[0] and 0 <= j + dy < img.shape[1]:
                            binary_code <<= 1
                            binary_code |= img[i + dx, j + dy] < center
                c_img[i, j] = binary_code
        return c_img

    Il_census = census_transform(Il)
    Ir_census = census_transform(Ir)


    
    cost_l = np.zeros((max_disp+1, h, w), dtype=np.float32)
    for d in range(max_disp+1):
       for i in range(h):
           for j in range(w):
               if j >= d:
                   cost_l[d, i, j] = np.count_nonzero(Il_census[i, j] != Ir_census[i, j-d])
               else:
                   cost_l[d, i, j] = np.count_nonzero(Il_census[i, j] != Ir_census[i, 0])
    
    cost_r = np.zeros((max_disp+1, h, w), dtype=np.float32)
    for d in range(max_disp+1):
        for i in range(h):
            for j in range(w):
                if j + d < w:
                    cost_r[d, i, j] = np.count_nonzero(Ir_census[i, j] != Il_census[i, j+d])
                else:
                    cost_r[d, i, j] = np.count_nonzero(Ir_census[i, j] != Il_census[i, w-1])
  
 
    

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    cost_aggregated_left = np.zeros((max_disp+1, h, w), dtype=np.float32)
    cost_aggregated_right = np.zeros((max_disp+1, h, w), dtype=np.float32)

    
    
    for d in range(max_disp+1):
        cost_aggregated_left[d] = xip.jointBilateralFilter(Il, cost_l[d], 30, 5, 5)
        cost_aggregated_right[d] = xip.jointBilateralFilter(Ir, cost_r[d], 30, 5, 5)  


    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    
    winner_L = np.argmin(cost_aggregated_left, axis=0)
    winner_R = np.argmin(cost_aggregated_right, axis=0)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    # >>> Disparity Refinement
    # Left-right consistency check
    for y in range(h):
      for x in range(w):
           if x-winner_L[y,x]>=0 and winner_L[y,x] == winner_R[y,x-winner_L[y,x]]:
               continue
           else: 
               winner_L[y,x]=-1

    for y in range(h):
     for x in range(w):
         if winner_L[y,x] == -1:
             l = 0
             r = 0
             while x-l>=0 and winner_L[y,x-l] == -1:
                 l+=1
             if x-l < 0:
                 FL = max_disp 
             else:
                 FL = winner_L[y,x-l]

             while x+r<=w-1 and winner_L[y,x+r] == -1:
                 r+=1
             if x+r > w-1:
                 FR = max_disp
             else:
                 FR = winner_L[y, x+r]
             winner_L[y,x] = min(FL, FR)

    labels = xip.weightedMedianFilter(Il.astype(np.uint8), winner_L.astype(np.uint8), 18, 1)
    return labels.astype(np.uint8)
