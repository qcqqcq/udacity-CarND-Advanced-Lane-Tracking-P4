import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import pdb

    

def get_img_from_filename(file_name):
    '''Define explicit function to use correct imread. 
    Avoids user using an imread that returns BGR instead of RGB
    '''

    return mpimg.imread(file_name)


class LaneVerifier:
    def __init__(self):
        self.left_history = []
        self.right_history = []
        self.history_length = 20
        self.sd_width_thresh = 80 # in pixels
        self.mean_width_thresh = 900 # in pixels
        
        self.last_left_good_poly_cols = None
        self.last_right_good_poly_cols = None

    def ingest_lanes(self,left_tracker,right_tracker):

        self.left_history.append(left_tracker)
        self.right_history.append(right_tracker)
        if len(self.left_history) > self.history_length:
            self.left_history.pop(0)
        if len(self.right_history) > self.history_length:
            self.right_history.pop(0)
                
        # Smoothen lane tracks using history        
        left_tracker.smooth_poly_cols = self.smooth_lane(self.left_history)
        right_tracker.smooth_poly_cols = self.smooth_lane(self.right_history)
        
        
        # Check consistency for the pair
        if self.pair_consistent(left_tracker.smooth_poly_cols,right_tracker.smooth_poly_cols):
            self.last_left_good_poly_cols = left_tracker.smooth_poly_cols
            self.last_right_good_poly_cols = right_tracker.smooth_poly_cols
            #print('consisten')
        
        elif (self.last_left_good_poly_cols is not None 
              and self.last_right_good_poly_cols is not None):
            
            left_tracker.smooth_poly_cols = self.last_left_good_poly_cols
            right_tracker.smooth_poly_cols = self.last_right_good_poly_cols
        
            #print('lanes no good')


    def smooth_lane(self,history):
        '''Take weighted average of last lane detections'''

        poly_cols = np.array([track.poly_cols for track in history]) 
        smooth_poly = np.average(poly_cols,axis=0
                                 ,weights=1+np.arange(len(history)))

        return smooth_poly
        
    def pair_consistent(self,left_smooth_poly_cols,right_smooth_poly_cols):
        '''Do checks at the pair level'''
        consistent = True
        
        # Ensure parallel
        self.pixel_width = right_smooth_poly_cols - left_smooth_poly_cols
        
        #print('SD %.2f'%np.std(self.pixel_width))
        #print('Mean %.2f'%np.average(self.pixel_width))
        if np.std(self.pixel_width) > self.sd_width_thresh:
            consistent = False
        elif np.average(self.pixel_width) > self.mean_width_thresh:
            consistent = False
        
        return consistent



class ImageProcessor():
    def __init__(self):
        self.mtx = None
        self.dist = None
        self.M = None
        self.Minv = None
        self.thresher = None

    def ingest_image(self,img):
        self.raw_img = img
        self.img = self.undistort_image(img)
        self.gray = cv2.cvtColor(self.img,cv2.COLOR_RGB2GRAY)

        self.num_rows = img.shape[0]
        self.num_cols = img.shape[1]

        ret = self.warp_perspective()
        if ret is None:
            print('Re-run ingest_image after setting warp points')

        ret = self.get_binary()
        if ret is None:
            print('Re-run ingest_image after ingesting a thresholder object ')

    def calibrate_from_existing(self,calibrated_processor):
        self.mtx = calibrated_processor.mtx
        self.dist = calibrated_processor.dist


    def calibrate_camera(self):
        ''' Calculates the correct camera matrix and distortion coefficients        using the calibration chessboard images
        '''

        # Make a list of calibration image
        calibration_image_files = glob.glob('./camera_cal/calibration*.jpg')
    
        # Arrays to store object points and image points from all the images
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # here we assume the object (checker board) is a flat surface, so the
        # z-values are all 0.  
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    
        # Step through the calibration images and search for chessboard corners
        for fname in calibration_image_files:
            # Read image from file and convert to gray
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
            # Find the chessboard corners using gray channel
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
    
                # Draw and display the corners
                #img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                #plt.imshow(img)
               
        # Use image and object points to calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

        self.mtx = mtx
        self.dist = dist
        
    
    def undistort_image(self,img):
        '''Undistored image to reduce radial and tangental distortion'''

        if (self.mtx is None) or (self.dist is None):
            print('Error: Camera not calibrated')
            raise Exception
    
        # Undistort
        undistorted = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        return undistorted

    def set_warp_source_points(self,top_row,bottom_row
                               ,bottom_left,top_left
                               ,top_right,bottom_right):
        '''Set source points for perspective warping'''

        # Arrange parameters into points
        src1 = (bottom_left,bottom_row) # Bottom left
        src2 = (top_left,top_row) # Top left
        src3 = (top_right,top_row) # Top right
        src4 = (bottom_right,bottom_row) # Bottom right

        # Save points as array of points
        self.src = np.float32([src1,src2,src3,src4]) 
    
        
        # Get perpective transform
        self.get_perspective_transforms()

    def get_perspective_transforms(self):
        '''Set destination points for perspective warping and 
        given src and dst points, calculate the perspective 
        transform matrix'''

        offset = 300

        # Arrange points based on image size and offset
        dst1 = (offset,self.num_rows) # Bottom left
        dst2 = (dst1[0],offset) # Top left
        dst3 = (self.num_cols-offset,dst2[1]) # Top right
        dst4 = (dst3[0],dst1[1]) # Bottom right
        
        # Save points as array of points
        self.dst = np.float32([dst1, dst2, dst3, dst4])

        # Get perspective transform
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def warp_perspective(self):
        '''Perform warp to get bird's eye perspective'''

        if self.M is None:
            print('Perspective warp not performed, transform matrix is None')
            return

        warped = cv2.warpPerspective(self.img, self.M, (self.num_cols,self.num_rows))
        self.warped = warped

        # Also make a gray version
        self.warped_gray = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)

        return warped


    def ingest_lanes(self,left_lane,right_lane,lane_verifier):

        lane_verifier.ingest_lanes(left_lane,right_lane)

        '''Fill area between lane'''

        # Create an image to draw the lines on
        zero_img = np.zeros_like(self.binary).astype(np.uint8)
        filled_lanes = np.dstack((zero_img, zero_img,zero_img))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_lane.smooth_poly_cols,left_lane.poly_rows]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.smooth_poly_cols, right_lane.poly_rows])))])
        #pts_right = np.array([np.flipud(np.transpose(np.vstack([poly_lanes['right'][1], poly_lanes['right'][0]])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(filled_lanes, np.int_([pts]), (0,255, 0))

        unwarped_lanes = cv2.warpPerspective(filled_lanes, self.Minv, (self.num_cols,self.num_rows))
        overlayed_image = cv2.addWeighted(self.img, 1, unwarped_lanes, 0.3, 0)

        self.overlayed = overlayed_image
        self.unwarped_lanes = unwarped_lanes
        self.get_offset_from_center()


        # Annotate image
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Left curve
        curve_str = 'Left Radius of Curvature: %im'%left_lane.radius_of_curve
        cv2.putText(overlayed_image,curve_str,(100,100),font,1,(255,255,255),2,cv2.LINE_AA)

        # Right curve
        curve_str = 'Right Radius of Curvature: %im'%right_lane.radius_of_curve
        cv2.putText(overlayed_image,curve_str,(100,200),font,1,(255,255,255),2,cv2.LINE_AA)

        # Distance to right
        distance_str = 'Offset from Center (+ is Right): %.1f'%self.meters_to_right
        cv2.putText(overlayed_image,distance_str,(100,300),font,1,(255,255,255),2,cv2.LINE_AA)

        



    def get_offset_from_center(self):
        _,hot_cols = self.unwarped_lanes[:,:,1].nonzero()

        # Lane width
        left_lane_px = min(hot_cols) 
        right_lane_px = max(hot_cols)
        lane_width_px = right_lane_px - left_lane_px
        
        # Meters per pixel
        lane_width_m = 3.7
        m_per_px = lane_width_m/lane_width_px
        
        # Lane center
        lane_center = (left_lane_px + right_lane_px)/2
        image_center = self.overlayed.shape[1]/2

        # Offset from center
        pixels_to_right = image_center - lane_center
        meters_to_right = pixels_to_right*m_per_px
        
        self.meters_to_right = meters_to_right

    def ingest_thresher(self,thresher):
        self.thresher = thresher

    def get_binary(self):
        if self.thresher is None: return
        self.binary = self.thresher.get_binary(self)
        return self.binary


class Thresher():
    ''' Thresholds images to form binaries
    Usually not used directly.  Just to visualization purposes'''

    def __init__(self,sobel_kernel,dir_thresh,mag_thresh,s_thresh):
        self.absgraddir = None

        # Set thresholds
        self.sobel_kernel = sobel_kernel
        self.dir_thresh = dir_thresh
        self.mag_thresh = mag_thresh
        self.s_thresh = s_thresh


    def calculate_sobel(self,gray):
        '''Calculates X and Y direction Sobels'''

        
        # Calcualte sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel) 


        # Calculate quantities derived from Sobel

        # Absolute direction of Sobel gradient
        self.absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        # Magnitude of Sobel gradient
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        self.gradmag = (gradmag/scale_factor).astype(np.uint8) 


    def sobel_direction_threshold(self,gray):
        '''Create binary based on Sobel direction thresholding'''
        
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result

        self.calculate_sobel(gray)
    
        thresh = self.dir_thresh
        binary_output =  np.zeros_like(self.absgraddir)
        binary_output[(self.absgraddir >= thresh[0]) & (self.absgraddir <= thresh[1])] = 1
        
        # Return the binary
        return binary_output
    
    
    def sobel_magnitude_threshold(self,gray):
        '''Create binary based on Sobel magnitude thresholding'''

        self.calculate_sobel(gray)
    
        thresh = self.mag_thresh
        binary_output = np.zeros_like(self.gradmag)
        binary_output[(self.gradmag >= thresh[0]) & (self.gradmag <= thresh[1])] = 1
    
        # Return the binary
        return binary_output
    
    def saturation_threshold(self,img):
        '''Threshold the S-channel of HLS'''
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)    
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        self.s_channel = s_channel
        
        thresh = self.s_thresh
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

        # Return the binary
        return binary_output
    
    def get_binary(self,img_processor,thresh=0.5):
        '''Main method to create a combined binary image'''

        sobel_direction_binary = self.sobel_direction_threshold(img_processor.warped_gray)
        sobel_magnitude_binary = self.sobel_magnitude_threshold(img_processor.warped_gray)
        saturation_binary = self.saturation_threshold(img_processor.warped)

        # Combine the averaging binaries
        self.ensemble = np.average([sobel_magnitude_binary,saturation_binary],weights=[1,1],axis=0)

        ensemble_binary = self.ensemble >= thresh

        combined_binary = np.zeros_like(self.s_channel)
        
        # Excluding direction binary for now
        combined_binary[ensemble_binary] = 1
   
        return combined_binary



class LaneTracker():
    def __init__(self,lane,xm_per_pix,ym_per_pix
                 ,window_height=40,window_width=150,minpix=50):
        

        self.window_height = window_height # number of pixels for window height
        self.window_width = window_width # number of pixels for window width
        self.minpix = minpix # number pixels needed to recenter window

        # Meters to pixels conversion
        self.xm_per_pix = xm_per_pix
        self.ym_per_pix = ym_per_pix

        if lane not in ['left','right']:
            print('Choose lane as left or right and not %s'%lane)
        self.lane = lane


        # Initialize tracking statistics from past detections
        self.polyfit = None

    def reset(self):
        self.polyfit = None

    def get_initial_lane_col(self):
        '''Make initial guess of lane position (column pixel)
        based on histogram of bottom half of binary image'''

        # Calculate bottom half histogram
        histogram = np.sum(self.bimg[int(self.bimg.shape[0]/2):]
                                       ,axis=0)
        self.histogram = histogram

        # Guess left and right lane position based on peaks in histogram
        midpoint= int(len(histogram)/2)
        if self.lane == 'left':
            lane_col = np.argmax(histogram[:midpoint])
        else: lane_col = np.argmax(histogram[midpoint:]) + midpoint

        return lane_col

    def _find_lane_in_window(self,hot_rows,hot_cols,window):

        '''Find hot indices within a small window of image'''

        # Indices for hot pixels between top and bottom row
        within_top_bot = (hot_rows <= window.bottom) & (hot_rows > window.top)
    
        # Indices for in the band and within Left window
        within_left_right = (hot_cols >= window.left) & (hot_cols < window.right)

        # Indices within top,bottom,left, and right
        # and thus within the window
        within_window = within_top_bot & within_left_right

        hot_inds, = within_window.nonzero()
        return hot_inds



    def search_entire_image(self,hot_rows,hot_cols,draw=False):
        '''Search entire image of lanes using iteration of sliding window'''

        # Get initial lane positions using histogram
        init_lane_col = self.get_initial_lane_col()
        bottom_row = self.bimg.shape[0]


        # Initialize window
        win = Window(self.window_height,self.window_width)
        win.update_position(init_lane_col,bottom_row)


        # Holder for all lane pixel indices throughout binary image
        lane_inds = []


        # Iterate through windows while window is within image
        while win.top > 0:

            # Draw for debug and documentation
            if draw:
                cv2.rectangle(self.out_img,(win.left,win.bottom)
                    ,(win.right,win.top),(0,255,0), 2)
            
            # Get indices of hot pixels within window
            hot_ind = self._find_lane_in_window(hot_rows,hot_cols,win)
            lane_inds.append(hot_ind)

            # If enough pixels are hot, shift the window
            if len(hot_ind) >= self.minpix:
                center_col = int(np.mean(hot_cols[hot_ind]))
            else: center_col = win.center_col

            # Update window position
            win.update_position(center_col)


        # Post-iteration cleanup
        # Combine hot indices
        lane_inds = np.concatenate(lane_inds) 

        # Get pixel values of left and right lanes
        lane_rows = hot_rows[lane_inds]
        lane_cols = hot_cols[lane_inds]

        return lane_rows,lane_cols

    def find_near_previous(self,hot_rows,hot_cols):
        '''Find hot pixels close to previously drawn polyfit'''

        # Set margin around previous lane to accept new pixels
        margin = 100
        
        # Determine polynomial line fit from previous fit
        previous_cols = self.apply_polyfit(hot_rows,self.polyfit)

        # Find indices where new lane is within margin of previous lane
        lane_inds = (hot_cols < (previous_cols + margin)) & (hot_cols > (previous_cols - margin))

        # Get pixel values of left and right lanes
        lane_rows = hot_rows[lane_inds]
        lane_cols = hot_cols[lane_inds]

        return lane_rows,lane_cols

    def find_lane(self,img_processor,draw=False):
        '''Iteratively search of lanes within small windows. Start from
        bottom of image and move windows towards top of image until no
        longer within image. 
        '''

        # Save incoming image
        self.bimg = img_processor.binary

        # Preprae output image on which overlays will be placed
        self.out_img = np.dstack((self.bimg,self.bimg,self.bimg))*255

        # Get "hot" rows and columns.  Hot means binary = 1
        # These are arrays of pixel values where the binary image
        # is not zero. Variable names ending in _ind represent
        # indices into these arrays
        hot_rows,hot_cols = self.bimg.nonzero()
        
        # The actual search for lane pixels
        if self.polyfit is None:
            lane_rows,lane_cols = self.search_entire_image(hot_rows,hot_cols,draw=draw)
        else:
            lane_rows,lane_cols = self.find_near_previous(hot_rows,hot_cols)


        # Get radius of curvature
        self.radius_of_curve = self.get_curvature(lane_rows,lane_cols)

        # Fit raw polynomial in pixel space
        self.poly_rows,self.poly_cols,self.polyfit = self.get_polyfit(lane_rows,lane_cols)

        # Draw lane pixels 
        if draw:
            # Draw hot pixels
            if self.lane == 'left': color = [255,0,0]
            else: color = [0,0,255]
            self.out_img[lane_rows,lane_cols] = color


    def apply_polyfit(self,poly_rows,polyfit):
        '''Return polynomial estimated columns for an array of rows'''

        # Polynomial of row array
        poly_rows_matrix = np.vstack([poly_rows**2,poly_rows,np.ones_like(poly_rows)]).T
    
        # Multiply polynomial with coefficients
        poly_cols = np.dot(poly_rows_matrix,polyfit)
    
        return poly_cols

    def get_polyfit(self,lane_rows,lane_cols):
        '''Fit a 2nd degree polynomial. Lane rows and cols can be in pixels
        or meters, it doesn't matter '''
        
        # Fit weighted polynomial
        polyfit = np.polyfit(lane_rows, lane_cols,2)
        
        # Create rows for which polynomial will be calculated for
        poly_rows = np.arange(self.out_img.shape[0])
        poly_cols = self.apply_polyfit(poly_rows,polyfit)
        

        return poly_rows,poly_cols,polyfit

    def get_curvature(self,lane_rows,lane_cols):

        # Pixels to meters conversion
        lane_rows_meters = lane_rows*self.ym_per_pix
        lane_cols_meters = lane_cols*self.xm_per_pix

        _,_,polyfit_meters = self.get_polyfit(lane_rows_meters,lane_cols_meters)
        # Calculate the radii of curvature
        # y_eval is y position at which curvature is to be calculated
        # Example values: 632.1 m    626.2 m
        y_eval = self.out_img.shape[0]
        y_eval_meters = y_eval*self.ym_per_pix

        radius_curve = ((1 + (2*polyfit_meters[0]*y_eval_meters + polyfit_meters[1])**2)**1.5) / np.absolute(2*polyfit_meters[0])

        return radius_curve


        
class Window():
    def __init__(self,height=40,width=150):
        
        # Set window parameters
        self.height = height # number of pixels for window height
        self.width = width # number of pixels for window width

        
    def update_position(self,center_col,bottom_row=None):

        # Save center column
        self.center_col = center_col

        # Left and right columns
        self.left = center_col - int(self.width/2)
        self.right = self.left + self.width

        # Window rows
        # If no bottom row is given, use the current top row
        if bottom_row is None:
            self.bottom = self.top
        else:
            self.bottom = bottom_row
        self.top = self.bottom - self.height

        if self.top <= 0:
            self.top = 0


    
if __name__ == '__main__':
    # Test code
    win = Window()
    
if __name__ == '__main__':
    # Test code
    tools = CameraTools()
    tools.calibrate_camera()
