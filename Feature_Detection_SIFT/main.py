import numpy as np
import cv2
import os
import argparse
"""Take the path of images. Train and Query images must be on the same directory"""
print("Program is starting")
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True, help="Images Path")
args = vars(ap.parse_args())

"""Minimum value of the matching """
MIN_MATCH_COUNT = 4

# queryPath= "C:\\Users\\AdemBerkAksoy\\Desktop\\matchingPics"
queryPath= args["query"]

"""Finding and reading query and train images. Train image's name must be wf.JPG"""
imagesList=[]
tags=[]
print("Finding the train and query images in the given path")
for i in os.listdir(queryPath):
    if not i == "wf.JPG":
        imagesList.append(queryPath + "\\" + i)
        tags.append(i)
    else:
        trainPath = (queryPath + "\\" + i)
print("Reading the query images")
readTrainImages=[]
for i in imagesList:
    readTrainImages.append(cv2.imread(i,0))

print("Reading the train image")
trainImage= cv2.imread(trainPath,0)

print("İnitiliaze the feature detection / SIFT ")
sift = cv2.xfeatures2d.SIFT_create()
kp=[]
des=[]

"""Calculating keypoints and descriptors by sift"""
print("Calucalte the keypoints and descriptors for query images")
for img in readTrainImages:
    kpImg, desImg = sift.detectAndCompute(img,None)
    kp.append(kpImg)
    des.append(desImg)

print("Calucalte the keypoints and descriptors for train image")
kpTrain, desTrain = sift.detectAndCompute(trainImage,None)

"""Matching using Flann based """
print("Adjusting the matcher paramethers")
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

print("looking the matching")
matches=[]
for i in des:
     matches.append(flann.knnMatch(i,desTrain,k=2))


"""Eliminate the matching using Lowe"""
good = []
goods=[]
for match in matches:
    for m,n in match:
        if m.distance < 0.7*n.distance:
            good.append(m)
    goods.append(good)
    good = []

"""after eliminate the matches apply the homograpy and find the exact place of the image"""
matchesMaskes=[]
for good in goods:
    if len(good)>MIN_MATCH_COUNT:
        srcPts = np.float32([ kp[goods.index(good)][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dstPts = np.float32([ kpTrain[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        matchesMaskes.append(matchesMask)

        h,w = readTrainImages[goods.index(good)].shape

        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(trainImage,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        text = tags[goods.index(good)]
        pos =tuple(dst[0][0])
        cv2.putText(img2,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,0),3)
        del srcPts, dstPts, M, mask, matchesMask, h,w,pts,dst
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
"""show the result"""
img2 = cv2.resize(img2, (1500, 800))
cv2.imshow("result", img2)
cv2.waitKey(0)

"""these lines for the draw matching"""
"""draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
img3 = cv2.resize(img3,(1500,800))
cv2.imshow("oldu",img3)
cv2.waitKey(0)"""