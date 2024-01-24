import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Loading image into list
#image_list_name = ["250.JPG", "1000.JPG", "4000.JPG", "15000.JPG"]
image_list_name = [r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0061.png", 
                   r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0062.png", 
                   r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0063.png", 
                   r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0064.png", 
                   r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0065.png", 
                   r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0066.png", 
                   r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0067.png", 
                   r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0068.png", 
                   r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0069.png", 
                   r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0070.png", 
                   r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0071.png", 
                   r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0072.png", 
                   r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0073.png", 
                   r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0074.png", 
                   r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0075.png", 
                   r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0076.png"]
img_list = [cv.imread(fn) for fn in image_list_name]

# Align
#alignMTB = cv.createAlignMTB()
#alignMTB.process(img_list,img_list)

#exposure_times = np.array([0.25, 1.0, 4.0, 15.0], dtype=np.float32)
exposure_times = np.array([(1/0.03125), (1/0.0625), (1/0.125), (1/0.25), (1/0.5), (1/1), (1/2), (1/4), (1/8), (1/16), (1/32), (1/64), (1/128), (1/256), (1/512), (1/1024)], dtype=np.float32)

# CRF Dev
calibrateDebevec = cv.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(img_list, exposure_times)


"""""
blue_responseRo = responseDebevec.reshape(255,3)

plt.plot(x,blue_responseRo)
plt.show()
"""""
# merge Dev
merge_debevec = cv.createMergeDebevec()
hdr_debevec = merge_debevec.process(img_list, exposure_times, responseDebevec)

#CRF Robertson
CalibrateRobertson =cv.createCalibrateRobertson()
responseRo = CalibrateRobertson.process(img_list,exposure_times)
# print(responseDebevec.shape)
q = cv.imread(r"C:\Users\admin\Desktop\ws2324\Computational_Imaging\exe\hdr\memorial_church\memorial0061.png")
print(q.shape)

# plot response curve
plt.figure(1)

b_Dev=responseDebevec[:,0,0]
g_Dev=responseDebevec[:,0,1]
r_Dev=responseDebevec[:,0,2]
x = np.arange(256)
plt.plot(x,b_Dev,x,g_Dev,x,r_Dev)
plt.axis([0, 256, 0, 11])
plt.xlabel('Pixel value')
plt.ylabel('Exposure')
plt.title('response curve of Devbec')
plt.savefig(r'C:\Users\admin\Documents\PythonScripts\hdr_church\responseDev.png')
plt.show()

plt.figure(2)
b_Ro=responseRo[:,0,0]
g_Ro=responseRo[:,0,1]
r_Ro=responseRo[:,0,2]
plt.plot(x,b_Ro,x,g_Ro,x,r_Ro)
plt.axis([0, 256, 0, 11])
plt.xlabel('Pixel value')
plt.ylabel('Exposure')
plt.title('response curve of Rorberson')
plt.savefig(r'C:\Users\admin\Documents\PythonScripts\hdr_church\responseRo.png')
plt.show()



"""""
# merge robortson
merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, exposure_times, responseRo)

#生成hdr图片
cv.imwrite("de.hdr",hdr_debevec)  
cv.imwrite("Ro.hdr",hdr_robertson)


# Tonemap HDR image
tonemap1 = cv.createTonemap(2.2)
res_debevec = tonemap1.process(hdr_debevec)
res_robertson = tonemap1.process(hdr_robertson)


# Convert datatype to 8-bit and save
res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')

cv.imwrite("ldr_debevec.jpg", res_debevec_8bit)
cv.imwrite("ldr_robertson.jpg", res_robertson_8bit)
cv.imwrite("ldr_d.png",res_debevec * 255)
cv.imwrite("ldr_r.png",res_robertson * 255)


# Exposure fusion using Mertens
merge_mertens = cv.createMergeMertens()
res_mertens = merge_mertens.process(img_list)
res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
cv.imwrite("fusion_mertens.jpg", res_mertens_8bit)




#tonemapping



# Tonemap using Reinhard's method to obtain 24-bit color image
tonemapReinhard = cv.createTonemapReinhard(2.2, 0,0.1,0)
ldrReinhard_dev = tonemapReinhard.process(hdr_debevec)
cv.imwrite("ldr-Reinhard_dev.jpg", ldrReinhard_dev * 255)

ldrReinhard_rob = tonemapReinhard.process(hdr_robertson)
cv.imwrite("ldr-Reinhard_rob.jpg", ldrReinhard_rob * 255)


"""""

# Print or use the obtained response and weight functions
# print("Response Function:", responseDebevec)
# print("Weight Function:", weight_function)

# import cv2
# import numpy as np

# # Read images at different exposures and their corresponding exposure times
# img1 = cv2.imread('250.JPG')  # Replace 'img1.jpg' with your image file name
# img2 = cv2.imread('1000.JPG')  # Replace 'img2.jpg' with your image file name
# img3 = cv2.imread('4000.JPG')  # Replace 'img3.jpg' with your image file name
# img4 = cv2.imread('15000.JPG')
# exposure_times = np.array([0.25, 1.0, 4.0, 15.0])  # Exposure times for the images

# # Convert images to 32-bit floats
# img1 = np.float32(img1)
# img2 = np.float32(img2)
# img3 = np.float32(img3)
# img4 = np.float32(img4)

# # Merge images to create an HDR image
# merge_debevec = cv2.createMergeDebevec()
# hdr = merge_debevec.process([img1, img2, img3, img4], times=exposure_times)

# # Tonemap HDR image to convert it to a viewable format
# tonemap = cv2.createTonemapDurand(gamma=2.2)
# ldr = tonemap.process(hdr)


# # Save the HDR and tonemapped LDR images
# cv2.imwrite('result.hdr', hdr)
# cv2.imwrite('result_ldr.jpg', np.clip(ldr * 255, 0, 255).astype('uint8'))

