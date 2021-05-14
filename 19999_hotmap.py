import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import io
import cv2 as cv
#sio.savemat('result/'+str(global_step) + '_yabx_rslt.mat',
#                        {'real': np.array(real_target_data_np), 'pred': np.array(real_pred_data_np),
#                        'real_org': samples['target'], 'input': samples['datum'], 'mask': samples['mask']})
data = io.loadmat('19999_yabx_rslt.mat')

real=data["real"]
pred=data["pred"]
real_org=data["real_org"]
#input=data["input"]
mask=data["mask"]

print(real.shape)
print(pred.shape)
print(real_org.shape)
#print(input.shape)
#print(mask[10,:,:,0])

index=11
#pic=np.zeros(35,180)
sns.set()
pic_real=real[index, :, :, 0]
#pic = pred[index, :, :, 0] * mask[index, :, :, 0] + real[index, :, :, 0] * (1 - mask[index, :, :, 0])
pic=real[index, :, :, 0] * (1-mask[index, :, :, 0])
rectan=mask[index,:,:,0]
print(rectan.shape)


fig,ax=plt.subplots()
ax = sns.heatmap(pic)
label_x=plt.xlabel("Time(h)")
label_y=plt.ylabel("Detectors")
plt.setp(label_y, rotation=90, horizontalalignment='right')
plt.setp(label_x, rotation=0, horizontalalignment='left')
plt.xticks([0, 12, 24, 36, 48, 60, 72, 84, 96, 108,120,132,144,156,168,180], ['7:00', '8', '9', '10', '11', '12', '13', '14', '15','16','17','18','19','20','21','22:00'],rotation='horizontal')
plt.yticks([5, 10, 15, 20, 25, 30, 35], ['5', '10', '15', '20', '25', '30', '35'],rotation='horizontal')

pos=np.argwhere(rectan!=0)
start=(pos[0][1],pos[0][0])
length=0
width=0
for i in range(len(pos)-1):
    if (pos[i+1][0]-pos[i][0]>=2 or pos[i+1][1]-pos[i][1]>=2):
        length=pos[i][1]-start[0]+1
        width=pos[i][0]-start[1]+1
        print(start,length,width)
        ax.add_patch(
            plt.Rectangle(start,length,width,
                 color='greenyellow',
                 linewidth=1.5,
                 fill=False
             )
        )
        start=(pos[i+1][1],pos[i+1][0])

i=len(pos)-1
length=pos[i][1]-start[0]+1
width=pos[i][0]-start[1]+1                          
print(start,length,width)
ax.add_patch(
    plt.Rectangle(start,length,width,
         color='greenyellow',
         linewidth=1.5,
         fill=False
     )
)

#plt.savefig("./19999-real-mask-pred/" + str(index + 1) + '-mask填充图.eps', dpi=1000)
#plt.savefig("./19999-real-mask-pred/" + str(index + 1) + '-mask填充图.jpg', dpi=1000)
plt.savefig("xxx1.jpg" )
plt.show()
