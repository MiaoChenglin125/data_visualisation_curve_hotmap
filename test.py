import matplotlib.pyplot as plt
import seaborn as sns
from scipy import io

#sio.savemat('result/'+str(global_step) + '_yabx_rslt.mat',
#                        {'real': np.array(real_target_data_np), 'pred': np.array(real_pred_data_np),
#                        'real_org': samples['target'], 'input': samples['datum'], 'mask': samples['mask']})
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
#data1 = io.loadmat('.\\result_mat\\yabx\\linear\\CE_8999_yabx_rslt.mat')
data = io.loadmat('.\\result_mat\\yabx\\linear\\Cnn1_19999_yabx_rslt.mat')
data2 = io.loadmat('.\\result_mat\\yabx\\linear\\Cnn3_19999_yabx_rslt.mat')
data3 = io.loadmat('.\\result_mat\\yabx\\linear\\Cnnbranch3_19999_yabx_rslt.mat')
data4 = io.loadmat('.\\result_mat\\yabx\\linear\\Cnnbranch3_fc_19999_yabx_rslt.mat')
data5 = io.loadmat('.\\result_mat\\yabx\\linear\\our_model_local+global_19999_yabx_rslt.mat')

real=data["real"]
real_org=data["real_org"]
mask=data["mask"]

#pred1=data1["pred"]
pred=data["pred"]
pred2=data2["pred"]
pred3=data3["pred"]
pred4=data4["pred"]
pred5=data5["pred"]
print(pred.shape)

sns.set()
for index in range(36):
    for j in range(35):
        pic_real = real[index, j, :, 0]
        #pic1 = pred1[index, j, :, 0] * mask[index, j, :, 0] + real[index, j, :, 0] * (1 - mask[index, j, :, 0])
        pic = pred[index, j, :, 0] * mask[index, j, :, 0] + real[index, j, :, 0] * (1 - mask[index, j, :, 0])
        pic2 = pred2[index, j, :, 0] * mask[index, j, :, 0] + real[index, j, :, 0] * (1 - mask[index, j, :, 0])
        pic3 = pred3[index, j, :, 0] * mask[index, j, :, 0] + real[index, j, :, 0] * (1 - mask[index, j, :, 0])
        pic4 = pred4[index, j, :, 0] * mask[index, j, :, 0] + real[index, j, :, 0] * (1 - mask[index, j, :, 0])
        pic5 = pred5[index, j, :, 0] * mask[index, j, :, 0] + real[index, j, :, 0] * (1 - mask[index, j, :, 0])

        # plt.plot(pic[0],pic[1],label='prediction'),linestyle='--'
        if (mask[index, j, :, 0].any() == 1):
            fig, ax = plt.subplots()

            # 设置放大区间
            gap = np.where(mask[index, j, :, 0])
            # zone_left = 100
            # zone_right = 150
            xlim0 = gap[0][0] - 1
            xlim1 = gap[0][-1] + 1
            list = np.hstack(
                (   #pic1[gap[0]],
                    pic[gap[0]],
                    pic2[gap[0]],
                    pic3[gap[0]],
                    pic4[gap[0]],
                    pic5[gap[0]],
                    pic_real[gap[0]]))

            ylim0 = min(list) - 1
            ylim1 = max(list) + 1

            list_all = np.hstack(
                (   #pic1,
                    pic,
                    pic2,
                    pic3,
                    pic4,
                    pic5,
                    pic_real))

            ylim_down = min(list_all) - 5
            ylim_up = max(list_all) + 5

            tx0 = xlim0
            tx1 = xlim1
            ty0 = ylim0
            ty1 = ylim1
            sx = [tx0, tx1, tx1, tx0, tx0]
            sy = [ty0, ty0, ty1, ty1, ty0]

            #ax.plot(pic1, linewidth=1.5)
            ax.plot(pic, linewidth=1.5)
            ax.plot(pic2, linewidth=1.5)
            ax.plot(pic3, linewidth=1.5)
            ax.plot(pic4, linewidth=1.5)
            ax.plot(pic5, linewidth=3, color='yellow')
            ax.plot(pic_real, lw=2.5, color='lightskyblue')
            plt.xlabel('Time(h)')
            plt.ylabel('Speed(km/h)')


            if xlim1-xlim0<40 and ylim1-ylim0<20:
                ax.plot(sx, sy, "white",lw=1)

                if xlim1<108 and ylim0>(ylim_up+ylim_down)/2:

                    axins = ax.inset_axes((0.5, 0.05, 0.4, 0.33))

                elif xlim1<108 and ylim1<(ylim_up+ylim_down)/2:
                    axins = ax.inset_axes((0.5, 0.65, 0.4, 0.33))

                elif xlim0>72 and ylim0>(ylim_up+ylim_down)/2:
                    axins = ax.inset_axes((0.1, 0.05, 0.4, 0.33))

                elif xlim0>72 and ylim1<(ylim_up+ylim_down)/2:
                    axins = ax.inset_axes((0.1, 0.65, 0.4, 0.33))

                else:
                    axins = ax.inset_axes((0.5, 0.05, 0.4, 0.33))
                xy = (xlim0, ylim0)
                xy2 = (xlim0, ylim0)
                con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                                      axesA=axins, axesB=ax)
                axins.add_artist(con)

                xy = (xlim1, ylim0)
                xy2 = (xlim1, ylim0)
                con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                                      axesA=axins, axesB=ax)
                axins.add_artist(con)

                # 调整子坐标系的显示范围
                axins.set_xticks([])

                #axins.plot(pic1, linewidth=2)
                axins.plot(pic, linewidth=2)
                axins.plot(pic2, linewidth=2)
                axins.plot(pic3, linewidth=2)
                axins.plot(pic4, linewidth=2)
                axins.plot(pic5, linewidth=4.5, color='yellow')
                axins.plot(pic_real, lw=3.5, color='lightskyblue')
                if xlim0 < xlim1 and ylim0 < ylim1:
                    axins.set_xlim(xlim0, xlim1)
                    axins.set_ylim(ylim0, ylim1)

            plt.legend([  #'CE_8999_yabx',
                'Cnn1_yabx',
                'Cnn3_yabx',
                'Cnnbranch3_yabx',
                'Cnnbranch3_fc_yabx',
                'our_model_yabx',
                'Ground truth'])
            #plt.yticks([0, 22, 44, 66, 88, 110], ['0', '22', '44', '66', '88', '110'])
            plt.xticks([0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180],
                       ['7:00', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
                        '22:00'],
                       rotation='horizontal')
            plt.savefig("./19999-yabx-linear/" + str(index + 1) +"第"+ str(j+1)+'检测器对比.eps', dpi=1000)
            plt.savefig("./19999-yabx-linear/" + str(index + 1) +"第"+ str(j+1)+'检测器对比.jpg', dpi=1000)
            plt.show()

