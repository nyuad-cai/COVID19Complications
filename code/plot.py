
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve, precision_recall_curve

def plot_calibration(label,true,predict, bins,set_name):

        """ This function plots the reliability curves based on the predictions"""

        colors=["#FF1F5B","#00CD6C","#009ADE","#AF58BA","#FFC61E","#F28522", "#A0B1BA"]

        plt.plot([0,1],[0,1], 'k--')
        count = 0
        for i, j, k in zip(label, true, predict):
            # slope,intercept=get_confidence(j, k, bins)
            fpr1, tpr1 = calibration_curve(j, k, n_bins=bins)
            text_a = label[count]
            plt.plot(tpr1, fpr1, label=text_a, color=colors[count])
            count = count +1

        ax = plt.gca()
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('1')
        ax.axis("on")
        plt.xlim(0,1)
        plt.ylim(0,1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor('white')
        plt.xlabel("Predicted Probablity")
        plt.ylabel("True Proability in each bin")
        plt.rcParams['figure.facecolor'] = 'none'



        plt.savefig('plots/Calibration Curvefinal'+set_name+'.pdf', format='pdf', dpi=1400,bbox_inches="tight")
        plt.show()

def plot_roc(label,true,predict,set_name):
        """ This function plots the ROC curves based on the predictions"""

        colors=["#FF1F5B", "#00CD6C", "#009ADE", "#AF58BA", "#FFC61E", "#F28522", "#A0B1BA"]

#
        plt.plot([0,1],[0,1], 'k--')
        count = 0
        for i, j, k in zip(label, true, predict):
            fpr1, tpr1,_ =  roc_curve(j, k)
            text_a = label[count]
            plt.plot(fpr1, tpr1, label=text_a ,color=colors[count])
            count = count +1

        ax = plt.gca()
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('1')
        ax.axis("on")
        plt.xlim(0,1)
        plt.ylim(0,1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor('white')
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")

        plt.rcParams['figure.facecolor'] = 'none'
#         plt.legend(fontsize=8)
        plt.savefig('plots/Roc_curve_final'+set_name+'.pdf', format='pdf', dpi=1400, bbox_inches="tight")

        plt.show()


def plot_PRC(label,true,predict,set_name):
        """ This function plots the PR curves based on the predictions"""

        count = 0
        colors=["#FF1F5B", "#00CD6C", "#009ADE", "#AF58BA", "#FFC61E", "#F28522", "#A0B1BA"]

        for i, j, k in zip(label, true, predict):
            precision, recall,_ =  precision_recall_curve(j, k)
            text_a = label[count]
            plt.plot(recall, precision, label=text_a, color=colors[count] )
            count = count +1

        ax = plt.gca()
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('1')
        ax.axis("on")
        plt.xlim(0,1)
        plt.ylim(0,1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor('white')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.rcParams['figure.facecolor'] = 'none'
        # plt.legend(loc = 1)
        # plt.legend(fontsize=8)
        plt.savefig('plots/PR_curve_final'+set_name+'.pdf', format='pdf', dpi=1400, bbox_inches="tight")

        plt.show()