import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from matplotlib.patches import Patch

rc('text', usetex=True)

import matplotlib.pylab as pylab

params = {'legend.fontsize': '14',
          'figure.figsize': (20,10),
          'axes.labelsize': 30,
          'axes.titlesize': 32,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20}
pylab.rcParams.update(params)


fig, ax=plt.subplots(1,1)

sigma_SrTiO=0.800
sigma_SrZrO=0.905
sigma_SrSnO=0.716
sigma_SrHfO=0.689
sigma_SrMnO=0.685

ax.plot([1,2],[sigma_SrTiO,sigma_SrTiO], 'k-')
ax.plot([0.7,1],[sigma_SrTiO-0.025,sigma_SrTiO],'k:')
ax.text(-0.2,sigma_SrTiO-0.035,'SrTiO$_3$',size=20)

ax.plot([1,2],[sigma_SrZrO,sigma_SrZrO], 'k-')
ax.plot([0.7,1],[sigma_SrZrO-0.005,sigma_SrZrO],'k:')
ax.text(-0.2,sigma_SrZrO-0.015,'SrZrO$_3$',size=20)

ax.plot([1,2],[sigma_SrSnO,sigma_SrSnO], 'k-')
ax.plot([0.7,1],[sigma_SrSnO-0.015,sigma_SrSnO],'k:')
ax.text(-0.2,sigma_SrSnO-0.025,'SrSnO$_3$',size=20)

ax.plot([1,2],[sigma_SrHfO,sigma_SrHfO], 'k-')
ax.plot([0.7,1],[sigma_SrHfO-0.015,sigma_SrHfO],'k:')
ax.text(-0.2,sigma_SrHfO-0.025,'SrHfO$_3$',size=20)

ax.plot([1,2],[sigma_SrMnO,sigma_SrMnO], 'k-')
ax.plot([0.7,1],[sigma_SrMnO-0.035,sigma_SrMnO],'k:')
ax.text(-0.2,sigma_SrMnO-0.045,'SrMnO$_3$',size=20)

#=====================================================

# sigma_CaTiO=1.083
# sigma_CaZrO=1.436
# sigma_CaSnO=1.567
# sigma_CaHfO=1.302
# sigma_CaMnO=0.863
#
# ax.plot([1,2],[sigma_CaTiO,sigma_CaTiO], '-',c='#F0810F')
# ax.plot([0.8,1],[sigma_CaTiO-0.025,sigma_CaTiO],':',c='#F0810F')
# ax.text(0.2,sigma_CaTiO-0.035,'CaTiO$_3$',color='#F0810F')
#
# ax.plot([1,2],[sigma_CaMnO,sigma_CaMnO], '-',c='#F0810F')
# ax.plot([0.8,1],[sigma_CaMnO-0.025,sigma_CaMnO],':',c='#F0810F')
# ax.text(0.2,sigma_CaMnO-0.035,'CaMnO$_3$',color='#F0810F')
#
# ax.plot([1,2],[sigma_CaZrO,sigma_CaZrO], '-',c='#F0810F')
# ax.plot([0.8,1],[sigma_CaZrO-0.025,sigma_CaZrO],':',c='#F0810F')
# ax.text(0.2,sigma_CaZrO-0.035,'CaZrO$_3$',color='#F0810F')
#
# ax.plot([1,2],[sigma_CaSnO,sigma_CaSnO], '-',c='#F0810F')
# ax.plot([0.8,1],[sigma_CaSnO-0.025,sigma_CaSnO],':',c='#F0810F')
# ax.text(0.2,sigma_CaSnO-0.035,'CaSnO$_3$',color='#F0810F')
#
# ax.plot([1,2],[sigma_CaHfO,sigma_CaHfO], '-',c='#F0810F')
# ax.plot([0.8,1],[sigma_CaHfO-0.025,sigma_CaHfO],':',c='#F0810F')
# ax.text(0.2,sigma_CaHfO-0.035,'CaHfO$_3$',color='#F0810F')

#=============================
ax.plot([3,6],[0.758,0.758], '-',c='#CB0000',lw=4)
ax.plot([2,3],[sigma_SrZrO,0.758],':',lw=3,c='#CB0000')
ax.plot([2,3],[sigma_SrTiO,0.758],':',lw=3,c='#CB0000')
ax.text(3,0.74,'Sr(Ti$_{0.8}$Zr$_{0.2}$)O$_3$',color='#CB0000',size=20)
#=============================
ax.plot([7,10],[0.756,0.756], '-',c='#CB0000',lw=4)
ax.plot([2,7],[sigma_SrHfO,0.756],':',lw=3,c='#CB0000')
ax.plot([6,7],[0.758,0.756],':',lw=3,c='#CB0000')
ax.text(7,0.74,'Sr(Ti$_{0.6}$Zr$_{0.2}$Hf$_{0.2}$)O$_3$',color='#CB0000',size=20)
#=============================
ax.plot([11,14],[0.692,0.692], '-',c='#CB0000',lw=4)
ax.plot([2,11],[sigma_SrMnO,0.692],':',lw=3,c='#CB0000')
ax.plot([10,11],[0.756,0.692],':',lw=3,c='#CB0000')
ax.text(11,0.675,'Sr(Ti$_{0.4}$Zr$_{0.2}$Hf$_{0.2}$Mn$_{0.2}$)O$_3$',color='#CB0000',size=20)
#=============================
ax.plot([15,18],[0.547,0.547], '-',c='#CB0000',lw=4)
ax.plot([2,15],[sigma_SrSnO,0.547],':',lw=3,c='#CB0000')
ax.plot([14,15],[0.692,0.547],':',lw=3,c='#CB0000')
ax.text(15,0.53,'Sr(Ti$_{0.2}$Zr$_{0.2}$Hf$_{0.2}$Mn$_{0.2}$Sn$_{0.2}$)O$_3$',color='#CB0000',size=20)




#ax.plot([2.5,3],[0.705,0.705], '-', c='#2d4262',lw=4)
#ax.plot([2,2.5],[sigma_SrTiO,0.705], ':', c='#2d4262',lw=2)
#ax.plot([2,2.5],[sigma_CaTiO,0.705], ':', c='#2d4262',lw=2)
#ax.text(2.5,0.65,'(Sr$_{0.5}$Ca$_{0.5})$TiO$_3$',color='#2d4262',size=20)


#ax.plot([2.5,3],[0.552,0.552], '-', c='#2d4262',lw=4)
#ax.plot([2,2.5],[sigma_SrZrO,0.552], ':', c='#2d4262',lw=2)
#ax.plot([2,2.5],[sigma_CaZrO,0.552], ':', c='#2d4262',lw=2)
#ax.text(2.5,0.512,'(Sr$_{0.5}$Ca$_{0.5})$ZrO$_3$',color='#2d4262',size=20)

#s=0.493
#ax.plot([2.5,3],[s,s], '-', c='#2d4262',lw=4)
#ax.plot([2,2.5],[sigma_SrMnO,s], ':', c='#2d4262',lw=2)
#ax.plot([2,2.5],[sigma_CaMnO,s], ':', c='#2d4262',lw=2)
#ax.text(2.5,s-0.05,'(Sr$_{0.5}$Ca$_{0.5})$MnO$_3$',color='#2d4262',size=20)

#ax.plot([3.5,6],[0.840,0.840], '-', c='#cb0000',lw=4)
#ax.plot([3,3.5],[0.705,0.840], ':', c='#cb0000',lw=2)
#ax.plot([3,3.5],[0.552,0.840], ':', c='#cb0000',lw=2)
#ax.text(3.5,0.81,'(Sr$_{0.5}$Ca$_{0.5})$(Ti$_{0.8}$Zr$_{0.2}$)O$_3$',color='#cb0000',size=15)


#ax.plot([7.5,10],[0.639,0.639], '-', c='#cb0000',lw=4)
#ax.plot([6,7.5],[0.840,0.639], ':', c='#cb0000',lw=2)
#ax.plot([3,3.5],[0.552,0.840], ':', c='#cb0000',lw=2)
#ax.text(7.5,0.61,'(Sr$_{0.5}$Ca$_{0.5})$(Ti$_{0.6}$Zr$_{0.2}$Hf$_{0.2}$)O$_3$',color='#cb0000',size=15)


ax.plot([0.5],[0.55],'o',c='#ffd900',ms=67*1.42,alpha=0.6)
ax.text(0.25,0.54,'Ti',size=40)

ax.plot([2.5],[0.55],'o',c='#ffd900',ms=69*1.5,alpha=0.6)
ax.text(2.15,0.54,'Sn',size=40)

ax.plot([4.5],[0.55],'o',c='#ffd900',ms=71*1.51,alpha=0.6)
ax.text(4.225,0.54,'Hf',size=40)

ax.plot([6.5],[0.55],'o',c='#ffd900',ms=72*1.53,alpha=0.6)
ax.text(6.25,0.54,'Zr',size=40)

ax.plot([8.5],[0.55],'o',c='#ffd900',ms=83*1.55,alpha=0.6)
ax.text(8.15,0.54,'Mn',size=40)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

#ax.set_ylim([0.5,1.7])
ax.set_ylabel('$\\sigma^{(2)}$')
#ax.plot([-0.5,19],[1,1],'k--')
ax.set_xlim([-0.5,19])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().tick_left()
plt.tight_layout()
#plt.show()

plt.savefig('/Users/z3079335/Desktop/hep.pdf')