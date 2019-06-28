import numpy as np
import matplotlib.pyplot as plt
import time

disWPPlist = [0.004530192322051672, 0.009747954194083279, 0.012751387990885899, 0.10905924470333368, 0.5285318494526505, 1.5423785744847043]
disWPP3list = [0.00018786022006641893, 0.0002405263420954613, 0.00024725835450047724, 0.0002886070727236045, 0.0004041537202577949, 0.00044650722420122525]
yErrorpplist = [0.013890180201837488, 0.01480815606447396, 0.022152074206704238, 0.059846402355113126, 0.776628273748261, 0.894524370866273]
yErrorpp3list = [0.010291153000042935, 0.010061539329193109, 0.01073366705337468, 0.010972335030792677, 0.011766735114489167, 0.012997530041471578]
x = [k for k in range(20, 31, 2)]
fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_yscale("log")
#
# ax.plot(x, disWPPlist, c='pink', linewidth=1, label='RRppLaplace')
# print(disWPPlist)
# print()
# ax.plot(x, disWPP3list, c='red', linewidth=1, label='RRppGaussion')
# #ax.plot(logy=True, legend=True)
# print(disWPP3list)
# print()
# ax_xlabel_text = ax.set_xlabel('Dimension', size=10, weight='bold')
# ax_ylabel_text = ax.set_ylabel('trainError', size=10, weight='bold')
# ax.set_title('Relationship between dimension $d$ and trainError', size=20, weight='bold')

bx = fig.add_subplot(111)
bx.set_yscale("log")
# bx.plot(x, yErrorlist, c='pink', linewidth=1, label='RR')
bx.plot(x, yErrorpplist, c='pink', linewidth=1, label='RRppLaplace')
print(yErrorpplist)
print()
bx.plot(x, yErrorpp3list, c='red', linewidth=1, label='RRppGaussion')
print(yErrorpp3list)
bx_xlabel_text = bx.set_xlabel('Dimension', size=10, weight='bold')
bx_ylabel_text = bx.set_ylabel('testError', size=10, weight='bold')
bx.set_title('Relationship between dimension $d$ and testError', size=20, weight='bold')

plt.legend()
# plt.ylim(0., 30.)
plt.show()