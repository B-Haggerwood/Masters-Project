# This code is used to analyse the test data in comparison to the real and input data.

import Pk_library as PKL
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

import matplotlib
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times"],
})
matplotlib.rcParams.update({'font.size': 15})

#%%

BoxSize = 1000.0 # Mpc/h
MAS = 'CIC' # MAS used to create the image; 'NGP', 'CIC', 'TSC', 'PCS' o 'None'
threads = 1 # number of openmp threads

def untile(field):
    field = field[::2,:]
    field = field[:,::2]
    
    return field

def unnormalize(image):
  
  image = ((((image+1)/2)**2)*2)-1

  return image

def sync(fieldTrue, field):
    
    trueStart = fieldTrue[0]
    fieldStart = field[0]
    
    ratio = trueStart/fieldStart
    
    syncField = field * ratio
        
    return syncField

#%% Pk Single

fieldInputBase = np.load('Test_Comparison_New/Test_Input_9.npy').reshape([256,256])
fieldInputBase = untile(fieldInputBase)
fieldPredictBase = np.load('Test_Comparison_New/Test_Prediction_9.npy').reshape([256,256]) 
fieldTruthBase = np.load('Test_Comparison_New/Test_Truth_9.npy').reshape([256,256]) 

deltaInput = fieldInputBase
deltaPredict = fieldPredictBase
deltaTruth = fieldTruthBase

Pk2DInput = PKL.Pk_plane(deltaInput, BoxSize, MAS, threads)
Pk2DPredict = PKL.Pk_plane(deltaPredict, BoxSize, MAS, threads)
Pk2DTruth = PKL.Pk_plane(deltaTruth, BoxSize, MAS, threads)

kInput      = Pk2DInput.k      #k in h/Mpc
PkInput     = Pk2DInput.Pk     #Pk in (Mpc/h)^2
NmodesInput = Pk2DInput.Nmodes #Number of modes in the different k bins

kPredict      = Pk2DPredict.k      #k in h/Mpc
PkPredict     = Pk2DPredict.Pk     #Pk in (Mpc/h)^2
NmodesPredict = Pk2DPredict.Nmodes #Number of modes in the different k bins

kTruth      = Pk2DTruth.k      #k in h/Mpc
PkTruth     = Pk2DTruth.Pk     #Pk in (Mpc/h)^2
NmodesTruth = Pk2DTruth.Nmodes #Number of modes in the different k bins

PkInput = sync(PkTruth, PkInput)
PkPredict = sync(PkTruth, PkPredict)
    
ratioLow = PkInput / PkTruth[0:len(PkInput)]
ratio = PkPredict / PkTruth
base = PkTruth / PkTruth

kInput = kInput[0:62]
PkInput = PkInput[0:62]
kPredict = kPredict[0:126]
PkPredict = PkPredict[0:126]
kTruth = kTruth[0:126]
PkTruth = PkTruth[0:126]

fig0, ax = plt.subplots(2, 1, sharex=True)
plt.tight_layout(h_pad = -1, w_pad = -1)
fig0.set_size_inches(5, 5)

ax[0].loglog(kInput, PkInput)
ax[0].loglog(kPredict, PkPredict)
ax[0].loglog(kTruth, PkTruth)
ax[0].legend(['Test Input', 'Model Prediction', 'Ground Truth'], fontsize = 13)
ax[0].set_ylabel('Pk (Arbitrary Units)')
ax[0].axvline(x=0.80424, color='gray', linestyle='--', linewidth=0.6)
ax[0].axis([None, None, 0.01, 100])

ratioLow = PkInput / PkTruth[0:len(PkInput)]
ratio = PkPredict / PkTruth
base = PkTruth / PkTruth

ax[1].plot(kInput, ratioLow)
ax[1].plot(kPredict, ratio)
ax[1].plot(kTruth, base)
ax[1].set_xscale('log')
ax[1].axis([None, None, 0.7, 1.3])
ax[1].set_xlabel('k ($hMpc^{-1}$)')
ax[1].set_ylabel('Ratio')
fig0.savefig('./Power_Spec_Single.png', bbox_inches='tight')

#%% Pk Average

PkInput1 = np.zeros([62,])
PkPredict1 = np.zeros([126,])
PkTruth1 = np.zeros([126,])

allFieldsTrue = np.empty([126,10])

allFieldsPred = np.empty([126,10])

avgPredict = np.zeros([256,256])
avgTruth = np.zeros([256,256])

for n in range(0,10):
        
    fieldInput = np.load('Test_Comparison_New/Test_Input_%d.npy' % (n)).reshape([256,256])
    fieldInput = untile(fieldInput)
    fieldPredict = np.load('Test_Comparison_New/Test_Prediction_%d.npy' % (n)).reshape([256,256]) 
    fieldTruth = np.load('Test_Comparison_New/Test_Truth_%d.npy' % (n)).reshape([256,256]) 
    
    deltaInput = fieldInput
    deltaPredict = fieldPredict
    deltaTruth = fieldTruth
        
    avgPredict += deltaPredict
    avgTruth += deltaTruth

    Pk2DInput = PKL.Pk_plane(deltaInput, BoxSize, MAS, threads)
    Pk2DPredict = PKL.Pk_plane(deltaPredict, BoxSize, MAS, threads)
    Pk2DTruth = PKL.Pk_plane(deltaTruth, BoxSize, MAS, threads)
    
    kInput      = Pk2DInput.k      #k in h/Mpc
    PkInput     = Pk2DInput.Pk     #Pk in (Mpc/h)^2
    NmodesInput = Pk2DInput.Nmodes #Number of modes in the different k bins
    
    kPredict      = Pk2DPredict.k      #k in h/Mpc
    PkPredict     = Pk2DPredict.Pk     #Pk in (Mpc/h)^2
    NmodesPredict = Pk2DPredict.Nmodes #Number of modes in the different k bins
    
    kTruth      = Pk2DTruth.k      #k in h/Mpc
    PkTruth     = Pk2DTruth.Pk     #Pk in (Mpc/h)^2
    NmodesTruth = Pk2DTruth.Nmodes #Number of modes in the different k bins
    
    PkInput = sync(PkTruth, PkInput)
    PkPredict = sync(PkTruth, PkPredict)
    
    PkInput1 += PkInput[0:62]
    PkPredict1 += PkPredict[0:126]
    PkTruth1 += PkTruth[0:126]
    
    allFieldsTrue[:,n] = PkTruth[0:126]
    allFieldsPred[:,n] = PkPredict[0:126]

    

PkInput1 = PkInput1[0:62]
PkPredict1 = PkPredict1[0:126]
PkTruth1 = PkTruth1[0:126]

kInput = kInput[0:62]
kPredict = kPredict[0:126]
kTruth = kTruth[0:126]

PkInput1 /= 10 
PkPredict1 /= 10 
PkTruth1 /= 10 

avgPredict /= 10
avgTruth /= 10
    

fig1, ax1 = plt.subplots(2, 1, sharex=True)
plt.tight_layout(h_pad = -0.9, w_pad = -1)
fig1.set_size_inches(5, 5)

ax1[0].loglog(kInput, PkInput1)
ax1[0].loglog(kPredict, PkPredict1)
ax1[0].loglog(kTruth, PkTruth1)
ax1[0].axvline(x=0.80424, color='gray', linestyle='--', linewidth=0.6)
ax1[0].axis([None, None, 0.01, 100])
ax1[0].legend(['Test Input', 'Model Prediction', 'Ground Truth'], fontsize = 13)
ax1[0].set_ylabel('Pk (Arbitrary Unit)')

stddevTrue = np.std(allFieldsTrue, axis=1)

errorTrue = stddevTrue/((n+1)**0.5)


ax1[0].fill_between(kTruth, PkTruth1 - errorTrue, PkTruth1 + errorTrue, color='green', alpha=0.2)

stddevPred = np.std(allFieldsPred, axis=1)

errorPred = stddevPred/((n+1)**0.5)


ax1[0].fill_between(kTruth, PkPredict1 - errorPred, PkPredict1 + errorPred, color='orange', alpha=0.2)


diffValue = abs(PkTruth1 - PkPredict1)

sigmaDiffValue = ((errorTrue**2)+(errorPred**2))**0.5

sigma = diffValue/sigmaDiffValue


ax1[1].plot(kTruth, sigma)
ax1[1].plot(kTruth, np.ones([len(kTruth),]), 'k-')
ax1[1].plot(kTruth, np.ones([len(kTruth),]) + 1, 'k-')
ax1[1].plot(kTruth, np.ones([len(kTruth),]) + 4, 'k-')
ax1[1].set_xscale('log')
ax1[1].axis([None, None, 0, 7])
ax1[1].set_yticks([0,1,2,3,4,5,6])
ax1[1].set_xlabel('k ($hMpc^{-1}$)')
ax1[1].set_ylabel('$\sigma$')

plt.savefig('./Power_Spec_Average.png', bbox_inches='tight')

ratioLow1 = PkInput1 / PkTruth1[0:len(PkInput1)]
ratio1 = PkPredict1 / PkTruth1
base1 = PkTruth1 / PkTruth1

fig100, ax100 = plt.subplots(1, 1)
fig1.set_size_inches(5, 5)

ax100.plot(kInput, ratioLow1)
ax100.plot(kPredict, ratio1)
ax100.plot(kTruth, base1)
ax100.set_yticks([0.9, 0.95, 1.0, 1.05, 1.1])
ax100.set_xscale('log')
ax100.axis([None, None, 0.85, 1.15])
ax100.set_xlabel('k ($hMpc^{-1}$)')
ax100.set_ylabel('Ratio')

plt.show()

#%% Power Spec Percent Diff

print()
print()
print()

realStart = PkTruth1[0:89] 
fakeStart = PkPredict1[0:89] 

dStart = abs(realStart - fakeStart)

PercentDiffStart = round(np.mean((dStart/realStart)*100),2)

print('For LR Reproduction (k<0.568689): ', PercentDiffStart, '%')

realEnd = PkTruth[90:] 
fakeEnd = PkPredict[90:] 

dEnd= abs(realEnd - fakeEnd)

PercentDiffEnd = round(np.mean((dEnd[:-1]/realEnd[:-1])*100),2)

print('For New Data (k>0.568689): ', PercentDiffEnd, '%')

#%% Mass histogram

number = 100

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)


fieldInput = fieldInput.flatten()
fieldPredict = fieldPredict.flatten()
fieldTruth = fieldTruth.flatten()

histInput = np.histogram(fieldInput * 0.5 + 0.5, number, density = True)
histPredict = np.histogram(fieldPredict * 0.5 + 0.5, number, density = True)
histTruth = np.histogram(fieldTruth * 0.5 + 0.5, number, density = True)

xInput = np.linspace(np.min(fieldInput), np.max(fieldInput), number)
ax3.plot(xInput * 0.5 + 0.5, histInput[0])

xPredict = np.linspace(np.min(fieldPredict), np.max(fieldPredict), number)
ax3.plot(xPredict * 0.5 + 0.5, histPredict[0])

xTruth = np.linspace(np.min(fieldTruth), np.max(fieldTruth), number)
ax3.plot(xTruth * 0.5 + 0.5, histTruth[0])

plt.legend(['Test Input', 'Prediction', 'Ground Truth'])
plt.xlabel('Mass Value (Arbitrary Unit)')
plt.ylabel('Pixel Density')
plt.axis([0, 1, 0, None])
plt.tight_layout()
plt.savefig('./Mass_Histogram_Average.png', bbox_inches='tight')

#%% KL Divergence/JensenShannon Distance

js_pq = jensenshannon(histTruth[0], histPredict[0], base=2)

print('The JensenShannon divergence is: ', js_pq)
print('The JensenShannon distance is: ', js_pq**0.5)


#%% Investigating Corralation Matrix

CorrReal = np.corrcoef(avgTruth)
CorrPredict = np.corrcoef(avgPredict)

fig4 = plt.figure()

ax4 = fig4.add_subplot(121)
ax4.imshow(CorrReal)
plt.title('Real')
plt.tight_layout()

ax5 = fig4.add_subplot(122)
ax5.imshow(CorrPredict)
plt.title('Prediction')
plt.tight_layout()
plt.savefig('./Correlation_Matrix.png', bbox_inches='tight', dpi=300)

true_p_val = np.mean(CorrReal)
predict_p_val = np.mean(CorrPredict)


print('The true p-value of the correlation matrix is: ', round(true_p_val,5))
print('The prediction p-value of the correlation matrix is: ', round(predict_p_val,5))

diff = (abs(true_p_val-predict_p_val)/true_p_val)*100

print('The percentage difference between the p-values is: ', round(diff,2), '%')

plt.figure()

im = plt.imshow(abs(CorrReal-CorrPredict), vmin=0, vmax=0.5)
plt.colorbar(im)
plt.tight_layout()
plt.savefig('./Correlation_Noise.png', bbox_inches='tight', dpi=300)
plt.show()


#%% LOSS PLOTS

L1_loss_array = np.load('./L1_loss_array.npy')
gen_gan_loss_array = np.load('./gen_gan_loss_array.npy')
disc_loss_array = np.load('./disc_loss_array.npy')

x = np.linspace(1, len(L1_loss_array), len(L1_loss_array))

fig9, ax9 = plt.subplots(3, 1, sharex=True)
plt.tight_layout(h_pad = -1, w_pad = -1)
fig9.set_size_inches(5, 5)


ax9[0].plot(x, disc_loss_array)
ax9[0].set_ylabel('D')
ax9[0].set_yticks([1.0,1.1,1.2,1.3,1.4])

ax9[1].plot(x, L1_loss_array)
ax9[1].set_ylabel('L1')
ax9[1].axis([None, None, 0, 0.1])
ax9[1].set_yticks([0.01,0.05,0.09])

ax9[2].plot(x, gen_gan_loss_array)
ax9[2].set_xlabel('Epoch')
ax9[2].set_ylabel('G')
plt.savefig('./Loss_Plots.png', bbox_inches='tight')

