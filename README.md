# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py


Training printout logs on Sentiment(SST2):

Epoch 1, loss 31.408166498040764, train accuracy: 50.89%
Validation accuracy: 49.00%
Best Valid accuracy: 49.00%
Epoch 2, loss 31.000659087832346, train accuracy: 56.00%
Validation accuracy: 59.00%
Best Valid accuracy: 59.00%
Epoch 3, loss 30.931291191877772, train accuracy: 50.89%
Validation accuracy: 49.00%
Best Valid accuracy: 59.00%
Epoch 4, loss 30.75163751552338, train accuracy: 56.22%
Validation accuracy: 55.00%
Best Valid accuracy: 59.00%
Epoch 5, loss 30.641835368491712, train accuracy: 56.44%
Validation accuracy: 50.00%
Best Valid accuracy: 59.00%
Epoch 6, loss 30.469559428199354, train accuracy: 58.00%
Validation accuracy: 52.00%
Best Valid accuracy: 59.00%
Epoch 7, loss 30.02914179067206, train accuracy: 62.22%
Validation accuracy: 57.00%
Best Valid accuracy: 59.00%
Epoch 8, loss 29.790876125289792, train accuracy: 62.67%
Validation accuracy: 57.00%
Best Valid accuracy: 59.00%
Epoch 9, loss 29.57564415760404, train accuracy: 64.44%
Validation accuracy: 53.00%
Best Valid accuracy: 59.00%
Epoch 10, loss 28.912644474457174, train accuracy: 67.33%
Validation accuracy: 55.00%
Best Valid accuracy: 59.00%
Epoch 11, loss 28.662613441634193, train accuracy: 68.00%
Validation accuracy: 64.00%
Best Valid accuracy: 64.00%
Epoch 12, loss 28.454380470664823, train accuracy: 65.56%
Validation accuracy: 62.00%
Best Valid accuracy: 64.00%
Epoch 13, loss 27.610726514041797, train accuracy: 72.67%
Validation accuracy: 61.00%
Best Valid accuracy: 64.00%
Epoch 14, loss 27.07318747375787, train accuracy: 70.44%
Validation accuracy: 63.00%
Best Valid accuracy: 64.00%
Epoch 15, loss 26.42236505588852, train accuracy: 74.44%
Validation accuracy: 56.00%
Best Valid accuracy: 64.00%
Epoch 16, loss 25.997119332119947, train accuracy: 70.67%
Validation accuracy: 61.00%
Best Valid accuracy: 64.00%
Epoch 17, loss 25.256172659333576, train accuracy: 74.67%
Validation accuracy: 67.00%
Best Valid accuracy: 67.00%
Epoch 18, loss 24.91994693296895, train accuracy: 74.44%
Validation accuracy: 65.00%
Best Valid accuracy: 67.00%
Epoch 19, loss 24.241589688034818, train accuracy: 72.67%
Validation accuracy: 62.00%
Best Valid accuracy: 67.00%
Epoch 20, loss 23.307578066232022, train accuracy: 75.78%
Validation accuracy: 65.00%
Best Valid accuracy: 67.00%
Epoch 21, loss 23.050170455906027, train accuracy: 77.33%
Validation accuracy: 59.00%
Best Valid accuracy: 67.00%
Epoch 22, loss 22.533198196751233, train accuracy: 77.33%
Validation accuracy: 66.00%
Best Valid accuracy: 67.00%
Epoch 23, loss 21.281787579561772, train accuracy: 78.44%
Validation accuracy: 61.00%
Best Valid accuracy: 67.00%
Epoch 24, loss 21.171938470596263, train accuracy: 78.67%
Validation accuracy: 59.00%
Best Valid accuracy: 67.00%
Epoch 25, loss 19.91883283023371, train accuracy: 80.67%
Validation accuracy: 66.00%
Best Valid accuracy: 67.00%
Epoch 26, loss 19.49764505430426, train accuracy: 81.56%
Validation accuracy: 63.00%
Best Valid accuracy: 67.00%
Epoch 27, loss 19.217253419645605, train accuracy: 78.89%
Validation accuracy: 69.00%
Best Valid accuracy: 69.00%
Epoch 28, loss 18.626102672212713, train accuracy: 82.00%
Validation accuracy: 70.00%
Best Valid accuracy: 70.00%
Epoch 29, loss 18.255671653640785, train accuracy: 83.11%
Validation accuracy: 70.00%
Best Valid accuracy: 70.00%
Epoch 30, loss 17.904460777219054, train accuracy: 82.00%
Validation accuracy: 53.00%
Best Valid accuracy: 70.00%
Epoch 31, loss 17.81472611516503, train accuracy: 81.11%
Validation accuracy: 62.00%
Best Valid accuracy: 70.00%
Epoch 32, loss 16.52153691202468, train accuracy: 83.11%
Validation accuracy: 61.00%
Best Valid accuracy: 70.00%
Epoch 33, loss 16.151134809267027, train accuracy: 82.67%
Validation accuracy: 64.00%
Best Valid accuracy: 70.00%
Epoch 34, loss 17.02581126914103, train accuracy: 82.44%
Validation accuracy: 70.00%
Best Valid accuracy: 70.00%
Epoch 35, loss 15.255103996230496, train accuracy: 83.11%
Validation accuracy: 59.00%
Best Valid accuracy: 70.00%
Epoch 36, loss 15.113158442390581, train accuracy: 85.56%
Validation accuracy: 66.00%
Best Valid accuracy: 70.00%
Epoch 37, loss 15.828452002717391, train accuracy: 80.67%
Validation accuracy: 62.00%
Best Valid accuracy: 70.00%
Epoch 38, loss 14.634010528904764, train accuracy: 83.56%
Validation accuracy: 70.00%
Best Valid accuracy: 70.00%
Epoch 39, loss 14.54434847606406, train accuracy: 86.44%
Validation accuracy: 62.00%
Best Valid accuracy: 70.00%
Epoch 40, loss 14.209664587255213, train accuracy: 84.67%
Validation accuracy: 63.00%
Best Valid accuracy: 70.00%
Epoch 41, loss 13.010536260082816, train accuracy: 85.56%
Validation accuracy: 59.00%
Best Valid accuracy: 70.00%
Epoch 42, loss 13.80239318873317, train accuracy: 85.78%
Validation accuracy: 61.00%
Best Valid accuracy: 70.00%
Epoch 43, loss 13.272993811262815, train accuracy: 86.67%
Validation accuracy: 69.00%
Best Valid accuracy: 70.00%
Epoch 44, loss 12.677869589333385, train accuracy: 85.33%
Validation accuracy: 61.00%
Best Valid accuracy: 70.00%
Epoch 45, loss 13.198199119174587, train accuracy: 83.56%
Validation accuracy: 61.00%
Best Valid accuracy: 70.00%
Epoch 46, loss 13.523575536892192, train accuracy: 85.11%
Validation accuracy: 62.00%
Best Valid accuracy: 70.00%
Epoch 47, loss 13.126116187898038, train accuracy: 85.78%
Validation accuracy: 62.00%
Best Valid accuracy: 70.00%
Epoch 48, loss 12.36412658152404, train accuracy: 86.44%
Validation accuracy: 70.00%
Best Valid accuracy: 70.00%
Epoch 49, loss 12.122952374531454, train accuracy: 86.00%
Validation accuracy: 55.00%
Best Valid accuracy: 70.00%
Epoch 50, loss 11.602077376381883, train accuracy: 87.11%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 51, loss 11.821491613565916, train accuracy: 85.78%
Validation accuracy: 70.00%
Best Valid accuracy: 71.00%
Epoch 52, loss 12.0920457095797, train accuracy: 84.67%
Validation accuracy: 65.00%
Best Valid accuracy: 71.00%
Epoch 53, loss 12.659623411532644, train accuracy: 85.11%
Validation accuracy: 64.00%
Best Valid accuracy: 71.00%
Epoch 54, loss 11.306248705156381, train accuracy: 86.89%
Validation accuracy: 59.00%
Best Valid accuracy: 71.00%
Epoch 55, loss 11.835124484813962, train accuracy: 85.11%
Validation accuracy: 65.00%
Best Valid accuracy: 71.00%
Epoch 56, loss 10.734557318568505, train accuracy: 86.89%
Validation accuracy: 70.00%
Best Valid accuracy: 71.00%
Epoch 57, loss 11.673080265971647, train accuracy: 86.22%
Validation accuracy: 63.00%
Best Valid accuracy: 71.00%
Epoch 58, loss 10.449539025507416, train accuracy: 86.00%
Validation accuracy: 57.00%
Best Valid accuracy: 71.00%
Epoch 59, loss 10.771163524949733, train accuracy: 86.89%
Validation accuracy: 59.00%
Best Valid accuracy: 71.00%
Epoch 60, loss 10.842887938921294, train accuracy: 85.56%
Validation accuracy: 68.00%
Best Valid accuracy: 71.00%
Epoch 61, loss 10.58855294704028, train accuracy: 87.11%
Validation accuracy: 66.00%
Best Valid accuracy: 71.00%
Epoch 62, loss 10.836103372915556, train accuracy: 86.67%
Validation accuracy: 61.00%
Best Valid accuracy: 71.00%
Epoch 63, loss 10.0000734702855, train accuracy: 89.11%
Validation accuracy: 69.00%
Best Valid accuracy: 71.00%
Epoch 64, loss 10.652247624791025, train accuracy: 87.56%
Validation accuracy: 63.00%
Best Valid accuracy: 71.00%
Epoch 65, loss 9.674432528935077, train accuracy: 87.11%
Validation accuracy: 68.00%
Best Valid accuracy: 71.00%
Epoch 66, loss 11.889486404732263, train accuracy: 84.22%
Validation accuracy: 68.00%
Best Valid accuracy: 71.00%
Epoch 67, loss 10.011039241526646, train accuracy: 88.00%
Validation accuracy: 67.00%
Best Valid accuracy: 71.00%
Epoch 68, loss 9.210640204299942, train accuracy: 86.22%
Validation accuracy: 68.00%
Best Valid accuracy: 71.00%
Epoch 69, loss 10.1114462445481, train accuracy: 86.89%
Validation accuracy: 64.00%
Best Valid accuracy: 71.00%
Epoch 70, loss 10.257779083408222, train accuracy: 89.56%
Validation accuracy: 61.00%
Best Valid accuracy: 71.00%
Epoch 71, loss 9.266263406536405, train accuracy: 88.00%
Validation accuracy: 61.00%
Best Valid accuracy: 71.00%
Epoch 72, loss 8.503271748576022, train accuracy: 90.00%
Validation accuracy: 68.00%
Best Valid accuracy: 71.00%
Epoch 73, loss 9.416803100507261, train accuracy: 88.44%
Validation accuracy: 65.00%
Best Valid accuracy: 71.00%
Epoch 74, loss 9.32623638041214, train accuracy: 87.78%
Validation accuracy: 60.00%
Best Valid accuracy: 71.00%
Epoch 75, loss 9.188669760276534, train accuracy: 88.89%
Validation accuracy: 64.00%
Best Valid accuracy: 71.00%
Epoch 76, loss 10.24278349034291, train accuracy: 86.67%
Validation accuracy: 66.00%
Best Valid accuracy: 71.00%
Epoch 77, loss 9.796802289031515, train accuracy: 86.00%
Validation accuracy: 63.00%
Best Valid accuracy: 71.00%
Epoch 78, loss 11.250662775710351, train accuracy: 83.56%
Validation accuracy: 64.00%
Best Valid accuracy: 71.00%
Epoch 79, loss 10.12890639130546, train accuracy: 84.44%
Validation accuracy: 68.00%
Best Valid accuracy: 71.00%
Epoch 80, loss 9.452902744044568, train accuracy: 86.89%
Validation accuracy: 63.00%
Best Valid accuracy: 71.00%
Epoch 81, loss 9.365260763662176, train accuracy: 86.67%
Validation accuracy: 64.00%
Best Valid accuracy: 71.00%
Epoch 82, loss 9.625917851430378, train accuracy: 86.00%
Validation accuracy: 63.00%
Best Valid accuracy: 71.00%
Epoch 83, loss 9.945376288266884, train accuracy: 85.33%
Validation accuracy: 59.00%
Best Valid accuracy: 71.00%
Epoch 84, loss 9.786669648317913, train accuracy: 86.00%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%



Training logs on Digit classification (MNIST):

Epoch 1 loss 2.2879235998670375 valid acc 2/16
Epoch 1 loss 11.474414593549547 valid acc 2/16
Epoch 1 loss 11.455250769734775 valid acc 2/16
Epoch 1 loss 11.404410328489126 valid acc 2/16
Epoch 1 loss 11.400432792103928 valid acc 3/16
Epoch 1 loss 11.093286008544666 valid acc 5/16
Epoch 1 loss 10.79861015516857 valid acc 8/16
Epoch 1 loss 10.213128603487105 valid acc 8/16
Epoch 1 loss 9.046812731564891 valid acc 12/16
Epoch 1 loss 7.3582559606306575 valid acc 7/16
Epoch 1 loss 6.745538865856421 valid acc 13/16
Epoch 1 loss 5.659467387436033 valid acc 12/16
Epoch 1 loss 5.56271512221329 valid acc 15/16
Epoch 1 loss 5.1312887482122544 valid acc 14/16
Epoch 1 loss 5.181182289211036 valid acc 12/16
Epoch 1 loss 4.032982118241202 valid acc 9/16
Epoch 1 loss 4.968002537678377 valid acc 12/16
Epoch 1 loss 4.607525284293134 valid acc 13/16
Epoch 1 loss 3.8958951322621056 valid acc 14/16
Epoch 1 loss 3.4699133645111426 valid acc 13/16
Epoch 1 loss 3.1276822730713993 valid acc 12/16
Epoch 1 loss 5.506592323975358 valid acc 14/16
Epoch 1 loss 2.2692294521682173 valid acc 15/16
Epoch 1 loss 3.3303126465082142 valid acc 13/16
Epoch 1 loss 2.7382921299716467 valid acc 12/16
Epoch 1 loss 3.6826145734470153 valid acc 10/16
Epoch 1 loss 4.434619216105338 valid acc 15/16
Epoch 1 loss 2.3639468113345554 valid acc 14/16
Epoch 1 loss 2.808773958049395 valid acc 14/16
Epoch 1 loss 1.9623693584823858 valid acc 12/16
Epoch 1 loss 3.3343886412413273 valid acc 13/16
Epoch 1 loss 3.312943735219782 valid acc 13/16
Epoch 1 loss 1.974216539183252 valid acc 11/16
Epoch 1 loss 3.9943998640897465 valid acc 13/16
Epoch 1 loss 4.779670706625998 valid acc 14/16
Epoch 1 loss 3.3820343007193223 valid acc 13/16
Epoch 1 loss 2.456511509675864 valid acc 13/16
Epoch 1 loss 3.0173858044657553 valid acc 13/16
Epoch 1 loss 3.084502981960034 valid acc 14/16
Epoch 1 loss 2.3112729532300547 valid acc 14/16
Epoch 1 loss 1.8343863140780043 valid acc 13/16
Epoch 1 loss 3.610480318935864 valid acc 15/16
Epoch 1 loss 2.4019579421774573 valid acc 15/16
Epoch 1 loss 2.3713830679567782 valid acc 15/16
Epoch 1 loss 3.9021493071839624 valid acc 13/16
Epoch 1 loss 2.8669550388384915 valid acc 15/16
Epoch 1 loss 3.171835554385624 valid acc 15/16
Epoch 1 loss 2.895120676469145 valid acc 15/16
Epoch 1 loss 2.8749595199582756 valid acc 15/16
Epoch 1 loss 1.4595130111134105 valid acc 15/16
Epoch 1 loss 2.892136191714777 valid acc 14/16
Epoch 1 loss 2.3400613921697633 valid acc 14/16
Epoch 1 loss 2.7208860081186748 valid acc 14/16
Epoch 1 loss 2.138471405124199 valid acc 15/16
Epoch 1 loss 3.0461389150699953 valid acc 12/16
Epoch 1 loss 1.976548665494659 valid acc 14/16
Epoch 1 loss 2.7869999504039216 valid acc 14/16
Epoch 1 loss 2.1205302286004306 valid acc 13/16
Epoch 1 loss 2.434422412869582 valid acc 14/16
Epoch 1 loss 2.006434645071007 valid acc 12/16
Epoch 1 loss 2.801579778129973 valid acc 13/16
Epoch 1 loss 2.2174012242080154 valid acc 13/16
Epoch 1 loss 3.276592559107602 valid acc 14/16
Epoch 2 loss 0.31894541540507304 valid acc 13/16
Epoch 2 loss 2.150187519893469 valid acc 14/16
Epoch 2 loss 3.2384375848775697 valid acc 15/16
Epoch 2 loss 2.696787990955471 valid acc 15/16
Epoch 2 loss 1.9779437817932668 valid acc 14/16
Epoch 2 loss 1.6788511920847804 valid acc 15/16
Epoch 2 loss 2.5480608412782924 valid acc 14/16
Epoch 2 loss 3.2833915510560936 valid acc 14/16
Epoch 2 loss 2.852530563866188 valid acc 14/16
Epoch 2 loss 2.0181109912905666 valid acc 14/16
Epoch 2 loss 1.9071306916066537 valid acc 15/16
Epoch 2 loss 4.6493416278214 valid acc 15/16
Epoch 2 loss 2.810988578566051 valid acc 15/16
Epoch 2 loss 2.8482471711600863 valid acc 15/16
Epoch 2 loss 3.506667382584223 valid acc 15/16
Epoch 2 loss 2.1517416049202693 valid acc 14/16
Epoch 2 loss 3.666426434994901 valid acc 15/16
Epoch 2 loss 2.854842176685924 valid acc 14/16
Epoch 2 loss 2.666711176877575 valid acc 14/16
Epoch 2 loss 1.7583245274371342 valid acc 15/16
Epoch 2 loss 1.8122240564780345 valid acc 15/16
Epoch 2 loss 2.258842916630784 valid acc 14/16
Epoch 2 loss 0.9117130888660302 valid acc 15/16
Epoch 2 loss 2.345972278084914 valid acc 15/16
Epoch 2 loss 1.737980187556265 valid acc 13/16
Epoch 2 loss 2.1488538081179627 valid acc 14/16
Epoch 2 loss 2.397426440339121 valid acc 15/16
Epoch 2 loss 2.283223843975316 valid acc 15/16
Epoch 2 loss 2.1387041432513345 valid acc 14/16
Epoch 2 loss 1.5282664451620367 valid acc 15/16
Epoch 2 loss 2.21674925858865 valid acc 14/16
Epoch 2 loss 1.9581404610714577 valid acc 14/16
Epoch 2 loss 1.1556775531289296 valid acc 14/16
Epoch 2 loss 1.9062745558085232 valid acc 15/16
Epoch 2 loss 4.0142573799470025 valid acc 15/16
Epoch 2 loss 2.951810758252619 valid acc 14/16
Epoch 2 loss 1.3660670284510377 valid acc 13/16
Epoch 2 loss 1.8149330633384633 valid acc 15/16
Epoch 2 loss 2.3310875658788155 valid acc 15/16
Epoch 2 loss 1.7907464996717373 valid acc 15/16
Epoch 2 loss 1.5680927103927476 valid acc 15/16
Epoch 2 loss 1.9637365243650862 valid acc 15/16
Epoch 2 loss 1.4054091107104592 valid acc 15/16
Epoch 2 loss 1.8298494444871216 valid acc 15/16
Epoch 2 loss 2.1291957019480687 valid acc 15/16
Epoch 2 loss 1.5774857229315193 valid acc 15/16
Epoch 2 loss 2.402769297661712 valid acc 15/16
Epoch 2 loss 2.6888216205935533 valid acc 15/16
Epoch 2 loss 1.8679405860422054 valid acc 15/16
Epoch 2 loss 1.2239625564971162 valid acc 15/16
Epoch 2 loss 2.1963831698693235 valid acc 15/16
Epoch 2 loss 2.0127222171927053 valid acc 15/16
Epoch 2 loss 2.2097198627883468 valid acc 14/16
Epoch 2 loss 1.6577490449124095 valid acc 14/16
Epoch 2 loss 2.725608136587956 valid acc 15/16
Epoch 2 loss 1.3213126331223948 valid acc 14/16
Epoch 2 loss 1.8463250972826548 valid acc 13/16
Epoch 2 loss 1.6070802936521604 valid acc 15/16
Epoch 2 loss 2.1576655835948277 valid acc 14/16
Epoch 2 loss 2.102498558714833 valid acc 13/16
Epoch 2 loss 1.9039446827254956 valid acc 13/16
Epoch 2 loss 1.4753353151594546 valid acc 13/16
Epoch 2 loss 2.8241460749180214 valid acc 13/16
Epoch 3 loss 0.155234071718093 valid acc 14/16
Epoch 3 loss 1.8288749701294456 valid acc 14/16
Epoch 3 loss 3.2272402245227334 valid acc 15/16
Epoch 3 loss 2.6768425757917886 valid acc 15/16
Epoch 3 loss 2.1464167642429612 valid acc 15/16
Epoch 3 loss 1.3521382027222095 valid acc 15/16
Epoch 3 loss 2.054123862723446 valid acc 15/16
Epoch 3 loss 2.1421033211935625 valid acc 15/16
Epoch 3 loss 1.9365270151198561 valid acc 15/16
Epoch 3 loss 1.1776836248849456 valid acc 14/16
Epoch 3 loss 2.258585882798087 valid acc 14/16
Epoch 3 loss 2.992052818061377 valid acc 15/16
Epoch 3 loss 2.587684030236943 valid acc 15/16
Epoch 3 loss 2.3224605149643422 valid acc 15/16
Epoch 3 loss 3.0661228574993094 valid acc 15/16
Epoch 3 loss 2.0877379645481406 valid acc 13/16
Epoch 3 loss 2.694445524509785 valid acc 14/16
Epoch 3 loss 2.984665446504172 valid acc 14/16
Epoch 3 loss 2.4050836822834114 valid acc 15/16
Epoch 3 loss 1.5513629726209515 valid acc 15/16
Epoch 3 loss 1.7429534079377769 valid acc 13/16
Epoch 3 loss 1.908915478932632 valid acc 15/16
Epoch 3 loss 0.8635770825443406 valid acc 15/16
Epoch 3 loss 1.705116199681541 valid acc 15/16
Epoch 3 loss 1.1702266924506826 valid acc 14/16
Epoch 3 loss 1.6135316489683045 valid acc 15/16
Epoch 3 loss 1.6518769255369914 valid acc 15/16
Epoch 3 loss 1.4289060426199944 valid acc 15/16
Epoch 3 loss 1.4191947252744301 valid acc 15/16
Epoch 3 loss 0.8830171284266639 valid acc 15/16
Epoch 3 loss 1.5214667942458944 valid acc 15/16
Epoch 3 loss 1.600678278841256 valid acc 14/16
Epoch 3 loss 0.852832651578491 valid acc 14/16
Epoch 3 loss 2.0676530038770298 valid acc 15/16
Epoch 3 loss 3.4378180681990367 valid acc 15/16
Epoch 3 loss 2.043685775248951 valid acc 15/16
Epoch 3 loss 1.584652435706425 valid acc 14/16
Epoch 3 loss 1.4666477822207784 valid acc 14/16
Epoch 3 loss 1.7918199297105513 valid acc 15/16
Epoch 3 loss 1.2560571148886568 valid acc 15/16
Epoch 3 loss 1.198705889315141 valid acc 15/16
Epoch 3 loss 1.5838137567129733 valid acc 15/16
Epoch 3 loss 1.065608969876699 valid acc 15/16
Epoch 3 loss 0.9940984493751692 valid acc 15/16
Epoch 3 loss 1.47993174563265 valid acc 15/16
Epoch 3 loss 1.0622758811205228 valid acc 15/16
Epoch 3 loss 1.7541981565327833 valid acc 15/16
Epoch 3 loss 2.426428596096884 valid acc 15/16
Epoch 3 loss 1.2820317544093338 valid acc 14/16
Epoch 3 loss 0.8431653897966682 valid acc 15/16
Epoch 3 loss 1.8558210496423326 valid acc 14/16
Epoch 3 loss 1.4359073815115326 valid acc 15/16
Epoch 3 loss 2.326243775097542 valid acc 14/16
Epoch 3 loss 1.6901697360496237 valid acc 15/16
Epoch 3 loss 2.191846613862082 valid acc 15/16
Epoch 3 loss 1.075872603205578 valid acc 14/16
Epoch 3 loss 1.3278701119997034 valid acc 14/16
Epoch 3 loss 1.4749548234189587 valid acc 15/16
Epoch 3 loss 2.1068376191511597 valid acc 14/16
Epoch 3 loss 1.3929778602621234 valid acc 12/16
Epoch 3 loss 1.5041606719041893 valid acc 13/16
Epoch 3 loss 0.7958823457349777 valid acc 14/16
Epoch 3 loss 2.079863944085749 valid acc 15/16
Epoch 4 loss 0.07298259932294471 valid acc 15/16
Epoch 4 loss 1.5329720310292565 valid acc 15/16
Epoch 4 loss 2.073158287452355 valid acc 15/16
Epoch 4 loss 2.326994758402292 valid acc 15/16
Epoch 4 loss 1.252910449646278 valid acc 15/16
Epoch 4 loss 1.2762632688731812 valid acc 15/16
Epoch 4 loss 1.5671246443161873 valid acc 15/16
Epoch 4 loss 1.8278006775143802 valid acc 15/16
Epoch 4 loss 1.396113494109433 valid acc 15/16
Epoch 4 loss 0.8798858256637466 valid acc 14/16
Epoch 4 loss 1.5950291592315797 valid acc 15/16
Epoch 4 loss 2.8567917949074335 valid acc 15/16
Epoch 4 loss 2.2387958943520805 valid acc 15/16
Epoch 4 loss 2.8556186917727695 valid acc 15/16
Epoch 4 loss 2.218571003451007 valid acc 15/16
Epoch 4 loss 1.2968185884495593 valid acc 15/16
Epoch 4 loss 2.2120611170523254 valid acc 15/16
Epoch 4 loss 2.0782765487221395 valid acc 14/16
Epoch 4 loss 1.6249826555964768 valid acc 15/16
Epoch 4 loss 1.2475652212794328 valid acc 15/16
Epoch 4 loss 1.321065893381384 valid acc 14/16
Epoch 4 loss 1.1496887274339507 valid acc 15/16
Epoch 4 loss 0.40064450684666475 valid acc 15/16
Epoch 4 loss 1.292075419086091 valid acc 15/16
Epoch 4 loss 0.4549704361031231 valid acc 15/16
Epoch 4 loss 1.0868417929955392 valid acc 15/16
Epoch 4 loss 1.4524356860226342 valid acc 15/16
Epoch 4 loss 1.370627469042702 valid acc 15/16
Epoch 4 loss 1.19061793013302 valid acc 15/16
Epoch 4 loss 0.4694160735066528 valid acc 15/16
Epoch 4 loss 1.0953339063895298 valid acc 15/16
Epoch 4 loss 0.8457170966496006 valid acc 15/16
Epoch 4 loss 0.4713119885621463 valid acc 15/16
Epoch 4 loss 0.9472298349178638 valid acc 15/16
Epoch 4 loss 2.381244432431791 valid acc 15/16
Epoch 4 loss 1.266815152039535 valid acc 15/16
Epoch 4 loss 0.9158938960728111 valid acc 15/16
Epoch 4 loss 0.9276028903704845 valid acc 15/16
Epoch 4 loss 1.1182361924223403 valid acc 15/16
Epoch 4 loss 0.8562404306771999 valid acc 15/16
Epoch 4 loss 0.5153665146431596 valid acc 15/16
Epoch 4 loss 1.0750955617703506 valid acc 15/16
Epoch 4 loss 0.492106017456453 valid acc 15/16
Epoch 4 loss 0.6594941908541365 valid acc 15/16
Epoch 4 loss 1.6748284377931306 valid acc 15/16
Epoch 4 loss 0.6715552062807202 valid acc 15/16
Epoch 4 loss 1.5490400832665536 valid acc 15/16
Epoch 4 loss 1.858636553204176 valid acc 15/16
Epoch 4 loss 0.8508024610840013 valid acc 15/16
Epoch 4 loss 0.742491358411192 valid acc 15/16
Epoch 4 loss 0.7012269070143182 valid acc 15/16
Epoch 4 loss 1.2859535698926567 valid acc 15/16
Epoch 4 loss 1.8059257348841395 valid acc 14/16
Epoch 4 loss 1.6217319984087255 valid acc 14/16
Epoch 4 loss 2.1457403745195816 valid acc 14/16
Epoch 4 loss 1.1817210744772155 valid acc 14/16
Epoch 4 loss 0.9606561463906949 valid acc 14/16
Epoch 4 loss 0.8873285608956831 valid acc 15/16
Epoch 4 loss 1.2048994156426889 valid acc 15/16
Epoch 4 loss 1.4801559822218389 valid acc 12/16
Epoch 4 loss 1.0340740334712044 valid acc 15/16
Epoch 4 loss 1.037689338506848 valid acc 15/16
Epoch 4 loss 1.8613268889839882 valid acc 13/16
Epoch 5 loss 0.21053738682737366 valid acc 15/16
Epoch 5 loss 1.2484557532413216 valid acc 15/16
Epoch 5 loss 1.904975160239823 valid acc 15/16
Epoch 5 loss 1.5148071995689314 valid acc 15/16
Epoch 5 loss 1.045095184935377 valid acc 15/16
Epoch 5 loss 1.0587800630515756 valid acc 15/16
Epoch 5 loss 1.4360240949244747 valid acc 15/16
Epoch 5 loss 1.8329531901154417 valid acc 15/16
Epoch 5 loss 1.1025960487603532 valid acc 15/16
Epoch 5 loss 0.7797617643044935 valid acc 15/16
Epoch 5 loss 1.1277536530025283 valid acc 15/16
Epoch 5 loss 2.5731072636352383 valid acc 15/16
Epoch 5 loss 1.6184442097679623 valid acc 15/16
Epoch 5 loss 1.32337182400489 valid acc 15/16
Epoch 5 loss 1.577721602234284 valid acc 15/16
Epoch 5 loss 0.9167155017578819 valid acc 15/16
Epoch 5 loss 1.9841834239437637 valid acc 15/16
Epoch 5 loss 1.6670233709231108 valid acc 15/16
Epoch 5 loss 1.4508140859512655 valid acc 15/16
Epoch 5 loss 0.8302557095320623 valid acc 15/16
Epoch 5 loss 1.0704099001244023 valid acc 14/16
Epoch 5 loss 1.028779789810461 valid acc 15/16
Epoch 5 loss 0.49215406821808366 valid acc 15/16
Epoch 5 loss 0.689635651496069 valid acc 15/16
Epoch 5 loss 0.4729461196140495 valid acc 15/16
Epoch 5 loss 0.6854728848307223 valid acc 15/16
Epoch 5 loss 1.0585813150231584 valid acc 15/16
Epoch 5 loss 1.2979213057242933 valid acc 15/16
Epoch 5 loss 1.0140499615311214 valid acc 15/16
Epoch 5 loss 0.6792609830844305 valid acc 13/16
Epoch 5 loss 0.8965842644383235 valid acc 15/16
Epoch 5 loss 0.7782356016734809 valid acc 15/16
Epoch 5 loss 0.5065290267099245 valid acc 15/16
Epoch 5 loss 0.8475264188759042 valid acc 15/16
Epoch 5 loss 2.136149020966861 valid acc 15/16
Epoch 5 loss 1.025997176198827 valid acc 15/16
Epoch 5 loss 0.7592703771690689 valid acc 15/16
Epoch 5 loss 0.9616068780957702 valid acc 15/16
Epoch 5 loss 1.0073246934558102 valid acc 15/16
Epoch 5 loss 0.5970140942995019 valid acc 15/16
Epoch 5 loss 0.27899550495006686 valid acc 15/16
Epoch 5 loss 0.7851808064928969 valid acc 15/16
Epoch 5 loss 0.7166087312619187 valid acc 15/16
Epoch 5 loss 0.7515326558220419 valid acc 15/16
Epoch 5 loss 1.440314365001321 valid acc 15/16
Epoch 5 loss 0.44513828353538687 valid acc 15/16
Epoch 5 loss 1.226896028171896 valid acc 15/16
Epoch 5 loss 1.9761301076104236 valid acc 15/16
Epoch 5 loss 0.824663907327587 valid acc 15/16
Epoch 5 loss 0.7237676932810915 valid acc 15/16
Epoch 5 loss 0.6561897869094027 valid acc 15/16
Epoch 5 loss 0.8249187659701943 valid acc 15/16
Epoch 5 loss 1.1467475697159109 valid acc 14/16
Epoch 5 loss 1.2280916241268542 valid acc 14/16
Epoch 5 loss 1.340390338643242 valid acc 14/16
Epoch 5 loss 0.9146582984174245 valid acc 14/16
Epoch 5 loss 0.9411097658341725 valid acc 15/16
Epoch 5 loss 1.0000677499219632 valid acc 15/16
Epoch 5 loss 1.1852642182001354 valid acc 15/16
Epoch 5 loss 0.5108205063348514 valid acc 15/16
Epoch 5 loss 0.9971708472435277 valid acc 15/16
Epoch 5 loss 0.7795194098326216 valid acc 15/16
Epoch 5 loss 1.4663830965020912 valid acc 14/16
Epoch 6 loss 0.060342159942321394 valid acc 15/16
Epoch 6 loss 1.0923976960938102 valid acc 15/16
Epoch 6 loss 1.6774784354869063 valid acc 15/16
Epoch 6 loss 1.152861137346363 valid acc 15/16
Epoch 6 loss 1.081420866067663 valid acc 15/16
Epoch 6 loss 1.0500660065951528 valid acc 14/16
Epoch 6 loss 1.0564561765776102 valid acc 14/16
Epoch 6 loss 1.3094978901771073 valid acc 15/16
Epoch 6 loss 0.6661548626562835 valid acc 15/16
Epoch 6 loss 0.4906097850976761 valid acc 15/16
Epoch 6 loss 1.2851203991273346 valid acc 15/16
Epoch 6 loss 2.492842806400655 valid acc 15/16
Epoch 6 loss 1.3706246260244201 valid acc 15/16
Epoch 6 loss 1.1596078814046638 valid acc 14/16
Epoch 6 loss 1.0265651771744324 valid acc 15/16
Epoch 6 loss 1.3744412558263264 valid acc 15/16
Epoch 6 loss 1.6108231936844377 valid acc 14/16
Epoch 6 loss 1.3434146933015423 valid acc 15/16
Epoch 6 loss 1.264360969321919 valid acc 15/16
Epoch 6 loss 0.5075089548483955 valid acc 15/16
Epoch 6 loss 0.9964714098059453 valid acc 14/16
Epoch 6 loss 0.7340143291139043 valid acc 15/16
Epoch 6 loss 0.3084304640505495 valid acc 15/16
Epoch 6 loss 0.6463315285807628 valid acc 15/16
Epoch 6 loss 0.5571588184368081 valid acc 15/16
Epoch 6 loss 0.7114996376847953 valid acc 15/16
Epoch 6 loss 0.7310404724183003 valid acc 15/16
Epoch 6 loss 1.118780731123988 valid acc 15/16
Epoch 6 loss 0.8075910672605806 valid acc 16/16
Epoch 6 loss 0.4207454624172158 valid acc 15/16
Epoch 6 loss 0.8026281055343909 valid acc 15/16
Epoch 6 loss 0.8244148445962151 valid acc 15/16
Epoch 6 loss 0.5113199903853212 valid acc 15/16
Epoch 6 loss 0.8176158041041048 valid acc 15/16
Epoch 6 loss 1.5805785598559845 valid acc 15/16
Epoch 6 loss 0.8043487565360776 valid acc 15/16
Epoch 6 loss 0.5366471288068994 valid acc 15/16
Epoch 6 loss 0.6618050088919674 valid acc 15/16
Epoch 6 loss 0.9051477153031475 valid acc 15/16
Epoch 6 loss 0.4100596712107284 valid acc 15/16
Epoch 6 loss 0.35435800260633693 valid acc 16/16
Epoch 6 loss 0.6821666643317572 valid acc 15/16
Epoch 6 loss 0.33598876228184543 valid acc 15/16
Epoch 6 loss 0.41288362386628147 valid acc 15/16
Epoch 6 loss 1.5326769233110022 valid acc 15/16
Epoch 6 loss 0.34376975430172063 valid acc 16/16
Epoch 6 loss 1.143421464009999 valid acc 15/16
Epoch 6 loss 1.9683028290242817 valid acc 15/16
Epoch 6 loss 0.8072380035668922 valid acc 14/16
Epoch 6 loss 0.7470016606047352 valid acc 14/16
Epoch 6 loss 0.6901917626530011 valid acc 15/16
Epoch 6 loss 0.9212231746237375 valid acc 15/16
Epoch 6 loss 0.9599083146688975 valid acc 14/16
Epoch 6 loss 0.7639981799794415 valid acc 15/16
Epoch 6 loss 0.9464308710847984 valid acc 14/16
Epoch 6 loss 0.6654335916457619 valid acc 15/16
Epoch 6 loss 0.8565342036319546 valid acc 15/16
Epoch 6 loss 1.01532025204331 valid acc 15/16
Epoch 6 loss 1.0393055852090938 valid acc 15/16
Epoch 6 loss 0.7386363439771427 valid acc 14/16
Epoch 6 loss 0.8695417895357032 valid acc 15/16
Epoch 6 loss 0.5015969267267822 valid acc 15/16
Epoch 6 loss 1.2798908521484513 valid acc 12/16
Epoch 7 loss 0.02772567456887126 valid acc 15/16
Epoch 7 loss 0.9989161635309842 valid acc 15/16
Epoch 7 loss 1.1778865748133664 valid acc 15/16
Epoch 7 loss 0.5961201773178999 valid acc 15/16
Epoch 7 loss 0.4715295639578598 valid acc 15/16
Epoch 7 loss 0.8410301241982823 valid acc 14/16
Epoch 7 loss 1.4716175100570694 valid acc 15/16
Epoch 7 loss 1.3875019906901171 valid acc 15/16
Epoch 7 loss 0.4560407512930299 valid acc 15/16
Epoch 7 loss 0.3418008177985337 valid acc 15/16
Epoch 7 loss 1.0009975744292592 valid acc 14/16
Epoch 7 loss 1.8411895562626621 valid acc 15/16
Epoch 7 loss 1.2097413104231463 valid acc 15/16
Epoch 7 loss 1.3638122188566206 valid acc 15/16
Epoch 7 loss 0.720103159997293 valid acc 15/16
Epoch 7 loss 0.8070320345024438 valid acc 15/16
Epoch 7 loss 1.3122972143924412 valid acc 15/16
Epoch 7 loss 1.53160831550255 valid acc 15/16
Epoch 7 loss 1.0602282764961097 valid acc 15/16
Epoch 7 loss 0.6752473787473228 valid acc 15/16
Epoch 7 loss 0.9332552219085843 valid acc 14/16
Epoch 7 loss 0.7249587695368209 valid acc 15/16
Epoch 7 loss 0.3377635129780142 valid acc 15/16
Epoch 7 loss 0.5179225629714037 valid acc 15/16
Epoch 7 loss 0.581946990474193 valid acc 15/16
Epoch 7 loss 0.8047917202502747 valid acc 15/16
Epoch 7 loss 0.5820721039311713 valid acc 15/16
Epoch 7 loss 0.5520208116292493 valid acc 15/16
Epoch 7 loss 0.9215700555150527 valid acc 16/16
Epoch 7 loss 0.2612732692472023 valid acc 16/16
Epoch 7 loss 0.6659587589908724 valid acc 15/16
Epoch 7 loss 0.6451883008370773 valid acc 15/16
Epoch 7 loss 0.2778197138504348 valid acc 15/16
Epoch 7 loss 0.3935292615977908 valid acc 15/16
Epoch 7 loss 1.9383362597910083 valid acc 16/16
Epoch 7 loss 0.6227911698847426 valid acc 15/16
Epoch 7 loss 0.5344737748991253 valid acc 15/16
Epoch 7 loss 0.6354547758145201 valid acc 15/16
Epoch 7 loss 0.6205659582987593 valid acc 15/16
Epoch 7 loss 0.5632742577984593 valid acc 16/16
Epoch 7 loss 0.23492026436098318 valid acc 15/16
Epoch 7 loss 0.8245962351337297 valid acc 15/16
Epoch 7 loss 0.26042214559418003 valid acc 15/16
Epoch 7 loss 0.41810569001120523 valid acc 15/16
Epoch 7 loss 0.8640067734144669 valid acc 16/16
Epoch 7 loss 0.618277528888773 valid acc 16/16
Epoch 7 loss 1.2163408976065881 valid acc 15/16
Epoch 7 loss 1.5453842767193646 valid acc 16/16
Epoch 7 loss 0.7357372300257603 valid acc 15/16
Epoch 7 loss 0.5241728646828651 valid acc 15/16
Epoch 7 loss 0.43456026988990337 valid acc 15/16
Epoch 7 loss 0.6807220725735854 valid acc 15/16
Epoch 7 loss 0.8671444766699179 valid acc 14/16
Epoch 7 loss 0.5519329950127043 valid acc 15/16
Epoch 7 loss 0.8951495950614143 valid acc 15/16
Epoch 7 loss 0.5513508594853036 valid acc 15/16
Epoch 7 loss 0.7043379879602201 valid acc 15/16
Epoch 7 loss 0.8715192710400363 valid acc 15/16
Epoch 7 loss 0.7050999642140949 valid acc 15/16
Epoch 7 loss 0.6256585565451245 valid acc 15/16
Epoch 7 loss 0.7866169959874194 valid acc 15/16
Epoch 7 loss 0.4043015787333787 valid acc 14/16
Epoch 7 loss 0.9417928636047881 valid acc 14/16
Epoch 8 loss 0.025604336487630996 valid acc 14/16
Epoch 8 loss 0.8167952254122148 valid acc 15/16
Epoch 8 loss 1.2201351158861236 valid acc 15/16
Epoch 8 loss 0.6872137088091979 valid acc 15/16
Epoch 8 loss 0.4524521287284887 valid acc 15/16
Epoch 8 loss 0.7461524032150804 valid acc 15/16
Epoch 8 loss 0.98150154115253 valid acc 14/16
Epoch 8 loss 1.3821702522921564 valid acc 15/16
Epoch 8 loss 0.45998286938578126 valid acc 15/16
Epoch 8 loss 0.45200410625811427 valid acc 15/16
Epoch 8 loss 0.5365116220942441 valid acc 15/16
Epoch 8 loss 1.1042636285696874 valid acc 15/16
Epoch 8 loss 0.9880900008758702 valid acc 15/16
Epoch 8 loss 1.6857085170068045 valid acc 15/16
Epoch 8 loss 0.8576292517885666 valid acc 15/16
Epoch 8 loss 0.7356899877005694 valid acc 15/16
Epoch 8 loss 1.4719446487940564 valid acc 15/16
Epoch 8 loss 1.3953084997614331 valid acc 15/16
Epoch 8 loss 0.9209727886202788 valid acc 15/16
Epoch 8 loss 0.46628488070263135 valid acc 15/16
Epoch 8 loss 0.7595414572290191 valid acc 15/16
Epoch 8 loss 0.5125106543404906 valid acc 16/16
Epoch 8 loss 0.3881196932139095 valid acc 16/16
Epoch 8 loss 0.6887681932713363 valid acc 15/16
Epoch 8 loss 0.6161037550176219 valid acc 15/16
Epoch 8 loss 0.5247879733327172 valid acc 14/16
Epoch 8 loss 0.7374204269634276 valid acc 15/16
Epoch 8 loss 0.6672282432083669 valid acc 15/16
Epoch 8 loss 0.4695612364402502 valid acc 15/16
Epoch 8 loss 0.1493303781472311 valid acc 16/16
Epoch 8 loss 0.5332300159355012 valid acc 16/16
Epoch 8 loss 0.5828311767380774 valid acc 15/16
Epoch 8 loss 0.3669670737304566 valid acc 14/16
Epoch 8 loss 0.464681279178061 valid acc 15/16
Epoch 8 loss 1.9067184401805468 valid acc 16/16
Epoch 8 loss 0.4525489404169752 valid acc 16/16
Epoch 8 loss 0.5112876467747997 valid acc 16/16
Epoch 8 loss 0.5234612427836391 valid acc 16/16
Epoch 8 loss 0.6572903931752342 valid acc 15/16
Epoch 8 loss 0.3189271833785885 valid acc 16/16
Epoch 8 loss 0.09701195945525715 valid acc 16/16
Epoch 8 loss 0.8416499153208552 valid acc 15/16
Epoch 8 loss 0.6660633599930545 valid acc 15/16
Epoch 8 loss 0.23769817616328942 valid acc 15/16
Epoch 8 loss 1.1503739222138998 valid acc 16/16
Epoch 8 loss 0.14379581215298198 valid acc 16/16
Epoch 8 loss 1.1804943568860227 valid acc 15/16
Epoch 8 loss 1.5568743246480217 valid acc 15/16
Epoch 8 loss 0.615416938235118 valid acc 14/16
Epoch 8 loss 0.2759427980105191 valid acc 14/16
Epoch 8 loss 0.36532810459775866 valid acc 15/16
Epoch 8 loss 0.6543172891742546 valid acc 15/16
Epoch 8 loss 0.7834276978880559 valid acc 14/16
Epoch 8 loss 0.4976176911297939 valid acc 15/16
Epoch 8 loss 0.6977223700436628 valid acc 15/16
Epoch 8 loss 0.44413938302985817 valid acc 15/16
Epoch 8 loss 0.6723730896693327 valid acc 15/16
Epoch 8 loss 0.7693722381491401 valid acc 15/16
Epoch 8 loss 0.7124607538197512 valid acc 15/16
Epoch 8 loss 0.33757791563028716 valid acc 15/16
Epoch 8 loss 0.9389824801990784 valid acc 15/16
Epoch 8 loss 0.5154802768012239 valid acc 15/16
Epoch 8 loss 0.7658758156335015 valid acc 15/16
Epoch 9 loss 0.014922489014241813 valid acc 15/16
Epoch 9 loss 0.9866861750686031 valid acc 15/16
Epoch 9 loss 1.2619606436882294 valid acc 15/16
Epoch 9 loss 0.6045430534528533 valid acc 15/16
Epoch 9 loss 0.4563379864496794 valid acc 15/16
Epoch 9 loss 0.6329938878006052 valid acc 15/16
Epoch 9 loss 0.7437220023653249 valid acc 15/16
Epoch 9 loss 1.1568231405775573 valid acc 15/16
Epoch 9 loss 0.3935297143978154 valid acc 15/16
Epoch 9 loss 0.42301409175773386 valid acc 15/16
Epoch 9 loss 0.7791284286932596 valid acc 15/16
Epoch 9 loss 1.4658005699827257 valid acc 15/16
Epoch 9 loss 1.0798093819563097 valid acc 15/16
Epoch 9 loss 1.540217102740924 valid acc 15/16
Epoch 9 loss 0.5720516235710069 valid acc 15/16
Epoch 9 loss 0.6074858687614131 valid acc 15/16
Epoch 9 loss 1.622163383192346 valid acc 14/16
Epoch 9 loss 1.4626373644846846 valid acc 14/16
Epoch 9 loss 1.0137572942507282 valid acc 15/16
Epoch 9 loss 0.6034053425050807 valid acc 15/16
Epoch 9 loss 0.565560071147133 valid acc 15/16
Epoch 9 loss 0.3909797888921209 valid acc 15/16
Epoch 9 loss 0.2262366629437449 valid acc 15/16
Epoch 9 loss 0.2450894295538561 valid acc 15/16
Epoch 9 loss 1.097370435702954 valid acc 15/16
Epoch 9 loss 0.7320565558618163 valid acc 14/16
Epoch 9 loss 0.8299440603836993 valid acc 15/16
Epoch 9 loss 0.45422860187822617 valid acc 15/16
Epoch 9 loss 0.5494393420989439 valid acc 15/16
Epoch 9 loss 0.11529736499934479 valid acc 15/16
Epoch 9 loss 0.637640505823116 valid acc 15/16
Epoch 9 loss 0.21053407550940445 valid acc 15/16
Epoch 9 loss 0.5552431311744657 valid acc 15/16
Epoch 9 loss 0.18547683164527123 valid acc 15/16
Epoch 9 loss 1.1748903696511632 valid acc 15/16
Epoch 9 loss 0.45473078803358136 valid acc 15/16
Epoch 9 loss 0.40396066069745723 valid acc 15/16
Epoch 9 loss 0.3040532762690391 valid acc 15/16
Epoch 9 loss 0.6197068069189421 valid acc 15/16
Epoch 9 loss 0.42241016627521694 valid acc 15/16
Epoch 9 loss 0.23738269521916522 valid acc 15/16
Epoch 9 loss 0.6878582199417211 valid acc 15/16
Epoch 9 loss 0.38099246113657775 valid acc 15/16
Epoch 9 loss 0.4461681035161498 valid acc 15/16
Epoch 9 loss 1.3976279243390721 valid acc 15/16
Epoch 9 loss 0.09146870849123306 valid acc 15/16
Epoch 9 loss 0.9739653109056436 valid acc 15/16
Epoch 9 loss 1.555224276714045 valid acc 15/16
Epoch 9 loss 0.8389589038143574 valid acc 15/16
Epoch 9 loss 0.5090610843403376 valid acc 15/16
Epoch 9 loss 0.2603800681134956 valid acc 15/16
Epoch 9 loss 0.4276818104301979 valid acc 15/16
Epoch 9 loss 0.8327061547139017 valid acc 15/16
Epoch 9 loss 0.39637815032885626 valid acc 15/16
Epoch 9 loss 0.39797770421458184 valid acc 15/16
Epoch 9 loss 1.1148506548961197 valid acc 15/16
Epoch 9 loss 0.7735067297182832 valid acc 15/16
Epoch 9 loss 0.5621455104936104 valid acc 15/16
Epoch 9 loss 0.5835682308219637 valid acc 15/16
Epoch 9 loss 0.2771385098103316 valid acc 15/16
Epoch 9 loss 0.6291909060358278 valid acc 15/16
Epoch 9 loss 0.508238829160764 valid acc 15/16
Epoch 9 loss 1.2876534363674816 valid acc 15/16
Epoch 10 loss 0.0408304510395569 valid acc 15/16
Epoch 10 loss 0.5372807129011387 valid acc 15/16
Epoch 10 loss 1.2463188577659479 valid acc 15/16
Epoch 10 loss 0.5808555313979662 valid acc 15/16
Epoch 10 loss 0.2588378243188887 valid acc 15/16
Epoch 10 loss 0.6552958742504291 valid acc 15/16
Epoch 10 loss 0.7584968136473637 valid acc 15/16
Epoch 10 loss 0.9850930866411911 valid acc 16/16
Epoch 10 loss 0.4360154508725448 valid acc 15/16
Epoch 10 loss 0.5019197753444813 valid acc 15/16
Epoch 10 loss 0.8338373836312669 valid acc 15/16
Epoch 10 loss 1.1716701402132028 valid acc 15/16
Epoch 10 loss 0.6214432222327811 valid acc 15/16
Epoch 10 loss 1.120918372531541 valid acc 15/16
Epoch 10 loss 0.5738802212596146 valid acc 15/16
Epoch 10 loss 0.7254154684369052 valid acc 15/16
Epoch 10 loss 1.446620849923324 valid acc 14/16
Epoch 10 loss 1.2820884852390653 valid acc 14/16
Epoch 10 loss 0.838362984942473 valid acc 15/16
Epoch 10 loss 0.521941920698513 valid acc 15/16
Epoch 10 loss 0.8920000715306322 valid acc 15/16
Epoch 10 loss 0.20367772209322266 valid acc 15/16
Epoch 10 loss 0.19781623288129419 valid acc 15/16
Epoch 10 loss 0.2592855170608293 valid acc 15/16
Epoch 10 loss 0.6585343591825521 valid acc 15/16
Epoch 10 loss 0.6369597957052611 valid acc 15/16
Epoch 10 loss 0.518366463868725 valid acc 15/16
Epoch 10 loss 0.39779250254531284 valid acc 15/16
Epoch 10 loss 0.7118597728001778 valid acc 15/16
Epoch 10 loss 0.12633121399106725 valid acc 15/16
Epoch 10 loss 0.8130696547430729 valid acc 15/16
Epoch 10 loss 0.26966613142645857 valid acc 15/16
Epoch 10 loss 0.32678794427554186 valid acc 15/16
Epoch 10 loss 0.1915902749989023 valid acc 15/16
Epoch 10 loss 1.632615826130634 valid acc 16/16
Epoch 10 loss 0.21643099212781197 valid acc 15/16
Epoch 10 loss 0.34466460370412966 valid acc 16/16
Epoch 10 loss 0.32550245930814486 valid acc 15/16
Epoch 10 loss 0.5250413880219604 valid acc 15/16
Epoch 10 loss 0.30635338745220964 valid acc 15/16
Epoch 10 loss 0.12591264256108836 valid acc 15/16
Epoch 10 loss 0.5085959033180489 valid acc 15/16
Epoch 10 loss 0.2225892043078872 valid acc 15/16
Epoch 10 loss 0.2926506822565237 valid acc 15/16
Epoch 10 loss 0.7976219626323168 valid acc 15/16
Epoch 10 loss 0.1118755773902062 valid acc 15/16
Epoch 10 loss 1.3700377347821595 valid acc 14/16
Epoch 10 loss 1.7776684441520454 valid acc 15/16
Epoch 10 loss 0.4975993977497844 valid acc 16/16
Epoch 10 loss 0.25278608152539256 valid acc 15/16
Epoch 10 loss 0.2856300112182027 valid acc 15/16
Epoch 10 loss 0.31130666159788967 valid acc 16/16
Epoch 10 loss 0.6820925179850846 valid acc 15/16
Epoch 10 loss 0.49531041604453835 valid acc 15/16
Epoch 10 loss 0.575664309915267 valid acc 15/16
Epoch 10 loss 0.4354555103940175 valid acc 15/16
Epoch 10 loss 0.46567279437834636 valid acc 15/16
Epoch 10 loss 0.772774072388975 valid acc 15/16
Epoch 10 loss 0.2934104347135424 valid acc 15/16
Epoch 10 loss 0.28793048131389837 valid acc 15/16
Epoch 10 loss 0.7628845341701971 valid acc 15/16
Epoch 10 loss 0.17367412613071415 valid acc 15/16
Epoch 10 loss 0.8772908902757354 valid acc 15/16
Epoch 11 loss 0.00932373526815955 valid acc 15/16
Epoch 11 loss 0.42944038219160297 valid acc 15/16
Epoch 11 loss 0.8223282652108569 valid acc 15/16
Epoch 11 loss 0.3759071262460023 valid acc 15/16
Epoch 11 loss 0.4008723323667958 valid acc 14/16
Epoch 11 loss 0.5783359224329883 valid acc 15/16
Epoch 11 loss 1.0619143698434432 valid acc 14/16
Epoch 11 loss 1.4748555208211356 valid acc 15/16
Epoch 11 loss 0.3437956827160192 valid acc 15/16
Epoch 11 loss 0.45587759806436556 valid acc 15/16
Epoch 11 loss 0.6585655869771737 valid acc 15/16
Epoch 11 loss 1.3565781727580841 valid acc 15/16
Epoch 11 loss 0.5224091471239529 valid acc 15/16
Epoch 11 loss 0.8852248961839991 valid acc 15/16
Epoch 11 loss 0.9163868191656898 valid acc 15/16
Epoch 11 loss 0.6876714066525793 valid acc 15/16
Epoch 11 loss 1.3245286011050783 valid acc 14/16
Epoch 11 loss 1.2896534290219526 valid acc 14/16
Epoch 11 loss 0.8457772000544053 valid acc 15/16
Epoch 11 loss 0.3678379333822294 valid acc 15/16
Epoch 11 loss 0.5660762271022218 valid acc 16/16
Epoch 11 loss 0.135289265736595 valid acc 16/16
Epoch 11 loss 0.1536634314173606 valid acc 15/16
Epoch 11 loss 0.12368322756802608 valid acc 15/16
Epoch 11 loss 0.18097135403950715 valid acc 15/16
Epoch 11 loss 0.43729630824125965 valid acc 15/16
Epoch 11 loss 0.4224690263544112 valid acc 15/16
Epoch 11 loss 0.2916953639116299 valid acc 15/16
Epoch 11 loss 0.6840875101745043 valid acc 15/16
Epoch 11 loss 0.09008833361652047 valid acc 15/16
Epoch 11 loss 0.55992287241889 valid acc 15/16
Epoch 11 loss 0.4041533535178898 valid acc 15/16
Epoch 11 loss 0.3691304189351279 valid acc 15/16
Epoch 11 loss 0.1887904978188914 valid acc 15/16
Epoch 11 loss 1.2846434647201819 valid acc 16/16
Epoch 11 loss 0.23346772348039124 valid acc 16/16
Epoch 11 loss 0.43973775230542705 valid acc 16/16
Epoch 11 loss 0.14788424905686148 valid acc 16/16
Epoch 11 loss 0.4265584314221016 valid acc 15/16
Epoch 11 loss 0.16828602058668712 valid acc 15/16
Epoch 11 loss 0.06198488589886364 valid acc 15/16
Epoch 11 loss 0.6704096270483448 valid acc 15/16
Epoch 11 loss 0.35852492653160406 valid acc 15/16
Epoch 11 loss 0.23667819026533576 valid acc 15/16
Epoch 11 loss 0.748486864990277 valid acc 16/16
Epoch 11 loss 0.12562039628781957 valid acc 16/16
Epoch 11 loss 0.8611536484764353 valid acc 16/16
Epoch 11 loss 1.3804925909778165 valid acc 16/16
Epoch 11 loss 0.477610572677748 valid acc 16/16
Epoch 11 loss 0.1980422740648009 valid acc 15/16
Epoch 11 loss 0.24791596552279782 valid acc 15/16
Epoch 11 loss 0.20559863910523837 valid acc 15/16
Epoch 11 loss 0.4708416073552817 valid acc 15/16
Epoch 11 loss 0.3594651941766186 valid acc 15/16
Epoch 11 loss 0.7242719187946206 valid acc 15/16
Epoch 11 loss 0.332127920851399 valid acc 15/16
Epoch 11 loss 0.563815239741784 valid acc 15/16
Epoch 11 loss 0.6678379587874186 valid acc 15/16
Epoch 11 loss 0.28429375130472456 valid acc 15/16
Epoch 11 loss 0.19696949224966542 valid acc 15/16
Epoch 11 loss 0.3182741763410874 valid acc 15/16
Epoch 11 loss 0.4028367158426178 valid acc 15/16
Epoch 11 loss 0.7737726903409707 valid acc 15/16
Epoch 12 loss 0.00916717479757706 valid acc 16/16
Epoch 12 loss 0.9157510784041234 valid acc 13/16
Epoch 12 loss 1.387350578610425 valid acc 15/16
Epoch 12 loss 0.3173745145317076 valid acc 14/16
Epoch 12 loss 0.28497893209574415 valid acc 15/16
Epoch 12 loss 0.5428709496100748 valid acc 15/16
Epoch 12 loss 0.7857994078495076 valid acc 15/16
Epoch 12 loss 0.8769382451259113 valid acc 16/16
Epoch 12 loss 0.6211248126060623 valid acc 14/16
Epoch 12 loss 0.44511045181894277 valid acc 14/16
Epoch 12 loss 0.7340475593512747 valid acc 14/16
Epoch 12 loss 1.12070990429954 valid acc 16/16
Epoch 12 loss 1.398995675482576 valid acc 15/16
Epoch 12 loss 0.8854013437966426 valid acc 15/16
Epoch 12 loss 0.8931800072117838 valid acc 16/16
Epoch 12 loss 0.5928669534033856 valid acc 14/16
Epoch 12 loss 1.4837108788306808 valid acc 14/16
Epoch 12 loss 0.5021646365411887 valid acc 14/16
Epoch 12 loss 0.7519322535918976 valid acc 15/16
Epoch 12 loss 0.5272117467261261 valid acc 14/16
Epoch 12 loss 0.9186897080960301 valid acc 15/16
Epoch 12 loss 0.37414889670892865 valid acc 16/16
Epoch 12 loss 0.15500682297967447 valid acc 16/16
Epoch 12 loss 0.5338212478392774 valid acc 16/16
Epoch 12 loss 0.1996794390081651 valid acc 16/16
Epoch 12 loss 0.6201072111448565 valid acc 15/16
Epoch 12 loss 0.3559985548142429 valid acc 15/16
Epoch 12 loss 0.3921496493165641 valid acc 15/16
Epoch 12 loss 0.6267858564110549 valid acc 14/16
Epoch 12 loss 0.6564985252539624 valid acc 15/16
Epoch 12 loss 0.5954026790620275 valid acc 15/16
Epoch 12 loss 0.6122212340149734 valid acc 15/16
Epoch 12 loss 0.6617093794658444 valid acc 15/16
Epoch 12 loss 0.947209625475059 valid acc 15/16
Epoch 12 loss 1.3437899110700544 valid acc 15/16
Epoch 12 loss 0.8789017808803148 valid acc 16/16
Epoch 12 loss 0.4437516199558404 valid acc 15/16
Epoch 12 loss 0.3480429851911772 valid acc 16/16
Epoch 12 loss 0.6708367708743901 valid acc 16/16
Epoch 12 loss 1.006039957264476 valid acc 16/16
Epoch 12 loss 0.4496681313942752 valid acc 16/16
Epoch 12 loss 0.5268377484059401 valid acc 16/16
Epoch 12 loss 0.541287471400175 valid acc 15/16
Epoch 12 loss 0.17084110872344382 valid acc 15/16
Epoch 12 loss 0.5196122279665398 valid acc 15/16
Epoch 12 loss 0.11096557272089408 valid acc 15/16
Epoch 12 loss 1.0890767787152722 valid acc 15/16
Epoch 12 loss 0.7739538404406083 valid acc 15/16
Epoch 12 loss 0.5682432323440652 valid acc 16/16
Epoch 12 loss 0.38909698639439505 valid acc 15/16
Epoch 12 loss 0.36350424607105264 valid acc 15/16
Epoch 12 loss 0.7023492477147848 valid acc 15/16
Epoch 12 loss 1.2046297961611399 valid acc 15/16
Epoch 12 loss 0.4606246542270967 valid acc 15/16
Epoch 12 loss 0.5886264906064582 valid acc 15/16
Epoch 12 loss 0.694525572869805 valid acc 15/16
Epoch 12 loss 0.6468177387261274 valid acc 14/16
Epoch 12 loss 0.6919377006875774 valid acc 14/16
Epoch 12 loss 0.5580367792604528 valid acc 15/16
Epoch 12 loss 0.1998173854096525 valid acc 15/16
Epoch 12 loss 0.35070907887366015 valid acc 15/16
Epoch 12 loss 0.6500526078214716 valid acc 15/16
Epoch 12 loss 0.7885428877601397 valid acc 16/16
Epoch 13 loss 0.02935408663553085 valid acc 16/16
Epoch 13 loss 0.9285755111368827 valid acc 14/16
Epoch 13 loss 1.041133831910975 valid acc 15/16
Epoch 13 loss 0.41306666176379114 valid acc 15/16
Epoch 13 loss 0.4133783310288163 valid acc 14/16
Epoch 13 loss 0.570608090434223 valid acc 15/16
Epoch 13 loss 1.3631335560028792 valid acc 16/16
Epoch 13 loss 0.7838554495440083 valid acc 16/16
Epoch 13 loss 0.3699266367497937 valid acc 16/16
Epoch 13 loss 0.31909639820531543 valid acc 15/16
Epoch 13 loss 0.6669311989239819 valid acc 15/16
Epoch 13 loss 0.7939247920068415 valid acc 16/16
Epoch 13 loss 1.1801551461045146 valid acc 15/16
Epoch 13 loss 0.9653719265416619 valid acc 13/16
Epoch 13 loss 1.3050566069684506 valid acc 16/16
Epoch 13 loss 0.8355413590278699 valid acc 14/16
Epoch 13 loss 1.2875342222363566 valid acc 13/16
Epoch 13 loss 0.6614104847645703 valid acc 14/16
Epoch 13 loss 0.712412270546757 valid acc 15/16
Epoch 13 loss 0.27443408918236195 valid acc 14/16
Epoch 13 loss 0.544547869305183 valid acc 16/16
Epoch 13 loss 0.21521299490717094 valid acc 16/16
Epoch 13 loss 0.1519596011865132 valid acc 16/16
Epoch 13 loss 0.44208064755182125 valid acc 16/16
Epoch 13 loss 0.45839816443877157 valid acc 15/16
Epoch 13 loss 0.3424260654473707 valid acc 14/16
Epoch 13 loss 0.4337987553760022 valid acc 15/16
Epoch 13 loss 0.44310968476373824 valid acc 15/16
Epoch 13 loss 0.6434618731578164 valid acc 15/16
Epoch 13 loss 0.14901385863045274 valid acc 15/16
Epoch 13 loss 0.28372596608672995 valid acc 14/16
Epoch 13 loss 0.33243645009083517 valid acc 15/16
Epoch 13 loss 0.3229213787140866 valid acc 16/16
Epoch 13 loss 0.8973306923346452 valid acc 16/16
Epoch 13 loss 1.5129798641624475 valid acc 15/16
Epoch 13 loss 0.37003960251300716 valid acc 16/16
Epoch 13 loss 0.3372193243110342 valid acc 15/16
Epoch 13 loss 0.26833325954779247 valid acc 15/16
Epoch 13 loss 0.7452674062703801 valid acc 16/16
Epoch 13 loss 0.682344365748232 valid acc 16/16
Epoch 13 loss 0.35026029673012515 valid acc 16/16
Epoch 13 loss 0.4387014021791343 valid acc 16/16
Epoch 13 loss 0.48440438815988957 valid acc 15/16
Epoch 13 loss 0.21275829964067233 valid acc 15/16
Epoch 13 loss 0.8869018997904915 valid acc 15/16
Epoch 13 loss 0.06262285462423023 valid acc 15/16
Epoch 13 loss 0.6225488747850378 valid acc 15/16
Epoch 13 loss 1.0907918115473394 valid acc 16/16
Epoch 13 loss 0.5902918383574567 valid acc 16/16
Epoch 13 loss 0.33213817303548165 valid acc 16/16
Epoch 13 loss 0.45035572320771755 valid acc 15/16
Epoch 13 loss 0.5750977627803772 valid acc 15/16
Epoch 13 loss 0.8922537021460105 valid acc 15/16
Epoch 13 loss 0.3462584290102052 valid acc 15/16
Epoch 13 loss 0.5236411698375908 valid acc 15/16
Epoch 13 loss 0.717354170594463 valid acc 15/16
Epoch 13 loss 0.7141047573018304 valid acc 15/16
Epoch 13 loss 0.5761398423033639 valid acc 15/16
Epoch 13 loss 0.4388383416233516 valid acc 15/16
Epoch 13 loss 0.4252698078484826 valid acc 14/16
Epoch 13 loss 0.3253344073578889 valid acc 14/16
Epoch 13 loss 0.6150135858219731 valid acc 16/16
Epoch 13 loss 0.6648540007766077 valid acc 16/16
Epoch 14 loss 0.024778510722561653 valid acc 16/16
Epoch 14 loss 0.7432300318369265 valid acc 15/16
Epoch 14 loss 0.932811628271889 valid acc 16/16
Epoch 14 loss 0.5395518973586743 valid acc 15/16
Epoch 14 loss 0.2856928534959323 valid acc 15/16
Epoch 14 loss 0.4567892558534083 valid acc 15/16
Epoch 14 loss 0.48390159910185243 valid acc 16/16
Epoch 14 loss 0.9880307787566668 valid acc 15/16
Epoch 14 loss 0.5764116750076926 valid acc 14/16
Epoch 14 loss 0.4952927859104841 valid acc 15/16
Epoch 14 loss 0.6390015429919607 valid acc 15/16
Epoch 14 loss 1.7378784613960199 valid acc 16/16
Epoch 14 loss 1.6868710808778473 valid acc 14/16
Epoch 14 loss 0.7921081291173466 valid acc 14/16
Epoch 14 loss 1.1159080217936548 valid acc 16/16
Epoch 14 loss 0.5943845293315482 valid acc 14/16
Epoch 14 loss 1.7017066179087248 valid acc 14/16
Epoch 14 loss 0.6421663155206667 valid acc 14/16
Epoch 14 loss 0.6980970132612276 valid acc 15/16
Epoch 14 loss 0.38083197227007604 valid acc 15/16
Epoch 14 loss 0.5159012003884509 valid acc 16/16
Epoch 14 loss 0.21889323727335586 valid acc 16/16
Epoch 14 loss 0.33599227422864275 valid acc 15/16
Epoch 14 loss 0.26667894981834217 valid acc 16/16
Epoch 14 loss 0.3304779427514445 valid acc 16/16
Epoch 14 loss 0.72408038640981 valid acc 14/16
Epoch 14 loss 0.45497011393640086 valid acc 15/16
Epoch 14 loss 0.9717614254309175 valid acc 15/16
Epoch 14 loss 0.5151293485368125 valid acc 15/16
Epoch 14 loss 0.20990630913851666 valid acc 15/16
Epoch 14 loss 0.1932656043388546 valid acc 15/16
Epoch 14 loss 0.3814149044538098 valid acc 15/16
Epoch 14 loss 0.6215042695775435 valid acc 16/16
Epoch 14 loss 0.497959142383574 valid acc 16/16
Epoch 14 loss 1.8075788965177182 valid acc 16/16
Epoch 14 loss 0.8963665157473707 valid acc 16/16
Epoch 14 loss 0.24462914587408574 valid acc 16/16
Epoch 14 loss 0.3899530332450481 valid acc 16/16
Epoch 14 loss 0.9632653362320251 valid acc 16/16
Epoch 14 loss 0.9029017562673172 valid acc 16/16
Epoch 14 loss 0.46915978358854493 valid acc 16/16
Epoch 14 loss 0.23785837146610958 valid acc 16/16
Epoch 14 loss 0.6277583860018804 valid acc 15/16
Epoch 14 loss 0.6287058110250145 valid acc 16/16
Epoch 14 loss 0.7071008916982036 valid acc 16/16
Epoch 14 loss 0.3240455780622106 valid acc 16/16
Epoch 14 loss 0.3913328509336011 valid acc 15/16
Epoch 14 loss 0.849987745292262 valid acc 16/16
Epoch 14 loss 0.4823119842925884 valid acc 16/16
Epoch 14 loss 0.09775924826258825 valid acc 15/16
Epoch 14 loss 0.2681443566108898 valid acc 16/16
Epoch 14 loss 0.8807747065443328 valid acc 16/16
Epoch 14 loss 0.7543798175401129 valid acc 15/16
Epoch 14 loss 0.4401940821089378 valid acc 16/16
Epoch 14 loss 0.6485582466561366 valid acc 15/16
Epoch 14 loss 0.2755484460314342 valid acc 15/16
Epoch 14 loss 0.7006021706016903 valid acc 16/16
Epoch 14 loss 0.3710941260563252 valid acc 15/16
Epoch 14 loss 0.3826181701710704 valid acc 15/16
Epoch 14 loss 0.38137433187883374 valid acc 15/16
Epoch 14 loss 0.16345314498607605 valid acc 14/16
Epoch 14 loss 0.24809021095156558 valid acc 16/16
Epoch 14 loss 1.2192851217870129 valid acc 16/16
Epoch 15 loss 0.004931703174661739 valid acc 16/16
Epoch 15 loss 0.7022973714792361 valid acc 15/16
Epoch 15 loss 0.9812411154319205 valid acc 16/16
Epoch 15 loss 0.44088278289777044 valid acc 15/16
Epoch 15 loss 0.24155864794790993 valid acc 16/16
Epoch 15 loss 0.20263939533492026 valid acc 16/16
Epoch 15 loss 0.8665524180460933 valid acc 16/16
Epoch 15 loss 0.7730590602549116 valid acc 16/16
Epoch 15 loss 0.6374039829169638 valid acc 13/16
Epoch 15 loss 0.30713940707292964 valid acc 13/16
Epoch 15 loss 0.5889497239729009 valid acc 15/16
Epoch 15 loss 0.8873000409174103 valid acc 16/16
Epoch 15 loss 0.8990193252098697 valid acc 15/16
Epoch 15 loss 0.8666375392970831 valid acc 16/16
Epoch 15 loss 0.7973074318421693 valid acc 16/16
Epoch 15 loss 0.6267471218643765 valid acc 14/16
Epoch 15 loss 0.8657678215463259 valid acc 14/16
Epoch 15 loss 0.38856174596306103 valid acc 15/16
Epoch 15 loss 0.8629865566258185 valid acc 15/16
Epoch 15 loss 0.32853107903653345 valid acc 15/16
Epoch 15 loss 0.6484832955161192 valid acc 15/16
Epoch 15 loss 0.20035940440561933 valid acc 15/16
Epoch 15 loss 0.20601631505036683 valid acc 16/16
Epoch 15 loss 0.3112043002292311 valid acc 16/16
Epoch 15 loss 0.31538150048223934 valid acc 16/16
Epoch 15 loss 0.6365940381282484 valid acc 16/16
Epoch 15 loss 0.5857113796316611 valid acc 16/16
Epoch 15 loss 0.20732568344886854 valid acc 16/16
Epoch 15 loss 0.801239530683121 valid acc 15/16
Epoch 15 loss 0.233001965155557 valid acc 16/16
Epoch 15 loss 0.1419576382917772 valid acc 15/16
Epoch 15 loss 0.18919124983488722 valid acc 15/16
Epoch 15 loss 0.4577385546214434 valid acc 16/16
Epoch 15 loss 0.48833902283258196 valid acc 16/16
Epoch 15 loss 1.4164714513282974 valid acc 16/16
Epoch 15 loss 0.6633071252863503 valid acc 16/16
Epoch 15 loss 0.2813683784101563 valid acc 16/16
Epoch 15 loss 0.16758859069255777 valid acc 16/16
Epoch 15 loss 0.3352644114951014 valid acc 16/16
Epoch 15 loss 0.4804425785221604 valid acc 16/16
Epoch 15 loss 0.23608985774304464 valid acc 16/16
Epoch 15 loss 0.31691543349293927 valid acc 16/16
Epoch 15 loss 0.3006117795958293 valid acc 16/16
Epoch 15 loss 0.33355559717586786 valid acc 16/16
Epoch 15 loss 0.9064142048380095 valid acc 16/16
Epoch 15 loss 0.09909553539634619 valid acc 16/16
Epoch 15 loss 0.7179027213726886 valid acc 16/16
Epoch 15 loss 0.6353659776480751 valid acc 16/16
Epoch 15 loss 0.6492131095582729 valid acc 16/16
Epoch 15 loss 0.35322465954142557 valid acc 15/16
Epoch 15 loss 0.4977915794682515 valid acc 15/16
Epoch 15 loss 0.4315849902516626 valid acc 16/16
Epoch 15 loss 0.3632565012875055 valid acc 16/16
Epoch 15 loss 0.205847330700626 valid acc 16/16
Epoch 15 loss 0.5344227489716595 valid acc 15/16
Epoch 15 loss 0.5238704543349126 valid acc 16/16
Epoch 15 loss 0.47550305522992586 valid acc 16/16
Epoch 15 loss 0.35280975703408957 valid acc 16/16
Epoch 15 loss 0.47080799374837146 valid acc 15/16
Epoch 15 loss 0.3094826504643165 valid acc 15/16
Epoch 15 loss 0.22148982279211865 valid acc 15/16
Epoch 15 loss 0.3954127492988396 valid acc 16/16
Epoch 15 loss 0.5967444896655412 valid acc 16/16
Epoch 16 loss 0.007465747693595426 valid acc 16/16
Epoch 16 loss 0.6519394792802651 valid acc 16/16
Epoch 16 loss 0.910360552953905 valid acc 16/16
Epoch 16 loss 0.29631446851737386 valid acc 16/16
Epoch 16 loss 0.40641138318058434 valid acc 16/16
Epoch 16 loss 0.17332919479672682 valid acc 16/16
Epoch 16 loss 0.7346963976841208 valid acc 16/16
Epoch 16 loss 0.9056807052850604 valid acc 15/16
Epoch 16 loss 0.7200004450139974 valid acc 15/16
Epoch 16 loss 0.17905805136363623 valid acc 15/16
Epoch 16 loss 0.425502001172524 valid acc 15/16
Epoch 16 loss 1.2073874120094228 valid acc 15/16
Epoch 16 loss 0.8590834671583649 valid acc 15/16
Epoch 16 loss 0.459816071150996 valid acc 15/16
Epoch 16 loss 0.9487265140923573 valid acc 15/16
Epoch 16 loss 0.7781596297136923 valid acc 14/16
Epoch 16 loss 0.755915507283023 valid acc 14/16
Epoch 16 loss 0.41711604595223095 valid acc 15/16
Epoch 16 loss 0.6988025865223904 valid acc 15/16
Epoch 16 loss 0.3804047370720359 valid acc 15/16
Epoch 16 loss 0.5577241372248976 valid acc 16/16
Epoch 16 loss 0.1916551691832018 valid acc 16/16
Epoch 16 loss 0.23618381135581357 valid acc 16/16
Epoch 16 loss 0.17803883180221947 valid acc 16/16
Epoch 16 loss 0.4841169050426928 valid acc 16/16
Epoch 16 loss 0.5350411095410634 valid acc 15/16
Epoch 16 loss 0.3064197861165894 valid acc 15/16
Epoch 16 loss 0.14642744280876382 valid acc 16/16
Epoch 16 loss 0.585153402631502 valid acc 15/16
Epoch 16 loss 0.3413081873319411 valid acc 16/16
Epoch 16 loss 0.3232775133520329 valid acc 16/16
Epoch 16 loss 0.20420947796326622 valid acc 16/16
Epoch 16 loss 0.47938166944564825 valid acc 16/16
Epoch 16 loss 0.6300400668610476 valid acc 16/16
Epoch 16 loss 1.4724059116028383 valid acc 16/16
Epoch 16 loss 0.561099115129208 valid acc 16/16
Epoch 16 loss 0.1287315979897054 valid acc 16/16
Epoch 16 loss 0.23748266816968178 valid acc 16/16
Epoch 16 loss 0.25514477502598165 valid acc 16/16
Epoch 16 loss 0.25760454966379365 valid acc 16/16
Epoch 16 loss 0.09687569907964333 valid acc 16/16
Epoch 16 loss 0.5757091707980727 valid acc 16/16
Epoch 16 loss 0.26542998319211863 valid acc 16/16
Epoch 16 loss 0.47817869092789395 valid acc 16/16
Epoch 16 loss 0.8258755394734664 valid acc 16/16
Epoch 16 loss 0.2283362870981901 valid acc 16/16
Epoch 16 loss 0.1982661341873231 valid acc 16/16
Epoch 16 loss 0.8682351692514522 valid acc 16/16
Epoch 16 loss 0.5356673522852825 valid acc 16/16
Epoch 16 loss 0.20851283712559204 valid acc 16/16
Epoch 16 loss 0.3492132318934091 valid acc 16/16
Epoch 16 loss 0.4935832261411338 valid acc 15/16
Epoch 16 loss 0.5801444230454786 valid acc 16/16
Epoch 16 loss 0.20964719711544993 valid acc 16/16
Epoch 16 loss 0.6970917857408321 valid acc 16/16
Epoch 16 loss 0.22556286897874556 valid acc 16/16
Epoch 16 loss 0.5133049897898647 valid acc 16/16
Epoch 16 loss 0.1450144745721434 valid acc 16/16
Epoch 16 loss 0.4210355037551808 valid acc 15/16
Epoch 16 loss 0.2793258819563287 valid acc 15/16
Epoch 16 loss 0.20996606185174377 valid acc 15/16
Epoch 16 loss 0.24162054246206066 valid acc 16/16
Epoch 16 loss 0.4030624169436349 valid acc 16/16
Epoch 17 loss 0.05963568588145712 valid acc 16/16
Epoch 17 loss 0.4885514230106732 valid acc 16/16
Epoch 17 loss 0.508939568279096 valid acc 16/16
Epoch 17 loss 0.36640462499474513 valid acc 16/16
Epoch 17 loss 0.5923997348696888 valid acc 16/16
Epoch 17 loss 0.2565565764764784 valid acc 16/16
Epoch 17 loss 0.6227227154991477 valid acc 16/16
Epoch 17 loss 0.7408982268681223 valid acc 16/16
Epoch 17 loss 0.35814713199368375 valid acc 16/16
Epoch 17 loss 0.13044447923962632 valid acc 16/16
Epoch 17 loss 0.3247824941022034 valid acc 16/16
Epoch 17 loss 0.5512845680712637 valid acc 16/16
Epoch 17 loss 0.8960180945139357 valid acc 15/16
Epoch 17 loss 0.5130898293925873 valid acc 15/16
Epoch 17 loss 0.875652799019939 valid acc 16/16
Epoch 17 loss 0.5123764539232369 valid acc 15/16
Epoch 17 loss 1.152389534198151 valid acc 15/16
Epoch 17 loss 0.29193333990823295 valid acc 15/16
Epoch 17 loss 0.36877748881181854 valid acc 16/16
Epoch 17 loss 0.3247311318935754 valid acc 16/16
Epoch 17 loss 0.6164584435593363 valid acc 16/16
Epoch 17 loss 0.10389619280201273 valid acc 16/16
Epoch 17 loss 0.3304706840117042 valid acc 16/16
Epoch 17 loss 0.2583691425414369 valid acc 16/16
Epoch 17 loss 0.1641556407044198 valid acc 16/16
Epoch 17 loss 0.8015694063657255 valid acc 15/16
Epoch 17 loss 0.4536927551791265 valid acc 15/16
Epoch 17 loss 0.68197054360873 valid acc 16/16
Epoch 17 loss 0.27318754183207533 valid acc 15/16
Epoch 17 loss 0.08968621740076627 valid acc 16/16
Epoch 17 loss 0.23606138879913913 valid acc 15/16
Epoch 17 loss 0.36910779590220566 valid acc 16/16
Epoch 17 loss 0.5888297991010203 valid acc 15/16
Epoch 17 loss 1.1196813157711718 valid acc 16/16
Epoch 17 loss 1.2720282621645644 valid acc 15/16
Epoch 17 loss 0.5927338588457626 valid acc 16/16
Epoch 17 loss 0.4757541182541878 valid acc 16/16
Epoch 17 loss 0.36273183131884057 valid acc 15/16
Epoch 17 loss 0.4301923752341553 valid acc 16/16
Epoch 17 loss 0.20577476818954216 valid acc 16/16
Epoch 17 loss 0.24005924240105259 valid acc 15/16
Epoch 17 loss 0.35021397392421066 valid acc 16/16
Epoch 17 loss 0.22453173826637063 valid acc 16/16
Epoch 17 loss 0.11782399482352424 valid acc 16/16
Epoch 17 loss 0.8144298464279696 valid acc 15/16
Epoch 17 loss 0.1490075593836762 valid acc 16/16
Epoch 17 loss 0.7412687427815721 valid acc 16/16
Epoch 17 loss 0.6188886334217103 valid acc 15/16
Epoch 17 loss 0.5833770951711377 valid acc 15/16
Epoch 17 loss 0.2120711579188815 valid acc 15/16
Epoch 17 loss 0.16725709352276125 valid acc 15/16
Epoch 17 loss 0.36212623342120603 valid acc 16/16
Epoch 17 loss 0.41008192347335487 valid acc 16/16
Epoch 17 loss 0.21773207185514243 valid acc 16/16
Epoch 17 loss 0.5080303034570431 valid acc 15/16
Epoch 17 loss 0.2809991954799972 valid acc 15/16
Epoch 17 loss 0.5345167880276915 valid acc 16/16
Epoch 17 loss 0.3921596600881381 valid acc 16/16
Epoch 17 loss 0.5717333057254037 valid acc 14/16
Epoch 17 loss 0.45773258086335056 valid acc 15/16
Epoch 17 loss 0.3403322573977371 valid acc 15/16
Epoch 17 loss 0.26076028527424505 valid acc 15/16
Epoch 17 loss 0.41226573579994635 valid acc 16/16
Epoch 18 loss 0.011066440885779683 valid acc 16/16
Epoch 18 loss 0.6363949969600611 valid acc 15/16
Epoch 18 loss 0.5319882962770341 valid acc 15/16
Epoch 18 loss 0.4947231120315246 valid acc 15/16
Epoch 18 loss 0.247852601897671 valid acc 15/16
Epoch 18 loss 0.14494817522543885 valid acc 16/16
Epoch 18 loss 0.3301186111530937 valid acc 16/16
Epoch 18 loss 0.6706986386238981 valid acc 15/16
Epoch 18 loss 0.48912468999801156 valid acc 15/16
Epoch 18 loss 0.28237566104618317 valid acc 14/16
Epoch 18 loss 0.30140746776271926 valid acc 15/16
Epoch 18 loss 1.052718069181187 valid acc 15/16
Epoch 18 loss 0.9930483165971391 valid acc 15/16
Epoch 18 loss 0.7786598261871043 valid acc 16/16
Epoch 18 loss 0.6099004197351643 valid acc 16/16
Epoch 18 loss 0.709203062211176 valid acc 15/16
Epoch 18 loss 0.9230840524513492 valid acc 15/16
Epoch 18 loss 0.3690659109671221 valid acc 15/16
Epoch 18 loss 0.8593988081264035 valid acc 15/16
Epoch 18 loss 0.16826531326793842 valid acc 15/16
Epoch 18 loss 0.6284555193597936 valid acc 16/16
Epoch 18 loss 0.14600406682170536 valid acc 16/16
Epoch 18 loss 0.10257151693544808 valid acc 16/16
Epoch 18 loss 0.21611192498845333 valid acc 16/16
Epoch 18 loss 0.12238638067340363 valid acc 16/16
Epoch 18 loss 0.7756586747851366 valid acc 16/16
Epoch 18 loss 0.38854178517011617 valid acc 16/16
Epoch 18 loss 0.45316580947845675 valid acc 16/16
Epoch 18 loss 0.4198784861459308 valid acc 16/16
Epoch 18 loss 0.11580492269399989 valid acc 16/16
Epoch 18 loss 0.1328828329825834 valid acc 16/16
Epoch 18 loss 0.17353037415046385 valid acc 15/16
Epoch 18 loss 0.2220344502838727 valid acc 16/16
Epoch 18 loss 0.3259587732750216 valid acc 16/16
Epoch 18 loss 1.4181939616370822 valid acc 16/16
Epoch 18 loss 0.44689701794006925 valid acc 16/16
Epoch 18 loss 0.42967636302506756 valid acc 16/16
Epoch 18 loss 0.2845655075579273 valid acc 16/16
Epoch 18 loss 0.13165906122541915 valid acc 16/16
Epoch 18 loss 0.2336124582942337 valid acc 16/16
Epoch 18 loss 0.11646221446441579 valid acc 16/16
Epoch 18 loss 0.32570577965669223 valid acc 16/16
Epoch 18 loss 0.3530025427370567 valid acc 16/16
Epoch 18 loss 0.19314533535450312 valid acc 16/16
Epoch 18 loss 1.2464761327089582 valid acc 15/16
Epoch 18 loss 0.08389907677862601 valid acc 16/16
Epoch 18 loss 0.3445389213000779 valid acc 16/16
Epoch 18 loss 0.33974248905230814 valid acc 16/16
Epoch 18 loss 0.6132311375151611 valid acc 16/16
Epoch 18 loss 0.1696878238252944 valid acc 15/16
Epoch 18 loss 0.21099180870473916 valid acc 15/16
Epoch 18 loss 0.2924922994182026 valid acc 16/16
Epoch 18 loss 0.3383803285980804 valid acc 16/16
Epoch 18 loss 0.09873198200787398 valid acc 16/16
Epoch 18 loss 0.2502873476506037 valid acc 16/16
Epoch 18 loss 0.3172504633764961 valid acc 16/16
Epoch 18 loss 0.7405065507478268 valid acc 16/16
Epoch 18 loss 0.42152850740735176 valid acc 16/16
Epoch 18 loss 0.5334898012122625 valid acc 15/16
Epoch 18 loss 0.17351423126550664 valid acc 16/16
Epoch 18 loss 0.16760988611427224 valid acc 15/16
Epoch 18 loss 0.293346103268847 valid acc 16/16
Epoch 18 loss 0.3771162213356167 valid acc 16/16
Epoch 19 loss 0.002749103808387776 valid acc 16/16
Epoch 19 loss 0.4037387279926327 valid acc 15/16
Epoch 19 loss 0.9051047910790331 valid acc 16/16
Epoch 19 loss 0.24574311447982675 valid acc 16/16
Epoch 19 loss 0.2675595829425695 valid acc 16/16
Epoch 19 loss 0.16521688995532063 valid acc 16/16
Epoch 19 loss 0.6501894787116221 valid acc 16/16
Epoch 19 loss 0.5556833274608448 valid acc 15/16
Epoch 19 loss 0.2373057481050007 valid acc 15/16
Epoch 19 loss 0.18369263551419143 valid acc 15/16
Epoch 19 loss 0.3929333952705219 valid acc 15/16
Epoch 19 loss 0.7525097634260247 valid acc 15/16
Epoch 19 loss 0.625825903540886 valid acc 15/16
Epoch 19 loss 0.6812490109931209 valid acc 15/16
Epoch 19 loss 0.6873016333622047 valid acc 16/16
Epoch 19 loss 0.39380109994559453 valid acc 15/16
Epoch 19 loss 0.8868493353565725 valid acc 16/16
Epoch 19 loss 0.1795222143385555 valid acc 16/16
Epoch 19 loss 0.4752908098259031 valid acc 16/16
Epoch 19 loss 0.20763501570794035 valid acc 16/16
Epoch 19 loss 0.7173989752175208 valid acc 16/16
Epoch 19 loss 0.1637196051027736 valid acc 16/16
Epoch 19 loss 0.17748949541955855 valid acc 16/16
Epoch 19 loss 0.20757066841096325 valid acc 16/16
Epoch 19 loss 0.2537203819753036 valid acc 16/16
Epoch 19 loss 0.1936141389058198 valid acc 16/16
Epoch 19 loss 0.17063974135631943 valid acc 16/16
Epoch 19 loss 0.4090103664933065 valid acc 16/16
Epoch 19 loss 0.38593089985522616 valid acc 15/16
Epoch 19 loss 0.06849485182424991 valid acc 16/16
Epoch 19 loss 0.11811934661994339 valid acc 16/16
Epoch 19 loss 0.16521929988672113 valid acc 16/16
Epoch 19 loss 0.4006256621939134 valid acc 16/16
Epoch 19 loss 0.16391813588579918 valid acc 16/16
Epoch 19 loss 1.1844939393537035 valid acc 16/16
Epoch 19 loss 0.20013142654262298 valid acc 16/16
Epoch 19 loss 0.36979984772132496 valid acc 16/16
Epoch 19 loss 0.1554657439603751 valid acc 16/16
Epoch 19 loss 0.22047160129395033 valid acc 16/16
Epoch 19 loss 0.19027682111564082 valid acc 16/16
Epoch 19 loss 0.05947521545367496 valid acc 16/16
Epoch 19 loss 0.3780142897412423 valid acc 16/16
Epoch 19 loss 0.2685565680189923 valid acc 16/16
Epoch 19 loss 0.1138997839893165 valid acc 16/16
Epoch 19 loss 0.4559963034453761 valid acc 16/16
Epoch 19 loss 0.0238203687940709 valid acc 16/16
Epoch 19 loss 0.6818732580369147 valid acc 16/16
Epoch 19 loss 0.3688895489680891 valid acc 16/16
Epoch 19 loss 0.4145717610680092 valid acc 16/16
Epoch 19 loss 0.11382571776772182 valid acc 16/16
Epoch 19 loss 0.14077628318870078 valid acc 16/16
Epoch 19 loss 0.5127296837750412 valid acc 16/16
Epoch 19 loss 0.4415109587889395 valid acc 16/16
Epoch 19 loss 0.06951861817691912 valid acc 16/16
Epoch 19 loss 0.28628726073885047 valid acc 16/16
Epoch 19 loss 0.518442540943528 valid acc 16/16
Epoch 19 loss 0.4851170661374481 valid acc 16/16
Epoch 19 loss 0.3198318682775017 valid acc 16/16
Epoch 19 loss 0.642361143714164 valid acc 16/16
Epoch 19 loss 0.27057641269803856 valid acc 15/16
Epoch 19 loss 0.18893309703633965 valid acc 15/16
Epoch 19 loss 0.24604371220369697 valid acc 15/16
Epoch 19 loss 0.29646586893838406 valid acc 16/16
Epoch 20 loss 0.005564288972878251 valid acc 16/16
Epoch 20 loss 0.26753101873492535 valid acc 15/16
Epoch 20 loss 0.3317307269616002 valid acc 15/16
Epoch 20 loss 0.1543594184723283 valid acc 15/16
Epoch 20 loss 0.08985442358507495 valid acc 15/16
Epoch 20 loss 0.12490504198727381 valid acc 15/16
Epoch 20 loss 0.28442023359459145 valid acc 16/16
Epoch 20 loss 0.9293850601574125 valid acc 16/16
Epoch 20 loss 0.24277764066375174 valid acc 16/16
Epoch 20 loss 0.247355028521428 valid acc 15/16
Epoch 20 loss 0.29431323696108763 valid acc 15/16
Epoch 20 loss 0.744666200559307 valid acc 15/16
Epoch 20 loss 0.6124629485511215 valid acc 16/16
Epoch 20 loss 0.4523798323264129 valid acc 15/16
Epoch 20 loss 0.40379368550580097 valid acc 16/16
Epoch 20 loss 0.5281672414960298 valid acc 15/16
Epoch 20 loss 0.4297757005810586 valid acc 15/16
Epoch 20 loss 0.53877641890972 valid acc 13/16
Epoch 20 loss 0.47742475936811607 valid acc 15/16
Epoch 20 loss 0.2634319104993944 valid acc 16/16
Epoch 20 loss 0.49184841464151113 valid acc 16/16
Epoch 20 loss 0.18726595272402263 valid acc 16/16
Epoch 20 loss 0.10734655537862947 valid acc 15/16
Epoch 20 loss 0.09741768733436479 valid acc 15/16
Epoch 20 loss 0.2007821421115087 valid acc 16/16
Epoch 20 loss 0.308210686591323 valid acc 16/16
Epoch 20 loss 0.32488350516984327 valid acc 16/16
Epoch 20 loss 0.13751444459761736 valid acc 15/16
Epoch 20 loss 0.18624457753360302 valid acc 14/16
Epoch 20 loss 0.11461330810198683 valid acc 15/16
Epoch 20 loss 0.14252923410670493 valid acc 16/16
Epoch 20 loss 0.06362979401399732 valid acc 16/16
Epoch 20 loss 0.13035387963921996 valid acc 14/16
Epoch 20 loss 0.47866221666118325 valid acc 16/16
Epoch 20 loss 1.470422993129961 valid acc 16/16
Epoch 20 loss 0.21922516846675677 valid acc 16/16
Epoch 20 loss 0.18634874241681193 valid acc 16/16
Epoch 20 loss 0.09550692013331785 valid acc 16/16
Epoch 20 loss 0.47062300677639973 valid acc 15/16
Epoch 20 loss 0.2072137257427717 valid acc 16/16
Epoch 20 loss 0.06489150654527276 valid acc 16/16
Epoch 20 loss 0.15637973238683867 valid acc 16/16
Epoch 20 loss 0.16789114703149782 valid acc 16/16
Epoch 20 loss 0.07732778290020054 valid acc 16/16
Epoch 20 loss 0.4294644477466204 valid acc 15/16
Epoch 20 loss 0.11835087496585206 valid acc 16/16
Epoch 20 loss 0.590662341108621 valid acc 16/16
Epoch 20 loss 0.6822688345671541 valid acc 15/16
Epoch 20 loss 0.5069807584483039 valid acc 15/16
Epoch 20 loss 0.08561703134142379 valid acc 15/16
Epoch 20 loss 0.3786200890765353 valid acc 15/16
Epoch 20 loss 0.46527683127660635 valid acc 16/16
Epoch 20 loss 0.34789210139539894 valid acc 16/16
Epoch 20 loss 0.14266210265176876 valid acc 16/16
Epoch 20 loss 0.46118712419732866 valid acc 16/16
Epoch 20 loss 0.20302466097952898 valid acc 15/16
Epoch 20 loss 0.5486177759334895 valid acc 15/16
Epoch 20 loss 0.11452968849153089 valid acc 16/16
Epoch 20 loss 0.2133450637388794 valid acc 15/16
Epoch 20 loss 0.19465050869314698 valid acc 15/16
Epoch 20 loss 0.2208141169647192 valid acc 15/16
Epoch 20 loss 0.25722514313708095 valid acc 15/16
Epoch 20 loss 0.36983004467133285 valid acc 16/16
Epoch 21 loss 0.004346221146959761 valid acc 16/16
Epoch 21 loss 0.5928345127568024 valid acc 14/16
Epoch 21 loss 0.5907388494432175 valid acc 15/16
Epoch 21 loss 0.18879552063969401 valid acc 15/16
Epoch 21 loss 0.22026937173851052 valid acc 15/16
Epoch 21 loss 0.087656701793076 valid acc 16/16
Epoch 21 loss 0.7153791626947757 valid acc 15/16
Epoch 21 loss 0.6637588825188523 valid acc 15/16
Epoch 21 loss 0.47567485597730724 valid acc 15/16
Epoch 21 loss 0.13869522119982508 valid acc 15/16
Epoch 21 loss 0.2676006188310325 valid acc 15/16
Epoch 21 loss 0.5229082004116188 valid acc 14/16
Epoch 21 loss 0.5163863215460757 valid acc 14/16
Epoch 21 loss 0.3082782587388902 valid acc 14/16
Epoch 21 loss 0.47685868543016247 valid acc 15/16
Epoch 21 loss 0.734812112290546 valid acc 16/16
Epoch 21 loss 0.7172930978353664 valid acc 14/16
Epoch 21 loss 0.31341459803663113 valid acc 14/16
Epoch 21 loss 0.3503061188042631 valid acc 16/16
Epoch 21 loss 0.1830771928471311 valid acc 16/16
Epoch 21 loss 0.45336372077191367 valid acc 16/16
Epoch 21 loss 0.18260808750489083 valid acc 16/16
Epoch 21 loss 0.07966176438653921 valid acc 16/16
Epoch 21 loss 0.20700180806197327 valid acc 16/16
Epoch 21 loss 0.29274142813319015 valid acc 16/16
Epoch 21 loss 0.44278223896066254 valid acc 15/16
Epoch 21 loss 0.23465804451167627 valid acc 16/16
Epoch 21 loss 0.06736546651608927 valid acc 16/16
Epoch 21 loss 0.4489330984095535 valid acc 14/16
Epoch 21 loss 0.44629705844809386 valid acc 16/16
Epoch 21 loss 0.06968258750295508 valid acc 16/16
Epoch 21 loss 0.17372053485472977 valid acc 16/16
Epoch 21 loss 0.2329981149689962 valid acc 16/16
Epoch 21 loss 0.060251777239200666 valid acc 16/16
Epoch 21 loss 1.173732544941568 valid acc 16/16
Epoch 21 loss 0.46427793566077835 valid acc 16/16
Epoch 21 loss 0.06806454841466669 valid acc 16/16
Epoch 21 loss 0.342376996053625 valid acc 16/16
Epoch 21 loss 0.4968217931790033 valid acc 16/16
Epoch 21 loss 0.0832258048307547 valid acc 16/16
Epoch 21 loss 0.08810289756144646 valid acc 16/16
Epoch 21 loss 0.12225930805326468 valid acc 16/16
Epoch 21 loss 0.08853686064156213 valid acc 16/16
Epoch 21 loss 0.1522235230902254 valid acc 16/16
Epoch 21 loss 0.44795721645178077 valid acc 16/16
Epoch 21 loss 0.07524483997238396 valid acc 16/16
Epoch 21 loss 0.38701428788823516 valid acc 16/16
Epoch 21 loss 0.34388062113078727 valid acc 16/16
Epoch 21 loss 0.1832117858106678 valid acc 16/16
Epoch 21 loss 0.14113534754723034 valid acc 15/16
Epoch 21 loss 0.22196544407955227 valid acc 16/16
Epoch 21 loss 0.24320361616897357 valid acc 16/16
Epoch 21 loss 0.38218437310929027 valid acc 16/16
Epoch 21 loss 0.1733960030713521 valid acc 16/16
Epoch 21 loss 0.2600678843007367 valid acc 16/16
Epoch 21 loss 0.11997802860702111 valid acc 16/16
Epoch 21 loss 0.5776931370445868 valid acc 16/16
Epoch 21 loss 0.06189595002797654 valid acc 16/16
Epoch 21 loss 0.34448823115313393 valid acc 15/16
Epoch 21 loss 0.35732691793231314 valid acc 16/16
Epoch 21 loss 0.32371131160110367 valid acc 15/16
Epoch 21 loss 0.13591345649095682 valid acc 15/16
Epoch 21 loss 0.13742334977737386 valid acc 15/16
Epoch 22 loss 0.0016724239360408344 valid acc 15/16
Epoch 22 loss 0.5449022291228458 valid acc 16/16
Epoch 22 loss 0.8089719195678786 valid acc 16/16
Epoch 22 loss 0.15668749195342901 valid acc 16/16
Epoch 22 loss 0.15757087218044097 valid acc 15/16
Epoch 22 loss 0.21369794587119229 valid acc 15/16
Epoch 22 loss 0.9444397892811237 valid acc 16/16
Epoch 22 loss 0.6373990935675582 valid acc 15/16
Epoch 22 loss 0.2490042967187519 valid acc 15/16
Epoch 22 loss 0.17495277953268912 valid acc 15/16
Epoch 22 loss 0.22613393627912387 valid acc 15/16
Epoch 22 loss 0.5437088990915905 valid acc 15/16
Epoch 22 loss 0.46216780541361496 valid acc 15/16
Epoch 22 loss 0.5027837798060741 valid acc 16/16
Epoch 22 loss 0.4381654857075307 valid acc 16/16
Epoch 22 loss 0.4880372473094008 valid acc 15/16
Epoch 22 loss 0.33346268045678923 valid acc 15/16
Epoch 22 loss 0.38900509435200487 valid acc 16/16
Epoch 22 loss 0.4024434113814541 valid acc 16/16
Epoch 22 loss 0.21897459926514723 valid acc 16/16
Epoch 22 loss 0.5581218000317779 valid acc 16/16
Epoch 22 loss 0.09857982963681294 valid acc 16/16
Epoch 22 loss 0.1720814584025981 valid acc 16/16
Epoch 22 loss 0.05126515813301463 valid acc 16/16
Epoch 22 loss 0.16170788058152202 valid acc 16/16
Epoch 22 loss 0.36871389014252776 valid acc 16/16
Epoch 22 loss 0.32047565646504134 valid acc 16/16
Epoch 22 loss 0.3707685369327501 valid acc 16/16
Epoch 22 loss 0.29187532319089593 valid acc 15/16
Epoch 22 loss 0.1035896599718451 valid acc 15/16
Epoch 22 loss 0.12579688469669914 valid acc 16/16
Epoch 22 loss 0.2323982362915476 valid acc 16/16
Epoch 22 loss 0.12033248222827592 valid acc 16/16
Epoch 22 loss 0.42363797846571777 valid acc 16/16
Epoch 22 loss 1.2876383881871418 valid acc 16/16
Epoch 22 loss 0.2567851085934416 valid acc 16/16
Epoch 22 loss 0.31521859973398153 valid acc 16/16
Epoch 22 loss 0.09375133491489568 valid acc 16/16
Epoch 22 loss 0.22532900730770555 valid acc 16/16
Epoch 22 loss 0.20520366785683714 valid acc 16/16
Epoch 22 loss 0.12898757704132457 valid acc 16/16
Epoch 22 loss 0.2907198955530812 valid acc 16/16
Epoch 22 loss 0.12369670700752211 valid acc 16/16
Epoch 22 loss 0.4743129309812456 valid acc 16/16
Epoch 22 loss 0.3110909030876421 valid acc 16/16
Epoch 22 loss 0.16066082711238566 valid acc 16/16
Epoch 22 loss 0.5884851787988048 valid acc 16/16
Epoch 22 loss 0.3024077051948496 valid acc 16/16
Epoch 22 loss 0.3124946106616235 valid acc 16/16
Epoch 22 loss 0.28295171583616335 valid acc 16/16
Epoch 22 loss 0.3054705107147311 valid acc 16/16
Epoch 22 loss 0.25842413846781936 valid acc 16/16
Epoch 22 loss 0.3099883934881523 valid acc 16/16
Epoch 22 loss 0.12642961104603279 valid acc 16/16
Epoch 22 loss 0.2426015712871073 valid acc 16/16
Epoch 22 loss 0.21452129074952866 valid acc 16/16
Epoch 22 loss 0.623989562834639 valid acc 16/16
Epoch 22 loss 0.21315019687431835 valid acc 16/16
Epoch 22 loss 0.4714180056392351 valid acc 15/16
Epoch 22 loss 0.39662461455281417 valid acc 16/16
Epoch 22 loss 0.1744185473663183 valid acc 15/16
Epoch 22 loss 0.15459211469576542 valid acc 16/16
Epoch 22 loss 0.3653011679670448 valid acc 16/16
Epoch 23 loss 0.0012906054381730536 valid acc 16/16
Epoch 23 loss 0.36207240173638633 valid acc 16/16
Epoch 23 loss 0.28385412853756914 valid acc 16/16
Epoch 23 loss 0.19619825699005167 valid acc 16/16
Epoch 23 loss 0.17546106266177203 valid acc 15/16
Epoch 23 loss 0.34449869155369534 valid acc 16/16
Epoch 23 loss 0.40666406056706883 valid acc 16/16
Epoch 23 loss 0.6898605797484691 valid acc 16/16
Epoch 23 loss 0.1950930687558431 valid acc 16/16
Epoch 23 loss 0.20258700443233252 valid acc 15/16
Epoch 23 loss 0.45673845615905034 valid acc 15/16
Epoch 23 loss 0.4976678756793372 valid acc 16/16
Epoch 23 loss 0.6286870209610231 valid acc 15/16
Epoch 23 loss 0.13389321994668668 valid acc 15/16
Epoch 23 loss 0.6171194811805107 valid acc 16/16
Epoch 23 loss 0.5243374316670354 valid acc 15/16
Epoch 23 loss 0.6502543489676892 valid acc 15/16
Epoch 23 loss 0.22110671118514075 valid acc 15/16
Epoch 23 loss 0.27827458560768437 valid acc 15/16
Epoch 23 loss 0.06979184809335845 valid acc 15/16
Epoch 23 loss 0.4943284807891871 valid acc 16/16
Epoch 23 loss 0.05077484972518048 valid acc 16/16
Epoch 23 loss 0.13180537876480175 valid acc 16/16
Epoch 23 loss 0.1627028033607622 valid acc 16/16
Epoch 23 loss 0.14613096598306977 valid acc 16/16
Epoch 23 loss 0.8346119545342723 valid acc 15/16
Epoch 23 loss 0.4618548655846059 valid acc 15/16
Epoch 23 loss 0.173715947984873 valid acc 14/16
Epoch 23 loss 0.07210942369956375 valid acc 14/16
Epoch 23 loss 0.09489801933127551 valid acc 15/16
Epoch 23 loss 0.07475168050865483 valid acc 16/16
Epoch 23 loss 0.11327376667212075 valid acc 16/16
Epoch 23 loss 0.05832077192523544 valid acc 16/16
Epoch 23 loss 0.2760569724602122 valid acc 16/16
Epoch 23 loss 1.4399807614694895 valid acc 15/16
Epoch 23 loss 0.4181990883050455 valid acc 16/16
Epoch 23 loss 0.26487732512967743 valid acc 16/16
Epoch 23 loss 0.35248567667578456 valid acc 16/16
Epoch 23 loss 0.16342797830398093 valid acc 16/16
Epoch 23 loss 0.3709310222598321 valid acc 16/16
Epoch 23 loss 0.06962900304406783 valid acc 16/16
Epoch 23 loss 0.12950211906332898 valid acc 16/16
Epoch 23 loss 0.1627387824809166 valid acc 16/16
Epoch 23 loss 0.19025711905185405 valid acc 16/16
Epoch 23 loss 0.6282269152895028 valid acc 16/16
Epoch 23 loss 0.07559920329940506 valid acc 16/16
Epoch 23 loss 0.7262435886057848 valid acc 16/16
Epoch 23 loss 0.6274931573373455 valid acc 16/16
Epoch 23 loss 0.3262835686215713 valid acc 16/16
Epoch 23 loss 0.1020447217513411 valid acc 16/16
Epoch 23 loss 0.21357097268959951 valid acc 16/16
Epoch 23 loss 0.4630114080148095 valid acc 16/16
Epoch 23 loss 0.19619372533467028 valid acc 16/16
Epoch 23 loss 0.17106186646181176 valid acc 16/16
Epoch 23 loss 0.5027030320930124 valid acc 16/16
Epoch 23 loss 0.2312316786022527 valid acc 16/16
Epoch 23 loss 0.5739238602630315 valid acc 16/16
Epoch 23 loss 0.049656284101770476 valid acc 16/16
Epoch 23 loss 0.33360785766121553 valid acc 16/16
Epoch 23 loss 0.3009902784445441 valid acc 15/16
Epoch 23 loss 0.09301220548051659 valid acc 15/16
Epoch 23 loss 0.12694578011663638 valid acc 15/16
Epoch 23 loss 0.2650605440102365 valid acc 16/16
Epoch 24 loss 0.0026653062330784794 valid acc 16/16
Epoch 24 loss 0.08571684230503857 valid acc 16/16
Epoch 24 loss 0.15289679811756274 valid acc 16/16
Epoch 24 loss 0.33256528211590525 valid acc 16/16
Epoch 24 loss 0.11577885590008496 valid acc 16/16
Epoch 24 loss 0.15877592499479087 valid acc 16/16
Epoch 24 loss 0.13994389041709507 valid acc 16/16
Epoch 24 loss 0.37621526039537684 valid acc 16/16
Epoch 24 loss 0.23070786868829746 valid acc 16/16
Epoch 24 loss 0.017049181518652756 valid acc 16/16
Epoch 24 loss 0.38105746962689263 valid acc 15/16
Epoch 24 loss 0.31640153679362576 valid acc 15/16
Epoch 24 loss 0.29221975161227753 valid acc 15/16
Epoch 24 loss 0.36370932498709074 valid acc 15/16
Epoch 24 loss 0.20409449497415405 valid acc 16/16
Epoch 24 loss 0.43788629229766174 valid acc 15/16
Epoch 24 loss 0.7856307289506879 valid acc 16/16
Epoch 24 loss 0.2543806027422666 valid acc 15/16
Epoch 24 loss 0.17073532462185925 valid acc 15/16
Epoch 24 loss 0.24765413229060648 valid acc 15/16
Epoch 24 loss 0.3642696192200047 valid acc 16/16
Epoch 24 loss 0.12153997935385974 valid acc 16/16
Epoch 24 loss 0.1589267145974046 valid acc 16/16
Epoch 24 loss 0.06814460287004448 valid acc 16/16
Epoch 24 loss 0.13258586663672306 valid acc 16/16
Epoch 24 loss 0.4606569398217457 valid acc 16/16
Epoch 24 loss 0.19447888093743748 valid acc 16/16
Epoch 24 loss 0.18445107940620692 valid acc 16/16
Epoch 24 loss 0.21856494260063608 valid acc 15/16
Epoch 24 loss 0.12263753619456258 valid acc 16/16
Epoch 24 loss 0.2849636463599184 valid acc 16/16
Epoch 24 loss 0.03906751449622181 valid acc 15/16
Epoch 24 loss 0.24280352349800094 valid acc 16/16
Epoch 24 loss 0.20622904446007184 valid acc 16/16
Epoch 24 loss 1.2330520037451196 valid acc 16/16
Epoch 24 loss 0.18944512144150993 valid acc 16/16
Epoch 24 loss 0.1293999315374998 valid acc 16/16
Epoch 24 loss 0.04401614760899436 valid acc 16/16
Epoch 24 loss 0.06862291308285123 valid acc 16/16
Epoch 24 loss 0.11052174189500069 valid acc 16/16
Epoch 24 loss 0.24089471000169604 valid acc 16/16
Epoch 24 loss 0.2266658606147982 valid acc 16/16
Epoch 24 loss 0.13646600328651692 valid acc 16/16
Epoch 24 loss 0.22782599635370404 valid acc 16/16
Epoch 24 loss 0.2883474030767701 valid acc 16/16
Epoch 24 loss 0.2267264936504929 valid acc 16/16
Epoch 24 loss 0.9286241963431514 valid acc 16/16
Epoch 24 loss 0.3531511284142537 valid acc 16/16
Epoch 24 loss 0.4326397675278885 valid acc 15/16
Epoch 24 loss 0.05295013357290773 valid acc 16/16
Epoch 24 loss 0.4282265285624726 valid acc 16/16
Epoch 24 loss 0.5663862265516902 valid acc 16/16
Epoch 24 loss 0.4423127951227407 valid acc 16/16
Epoch 24 loss 0.1648385771510228 valid acc 16/16
Epoch 24 loss 0.17609793068552304 valid acc 16/16
Epoch 24 loss 0.296837706437389 valid acc 16/16
Epoch 24 loss 0.5711636503386526 valid acc 16/16
Epoch 24 loss 0.24463228325148573 valid acc 16/16
Epoch 24 loss 0.2579086802305357 valid acc 16/16
Epoch 24 loss 0.21836224824736622 valid acc 16/16
Epoch 24 loss 0.10690355975614302 valid acc 15/16
Epoch 24 loss 0.2943739807012518 valid acc 16/16
Epoch 24 loss 0.25047151261723233 valid acc 16/16
Epoch 25 loss 0.0074054234854594725 valid acc 16/16
Epoch 25 loss 0.5808456161177478 valid acc 16/16
Epoch 25 loss 0.45154564892392596 valid acc 16/16
Epoch 25 loss 0.14085645068917257 valid acc 16/16
Epoch 25 loss 0.0415483531803425 valid acc 16/16
Epoch 25 loss 0.11721663085836104 valid acc 16/16
Epoch 25 loss 0.3025611845595056 valid acc 16/16
Epoch 25 loss 0.22966543208816442 valid acc 16/16
Epoch 25 loss 0.0875515594464743 valid acc 16/16
Epoch 25 loss 0.34207981994715353 valid acc 15/16
Epoch 25 loss 0.3598270411984638 valid acc 16/16
Epoch 25 loss 0.4976957448341302 valid acc 16/16
Epoch 25 loss 0.2113586455795461 valid acc 16/16
Epoch 25 loss 0.11043382746430908 valid acc 16/16
Epoch 25 loss 0.3994477869378393 valid acc 16/16
Epoch 25 loss 0.4478799518471821 valid acc 16/16
Epoch 25 loss 0.5060375905692984 valid acc 15/16
Epoch 25 loss 0.1122961067715505 valid acc 15/16
Epoch 25 loss 0.5269479519039639 valid acc 15/16
Epoch 25 loss 0.07596413220132164 valid acc 15/16
Epoch 25 loss 0.09943614128751344 valid acc 16/16
Epoch 25 loss 0.17370047422800128 valid acc 16/16
Epoch 25 loss 0.06377718722112563 valid acc 16/16
Epoch 25 loss 0.08891438535224999 valid acc 16/16
Epoch 25 loss 0.07803133888264546 valid acc 16/16
Epoch 25 loss 0.22035630617830437 valid acc 16/16
Epoch 25 loss 0.03788547230712519 valid acc 16/16
Epoch 25 loss 0.0800190626275864 valid acc 16/16
Epoch 25 loss 0.24920550900575344 valid acc 15/16
Epoch 25 loss 0.09815827498814378 valid acc 16/16
Epoch 25 loss 0.2838477071920864 valid acc 16/16
Epoch 25 loss 0.05215888513219402 valid acc 15/16
Epoch 25 loss 0.2528190929156994 valid acc 16/16
Epoch 25 loss 0.41371118011683905 valid acc 16/16
Epoch 25 loss 0.9230674001811642 valid acc 16/16
Epoch 25 loss 0.27185431017590567 valid acc 16/16
Epoch 25 loss 0.09555527044028667 valid acc 16/16
Epoch 25 loss 0.19969612168986106 valid acc 16/16
Epoch 25 loss 0.43134961242546227 valid acc 15/16
Epoch 25 loss 0.8282934111820441 valid acc 16/16
Epoch 25 loss 0.012669386822017348 valid acc 16/16
Epoch 25 loss 0.16301610132815753 valid acc 16/16
Epoch 25 loss 0.6201260851549666 valid acc 16/16
Epoch 25 loss 0.3113417287834116 valid acc 16/16
Epoch 25 loss 1.412363653571393 valid acc 15/16
Epoch 25 loss 0.03705248953410234 valid acc 15/16
Epoch 25 loss 0.34945256728154755 valid acc 15/16
Epoch 25 loss 0.18246371452024795 valid acc 16/16
Epoch 25 loss 0.23803444657030384 valid acc 16/16
Epoch 25 loss 0.5509464921056881 valid acc 16/16
Epoch 25 loss 0.34994642944604964 valid acc 16/16
Epoch 25 loss 0.2358887428262798 valid acc 16/16
Epoch 25 loss 0.23051038824657555 valid acc 16/16
Epoch 25 loss 0.06792448092117687 valid acc 16/16
Epoch 25 loss 0.3629936312008563 valid acc 16/16
Epoch 25 loss 0.03247923970843014 valid acc 16/16
Epoch 25 loss 0.561879342042485 valid acc 16/16
Epoch 25 loss 0.03736868936304194 valid acc 16/16
Epoch 25 loss 0.3101094068050756 valid acc 16/16
Epoch 25 loss 0.40432969242986905 valid acc 16/16
Epoch 25 loss 0.2205321851849456 valid acc 16/16
Epoch 25 loss 0.2155543176041667 valid acc 16/16
Epoch 25 loss 0.15846746168680725 valid acc 16/16