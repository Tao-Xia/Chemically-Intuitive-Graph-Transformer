# Chemically Intuitive Graph Transformer Based on Valence Bond Theory
This is the official implement of chemically intuitive graph transformer to predict the valence bond(vb) structure weight and select the important vb structures to construct compact wavefunction within chemical accuracy compared to full structure calculation. 

For more details about valence bond theory and XMVB, please refer to [XMVB Documentation](https://xacs.xmu.edu.cn/docs/xmvb/)

> This implement is based on GraphGPS [Rampasek et al., 2022.](https://github.com/rampasek/GraphGPS)

## Usage
1. You can predict on specific structure subspace. 
```python
# 1. define the molecule (xyz format and gamess format are supported)
molecule = """
C        -0.545879        1.285390        0.000006
C         0.840247        1.115444       -0.000011
C        -1.386138        0.169949       -0.000006
C         1.386138       -0.169959        0.000037
C        -0.840246       -1.115442       -0.000016
C         0.545876       -1.285390       -0.000007
H        -0.970539        2.285347        0.000018
H         1.493900        1.983187       -0.000017
H        -2.464448        0.302182       -0.000027
H         2.464459       -0.302140        0.000051
H        -1.493909       -1.983181       -0.000012
H         0.970553       -2.285342       -0.000034
"""     
# 2. define the valence bond structure space
# ('COV', '0-1', '0-2', '0-3', 'FULL')
# If you want predict the assigned vb structures, you have to write the structure by yourself, same as XMVB $Str input, but just the active active
vb_str_space = "FULL"
# 3. define the spin multiplicity and active space
numl = 1 
nao = 6
nae = 6
# 4. define the active orbital of the molecule, same as XMVB active orbital input (like $Orb part, but just active orbital)
active_orbital =  [5,6,4,2,1,3]
```
After defining the molecule, active space and structure subspace, you can run this command to predict the valence bond structure weight
```
python main.py --cfg configs/predict/predict_weight.yaml
```
or run this command to get the important vb structures that can be used to construct compact wavefunction
```
python main.py --cfg configs/predict/select_structure.yaml
```
2. You can write VB structure by yourself, same as XMVB $Str input, but just the active active.
CIGT/loader/predict_vb.py
```python
# 1. define the molecule (xyz format and gamess format are supported)
molecule = """
C        -0.545879        1.285390        0.000006
C         0.840247        1.115444       -0.000011
C        -1.386138        0.169949       -0.000006
C         1.386138       -0.169959        0.000037
C        -0.840246       -1.115442       -0.000016
C         0.545876       -1.285390       -0.000007
H        -0.970539        2.285347        0.000018
H         1.493900        1.983187       -0.000017
H        -2.464448        0.302182       -0.000027
H         2.464459       -0.302140        0.000051
H        -1.493909       -1.983181       -0.000012
H         0.970553       -2.285342       -0.000034
"""     
# 2. define the valence bond structure space
# If you want predict the assigned vb structures, you have to write the structure by yourself, same as XMVB $Str input, but just the active active
vb_str_space = "FULL"
# 3. define the spin multiplicity and active space
numl = 1 
nao = 6
nae = 6
# 4. define the VB structure by yourself, same as XMVB $Str input, but just the active active
active_orbital = [[21  22  20  23  19  24],
          [20  21  22  23  19  24],
          [20  21  19  22  23  24],
          [19  20  22  23  21  24],
          [19  20  21  22  23  24],]
```
After defining the molecule, active space and structure you want to predict, you can run this command to predict the valence bond structure weight
```
python main.py --cfg configs/predict/predict_weight.yaml
```
or run this command to get important vb structures that can be used to construct compact wavefunction
```
python main.py --cfg configs/predict/select_structure.yaml
```    

      
      
      