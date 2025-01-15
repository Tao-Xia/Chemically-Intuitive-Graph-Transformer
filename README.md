# Chemically Intuitive Graph Transformer Based on Valence Bond Theory
This is the official implement of chemically intuitive graph transformer to predict the valence bond(vb) structure weight and select the important vb structures to construct compact wavefunction within chemical accuracy compared to full structure calculation.
> This implement is based on GraphGPS [Rampasek et al., 2022.](https://github.com/rampasek/GraphGPS)

You can predict on specific active space. 
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
