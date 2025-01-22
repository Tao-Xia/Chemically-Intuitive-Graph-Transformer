# Chemically Intuitive Graph Transformer Based on Valence Bond Theory
This repository contains the official implementation of a Chemically Intuitive Graph Transformer (CIGT) designed to predict the Valence Bond (VB) structure weights and select important VB structures. These structures can then be used to construct a compact wavefunction in Xiamen Valence Bond (XMVB) program, compared to a full structure calculation.

For detailed information on VB theory and XMVB, please refer to [XMVB Documentation](https://xacs.xmu.edu.cn/docs/xmvb/)

> This implement is based on GraphGPS [Rampasek et al., 2022.](https://github.com/rampasek/GraphGPS)

## Usage
1. Predicting Weight for each specific VB structure
```python
# 1. Provide the molecule and geometry. Here’s an example of a molecule's geometry:
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
# 2. Provide the spin multiplicity and active space, e.g.
numl = 1 
nao = 6
nae = 6
# 3. Define the active orbitals, e.g.
# active_orbital =  [5,6,4,2,1,3]
# Each number represents the atom from which the corresponding orbital originates.
# 4. Define the set of VB structures (This functionality is currently supported only when used in conjunction with XMVB, more detailed information please refer to XMVB manual)
# ('COV', '0-1', '0-2', '0-3', ...,'FULL')
# Users can also write the VB structure by yourself.
# The format is the same as the $Str section in XMVB input file, but only the pairing of active orbitals are needed to provide.
```
After defining the molecule, active space and structure set, you can run the following command to predict the VB structure weights
```
python main.py --cfg configs/predict/predict_weight.yaml
```
or run the following command to get the important VB structures that can be used to construct a compact VB wavefunction
```
python main.py --cfg configs/predict/select_structure.yaml
```
2. This is an example of benzene molecule to predict weights of structures written by the user
```python
# 1. Provide the molecule and geometry
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
# 2. Provide the spin multiplicity and active space, e.g.
numl = 1 
nao = 6
nae = 6
# 3. Specify the VB structures for which you wish to predict the weights 
# (more detailed information please refer to XMVB manual)
structure=[[21  22  20  23  19  24],
          [20  21  22  23  19  24],
          [20  21  19  22  23  24],
          [19  20  22  23  21  24],
          [19  20  21  22  23  24],]
```
After defining the molecule, active space and structure set, you can run the following command to predict the VB structure weights
```
python main.py --cfg configs/predict/predict_weight.yaml
```
or run the following command to get the important VB structures that can be used to construct a compact VB wavefunction
```
python main.py --cfg configs/predict/select_structure.yaml
```    

      
      
      