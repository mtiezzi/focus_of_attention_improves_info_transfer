# Focus of Attention Improves Information Transfer in Visual Features

This repo contains the code to reproduce the experimental results of our paper [Focus of Attention Improves Information Transfer in Visual Features](https://arxiv.org/abs/2006.09229), accepted for publication at NeuriPS2020.


[Technical report here](https://arxiv.org/abs/2006.092294). 

Authors: [Matteo Tiezzi](https://mtiezzi.github.io/), [Stefano Melacci](https://sailab.diism.unisi.it/people/stefano-melacci/), [Alessandro Betti](https://sailab.diism.unisi.it/people/alessandro-betti/), [Marco Gori](https://sailab.diism.unisi.it/people/marco-gori/)


## Requirements
To install the requirements, use the following code:

```
pip install -r requirements.txt
```

## How to Run an Experiment

The script `runner_mi.py` allows you to run a batch of experiments on a visual
stream of the paper using all the probability density functions (UNI,FOA,FOAW).

Parameters name are the same that can be found in the paper. You can specify,
as argument by command line, specific values for the hyperparameters
(`lambda_c, lambda_e, lambda_s, zeta_s`), choose the architecture (argument
`arch`, choice is among `S, D,DL`) and the video stream to be processed
(argument `-stream`, choice is among `sparsemnist,carpark,call`).

Moreover, by default the SOTA FOA predictor is used to predict the scanpath.
Otherwise, using the `-rnd` argument a random scanpath can be exploited.

The PyToch device is chosen through the `-device` argument (`"cpu", "cuda:0",
"cuda:1"`, etc.).

Usage:

```
    runner_mi.py [-h] [-device DEVICE] [-l_c LAMBDA_C] [-l_e LAMBDA_E]
                        [-l_s LAMBDA_S] [-z_s ZETA_S]
                        [-stream {sparsemnist,carpark,call}] [-arch {S,D,DL}]
                        [-rnd]


```
optional arguments:




```
    -h, --help            show this help message and exit
      -device DEVICE, --device DEVICE
                            device id
      -l_c LAMBDA_C, --lambda_c LAMBDA_C
                            Lambda_c value
      -l_e LAMBDA_E, --lambda_e LAMBDA_E
                            Lambda_e value
      -l_s LAMBDA_S, --lambda_s LAMBDA_S
                            Lambda_s value
      -z_s ZETA_S, --zeta_s ZETA_S
                            Zeta_s value
      -stream {sparsemnist,carpark,call}, 
                            Video stream
      -arch {S,D,DL}, --arch {S,D,DL}
                            architecture
      -rnd, --rnd           random foa scanpath flag
```


### GETTING THE MEASURED MUTUAL INFORMATION VALUES IN THE UNI,FOA,FOAW CASES

The statistics of each experiment of the batch are stored in a specific folder.
The runner creates several logs where the evaluation measures (and others)
are collected.

Roughly speaking, the log file from which results of the paper can be taken
is stored in the "test" subfolder of the main experiment folder. Then, diving
into the folder with the output data, the log folder can be found.

For example, the log folder of an experiment based on model with architecture
D and some parameter configuration is:

```
   exp_mi_D/..../test/output_sparse1_MIFOA_Layers7_Order2_Var_SeedA/logs
```

In detail, in what follows the notation `{var}` means that the values inside
the `{}` takes the values of the respective variable passed as arguments).

The folder `exp_mi_{arch}` (example: `exp_mi_D`) is created after launch,
containing the experiment folder:

```
   lambda_c_lambda_e_{l_c}-{l_e}_lambda_s_{zeta_s}_{l_s}-{z_s}_arch_{arch}_foa_{rnd}
```

For example:
```
   lambda_c_lambda_e_1000.0-2000.0_lambda_s_zeta_s_1000.0-0.01_arch_D_foa_regular
```
There are two subfolders, `train` and `test`, containing the logs related to
these two stages. In particular, they contain two sets of 9 folders, since an
experiment comprises 9 different combinations of spatio-temporal potentials.
As reported in the paper, the spatio-temporal density can be defined as
`[UNI, FOA, FOAW]`, whilst the temporal locality can be defined as `[PLA, AVG,
VAR]`. Hence, there are 9 possible combinations of potential definition.

The runner executes all the possible 9 combinations, producing two sets of
folders:
```
   model_* are the model files.
   output_* contains, in the subfolder "logs", all the measures 
```
Example: 
```
   model_sparse1_MIFOA_Layers7_Order2_Var_SeedA
   output_sparse1_MIFOA_Layers7_Order2_Var_SeedA
```
The "logs" folder contains two types of files:
```
   cal.{layerID}.txt
   worker.txt (containing FOA statistics and other stuff)
```
The `cal.{}.txt` files having the greater `layerID` is the one containing the
measures. These files are structured files (readable as CSV separated by
commas), with the following header:

```
frame,ca,cad,lagr,mi,ce,e,sdotp,coher,coherk,n2,n1,nm,n,f,sup,nsup,evalmi,
evalmis,evalmifoa,evalmisfoa,evalmifoaw,evalmisfoaw,evalgmi,evalgmis,evalgmiz,
evalgmisz,evalgmifoa,evalgmisfoa,evalgmifoaz,evalgmisfoaz,evalgmifoaw,
evalgmisfoaw,evalgmifoawz,evalgmisfoawz,evalgcoher,evalgcoherk,evalcohers,
evalgcohers,evalgcohersz,evalaccsup,evalsup
```
Each line contains several statistics computed for the current frame (indexed
in column `frame`).

The three measures that in the paper we call `[UNI, FOA, FOAW]` can be found in
columns : `["evalgmi", "evalgmifoa", "evalgmifoaw"]`. They contain the MI index
computed in the interval `[0,frame]`.

Summarizing, for an experiment launched on SparseMNIST video stream, with
`architecture D, lambda_c==1000.0, lambda_e==2000.0, lambda_s==1000.0,
zeta_s==0.01, regular foa`, the test stage logs of the model defined with the
spatio-temporal density `FOA` and `VAR` temporal locality can be found in:
```
   exp_mi_D/lambda_c_lambda_e_1000.0-2000.0_lambda_s_zeta_s_1000.0-0.01_arch_D_foa_regular/test/output_sparse1_MIFOA_Layers7_Order2_Var_SeedA/logs/cal.6.txt
```
