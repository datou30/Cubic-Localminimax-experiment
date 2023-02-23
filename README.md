# Cubic-Localminimax-experiment
This repo is the Pytorch Implementation of paper "A Cubic Regularization Approach for Finding Local Minimax Points in Nonconvex Minimax Optimization". 

But this code is the first work of an undergraduate student, so the structure of code is a bit chaotic. I will modify it in about one week, and if you have any question of this code, please contact me at datou30@mail.ustc.edu.cn

To run this code, firstly run init_net.py to generate a initialize net, (since we need to make sure all the algorithms we compare are on the same initialization), then just run main.py and the MINST dataset we use will automatically installed "./data".

Currently it contrasts our Cubic-Localminimx algorithm with classical GDA, but in the paper I contrast with several other algorithms: IMCN, GDN, FR, CN, TGDA, later I will upload them.
